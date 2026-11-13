"""Artifact-free loader fixtures; these random tensors are not research weights."""
import hashlib
import pickle
from copy import deepcopy
import pytest
import torch
from torch.torch_version import TorchVersion
from synaptic_lens import checkpoints as c
from synaptic_lens.observer import Lens


def fixture_schema():
    rows=[]
    cursor=0
    for i in range(32):
        n=19627 if i==31 else 1
        rows.append({'parameter_name':f'p{i}','shape':[n],'raw_flat_index_range':[cursor,cursor+n],
                     'coordinate_count':n,'dtype':'torch.float32'})
        cursor+=n
    return {'parameter_count':19658,'parameters':rows}


def test_verified_bytes_rejects_hash_and_size(tmp_path):
    p=tmp_path/'bytes';p.write_bytes(b'abc'); sha=hashlib.sha256(b'abc').hexdigest()
    assert c.verified_bytes(p,sha)==b'abc'
    with pytest.raises(ValueError,match='mismatch'):c.verified_bytes(p,'0'*64)
    with pytest.raises(ValueError,match='size'):c.verified_bytes(p,sha,max_bytes=2)
    with pytest.raises(ValueError,match='hex'):c.verified_bytes(p,'bad')


def test_scoped_legacy_version(tmp_path):
    p=tmp_path/'target.pt';torch.save({'version':TorchVersion('2.14.0'),'x':torch.ones(2)},p)
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    before=torch.serialization.get_safe_globals()[:]
    with pytest.raises(pickle.UnpicklingError):c.load_verified(p,sha)
    assert str(c.load_verified(p,sha,legacy_torch_version=True)['version'])=='2.14.0'
    assert torch.serialization.get_safe_globals()==before


class UnreviewedCheckpointMetadata:
    pass

def test_unknown_global_remains_rejected(tmp_path):
    p=tmp_path/'bad.pt';torch.save({'x':UnreviewedCheckpointMetadata()},p)
    with pytest.raises(pickle.UnpicklingError):
        c.load_verified(p,hashlib.sha256(p.read_bytes()).hexdigest(),legacy_torch_version=True)


def test_ordered_weights_only_features():
    schema=fixture_schema();state={r['parameter_name']:torch.arange(r['coordinate_count'],dtype=torch.float32) for r in schema['parameters']}
    x,layout=c.flatten_target(state,schema)
    assert x.shape==(1,19658) and len(layout)==32
    assert torch.equal(x[0,-19627:],state['p31'])


@pytest.mark.parametrize('mutation',['gap','duplicate','shape','nonfinite','dtype','extra'])
def test_invalid_schema_or_tensor_rejected(mutation):
    schema=fixture_schema();state={r['parameter_name']:torch.zeros(r['shape']) for r in schema['parameters']}
    if mutation=='extra':state['extra']=torch.zeros(1)
    if mutation=='gap':schema['parameters'][1]['raw_flat_index_range'][0]=0
    if mutation=='duplicate':schema['parameters'][1]['parameter_name']='p0'
    if mutation=='shape':state['p0']=torch.zeros(2)
    if mutation=='nonfinite':state['p0'][0]=float('nan')
    if mutation=='dtype':state['p0']=state['p0'].double()
    with pytest.raises(ValueError):c.flatten_target(state,schema)


def test_observer_loader_strict_fixture(tmp_path,monkeypatch):
    model=Lens(192)
    bundle={'schema':'mnist-direct-weight-lens-bundle-v1','features':192,'state_dict':model.state_dict()}
    p=tmp_path/'fixture.pt';torch.save(bundle,p)
    monkeypatch.setitem(c.CHECKPOINTS,'mnist-statistics',hashlib.sha256(p.read_bytes()).hexdigest())
    assert c.load_milestone('mnist-statistics',p)['features']==192
    with pytest.raises(ValueError,match='Unknown'):c.load_milestone('unknown',p)


def test_mnist_end_to_end_tensor_fixture(tmp_path,monkeypatch):
    import json
    schema=fixture_schema();schema['schema_sha256']='synthetic-layout'
    sp=tmp_path/'schema.json';sp.write_text(json.dumps(schema))
    monkeypatch.setattr(c,'MNIST_SCHEMA_SHA256',hashlib.sha256(sp.read_bytes()).hexdigest())
    state={r['parameter_name']:torch.zeros(r['shape']) for r in schema['parameters']}
    tp=tmp_path/'target.pt'
    torch.save({'model_state':state,'coordinate_schema_sha256':'synthetic-layout'},tp)
    sha=hashlib.sha256(tp.read_bytes()).hexdigest()
    model=Lens(19658)
    bundle={'features':19658,'state_dict':model.state_dict(),'mean':torch.zeros(19658),
            'scale':torch.ones(19658),'y_mean':torch.tensor(1.),'y_scale':torch.tensor(1.)}
    assert isinstance(c.predict_mnist('mnist-raw',bundle,tp,sha,sp),float)
    with pytest.raises(ValueError,match='Unknown|Expected'):c.predict_mnist('other',bundle,tp,sha,sp)
    with pytest.raises(ValueError,match='mismatch'):c.predict_mnist('mnist-raw',bundle,tp,'0'*64,sp)


def test_syn53_model_fixture_load_and_predict(tmp_path,monkeypatch):
    from dataclasses import asdict
    from torch import nn
    from synaptic_lens.transformer_lens import DedicatedBoundaryReaderConfig,TransformerLens,LensQuery
    from synaptic_lens.target_records import affine_relu_records
    config=DedicatedBoundaryReaderConfig(width=8,heads=2,blocks=1,max_tokens=256)
    model=TransformerLens(config)
    p=tmp_path/'reader.pt';torch.save({'model_config':asdict(config),'model':model.state_dict()},p)
    monkeypatch.setitem(c.CHECKPOINTS,'syn53-regional',hashlib.sha256(p.read_bytes()).hexdigest())
    restored=c.load_milestone('syn53-regional',p)
    assert not restored.training and not any(p.requires_grad for p in restored.parameters())
    target=nn.Sequential(nn.Linear(20,2),nn.ReLU(),nn.Linear(2,19))
    with torch.inference_mode():
        response=restored.predict(restored.encode((affine_relu_records(target),)),LensQuery(torch.zeros(1,20)))
    assert response.status=='supported' and torch.isfinite(response.values).all()

def test_inference_export_wrong_source_rejected(monkeypatch):
    monkeypatch.setattr(c, 'load_verified', lambda *a, **kw: {'source_checkpoint_sha256': '0' * 64})
    with pytest.raises(ValueError, match='source identity'):
        c.load_milestone('syn53-control', 'unused', inference_export=True)
