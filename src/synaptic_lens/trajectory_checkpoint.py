"""Safe numeric-only SYN88 prototype loader; no artifact pickle imports."""
import hashlib
import io
import numpy as np
import torch
from .trajectory import OutputCentered, canonicalize, normalize_parts

def load_model(path, sha256):
    with open(path, "rb") as source:
        payload = source.read(8 * 1024 * 1024 + 1)
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError("Checkpoint exceeds inference size limit")
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError("Checkpoint SHA-256 mismatch")
    model = OutputCentered()
    with np.load(io.BytesIO(payload), allow_pickle=False) as f:
        if any(f[k].dtype.hasobject for k in f.files):
            raise ValueError('Object dtype prohibited')
        state = {k[6:]: torch.from_numpy(f[k].copy()) for k in f.files if k.startswith('model/')}
        norm = {k[5:]: torch.from_numpy(f[k].copy()) for k in f.files if k.startswith('norm/')}
        if len(state)+len(norm)!=len(f.files):
            raise ValueError('Unknown archive entries')
    model.load_state_dict(state, strict=True)
    expected = {'feature_mean':(21,), 'feature_scale':(21,), 'edge_mean':(1,), 'edge_scale':(1,), 'bias_mean':(1,), 'bias_scale':(1,), 'activation_mean':(1,), 'activation_scale':(1,)}
    if set(norm)!=set(expected): raise ValueError('Normalization keys')
    for k,shape in expected.items():
        if norm[k].shape!=shape or not torch.isfinite(norm[k]).all(): raise ValueError(k)
        if k.endswith('scale') and not (norm[k]>0).all(): raise ValueError(k)
    if any(not torch.isfinite(v).all() for v in state.values()): raise ValueError('Nonfinite model')
    return model.eval(), norm

@torch.no_grad()
def predict(model,norm,weights):
    weights = torch.as_tensor(weights, device='cpu')
    if weights.ndim < 1 or weights.shape[-1]!=1299: raise ValueError('Expected 1299 target coefficients')
    if weights.numel() == 0 or not torch.isfinite(weights).all(): raise ValueError('Empty or nonfinite target')
    pairs=torch.cartesian_prod(torch.arange(10),torch.arange(10))
    queries=torch.nn.functional.one_hot(pairs,num_classes=10).float().reshape(100,20)
    answers=pairs.sum(-1)
    flat=weights.reshape(-1,1299)
    ci,cb,ce,ob,act=canonicalize(flat,queries)
    parts={'incoming':ci,'hidden_bias':cb,'edges':ce,'output_bias':ob,'activation':act}
    features={k:v.float() for k,v in normalize_parts(parts,norm).items()}
    device=next(model.parameters()).device
    if device.type=='cuda' and torch.get_float32_matmul_precision()!='highest':
        raise ValueError('SYN88 requires highest float32 matmul precision for historical fidelity')
    features={k:v.to(device) for k,v in features.items()}
    queries=queries.to(device);answers=answers.to(device)
    result=[]
    for start in range(0,len(flat),8):
        stop=start+8; chunks=[]
        for lo in range(0,100,10):
            scores=model.score_outputs(features['neurons'][start:stop],features['edges'][start:stop],features['bias'][start:stop],queries[lo:lo+10],features['activation'][start:stop,lo:lo+10])
            idx=answers[lo:lo+10].view(1,-1,1).expand(scores.shape[0],-1,-1)
            chunks.append(scores.gather(-1,idx).squeeze(-1)-scores.scatter(-1,idx,float('-inf')).amax(-1))
        result.append(torch.cat(chunks,-1))
    return torch.cat(result).cpu().numpy().reshape(*weights.shape[:-1],100)
