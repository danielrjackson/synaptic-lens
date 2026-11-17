"""Validate the frozen static demo contract; no training or GPU required."""
import hashlib,json,math
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'docs/assets/no-sort-example.json'
d=json.loads(p.read_text(encoding='utf-8'))
assert d['schema']=='synaptic-lens-no-sort-example-v1'
assert d['target']=={'root':896,'rule':'addition'}
assert len(d['slots'])==32 and sorted(d['saved_permutation'])==list(range(32))
assert d['query_pairs']==[[a,b] for a in range(10) for b in range(10)]
assert d['original_answers']==[a+b for a,b in d['query_pairs']]
for i,slot in enumerate(d['slots']):
    assert slot['physical_slot']==i and slot['organic_neuron']==i
    scale=d['scales'][i]
    assert math.isfinite(scale) and scale>0
    actual=d['normalized_neurons'][i]
    original=d['original_neurons'][slot['organic_neuron']]
    assert len(actual)==len(original)==40
    assert max(abs((v*scale if j<21 else v/scale)-original[j]) for j,v in enumerate(actual))<1e-5
    assert set(slot['methods'])=={'zero','slot_mean','observed_mean','canonical','organic','unaugmented'}
    for name,m in slot['methods'].items():
        assert len(m['answers'])==100 and all(isinstance(x,int) and 0<=x<=18 for x in m['answers'])
        assert len(m['normalized_prediction'])==len(m['organic_prediction'])==40
        assert max(abs((v*scale if j<21 else v/scale)-m['organic_prediction'][j]) for j,v in enumerate(m['normalized_prediction']))<1e-5
        mse=sum((a-b)**2 for a,b in zip(actual,m['normalized_prediction']))/40
        assert abs(mse-m['normalized_neuron_mse'])<1e-5
        accuracy=sum(a==b for a,b in zip(m['answers'],d['original_answers']))/100
        assert abs(accuracy-m['native_agreement'])<1e-6
assert d['seed']==161120
assert 'canonical_to_organic' not in d
assert all(key.endswith('_seed_161120') for key in d['method_sources'].values() if '_seed_' in key)
print(json.dumps({'status':'passed','slots':32,'methods':6,'saved_answers':19200,'weight_values':7680,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
