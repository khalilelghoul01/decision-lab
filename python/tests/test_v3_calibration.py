import pytest
from decision_lab.evaluate import calibrate,derived
from decision_lab.train import metrics

def test_temperature_fit_reduces_nll_on_overconfident_errors():
    records=[{'id':str(i),'task':'synthetic_calibration' if i<20 else 'public','target':int(i%4==0),'prediction':0,
              'probabilities':[.999,.001],'logits':[[8.,0.],[8.,0.]],'truncated':False} for i in range(40)]
    result=calibrate(records)
    assert result['nll_after']<result['nll_before']
    assert result['temperature']>1
    assert all(r['prediction']==0 for r in derived(records,2,result['temperature']))

def test_ece_assigns_boundary_confidence_to_one_bin():
    records=[{'task':'public','probabilities':[.5,.5],'prediction':0,'target':0,'truncated':False},
             {'task':'synthetic_x','probabilities':[.5,.5],'prediction':0,'target':0,'truncated':False}]
    assert metrics(records)['overall']['ece']==pytest.approx(.5)
