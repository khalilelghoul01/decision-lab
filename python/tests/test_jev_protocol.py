import copy
import math
from types import SimpleNamespace
import pytest
from jsonschema import ValidationError
from decision_lab.protocol import compile_request, build_response, distribution_confidence, prepare_strict


@pytest.fixture
def payload():
    return {'model':'jev-latest','state':'The product is broken.', 'questions':{
        'department':{'type':'choice','instructions':'Which department?','criteria':{'billing':None,'technical':'Software failures','sales':'Purchase enquiries'}},
        'urgent':{'type':'noul','instructions':'Is this urgent?'},
        'severity':{'type':'score','instructions':'Rate severity','criteria':['Cosmetic','Degraded','Unavailable']}}}


def test_shapes_probability_semantics_and_score(payload):
    result=build_response(payload,[[.05,.9,.05],[.8,.2],[.1,.6,.3]],123)
    assert set(result)=={'model','answers','usage'}
    assert result['model']=='decision-lab-v2'
    assert result['answers']['urgent']=={'type':'noul','noul':.2}
    assert result['answers']['department']['choice']=='technical'
    assert result['answers']['severity']['score']==pytest.approx(1.2)
    assert result['answers']['severity']['legend']=={'0':'Cosmetic','1':'Degraded','2':'Unavailable'}
    assert result['usage']=={'input_tokens':123,'output_tokens':0}
    assert distribution_confidence([.5,.5])==pytest.approx(0)
    assert distribution_confidence([0,1])==pytest.approx(1)


def test_question_ids_never_enter_model_context(payload):
    renamed=copy.deepcopy(payload)
    renamed['questions']={f'ignore_rules_{i}':v for i,v in enumerate(payload['questions'].values())}
    assert compile_request(payload)==compile_request(renamed)
    assert all(row['context']==payload['state'] for row in compile_request(payload))


def test_criteria_and_structured_state_are_preserved(payload):
    payload['state']={'messages':[{'role':'user','content':'Please help'}]}
    payload['questions']['urgent']['criteria']={'false':'Routine payload','true':'Immediate deadline'}
    rows=compile_request(payload)
    assert rows[1]['options']==['No: Routine payload','Yes: Immediate deadline']
    assert rows[0]['options'][1]=='technical: Software failures'
    assert rows[2]['options']==['Cosmetic','Degraded','Unavailable']
    assert 'Please help' in rows[0]['context']


def test_unsupported_local_option_count_rejected(payload):
    payload['questions']['department']['criteria']={str(i):None for i in range(5)}
    with pytest.raises(ValueError,match='at most 4'):compile_request(payload)


def test_v3_eight_choice_and_score_readouts():
    req={'model':'decision-lab-v3','state':'x','questions':{'rating':{'type':'score','instructions':'Rate x','criteria':[str(i) for i in range(8)]}}}
    assert len(compile_request(req,8)[0]['options'])==8
    response=build_response(req,[[0,0,0,0,0,0,.25,.75]],20,'decision-lab-v3')
    assert response['model']=='decision-lab-v3'
    assert response['answers']['rating']['score']==pytest.approx(6.75)


@pytest.mark.parametrize('probabilities', [[[float('nan'),1]], [[-.2,1.2]], [[0,0]]])
def test_invalid_distributions_rejected(probabilities):
    req={'model':'decision-lab-v2','state':'x','questions':{'x':{'type':'noul','instructions':'x?'}}}
    with pytest.raises(ValueError):build_response(req,probabilities,2)


def test_overflow_rejected_instead_of_silently_losing_evidence():
    model=SimpleNamespace(cfg=SimpleNamespace(max_length=512),tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:list(range(513))))
    with pytest.raises(ValueError,match='No silent truncation'):
        prepare_strict(model,[{'context':'long','question':'q','options':['No','Yes']}],2)
