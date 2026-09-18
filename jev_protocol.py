"""Jev-compatible JSON shapes, backed by a local model, never Jev's API.

V2 supports 4 choices and 512 tokens; V3 supports 8 choices and 1024 tokens.
Confidence is explicitly
normalized inverse entropy; TypeSafe's exact confidence formula is not public.
"""
import json
import math
from pathlib import Path
from jsonschema import Draft202012Validator

from decision_engine import Prepared, aligned_probabilities

ROOT = Path(__file__).parent / 'protocol'
REQUEST = json.loads((ROOT/'jev-request.schema.json').read_text())
RESPONSE = json.loads((ROOT/'jev-response.schema.json').read_text())


def text(value):
    return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def compile_request(request, max_choices=4):
    Draft202012Validator(REQUEST).validate(request)
    state = text(request['state'])
    rows = []
    for question in request['questions'].values():
        kind = question['type']
        if kind=='choice':
            options = [key if description is None else f'{key}: {text(description)}' for key,description in question['criteria'].items()]
        elif kind=='score':
            options = [text(level) for level in question['criteria']]
        else:
            options = ['No','Yes'] if 'criteria' not in question else [f"No: {question['criteria']['false']}",f"Yes: {question['criteria']['true']}"]
        if len(options)>max_choices:raise ValueError(f'This loaded model supports at most {max_choices} choices')
        rows.append({'context':state,'question':text(question['instructions']),'options':options})
    return rows


def prepare_strict(model, rows, orders):
    sequences,sizes=[],[]
    for reverse in range(orders):
        for row in rows:
            options=list(reversed(row['options'])) if reverse else row['options']
            content='Context:\n'+row['context']+'\nQuestion: '+row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(options))+'\nReply with only the letter of the correct choice.'
            ids=model.tokenizer.apply_chat_template([{'role':'system','content':'You answer multiple-choice questions.'},{'role':'user','content':content}],add_generation_prompt=True,tokenize=True,return_dict=False)
            if len(ids)>model.cfg.max_length:
                raise ValueError(f'Request exceeds local {model.cfg.max_length}-token limit; shorten state or question. No silent truncation.')
            sequences.append(ids);sizes.append(len(options))
    prefix=0
    for tokens in zip(*sequences):
        if len(set(tokens))!=1:break
        prefix+=1
    return Prepared(sequences,sizes,len(rows),orders,min(prefix,min(map(len,sequences))-1))


def distribution_confidence(p):
    return min(1.,max(0.,1+sum(x*math.log(x) for x in p if x>0)/math.log(len(p))))


def build_response(request, probabilities, input_tokens, model_name='decision-lab-v2'):
    answers={}
    for (key,question),values in zip(request['questions'].items(),probabilities,strict=True):
        count=2 if question['type']=='noul' else len(question['criteria'])
        p=[float(x) for x in values[:count]]
        if len(p)!=count or any(not math.isfinite(x) or x<0 for x in p) or sum(p)<=0:
            raise ValueError('Invalid model probability distribution')
        total=sum(p);p=[x/total for x in p]
        if question['type']=='noul':
            answer={'type':'noul','noul':p[1]}
        else:
            keys=list(question['criteria']) if question['type']=='choice' else [str(i) for i in range(count)]
            answer={'type':question['type'],'probabilities':dict(zip(keys,p)), 'confidence':distribution_confidence(p)}
            if question['type']=='choice':answer['choice']=keys[max(range(count),key=p.__getitem__)]
            else:
                answer['score']=sum(i*v for i,v in enumerate(p))
                answer['legend']=dict(zip(keys,question['criteria']))
        answers[key]=answer
    result={'model':model_name,'answers':answers,'usage':{'input_tokens':input_tokens,'output_tokens':0}}
    Draft202012Validator(RESPONSE).validate(result)
    return result


def system_one(engine, request, fast=False):
    rows=compile_request(request,getattr(engine.model.cfg,'max_choices',4))
    prepared=prepare_strict(engine.model,rows,1 if fast else 2)
    p=aligned_probabilities(engine.logits(prepared),prepared,engine.model.temperature).cpu().tolist()
    prefix=prepared.prefix_length
    input_tokens=prefix+sum(len(s)-prefix for s in prepared.sequences)
    return build_response(request,p,input_tokens,getattr(engine.model,'model_name','decision-lab-v2'))


if __name__=='__main__':
    import argparse
    from decision_engine import DecisionEngine
    parser=argparse.ArgumentParser();parser.add_argument('request');parser.add_argument('--fast',action='store_true')
    parser.add_argument('--checkpoint',default='runs/v2',help='Local checkpoint directory, e.g. runs/v3')
    args=parser.parse_args()
    print(json.dumps(system_one(DecisionEngine.load(args.checkpoint,precision='fp16'),json.loads(Path(args.request).read_text()),args.fast),indent=2))
