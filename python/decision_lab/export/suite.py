"""Export known regression scenarios using the same typed API as the UI."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from decision_lab.engine import DecisionEngine
from decision_lab.protocol import system_one

def main(checkpoint,output):
    source=Path('research/evals/jev-usecases-v2.json');suite=json.loads(source.read_text())
    engine=DecisionEngine.load(checkpoint,precision='fp16');cases=[]
    model_name=getattr(engine.model,'model_name','decision-lab-v2')
    for case in suite['cases']:
        questions={}
        for name,spec in case['schema']['properties'].items():
            questions[name]={'type':'noul','instructions':spec['description']} if spec['type']=='boolean' else {
                'type':'choice','instructions':spec['description'],'criteria':{str(v):None for v in spec['enum']}}
        request={'model':model_name,'state':case['context'],'questions':questions}
        item={**case,'request':request,'native_value':{},'native_fields':{}}
        try:
            response=system_one(engine,request)
            for name,spec in case['schema']['properties'].items():
                answer=response['answers'][name];values=[False,True] if spec['type']=='boolean' else spec['enum']
                p=[1-answer['noul'],answer['noul']] if answer['type']=='noul' else [answer['probabilities'][str(v)] for v in values]
                item['native_value'][name]=values[max(range(len(p)),key=p.__getitem__)]
                item['native_fields'][name]={'confidence':max(p),'probabilities':[{'value':v,'probability':prob} for v,prob in zip(values,p)]}
        except ValueError as e:item['native_error']=str(e)
        cases.append(item)
    result={'suite_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'configuration':'system_one_two_orders',
            'reference':model_name+' native FP16 with shared cache','cases':cases,
            'scope':'Previously inspected regression cases; oversized inputs are rejected explicitly'}
    Path(output).write_text(json.dumps(result,indent=2));print('Exported',len(cases),'cases to',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/v3');p.add_argument('--output',default='browser/public/jev-suite-v3.json');a=p.parse_args();main(a.checkpoint,a.output)
