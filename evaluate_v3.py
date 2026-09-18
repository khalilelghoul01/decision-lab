"""Post-selection calibration and untouched-test evaluation on the Colab GPU.

Prespecified one-order and two-order readouts share the same measured logits.
The training process receives neither this payload nor these test labels.
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from decision_v3 import ConfigV3,PolicyV3,load_v3,metrics,predictions

def derived(records,orders,temperature):
    output=[]
    for row in records:
        logits=torch.tensor(row['logits'][:orders],dtype=torch.float64)
        p=(logits/temperature).softmax(-1).mean(0).tolist()
        output.append({**row,'probabilities':p,'prediction':int(np.argmax(p))})
    return output

def calibrate(records):
    width=max(len(r['probabilities']) for r in records)
    logits=torch.full((len(records),2,width),-1e4,dtype=torch.float64)
    for i,r in enumerate(records):logits[i,:,:len(r['probabilities'])]=torch.tensor(r['logits'],dtype=torch.float64)
    target=torch.tensor([r['target'] for r in records]);log_t=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_t],lr=.2,max_iter=100,line_search_fn='strong_wolfe')
    def nll():
        logp=(logits/log_t.exp().clamp(.1,10)).log_softmax(-1)
        averaged=torch.logsumexp(logp,dim=1)-math.log(2)
        return -averaged[torch.arange(len(records)),target].mean()
    before=nll().item()
    def closure():
        optimizer.zero_grad();loss=nll();loss.backward();return loss
    optimizer.step(closure)
    temperature=log_t.exp().clamp(.1,10).item()
    return {'temperature':temperature,'calibration_examples':len(records),'objective':'Two-order multiclass negative log likelihood',
            'nll_before':before,'nll_after':nll().item(),'bounds':[.1,10]}

def save_report(root,tag,records,temperature):
    reports={}
    for orders,name in [(1,'fast'),(2,'accurate')]:
        rows=derived(records,orders,temperature);report=metrics(rows)
        report.update(orders=orders,temperature=temperature)
        (root/f'{tag}-{name}.json').write_text(json.dumps(report,indent=2))
        (root/f'{tag}-{name}.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
        reports[name]=report
        print(tag,name,'public',report['public_macro_accuracy'],'synthetic',report['synthetic_macro_accuracy'],flush=True)
    return reports

def known_suite(model,suite_path):
    cases=json.loads(Path(suite_path).read_text())['cases'];rows=[];refs=[]
    for case in cases:
        for field,spec in case['schema']['properties'].items():
            values=[False,True] if spec['type']=='boolean' else spec['enum']
            opts=['No','Yes'] if spec['type']=='boolean' else [str(x) for x in values]
            rows.append({'id':case['id']+':'+field,'task':case['group'],'context':case['context'],'question':spec['description'],
                         'options':opts,'target':values.index(case['expected'][field])})
            refs.append({'case':case['id'],'split':case['split'],'group':case['group'],'field':field,'values':values})
    out=predictions(model,rows,orders=2,batch_size=8,temperature=model.temperature)
    results=[]
    for prediction,reference in zip(out,refs):results.append({**prediction,**reference})
    groups={}
    for split in sorted({c['split'] for c in cases}):
        subset=[x for x in results if x['split']==split];by_case={}
        for r in subset:by_case.setdefault(r['case'],[]).append(r)
        groups[split]={'fields':len(subset),'correct_fields':sum(x['prediction']==x['target'] for x in subset),
                       'cases':len(by_case),'exact_cases':sum(all(x['prediction']==x['target'] for x in records) for records in by_case.values()),
                       'truncated_fields':sum(x['truncated'] for x in subset)}
    return {'scope':'Previously inspected regression suite; questions-only prompt, two orders, not a new test', 'splits':groups,'predictions':results}

def main(root,data_path,suite_path):
    root=Path(root)
    if not (root/'training-complete.json').exists():raise RuntimeError('Finish checkpoint selection before opening the test payload.')
    if (root/'final-summary.json').exists():raise FileExistsError('Final evaluation already exists; do not overwrite.')
    data=json.loads(Path(data_path).read_text());assert 'train' not in data
    torch.set_num_threads(2);model=load_v3(root,'cuda',attention='sdpa')
    start=time.monotonic()
    calibration=predictions(model,data['calibration'],orders=2,batch_size=8)
    fitted=calibrate(calibration);model.temperature=fitted['temperature']
    (root/'calibration.json').write_text(json.dumps(fitted,indent=2))
    (root/'calibration-logits.json').write_text(json.dumps(calibration))
    print('CALIBRATION',json.dumps(fitted),flush=True)
    final_records=predictions(model,data['test'],orders=2,batch_size=8,temperature=model.temperature)
    selected=save_report(root,'test',final_records,model.temperature)
    regression=known_suite(model,suite_path)
    (root/'jev-regression.json').write_text(json.dumps(regression,indent=2));print('JEV REGRESSION',json.dumps(regression['splits']),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
    cfg=ConfigV3(**json.loads((root/'config.json').read_text()))
    base=PolicyV3(cfg).to('cuda').eval()
    baseline=save_report(root,'base-test',predictions(base,data['test'],orders=2,batch_size=8),1.)
    summary={'selected':selected,'base':baseline,'calibration':fitted,'jev_regression':regression['splits'],
             'evaluation_seconds':time.monotonic()-start,'dataset_sha256':data['metadata']['full_dataset_sha256']}
    (root/'final-summary.json').write_text(json.dumps(summary,indent=2));print('FINAL EVALUATION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='run');p.add_argument('--data',default='evaluation-data.json');p.add_argument('--suite',default='jev-usecases-v2.json');a=p.parse_args()
    main(a.root,a.data,a.suite)
