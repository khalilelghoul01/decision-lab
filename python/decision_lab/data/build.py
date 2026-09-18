"""Freeze public + synthetic splits before training. Group splits by context.

Previously inspected V1/V2 material and the Jev regression suite cannot enter the
new test/calibration sets. Test labels are never used for checkpoint selection.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from datasets import load_dataset
from decision_lab.legacy.v1 import SOURCES as OLD_SOURCES, normalize
from decision_lab.data.synthetic import generate

ROOT=Path('runs/v3'); SEED=2026091803
SOURCES={**{k:(v[0],v[1],None) for k,v in OLD_SOURCES.items()},
 'banking77':('mteb/banking77','18072d2685ea682290f7b8924d94c62acc19c0b2',None),
 'emotion':('dair-ai/emotion','cab853a1dbdf4c42c2b3ef2173804746df8825fe','split'),
 'piqa':('nthngdy/piqa','467437b6dc793b01c07946dd3e800b9bd3199993',None),
 'absa':('tomaarsen/setfit-absa-semeval-restaurants','8885372fe73256f96bb60f65b550538ac5c26047',None)}

def key(text): return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()

def convert(task,row,index,rng,labels):
    if task in OLD_SOURCES: item=normalize(task,row,index)
    else:
        if task=='banking77':
            answer=row['label_text'].replace('_',' ');n=rng.choice([4,6,8])
            candidates=[x for x in labels if x!=answer]
            ranked=sorted(candidates,key=lambda x:len(set(x.split())&set(answer.split())),reverse=True)
            hard=ranked[:min(2,n-1)]
            opts=hard+rng.sample([x for x in candidates if x not in hard],n-1-len(hard))+[answer];rng.shuffle(opts)
            context=row['text'];question='Which intent best matches this customer message?';target=opts.index(answer)
        elif task=='emotion':
            context=row['text'];question='Which emotion is expressed most strongly?';opts=labels;target=int(row['label'])
        elif task=='piqa':
            context=row['goal'];question='Which solution is more likely to accomplish the stated goal?';opts=[row['sol1'],row['sol2']];target=int(row['label'])
        else:
            context=row['text'];question=f'What sentiment is expressed toward "{row["span"]}" specifically? Ignore sentiment toward other aspects.'
            opts=['negative','neutral','positive','conflict'];target=opts.index(row['label']) if row['label'] in opts else -1
        item={'id':f'{task}:{index}','task':task,'context':context,'question':question,'options':opts,'target':target}
    item['synthetic']=False;item['source']=SOURCES[task][0]
    return item

def build():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'data.json').exists(): raise FileExistsError('Frozen data already exists; do not overwrite it.')
    old_all=set();old_eval=set()
    for path in [Path('runs/decision-lab-first/data.json'),Path('runs/v2/data.json')]:
        if path.exists():
            d=json.loads(path.read_text())
            for split in ('train','dev','calibration','test'):
                for row in d.get(split,[]):
                    old_all.add(key(row['context']))
                    if split!='train':old_eval.add(key(row['context']))
    if Path('research/evals/v3-exclusion-contexts.json').exists():
        exclusions=json.loads(Path('research/evals/v3-exclusion-contexts.json').read_text())
        old_all.update(exclusions['all_previous']);old_eval.update(exclusions['previous_eval'])
    result={s:[] for s in ('train','dev','calibration','test')}
    used={s:set() for s in result}
    for task,(repo,rev,config) in SOURCES.items():
        print('Loading',task,flush=True);ds=load_dataset(repo,config,revision=rev)
        ds={k:v.add_column('__source_index',list(range(len(v)))) for k,v in ds.items()}
        labels=sorted({r['label_text'].replace('_',' ') for r in ds['train']}) if task=='banking77' else ds['train'].features['label'].names if task=='emotion' else []
        # This ABSA mirror's test labels are blank. Reserve labeled training
        # contexts instead and disclose that exception in the dataset manifest.
        held='train' if task=='absa' else 'test' if task in ('ag_news','snli','banking77','emotion') else 'validation'
        rng=random.Random(SEED+len(result['train']))
        def collect(split, source_split, count, exclusions):
            selected=[];seen=set()
            for idx,row in enumerate(ds[source_split].shuffle(seed=SEED+['test','calibration','dev','train'].index(split))):
                item=convert(task,row,f'{source_split}:{row["__source_index"]}:v3',rng,labels);k=key(item['context'])
                if item['target']<0 or k in exclusions or k in seen: continue
                item['evidence']={'method':'Original public dataset annotation; not an invented explanation',
                                  'dataset':repo,'revision':rev,'source_split':source_split,'source_row':row['__source_index'],
                                  'answer':item['options'][item['target']]}
                seen.add(k);selected.append(item)
                if len(selected)>=count:break
            if len(selected)<min(count,500):raise ValueError(f'Insufficient unique {task}/{split}: {len(selected)}')
            used[split]|=seen;result[split].extend(selected)
            print(task,split,len(selected),flush=True)
        collect('test',held,200,old_all|set.union(*used.values()))
        collect('calibration','train',100,old_all|set.union(*used.values()))
        collect('dev','train',75,old_all|set.union(*used.values()))
        wanted={'boolq':9000,'sst2':24000,'ag_news':36000,'snli':50000,'banking77':10000,'emotion':16000,'piqa':16000,'absa':4000}[task]
        collect('train','train',wanted,old_eval|set.union(*used.values()))
    synthetic=generate(SEED)
    removed={}
    for split in result:
        result[split].extend(synthetic[split]);random.Random(SEED).shuffle(result[split])
        unique={}
        for row in result[split]:
            prompt=(row['context'],row['question'],tuple(row['options']))
            if prompt in unique:assert unique[prompt]['target']==row['target'],'Conflicting annotations'
            else:unique[prompt]=row
        removed[split]=len(result[split])-len(unique);result[split]=list(unique.values())
        for row in result[split]:
            row['split']=split;row['context_group']=key(row['context']);row['answer']=row['options'][row['target']]
            row['decision_type']='noul' if row['options']==['No','Yes'] else 'choice'
            if row['options'] in (['None','Low','Medium','High'],['Negative','Neutral','Positive']):
                row['supported_readouts']=['choice','score'];row['score_target']=row['target']
            else:row['supported_readouts']=[row['decision_type']]
    keys={s:{key(r['context']) for r in rows} for s,rows in result.items()}
    for a in keys:
        for b in keys:
            if a!=b:assert not keys[a]&keys[b],f'Context leakage: {a}/{b}'
    assert not keys['test']&old_all
    result['metadata']={'seed':SEED,'sources':SOURCES,'synthetic':'v3_synthetic.py, split-specific templates/vocabulary; labels derived from stated facts',
       'absa_test_split':'Context-group holdout from labeled train; the source test annotations are blank and never used',
       'banking77_task':'Candidate selection among 4, 6, or 8 labels with two lexical hard negatives; not full 77-way accuracy',
       'exact_prompt_duplicates_removed':removed,
       'exclusion':'Normalized context SHA256; no V1/V2 previously seen examples in fresh test or calibration',
       'selection':'Development macro accuracy; calibration fits temperature only; test opened after selection',
       'counts':{s:dict(Counter(x['task'] for x in result[s])) for s in keys}}
    content=json.dumps(result,ensure_ascii=False)
    (ROOT/'data.json').write_text(content)
    (ROOT/'data.sha256').write_text(hashlib.sha256(content.encode()).hexdigest()+'\n')
    (ROOT/'data-manifest.json').write_text(json.dumps(result['metadata'],indent=2))
    # Train-side payload deliberately does not contain test or calibration labels.
    columns=['id','task','context','question','options','target','synthetic']
    training={s:[{k:r[k] for k in columns} for r in result[s]] for s in ('train','dev')}
    training['metadata']=result['metadata'];training['metadata']['full_dataset_sha256']=hashlib.sha256(content.encode()).hexdigest()
    (ROOT/'training-data.json').write_text(json.dumps(training,ensure_ascii=False))
    detailed=ROOT/'dataset';detailed.mkdir(exist_ok=True)
    for split in keys:
        with (detailed/f'{split}.jsonl').open('w') as f:
            for row in result[split]:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('Frozen:',{s:len(result[s]) for s in keys},flush=True)

if __name__=='__main__': build()
