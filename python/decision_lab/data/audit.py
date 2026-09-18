"""Validate the detailed JSONL release using only Python's standard library.

Usage: python -m decision_lab.data.audit runs/v3/dataset
After extracting the release: python -m decision_lab.data.audit dataset
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

def context_hash(text):
    return hashlib.sha256(' '.join(text.lower().split()).encode()).hexdigest()

def audit(directory):
    directory=Path(directory)
    splits={};contexts={};all_ids=set()
    for split in ('train','dev','calibration','test'):
        path=directory/f'{split}.jsonl';tasks=Counter();sources=Counter();counts=Counter();prompts=set();groups=set()
        with path.open() as f:
            for line_number,line in enumerate(f,1):
                row=json.loads(line);where=f'{path.name}:{line_number}'
                assert row['split']==split,where+' wrong split'
                assert row['id'] not in all_ids,where+' duplicate ID'
                all_ids.add(row['id'])
                assert isinstance(row['context'],str) and row['context'].strip(),where+' empty context'
                assert isinstance(row['question'],str) and row['question'].strip(),where+' empty question'
                options=row['options'];target=row['target']
                assert 2<=len(options)<=8 and all(isinstance(x,str) and x for x in options),where+' invalid choices'
                assert len(set(options))==len(options),where+' duplicate choice text'
                assert isinstance(target,int) and 0<=target<len(options),where+' invalid target'
                assert row['answer']==options[target],where+' answer and target disagree'
                assert row['evidence']['answer']==row['answer'],where+' evidence and target disagree'
                group=context_hash(row['context'])
                assert row['context_group']==group,where+' invalid context hash'
                prompt=json.dumps([row['context'],row['question'],options],ensure_ascii=False)
                digest=hashlib.sha256(prompt.encode()).hexdigest()
                assert digest not in prompts,where+' duplicate prompt'
                prompts.add(digest);groups.add(group)
                assert row['decision_type'] in ('choice','noul'),where+' invalid primitive'
                if row['decision_type']=='noul':assert options==['No','Yes'],where+' invalid Noul mapping'
                if 'score' in row['supported_readouts']:assert row['score_target']==target,where+' invalid Score mapping'
                if row['synthetic']:
                    assert row['evidence']['facts'] and row['evidence']['rule_or_question'],where+' missing generation support'
                else:
                    evidence=row['evidence']
                    assert evidence['dataset']==row['source'] and len(evidence['revision'])==40,where+' missing pinned source'
                    assert isinstance(evidence['source_row'],int) and evidence['source_row']>=0,where+' missing source row'
                tasks[row['task']]+=1;sources[row['source']]+=1
                counts['synthetic' if row['synthetic'] else 'public']+=1
        splits[split]={'records':sum(tasks.values()),**counts,'tasks':dict(tasks),'sources':dict(sources),
                       'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        contexts[split]=groups
    overlaps={}
    names=list(contexts)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            overlaps[f'{a}/{b}']=len(contexts[a]&contexts[b])
            assert not overlaps[f'{a}/{b}'],f'Context leakage: {a}/{b}'
    return {'valid':True,'records':len(all_ids),'splits':splits,'normalized_context_overlaps':overlaps,
            'scope':'Structural, provenance and exact-overlap checks; not independent semantic relabeling or near-duplicate detection'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='runs/v3/dataset');p.add_argument('--output');a=p.parse_args()
    report=audit(a.directory)
    if a.output:Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
