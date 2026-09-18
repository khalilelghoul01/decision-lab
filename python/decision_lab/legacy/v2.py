"""V2: preserve pretrained answer semantics, then learn and validate decisions.

Uses the same 230M backbone. Selection uses development accuracy only. Final
evaluation excludes every test example examined in V1. Confidence is experimental.
"""
from __future__ import annotations
import dataclasses
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from decision_lab.legacy.v1 import (MODEL, REVISION, SOURCES, LETTERS, N_BINS, normalize,
    example_key, device_for, seed_all, trainable_state, load_trainable,
    save_checkpoint, sampled_rl_loss, warmup_loss, metrics)


@dataclass
class ConfigV2:
    output: str = "runs/v2"
    previous_data: str = "runs/decision-lab-first/data.json"
    seed: int = 20260918
    max_length: int = 512
    batch_size: int = 4
    accumulate: int = 4
    train_per_task: int = 4000
    eval_per_task: int = 200
    warmup_steps: int = 600
    rl_steps: int = 120
    group_size: int = 32
    adapter_lr: float = 8e-5
    head_lr: float = 2e-5
    confidence_lr: float = 1e-4
    rl_lr_scale: float = 0.15
    entropy_weight: float = 0.001
    lora_rank: int = 8
    checkpoint_every: int = 50
    eval_every: int = 100
    log_every: int = 25
    max_minutes: float = 150
    device: str = "auto"


def prepare_v2(cfg):
    from datasets import load_dataset
    root=Path(cfg.output); root.mkdir(parents=True,exist_ok=True)
    cached=root/'data.json'
    if cached.exists():
        return json.loads(cached.read_text())
    old=json.loads(Path(cfg.previous_data).read_text())
    # The original development set stays a development set. Never train on it.
    dev=old['dev']
    old_seen={example_key(x) for s in ('train','dev','test') for x in old[s]}
    old_test={example_key(x) for x in old['test']}
    excluded_train={example_key(x) for s in ('dev','test') for x in old[s]}
    data={'train':[], 'dev':dev, 'calibration':[], 'test':[], 'sources':SOURCES,
          'old_test_keys':sorted(old_test), 'seed':cfg.seed}
    for task,(name,revision) in SOURCES.items():
        ds=load_dataset(name,revision=revision)
        held='test' if task in ('ag_news','snli') else 'validation'
        seen=set(old_seen)
        task_test=[]
        for i,row in enumerate(ds[held].shuffle(seed=cfg.seed)):
            item=normalize(task,row,f'{held}:{i}:v2')
            key=example_key(item)
            if item['target']<0 or key in seen:continue
            seen.add(key);task_test.append(item)
            if len(task_test)==cfg.eval_per_task:break
        assert len(task_test)==cfg.eval_per_task
        data['test'].extend(task_test)
        # Calibration sample is fresh even relative to V1 training.
        for i,row in enumerate(ds['train'].shuffle(seed=cfg.seed+1)):
            item=normalize(task,row,f'train:{i}:calibration')
            key=example_key(item)
            if item['target']<0 or key in seen:continue
            seen.add(key);data['calibration'].append(item)
            if sum(x['task']==task for x in data['calibration'])==cfg.eval_per_task:break
        blocked=excluded_train | {example_key(x) for s in ('test','calibration') for x in data[s]}
        count=0
        for i,row in enumerate(ds['train'].shuffle(seed=cfg.seed+2)):
            item=normalize(task,row,f'train:{i}:v2');key=example_key(item)
            if item['target']<0 or key in blocked:continue
            blocked.add(key);data['train'].append(item);count+=1
            if count==cfg.train_per_task:break
        assert count==cfg.train_per_task
    for split in ('train','dev','calibration','test'):
        random.Random(cfg.seed).shuffle(data[split])
    keys={s:set(map(example_key,data[s])) for s in ('train','dev','calibration','test')}
    for a in keys:
        for b in keys:
            if a!=b:assert not keys[a]&keys[b],f'Overlap {a}/{b}'
    assert not keys['test']&old_seen
    cached.write_text(json.dumps(data,ensure_ascii=False))
    print('V2 split sizes:',{s:len(data[s]) for s in keys},flush=True)
    return data


class PolicyV2(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from transformers import AutoModel,AutoTokenizer
        from peft import LoraConfig,get_peft_model
        self.cfg=cfg
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
        self.tokenizer.padding_side='right'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        base=AutoModel.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32,attn_implementation='eager')
        assert base.config.tie_word_embeddings, 'Read the actual LM head if this model stops tying weights.'
        base.config.use_cache=False
        token_ids=[self.tokenizer.encode(letter,add_special_tokens=False) for letter in LETTERS]
        assert all(len(ids)==1 for ids in token_ids)
        weights=base.get_input_embeddings().weight[[ids[0] for ids in token_ids]].detach().clone()
        self.backbone=get_peft_model(base,LoraConfig(r=cfg.lora_rank,lora_alpha=2*cfg.lora_rank,
            lora_dropout=0.0,target_modules='all-linear',bias='none'))
        self.choice=nn.Linear(base.config.hidden_size,4,bias=False)
        with torch.no_grad():self.choice.weight.copy_(weights)
        self.confidence=nn.Linear(base.config.hidden_size,4*N_BINS)
        nn.init.zeros_(self.confidence.weight);nn.init.zeros_(self.confidence.bias)
        self.register_buffer('bins',torch.linspace(0,1,N_BINS))
        self.temperature=1.0

    def encode(self,rows,rng=None):
        sequences,targets,sizes=[],[],[]
        tok=self.tokenizer
        for row in rows:
            order=list(range(len(row['options'])))
            if rng is not None:rng.shuffle(order)
            options=[row['options'][i] for i in order]
            targets.append(order.index(row.get('target',0)));sizes.append(len(options))
            prefix='Context:\n'
            suffix='\nQuestion: '+row['question']+'\n'+'\n'.join(f'{LETTERS[i]}. {s}' for i,s in enumerate(options))+'\nReply with only the letter of the correct choice.'
            # Build the actual model chat template; reserve the complete question/options.
            def format_context(context):
                return tok.apply_chat_template([{'role':'system','content':'You answer multiple-choice questions.'},
                    {'role':'user','content':prefix+context+suffix}],add_generation_prompt=True,tokenize=True,return_dict=False)
            overhead=len(format_context(''))
            budget=self.cfg.max_length-overhead-4
            if budget<8:raise ValueError('Question/options exceed context budget')
            context_ids=tok.encode(row['context'],add_special_tokens=False)[:budget]
            ids=format_context(tok.decode(context_ids,skip_special_tokens=True))
            while len(ids)>self.cfg.max_length:
                context_ids=context_ids[:-(len(ids)-self.cfg.max_length+2)]
                ids=format_context(tok.decode(context_ids,skip_special_tokens=True))
            sequences.append(ids)
        padded=tok.pad({'input_ids':sequences},padding=True,return_tensors='pt')
        return {**padded,'targets':torch.tensor(targets),'sizes':torch.tensor(sizes)}

    def forward(self,input_ids,attention_mask,sizes,**ignored):
        h=self.backbone(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state
        pooled=h[torch.arange(len(h),device=h.device),attention_mask.sum(-1)-1].float()
        logits=self.choice(pooled)
        logits=logits.masked_fill(torch.arange(4,device=h.device)[None]>=sizes[:,None],-1e4)
        raw=self.confidence(pooled).reshape(-1,4,N_BINS)
        # Initialize the report policy near pretrained choice probabilities.
        p=logits.detach().softmax(-1)
        reports=raw-(self.bins[None,None]-p[:,:,None]).square()/(2*0.15**2)
        return logits,reports


@torch.no_grad()
def predict_rows(model,rows,orders=2,temperature=None):
    """Average original and reversed option orders, aligned by semantic choice."""
    model.eval();device=next(model.parameters()).device
    probs=[];reports=[]
    temp=model.temperature if temperature is None else temperature
    for reverse in range(orders):
        mapped=[]
        for row in rows:
            x=dict(row)
            if reverse:
                x['options']=list(reversed(row['options']))
                x['target']=len(x['options'])-1-row.get('target',0)
            mapped.append(x)
        batch={k:v.to(device) for k,v in model.encode(mapped).items()}
        logits,confidence=model(**batch)
        p=(logits/temp).softmax(-1)
        q=(confidence.softmax(-1)*model.bins).sum(-1)
        if reverse:
            for i,row in enumerate(rows):
                size=len(row['options']);p[i,:size]=p[i,:size].flip(0);q[i,:size]=q[i,:size].flip(0)
        probs.append(p);reports.append(q)
    return torch.stack(probs).mean(0),torch.stack(reports).mean(0)


def records_metrics(records):
    return {'overall':metrics(records),'tasks':{task:metrics([r for r in records if r['task']==task]) for task in SOURCES}}


@torch.no_grad()
def evaluate_v2(model,rows,cfg,tag,orders=2,write=True):
    output=[];start=time.monotonic()
    for offset in range(0,len(rows),cfg.batch_size):
        chunk=rows[offset:offset+cfg.batch_size]
        p,q=predict_rows(model,chunk,orders)
        p=p.cpu();q=q.cpu()
        for i,row in enumerate(chunk):
            a=p[i].argmax().item()
            output.append({'id':row['id'],'task':row['task'],'prediction':a,'target':row['target'],
                'correct':int(a==row['target']), 'confidence':p[i,a].item(),
                'rl_confidence':q[i,a].item(),'policy_probability':p[i,a].item(),
                'choice_nll':-p[i,row['target']].clamp_min(1e-9).log().item()})
    result=records_metrics(output);result['seconds']=time.monotonic()-start;result['orders']=orders
    result['confidence_readout']='temperature-scaled choice probability; RL report saved separately'
    if write:
        root=Path(cfg.output)
        (root/f'{tag}.json').write_text(json.dumps(result,indent=2))
        (root/f'{tag}.jsonl').write_text('\n'.join(map(json.dumps,output)))
    print(tag,json.dumps({k:v for k,v in result['overall'].items() if k!='reliability'}),flush=True)
    return result


def _restore_rng(saved,rng,device):
    rng.setstate(saved['batch_rng']);random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
    if saved.get('cuda_rng') is not None and device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if saved.get('mps_rng') is not None and device.type=='mps':torch.mps.set_rng_state(saved['mps_rng'])


def fit(cfg):
    seed_all(cfg.seed);root=Path(cfg.output);root.mkdir(parents=True,exist_ok=True)
    conf=dataclasses.asdict(cfg)
    if (root/'config.json').exists():
        old=json.loads((root/'config.json').read_text())
        for k in conf.keys()-{'output','previous_data','device','max_minutes'}:
            assert old[k]==conf[k],f'Changed {k}; choose a new output folder'
    (root/'config.json').write_text(json.dumps(conf,indent=2))
    data=prepare_v2(cfg);device=device_for(cfg.device)
    model=PolicyV2(cfg).to(device)
    optimizer=torch.optim.AdamW([
        {'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')],'lr':cfg.adapter_lr},
        {'params':list(model.choice.parameters()),'lr':cfg.head_lr},
        {'params':list(model.confidence.parameters()),'lr':cfg.confidence_lr}],weight_decay=0.01)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    rng=random.Random(cfg.seed+5);stage='warmup';done=0
    latest=root/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        load_trainable(model,saved['trainable']);optimizer.load_state_dict(saved['optimizer'])
        _restore_rng(saved,rng,device);stage=saved['stage'];done=saved['step']
        print('Resume',stage,done,flush=True)
    import importlib.metadata
    env={'device':str(device),'model':MODEL,'revision':REVISION,
         'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
         'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','datasets')}}
    (root/'environment.json').write_text(json.dumps(env,indent=2));print(env,flush=True)
    if stage=='complete':
        load_trainable(model,torch.load(root/'best.pt',map_location='cpu',weights_only=False)['trainable'])
        return model
    if not (root/'best-dev.json').exists():
        score=evaluate_v2(model,data['dev'],cfg,'pretrained-dev')
        save_checkpoint(model,optimizer,scaler,cfg,'warmup',0,rng,'best.pt')
        (root/'best-dev.json').write_text(json.dumps({'stage':'pretrained','step':0,'accuracy':score['overall']['accuracy'],'brier':score['overall']['confidence_brier']}))
    start=time.monotonic()
    for phase,total in [('warmup',cfg.warmup_steps),('rl',cfg.rl_steps)]:
        if phase=='warmup' and stage=='rl':continue
        first=done if phase==stage else 0
        if phase=='rl' and first==0:
            # Start RL from the best development-selected supervised policy.
            chosen=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
            load_trainable(model,chosen['trainable'])
            optimizer.state.clear()
        scale=cfg.rl_lr_scale if phase=='rl' else 1.0
        for group,lr in zip(optimizer.param_groups,[cfg.adapter_lr,cfg.head_lr,cfg.confidence_lr]):group['lr']=lr*scale
        model.train()
        for step in range(first,total):
            optimizer.zero_grad(set_to_none=True);loss_value=0.;stats={}
            for micro in range(cfg.accumulate):
                rows=[data['train'][rng.randrange(len(data['train']))] for _ in range(cfg.batch_size)]
                batch={k:v.to(device) for k,v in model.encode(rows,rng).items()}
                logits,reports=model(**batch)
                if phase=='warmup':loss=warmup_loss(logits,reports,batch['targets'],model.bins)
                else:loss,stats=sampled_rl_loss(logits,reports,batch['targets'],model.bins,cfg.group_size,cfg.entropy_weight)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss; resume latest.pt')
                (loss/cfg.accumulate).backward();loss_value+=loss.item()/cfg.accumulate
            norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();n=step+1
            if n==1 or n%cfg.log_every==0:
                entry={'stage':phase,'step':n,'total':total,'loss':loss_value,'minutes':(time.monotonic()-start)/60,**stats}
                print(json.dumps(entry),flush=True)
                with (root/'training.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
            if n%cfg.checkpoint_every==0 or n==total:save_checkpoint(model,optimizer,scaler,cfg,phase,n,rng)
            if n%cfg.eval_every==0 or n==total:
                result=evaluate_v2(model,data['dev'],cfg,f'{phase}-{n}-dev')
                prev=json.loads((root/'best-dev.json').read_text())
                candidate=(result['overall']['accuracy'],-result['overall']['confidence_brier'])
                if candidate>(prev['accuracy'],-prev['brier']):
                    save_checkpoint(model,optimizer,scaler,cfg,phase,n,rng,'best.pt')
                    (root/'best-dev.json').write_text(json.dumps({'stage':phase,'step':n,'accuracy':candidate[0],'brier':-candidate[1]}))
                    print('New best development policy:',candidate,flush=True)
                model.train()
            if (time.monotonic()-start)/60>=cfg.max_minutes:
                save_checkpoint(model,optimizer,scaler,cfg,phase,n,rng)
                print('Time budget reached. Rerun fit(cfg) to resume.',flush=True);return model
        save_checkpoint(model,optimizer,scaler,cfg,phase,total,rng,f'{phase}.pt')
        if phase=='warmup':
            chosen=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
            torch.save(chosen,root/'best-warmup.pt')
            stage,done='rl',0
            save_checkpoint(model,optimizer,scaler,cfg,'rl',0,rng)
    save_checkpoint(model,optimizer,scaler,cfg,'complete',cfg.rl_steps,rng)
    chosen=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    load_trainable(model,chosen['trainable'])
    print('Training complete. Selected by development data:',json.loads((root/'best-dev.json').read_text()),flush=True)
    print('Final test remains unused. Run finalize(cfg) once after freezing the design.',flush=True)
    return model.eval()


def load_v2(path,device='auto'):
    root=Path(path);cfg=ConfigV2(**json.loads((root/'config.json').read_text()))
    model=PolicyV2(cfg).to(device_for(device))
    saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    load_trainable(model,saved['trainable'])
    if (root/'calibration.json').exists():model.temperature=json.loads((root/'calibration.json').read_text())['temperature']
    return model.eval()


@torch.no_grad()
def calibrate(model,rows,cfg):
    # Cache unscaled choice logits for both deployed option orders. Optimize only T.
    logits_all=[];labels=[];device=next(model.parameters()).device
    for offset in range(0,len(rows),cfg.batch_size):
        chunk=rows[offset:offset+cfg.batch_size];both=[]
        for reverse in (False,True):
            mapped=[]
            for x in chunk:
                item=dict(x)
                if reverse:item['options']=list(reversed(x['options']));item['target']=len(x['options'])-1-x['target']
                mapped.append(item)
            batch={k:v.to(device) for k,v in model.encode(mapped).items()}
            logits,_=model(**batch)
            if reverse:
                for i,x in enumerate(chunk):logits[i,:len(x['options'])]=logits[i,:len(x['options'])].flip(0)
            both.append(logits.cpu())
        logits_all.append(torch.stack(both,1));labels.extend(x['target'] for x in chunk)
    logits=torch.cat(logits_all);targets=torch.tensor(labels)
    candidates=torch.logspace(math.log10(0.4),math.log10(4),81)
    losses=[]
    for temp in candidates:
        probabilities=(logits/temp).softmax(-1).mean(1)
        losses.append(F.nll_loss(probabilities.clamp_min(1e-9).log(),targets).item())
    temperature=candidates[int(np.argmin(losses))].item()
    model.temperature=temperature
    (Path(cfg.output)/'calibration.json').write_text(json.dumps({'temperature':temperature,'n':len(rows),'method':'one scalar temperature chosen on separate calibration split by choice NLL'}))
    print('Calibration temperature',temperature,flush=True)
    return temperature


def finalize(cfg,old_checkpoint=None):
    root=Path(cfg.output)
    saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
    assert saved['stage']=='complete','Finish training before final evaluation'
    data=json.loads((root/'data.json').read_text());model=load_v2(root,cfg.device)
    if not (root/'calibration.json').exists():
        calibrate(model,data['calibration'],cfg)
    # Freeze the selected model and temperature before touching final labels.
    if (root/'final-test.json').exists():
        print('Reusing saved final results; completing any missing exports/checks.',flush=True)
        result=json.loads((root/'final-test.json').read_text())
    else:
        result=evaluate_v2(model,data['test'],cfg,'final-test')
    if not (root/'final-fast-test.json').exists():
        evaluate_v2(model,data['test'],cfg,'final-fast-test',orders=1)
    if not (root/'final-permuted-test.json').exists():
        permuted=[];rng=random.Random(9401)
        for row in data['test']:
            order=list(range(len(row['options'])));rng.shuffle(order)
            permuted.append({**row,'options':[row['options'][i] for i in order],'target':order.index(row['target'])})
        evaluate_v2(model,permuted,cfg,'final-permuted-test')
    counts={task:Counter(x['target'] for x in data['train'] if x['task']==task).most_common(1)[0][0] for task in SOURCES}
    baseline=sum(x['target']==counts[x['task']] for x in data['test'])/len(data['test'])
    (root/'majority-baseline.json').write_text(json.dumps({'accuracy':baseline,'n':len(data['test'])}))
    model.backbone.save_pretrained(root/'adapter',safe_serialization=True)
    model.tokenizer.save_pretrained(root/'adapter')
    if old_checkpoint and not (root/'v1-fresh-test.json').exists():
        from decision_lab.legacy.v1 import load_run,evaluate,Config
        old=load_run(old_checkpoint,cfg.device)
        old_cfg=Config(**json.loads((Path(old_checkpoint)/'config.json').read_text()))
        old_cfg.output=str(root)
        evaluate(old,data['test'],old_cfg,'v1-fresh','test')
        del old
    print('Fresh final accuracy:',result['overall']['accuracy'],'majority:',baseline,flush=True)
    return result


@torch.no_grad()
def decide_v2(model,context,questions,fast=False):
    if not questions:return []
    for q in questions:
        if not 2<=len(q['choices'])<=4 or len(set(q['choices']))!=len(q['choices']):raise ValueError('Use 2-4 distinct choices')
    rows=[{'context':context,'question':q['question'],'options':q['choices']} for q in questions]
    p,q=predict_rows(model,rows,orders=1 if fast else 2)
    output=[]
    for i,row in enumerate(rows):
        a=p[i].argmax().item()
        output.append({'type':'choice','question':row['question'],'choice':row['options'][a],
            'confidence':p[i,a].item(),'rl_confidence':q[i,a].item(),
            'probabilities':dict(zip(row['options'],p[i,:len(row['options'])].tolist()))})
    return output


def benchmark(model,iterations=30):
    device=next(model.parameters()).device
    def sync():
        if device.type=='cuda':torch.cuda.synchronize()
        elif device.type=='mps':torch.mps.synchronize()
    context='The film was charming and beautifully acted.'
    questions=[{'question':'What is the sentiment of this review?','choices':['Negative','Positive']}]
    output={}
    for fast in (True,False):
        for _ in range(5):decide_v2(model,context,questions,fast)
        times=[]
        for _ in range(iterations):
            sync();start=time.perf_counter();decide_v2(model,context,questions,fast);sync()
            times.append((time.perf_counter()-start)*1000)
        output['fast' if fast else 'accurate']={'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
            'iterations':iterations,'scope':'one short question; tokenization and forward passes; warmed model'}
    output['device']=str(device)
    return output
