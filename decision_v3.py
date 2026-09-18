"""Small non-generative decision model: LoRA, constrained logits, calibration.

Train only on training-data.json. No final-test access during model selection.
Eight pretrained A-H token rows are the only output projection computed.
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODEL='LiquidAI/LFM2.5-350M'
REVISION='9e6c6ccf47cd318696e137d381a7ded8fe4df09f'
LETTERS='ABCDEFGH'

@dataclass
class ConfigV3:
    model: str=MODEL
    revision: str=REVISION
    output: str='runs/v3'
    seed: int=2026091803
    max_length: int=1024
    max_choices: int=8
    batch_size: int=8
    accumulate: int=4
    steps: int=1800
    lora_rank: int=16
    adapter_lr: float=1e-4
    head_lr: float=2e-5
    eval_every: int=200
    save_every: int=100
    log_every: int=10
    max_minutes: float=150
    device: str='cuda'
    attention: str='sdpa'

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

class PolicyV3(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        from transformers import AutoModel,AutoTokenizer
        from peft import LoraConfig,get_peft_model
        self.cfg=cfg;self.temperature=1.;self.model_name='decision-lab-v3'
        self.tokenizer=AutoTokenizer.from_pretrained(cfg.model,revision=cfg.revision)
        self.tokenizer.padding_side='right'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        base=AutoModel.from_pretrained(cfg.model,revision=cfg.revision,dtype=torch.float32,attn_implementation=cfg.attention)
        assert base.config.tie_word_embeddings
        base.config.use_cache=False
        token_ids=[self.tokenizer.encode(x,add_special_tokens=False) for x in LETTERS]
        assert all(len(x)==1 for x in token_ids)
        weights=base.get_input_embeddings().weight[[x[0] for x in token_ids]].detach().clone()
        self.backbone=get_peft_model(base,LoraConfig(r=cfg.lora_rank,lora_alpha=2*cfg.lora_rank,lora_dropout=0.,target_modules='all-linear',bias='none'))
        self.choice=nn.Linear(base.config.hidden_size,cfg.max_choices,bias=False)
        with torch.no_grad():self.choice.weight.copy_(weights)

    def chat(self,context,question,options):
        content='Context:\n'+context+'\nQuestion: '+question+'\n'+'\n'.join(f'{LETTERS[i]}. {s}' for i,s in enumerate(options))+'\nReply with only the letter of the correct choice.'
        return self.tokenizer.apply_chat_template([{'role':'system','content':'You answer multiple-choice questions.'},{'role':'user','content':content}],add_generation_prompt=True,tokenize=True,return_dict=False)

    def encode(self,rows,rng=None):
        sequences=[];targets=[];sizes=[];truncated=[]
        for row in rows:
            order=list(range(len(row['options'])))
            if rng:rng.shuffle(order)
            opts=[row['options'][i] for i in order]
            assert 2<=len(opts)<=self.cfg.max_choices
            targets.append(order.index(row.get('target',0)));sizes.append(len(opts))
            ids=self.chat(row['context'],row['question'],opts);cut=len(ids)>self.cfg.max_length
            if cut:
                budget=self.cfg.max_length-len(self.chat('',row['question'],opts))-4
                if budget<8:raise ValueError('Question/options exceed token budget')
                ctx=self.tokenizer.encode(row['context'],add_special_tokens=False)[:budget]
                ids=self.chat(self.tokenizer.decode(ctx,skip_special_tokens=True),row['question'],opts)
                while len(ids)>self.cfg.max_length:
                    ctx=ctx[:-(len(ids)-self.cfg.max_length+2)]
                    ids=self.chat(self.tokenizer.decode(ctx,skip_special_tokens=True),row['question'],opts)
            sequences.append(ids);truncated.append(cut)
        result=self.tokenizer.pad({'input_ids':sequences},padding=True,return_tensors='pt')
        return {**result,'targets':torch.tensor(targets),'sizes':torch.tensor(sizes),'truncated':torch.tensor(truncated)}

    def forward(self,input_ids,attention_mask,sizes,**ignored):
        h=self.backbone(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state
        pooled=h[torch.arange(len(h),device=h.device),attention_mask.sum(-1)-1].float()
        with torch.autocast(device_type=h.device.type,enabled=False):
            logits=self.choice(pooled)
        return logits.masked_fill(torch.arange(self.cfg.max_choices,device=h.device)[None]>=sizes[:,None],-1e4)

def trainable_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def restore(model,state):
    params=dict(model.named_parameters())
    with torch.no_grad():
        for n,value in state.items():params[n].copy_(value.to(params[n]))

def atomic_save(value,path):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

@torch.inference_mode()
def predictions(model,rows,orders=1,batch_size=8,temperature=1.):
    model.eval();device=next(model.parameters()).device
    result=[]
    for start in range(0,len(rows),batch_size):
        chunk=rows[start:start+batch_size];all_logits=[];cuts=None
        for reverse in range(orders):
            mapped=[{**x,'options':list(reversed(x['options'])),'target':len(x['options'])-1-x.get('target',0)} if reverse else x for x in chunk]
            batch=model.encode(mapped);cuts=batch.pop('truncated').tolist();batch={k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(**batch).float().cpu()
            if reverse:
                for j,row in enumerate(chunk):logits[j,:len(row['options'])]=logits[j,:len(row['options'])].flip(0)
            all_logits.append(logits)
        stack=torch.stack(all_logits)
        probs=(stack/temperature).softmax(-1).mean(0)
        for j,row in enumerate(chunk):
            n=len(row['options']);p=probs[j,:n].tolist()
            result.append({'id':row['id'],'task':row['task'],'target':row['target'],'prediction':int(np.argmax(p)),
                           'probabilities':p,'logits':stack[:,j,:n].tolist(),'truncated':cuts[j]})
    return result

def metrics(records):
    groups=defaultdict(list)
    for r in records:groups[r['task']].append(r)
    def stats(items):
        correct=np.array([x['prediction']==x['target'] for x in items]);conf=np.array([max(x['probabilities']) for x in items])
        ece=0.
        for low in np.arange(0,1,.1):
            take=(conf>=low)&((conf<low+.1) if low<.9 else (conf<=1))
            if take.any():ece+=take.mean()*abs(correct[take].mean()-conf[take].mean())
        return {'count':len(items),'accuracy':float(correct.mean()),'nll':float(np.mean([-math.log(max(x['probabilities'][x['target']],1e-9)) for x in items])),
                'brier':float(np.mean([sum((p-(j==x['target']))**2 for j,p in enumerate(x['probabilities'])) for x in items])),
                'ece':float(ece),'high_confidence_errors':int(sum(not a and p>=.9 for a,p in zip(correct,conf))),
                'truncated':sum(x['truncated'] for x in items)}
    tasks={k:stats(v) for k,v in groups.items()}
    return {'overall':stats(records),'macro_accuracy':float(np.mean([x['accuracy'] for x in tasks.values()])),
            'public_macro_accuracy':float(np.mean([v['accuracy'] for k,v in tasks.items() if not k.startswith('synthetic_')])),
            'synthetic_macro_accuracy':float(np.mean([v['accuracy'] for k,v in tasks.items() if k.startswith('synthetic_')])), 'tasks':tasks}

def fit(cfg,data_path,resume=False):
    seed_all(cfg.seed);torch.set_num_threads(2)
    root=Path(cfg.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'latest.pt').exists() and not resume:raise FileExistsError('Refusing to overwrite an existing run; use --resume.')
    if resume:
        previous=json.loads((root/'config.json').read_text())
        assert previous==dataclasses.asdict(cfg),'Resume configuration must match exactly'
    (root/'config.json').write_text(json.dumps(dataclasses.asdict(cfg),indent=2))
    data=json.loads(Path(data_path).read_text());assert 'test' not in data and 'calibration' not in data
    model=PolicyV3(cfg).to(cfg.device);device=torch.device(cfg.device)
    import importlib.metadata as m
    env={'model':cfg.model,'revision':cfg.revision,'device':torch.cuda.get_device_name(0) if device.type=='cuda' else str(device),
         'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
         'packages':{x:m.version(x) for x in ('torch','transformers','peft','datasets')},'training_data_sha256':__import__('hashlib').sha256(Path(data_path).read_bytes()).hexdigest()}
    if resume:assert json.loads((root/'environment.json').read_text())['training_data_sha256']==env['training_data_sha256'],'Training data changed'
    else:(root/'environment.json').write_text(json.dumps(env,indent=2))
    print(json.dumps(env),flush=True)
    groups=defaultdict(list)
    for row in data['train']:groups[row['task']].append(row)
    task_names=sorted(groups);rng=random.Random(cfg.seed)
    optimizer=torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.')],'lr':cfg.adapter_lr},
                                {'params':model.choice.parameters(),'lr':cfg.head_lr}],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',init_scale=1024)
    start_step=0;elapsed_before=0.;skipped=0
    if resume:
        saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
        restore(model,saved['trainable']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        rng.setstate(saved['rng']);torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
        start_step=saved['step'];skipped=saved.get('skipped_steps',0)
        baseline=json.loads((root/'pretrained-dev.json').read_text())
        history=json.loads((root/'development.json').read_text()) if (root/'development.json').exists() else []
        history=[x for x in history if x['step']<=start_step]
        best_record=max([{'step':0,'macro_accuracy':baseline['macro_accuracy']},*history],key=lambda x:x['macro_accuracy'])
        best=best_record['macro_accuracy'];best_step=best_record['step']
        logs=[json.loads(x) for x in (root/'training.jsonl').read_text().splitlines()]
        (root/f'training-before-resume-{start_step}.jsonl').write_text((root/'training.jsonl').read_text())
        logs=[x for x in logs if x['step']<=start_step]
        elapsed_before=saved.get('elapsed_seconds',logs[-1]['elapsed_seconds'] if logs else 0.)
        (root/'training.jsonl').write_text('\n'.join(json.dumps(x) for x in logs)+'\n')
        print('RESUMED',start_step,'elapsed',elapsed_before,flush=True)
    else:
        baseline=metrics(predictions(model,data['dev'],orders=1,batch_size=cfg.batch_size))
        (root/'pretrained-dev.json').write_text(json.dumps(baseline,indent=2));best=baseline['macro_accuracy'];best_step=0
        atomic_save({'trainable':trainable_state(model),'step':0,'config':dataclasses.asdict(cfg)},root/'best.pt')
        print('Baseline dev',json.dumps({k:v for k,v in baseline.items() if k not in ('tasks','overall')}),flush=True)
        history=[]
    start=time.monotonic();sampled=start_step*cfg.batch_size*cfg.accumulate
    eligibility_path=root/'training-eligibility.json'
    eligibility=json.loads(eligibility_path.read_text()) if eligibility_path.exists() else None
    exclusions_applied=False
    for step in range(start_step+1,cfg.steps+1):
        if eligibility and not exclusions_applied and step-1>=eligibility['effective_after_step']:
            excluded=set(eligibility['excluded_ids'])
            groups={task:[r for r in rows if r['id'] not in excluded] for task,rows in groups.items()}
            exclusions_applied=True
            print('TRAINING_INPUT_EXCLUSIONS',len(excluded),'after step',step-1,flush=True)
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;correct=0;tokens=0
        lr_scale=min(1.,step/80)*(.12+.88*.5*(1+math.cos(math.pi*step/cfg.steps)))
        for group,lr in zip(optimizer.param_groups,[cfg.adapter_lr,cfg.head_lr]):group['lr']=lr*lr_scale
        for _ in range(cfg.accumulate):
            pool=[rng.choice(groups[rng.choice(task_names)]) for _ in range(cfg.batch_size*8)]
            pool.sort(key=lambda r:len(r['context'])+len(r['question'])+sum(map(len,r['options'])))
            bucket=rng.randrange(8);rows=pool[bucket*cfg.batch_size:(bucket+1)*cfg.batch_size]
            batch={k:v.to(device) for k,v in model.encode(rows,rng).items()}
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits=model(**batch)
                loss=F.cross_entropy(logits,batch['targets'])
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite forward loss; checkpoint preserved')
            scaler.scale(loss/cfg.accumulate).backward();loss_sum+=loss.item()/cfg.accumulate
            correct+=(logits.argmax(-1)==batch['targets']).sum().item();tokens+=batch['attention_mask'].sum().item()
        scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(norm):
            skipped+=1
            print('AMP_OVERFLOW_SKIPPED',step,'scale',scaler.get_scale(),'total',skipped,flush=True)
            if skipped>32:raise FloatingPointError('Repeated gradient overflow; inspect numerics before continuing')
        scaler.step(optimizer);scaler.update();sampled+=cfg.batch_size*cfg.accumulate
        if step%cfg.log_every==0 or step==1:
            event={'step':step,'loss':loss_sum,'batch_accuracy':correct/(cfg.batch_size*cfg.accumulate),'elapsed_seconds':elapsed_before+time.monotonic()-start,
                   'examples_seen':sampled,'lr_scale':lr_scale,'skipped_steps':skipped,'loss_scale':scaler.get_scale(),'peak_gpu_gb':torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else None}
            with (root/'training.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        end=step==cfg.steps or (elapsed_before+time.monotonic()-start)>cfg.max_minutes*60
        if step%cfg.eval_every==0 or end:
            record=metrics(predictions(model,data['dev'],orders=1,batch_size=cfg.batch_size));record['step']=step;history.append(record)
            (root/'development.json').write_text(json.dumps(history,indent=2))
            print('Development',step,record['macro_accuracy'],record['public_macro_accuracy'],record['synthetic_macro_accuracy'],flush=True)
            if record['macro_accuracy']>best:
                best=record['macro_accuracy'];best_step=step
                atomic_save({'trainable':trainable_state(model),'step':step,'config':dataclasses.asdict(cfg)},root/'best.pt')
        if step%cfg.save_every==0 or end:
            atomic_save({'trainable':trainable_state(model),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step,
                         'config':dataclasses.asdict(cfg),'rng':rng.getstate(),'torch_rng':torch.get_rng_state(),
                         'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                         'elapsed_seconds':elapsed_before+time.monotonic()-start,'skipped_steps':skipped},root/'latest.pt')
        if end:break
    summary={'selected_step':best_step,'selected_dev_macro_accuracy':best,'completed_steps':step,'skipped_steps':skipped,'optimizer_updates':step-skipped,'elapsed_seconds':elapsed_before+time.monotonic()-start}
    (root/'training-complete.json').write_text(json.dumps(summary,indent=2));print('TRAINING COMPLETE',json.dumps(summary),flush=True)

def load_v3(path='runs/v3',device='cpu',merged=False,attention='eager'):
    root=Path(path);cfg=ConfigV3(**json.loads((root/'config.json').read_text()));cfg.attention=attention
    model=PolicyV3(cfg);saved=torch.load(root/'best.pt',map_location='cpu',weights_only=False);restore(model,saved['trainable'])
    if (root/'calibration.json').exists():model.temperature=json.loads((root/'calibration.json').read_text())['temperature']
    if merged:model.backbone=model.backbone.merge_and_unload(safe_merge=True)
    return model.to(device).eval()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='runs/v3/training-data.json');p.add_argument('--output',default='runs/v3')
    p.add_argument('--steps',type=int,default=1800);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accumulate',type=int,default=4)
    p.add_argument('--max-minutes',type=float,default=150);p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();fit(ConfigV3(output=a.output,steps=a.steps,batch_size=a.batch_size,accumulate=a.accumulate,max_minutes=a.max_minutes,eval_every=a.eval_every),a.data,a.resume)
