"""Reconstruct sampled training IDs and verify the saved sampler RNG state."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import random
import torch

root=Path('runs/v3')
cfg=json.loads((root/'config.json').read_text())
saved=torch.load(root/'latest.pt',map_location='cpu',weights_only=False)
data=json.loads((root/'training-data.json').read_text())
groups=defaultdict(list)
for row in data['train']:groups[row['task']].append(row)
tasks=sorted(groups);rng=random.Random(cfg['seed']);draws=Counter();unique=defaultdict(set)
selected_step=json.loads((root/'training-complete.json').read_text())['selected_step']
selected_coverage=None
eligibility=json.loads((root/'training-eligibility.json').read_text()) if (root/'training-eligibility.json').exists() else None
for step in range(saved['step']):
    if eligibility and step==eligibility['effective_after_step']:
        excluded=set(eligibility['excluded_ids'])
        groups={task:[r for r in rows if r['id'] not in excluded] for task,rows in groups.items()}
    for _ in range(cfg['accumulate']):
        pool=[rng.choice(groups[rng.choice(tasks)]) for _ in range(cfg['batch_size']*8)]
        pool.sort(key=lambda r:len(r['context'])+len(r['question'])+sum(map(len,r['options'])))
        bucket=rng.randrange(8)
        for row in pool[bucket*cfg['batch_size']:(bucket+1)*cfg['batch_size']]:
            order=list(range(len(row['options'])));rng.shuffle(order)
            draws[row['task']]+=1;unique[row['task']].add(row['id'])
    if step+1==selected_step:
        selected_coverage={'step':selected_step,'sampling_draws':sum(draws.values()),'unique_training_records':sum(map(len,unique.values()))}
assert rng.getstate()==saved['rng'],'Sampler reconstruction differs from the saved run'
report={'through_step':saved['step'],'sampling_draws':sum(draws.values()),'unique_training_records':sum(map(len,unique.values())),
        'selected_checkpoint_coverage':selected_coverage,
        'rng_state_matches_checkpoint':True,
        'tasks':{task:{'draws':draws[task],'unique_records':len(unique[task]),'available_records':len(groups[task])} for task in tasks}}
(root/'training-coverage.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
