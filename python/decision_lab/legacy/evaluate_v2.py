"""Legacy 230M baseline on V3's fresh <=4-choice test subset. Run after selection."""
import json
from pathlib import Path
import torch
from decision_lab.legacy.v2 import load_v2,predict_rows
from decision_lab.train import metrics

root=Path('runs/v3')
if not (root/'training-complete.json').exists():raise RuntimeError('Wait for V3 model selection to finish.')
model=load_v2('runs/v2','mps')
rows=[r for r in json.loads((root/'data.json').read_text())['test'] if len(r['options'])<=4]
records=[]
for start in range(0,len(rows),8):
    chunk=rows[start:start+8];p,_=predict_rows(model,chunk,orders=2);p=p.cpu()
    for j,row in enumerate(chunk):
        # Native V2 truncates context; report this instead of hiding the limitation.
        content='Context:\n'+row['context']+'\nQuestion: '+row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(row['options']))+'\nReply with only the letter of the correct choice.'
        ids=model.tokenizer.apply_chat_template([{'role':'system','content':'You answer multiple-choice questions.'},{'role':'user','content':content}],add_generation_prompt=True,tokenize=True,return_dict=False)
        probs=p[j,:len(row['options'])].tolist()
        records.append({'id':row['id'],'task':row['task'],'target':row['target'],'prediction':int(p[j].argmax()),'probabilities':probs,'truncated':len(ids)>model.cfg.max_length})
    if start%400==0:print('V2 common',start,'/',len(rows),flush=True)
(root/'v2-common-test.json').write_text(json.dumps(metrics(records),indent=2))
(root/'v2-common-test.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
print('V2 common subset complete',len(records),flush=True)
