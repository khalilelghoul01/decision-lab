"""Tokenizer-only input-budget audit; no model predictions or label use."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

parser=argparse.ArgumentParser();parser.add_argument('--root',default='runs/v3');parser.add_argument('--effective-after-step',type=int,default=0);args=parser.parse_args()
root=Path(args.root);data=json.loads((root/'data.json').read_text())
tokenizer=AutoTokenizer.from_pretrained('LiquidAI/LFM2.5-350M',revision='9e6c6ccf47cd318696e137d381a7ded8fe4df09f')
output={}
for split in ('train','dev','calibration','test'):
    rows=data[split];rejected=[];maximum=0
    for start in range(0,len(rows),512):
        chunk=rows[start:start+512];messages=[]
        for row in chunk:
            content='Context:\n\nQuestion: '+row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {s}' for i,s in enumerate(row['options']))+'\nReply with only the letter of the correct choice.'
            messages.append([{'role':'system','content':'You answer multiple-choice questions.'},{'role':'user','content':content}])
        sequences=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=False)
        for row,ids in zip(chunk,sequences):
            maximum=max(maximum,len(ids))
            # Leave context room and a small boundary/permutation margin.
            if len(ids)>1000:rejected.append({'id':row['id'],'task':row['task'],'question_options_tokens':len(ids)})
    output[split]={'records':len(rows),'max_question_options_tokens':maximum,'excluded_from_1024_token_training':rejected}
    print(split,len(rows),'oversized',len(rejected),'max',maximum,flush=True)
(root/'input-length-audit.json').write_text(json.dumps(output,indent=2))
eligibility={'effective_after_step':args.effective_after_step,'reason':'Question and choices need more than 1000 tokens without context; preserve full choices and exclude from this small-model run',
             'excluded_ids':[r['id'] for r in output['train']['excluded_from_1024_token_training']]}
(root/'training-eligibility.json').write_text(json.dumps(eligibility,indent=2))
