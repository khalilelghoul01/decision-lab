"""Warm native V2/V3 timings on identical prompts, including tokenization."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import torch
from decision_lab.engine import DecisionEngine

QUESTIONS=[
 ('What is the sentiment of the review?',['Negative','Positive']),
 ('Was the acting good?',['No','Yes']),
 ('Was the plot engaging?',['No','Yes']),
 ('What is the topic?',['Politics','Sports','Movies','Business']),
 ('Is the reviewer recommending the film?',['No','Yes']),
 ('Was the soundtrack praised?',['No','Yes']),
 ('Was the pacing criticized?',['No','Yes']),
 ('How was the cinematography?',['Bad','Average','Good']),
]
short='The film had convincing acting and an engaging plot. The soundtrack was excellent. The pacing dragged and the cinematography was average. Despite those flaws, I recommend it.'
long=('The festival program featured several films, including dramas, documentaries, and comedies. The audience discussed them after the screening. '*12)+short

def sync(device):
    if device.type=='mps':torch.mps.synchronize()
    if device.type=='cuda':torch.cuda.synchronize()

parser=argparse.ArgumentParser()
parser.add_argument('--versions',nargs='+',choices=['v2','v3'],default=['v2','v3'])
parser.add_argument('--output',default='runs/v3/native-benchmarks.json')
args=parser.parse_args()
results=[];torch.set_num_threads(4)
for version in args.versions:
    path='runs/'+version
    engine=DecisionEngine.load(path,device='mps',precision='fp16')
    for length,context in [('short',short),('long',long)]:
        for fields in [1,4,8]:
            rows=[{'context':context,'question':q,'options':options} for q,options in QUESTIONS[:fields]]
            for orders in [1,2]:
                prepared=engine.prepare(rows,orders)
                record={'model':version,'context':length,'fields':fields,'orders':orders,'shared_prefix_tokens':prepared.prefix_length,
                        'max_input_tokens':max(map(len,prepared.sequences)),'precision':'fp16','device':'Apple M4 MPS'}
                for shared in [False,True]:
                    for _ in range(2):engine.score(rows,orders,shared).cpu();sync(engine.device)
                    samples=[]
                    for _ in range(7):
                        sync(engine.device);start=time.perf_counter();engine.score(rows,orders,shared).cpu();sync(engine.device)
                        samples.append((time.perf_counter()-start)*1000)
                    record['cached_ms' if shared else 'full_ms']=statistics.median(samples)
                record['speedup']=record['full_ms']/record['cached_ms'];results.append(record)
                print(json.dumps(record),flush=True)
    del engine;gc.collect();torch.mps.empty_cache()
Path(args.output).write_text(json.dumps(results,indent=2))
