"""Assess an INT4 candidate on calibration data before deployment selection."""
import argparse
import json
import time
from pathlib import Path
from types import MethodType,SimpleNamespace
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from decision_lab.train import PolicyV3,metrics
from decision_lab.evaluate import calibrate,derived

def collect(artifact,rows):
    root=Path(artifact);meta=json.loads((root/'manifest.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(root/'tokenizer');tokenizer.padding_side='right'
    encoder=SimpleNamespace(tokenizer=tokenizer,cfg=SimpleNamespace(max_length=meta['max_length'],max_choices=meta.get('max_choices',4)))
    encoder.chat=MethodType(PolicyV3.chat,encoder);encoder.encode=MethodType(PolicyV3.encode,encoder)
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(root/'decision.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    empty={key:np.zeros(shape,dtype=np.float16) for key,shape in zip(meta['state_names'],meta['state_shapes'])}
    rows=sorted(rows,key=lambda x:len(x['context'])+len(x['question'])+sum(map(len,x['options'])))
    results=[];start_time=time.monotonic()
    for start in range(0,len(rows),8):
        chunk=rows[start:start+8]
        mapped=chunk+[{**x,'options':list(reversed(x['options'])),'target':len(x['options'])-1-x['target']} for x in chunk]
        batch=encoder.encode(mapped)
        feeds={k:batch[k].numpy().astype(np.int64) for k in ['input_ids','attention_mask','sizes']}
        raw=session.run(['logits'],{**feeds,**empty})[0]
        for i,row in enumerate(chunk):
            n=len(row['options']);logits=[raw[i,:n].tolist(),raw[i+len(chunk),:n][::-1].tolist()]
            results.append({'id':row['id'],'task':row['task'],'target':row['target'],'prediction':0,'probabilities':[1/n]*n,
                            'logits':logits,'truncated':bool(batch['truncated'][i])})
        if start%80==0:print('ONNX',start,'/',len(rows),'seconds',round(time.monotonic()-start_time,1),flush=True)
    return results

def main(root,artifact,final_test=False):
    root=Path(root);artifact=Path(artifact);data=json.loads((root/'data.json').read_text())
    if final_test:
        if (root/'q4-test-accurate.json').exists():raise FileExistsError('Quantized final evaluation already exists')
        gate=json.loads((root/'q4-calibration.json').read_text())
        if not gate['accuracy_gate_passed']:raise RuntimeError('Quantized candidate failed the calibration gate')
        rows=collect(artifact,data['test']);t=gate['temperature']
        for orders,name in [(1,'fast'),(2,'accurate')]:
            records=derived(rows,orders,t)
            (root/f'q4-test-{name}.json').write_text(json.dumps({**metrics(records),'orders':orders,'temperature':t},indent=2))
            (root/f'q4-test-{name}.jsonl').write_text('\n'.join(json.dumps(x) for x in records)+'\n')
        return
    if not (root/'calibration-logits.json').exists():raise RuntimeError('First finish native calibration after checkpoint selection')
    if (root/'q4-calibration.json').exists():raise FileExistsError('Calibration assessment already exists')
    raw=collect(artifact,data['calibration']);fitted=calibrate(raw)
    records=derived(raw,2,fitted['temperature']);quant=metrics(records)
    base_t=json.loads((root/'calibration.json').read_text())['temperature']
    base=metrics(derived(json.loads((root/'calibration-logits.json').read_text()),2,base_t))
    passed=all(quant[key]>=base[key]-.01 for key in ['public_macro_accuracy','synthetic_macro_accuracy'])
    report={**fitted,'accuracy_gate_passed':passed,'gate':'No more than 1 pp lower macro accuracy in either public or synthetic calibration tasks',
            'quantized':quant,'fp16':base,'browser_gate':'Still requires actual WebGPU parity and latency checks'}
    (root/'q4-calibration.json').write_text(json.dumps(report,indent=2))
    (root/'q4-calibration-logits.json').write_text(json.dumps(raw))
    manifest=json.loads((artifact/'manifest.json').read_text());manifest['temperature']=fitted['temperature'];manifest['calibration_accuracy_gate_passed']=passed
    (artifact/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Reference fixture probabilities must use the newly fitted temperature.
    fixture=json.loads((artifact/'fixture.json').read_text());values=np.array(fixture['logits'])/fitted['temperature']
    p=np.exp(values-values.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);n=fixture['fields'];aligned=p[:n].copy()
    for i,k in enumerate(fixture['sizes'][:n]):aligned[i,:k]=(p[i,:k]+p[i+n,:k][::-1])/2
    fixture['probabilities']=aligned.tolist();(artifact/'fixture.json').write_text(json.dumps(fixture,indent=2))
    print('INT4 GATE',passed,'public',quant['public_macro_accuracy'],'synthetic',quant['synthetic_macro_accuracy'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/v3');p.add_argument('--artifact',default='browser/public/model-v3-q4');p.add_argument('--final-test',action='store_true');a=p.parse_args();main(a.root,a.artifact,a.final_test)
