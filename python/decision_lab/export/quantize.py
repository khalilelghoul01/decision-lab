"""Optional weight-only INT4 browser artifact; preserves FP16 caches/head.

This creates a candidate, not an accuracy claim. Validate on calibration data
and in the actual browser before selecting it for deployment.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

def quantize(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'decision.onnx').exists():raise FileExistsError('Choose a new quantized artifact directory')
    model=onnx.load(source/'decision.onnx');onnx.external_data_helper.convert_model_from_external_data(model)
    quantizer=MatMulNBitsQuantizer(model,bits=4,block_size=32,is_symmetric=True,accuracy_level=0,
                                 op_types_to_quantize=('MatMul',),nodes_to_exclude=['/choice/MatMul'])
    quantizer.process();model=quantizer.model.model
    onnx.save_model(model,output/'decision.onnx',save_as_external_data=True,all_tensors_to_one_file=True,location='decision.weights',size_threshold=1024)
    manifest=json.loads((source/'manifest.json').read_text())
    manifest.update(weight_quantization={'bits':4,'block_size':32,'symmetric':True,'head':'fp32','embeddings':'fp16'},
                    weights_bytes=(output/'decision.weights').stat().st_size,graph_bytes=(output/'decision.onnx').stat().st_size)
    fixture=json.loads((source/'fixture.json').read_text());sequences=fixture['sequences']
    length=max(map(len,sequences));ids=np.full((len(sequences),length),manifest['pad_token_id'],dtype=np.int64);mask=np.zeros_like(ids)
    for i,row in enumerate(sequences):ids[i,:len(row)]=row;mask[i,:len(row)]=1
    feeds={'input_ids':ids,'attention_mask':mask,'sizes':np.array(fixture['sizes'],dtype=np.int64)}
    feeds.update({k:np.zeros(shape,dtype=np.float16) for k,shape in zip(manifest['state_names'],manifest['state_shapes'])})
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(output/'decision.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    logits=session.run(['logits'],feeds)[0];p=logits/manifest['temperature'];p=np.exp(p-p.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    fields=fixture['fields'];aligned=p[:fields].copy()
    if fixture['orders']==2:
        for i,n in enumerate(fixture['sizes'][:fields]):aligned[i,:n]=(p[i,:n]+p[i+fields,:n][::-1])/2
    reference=fixture['probabilities'];fixture['fp16_reference_probabilities']=reference
    fixture['probabilities']=aligned.tolist();fixture['logits']=logits.tolist()
    manifest['quantization_fixture']={'max_probability_delta':float(np.max(np.abs(aligned-np.array(reference)))),
                                     'answers_agree':bool(np.array_equal(aligned.argmax(-1),np.array(reference).argmax(-1))),
                                     'scope':'Small runtime fixture only; not general accuracy'}
    (output/'fixture.json').write_text(json.dumps(fixture,indent=2));(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copytree(source/'tokenizer',output/'tokenizer',dirs_exist_ok=True)
    for name in ('BASE_MODEL_LICENSE','BASE_MODEL_README.md','MODIFICATIONS.md'):
        if (source/name).exists():shutil.copy2(source/name,output/name)
    with (output/'MODIFICATIONS.md').open('a') as f:
        f.write('\nThis candidate additionally quantizes constant matrix multiplications to signed\n'
                '4-bit weights in blocks of 32. Embeddings, activations, and caches remain FP16;\n'
                'the answer-token head remains FP32. Its calibration is fitted separately.\n')
    print(json.dumps({'weights_mb':manifest['weights_bytes']/1e6,**manifest['quantization_fixture']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();quantize(a.source,a.output)
