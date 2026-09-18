"""Export the frozen trained model; verify full and shared-cache ONNX inference."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from decision_engine import DecisionEngine, aligned_probabilities
from decision_v2 import REVISION as V2_REVISION
from portable_model import PortableDecision


EXAMPLE_ROWS = [
    {'context': 'The film was charming and beautifully acted.', 'question': 'What is the sentiment of this review?', 'options': ['Negative', 'Positive']},
    {'context': 'The film was charming and beautifully acted.', 'question': 'Was the acting good?', 'options': ['No', 'Yes']},
    {'context': 'The film was charming and beautifully acted.', 'question': 'What is this text about?', 'options': ['Politics', 'Sports', 'Movies', 'Business']},
]


@torch.inference_mode()
def export(output, precision='fp16', checkpoint='runs/v2'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    engine = DecisionEngine.load(path=checkpoint,device='cpu', precision=precision)
    model = engine.model
    graph = PortableDecision(model).eval()
    rows=list(EXAMPLE_ROWS)
    if model.choice.out_features>4:
        rows.append({'context':EXAMPLE_ROWS[0]['context'],'question':'Which category best describes this review?',
                     'options':['Travel','Politics','Sports','Movies','Business','Education','Food','Technology']})
    prepared = engine.prepare(rows, 2)
    batch = engine._pad(prepared.sequences)
    sizes = torch.tensor(prepared.sizes)
    inputs = (batch['input_ids'], batch['attention_mask'], sizes, *graph.empty_cache())
    expected = engine.logits(prepared, shared=False)
    actual = graph(*inputs)[0]
    print('Portable full logit delta', (expected-actual).abs().max().item(), flush=True)
    torch.testing.assert_close(actual, expected, atol=0.1 if precision == 'fp16' else 2e-4, rtol=1e-3)
    prefix = prepared.prefix_length
    prefill_inputs = (torch.tensor([prepared.sequences[0][:prefix]]), torch.ones(1, prefix, dtype=torch.int64), torch.tensor([4]), *graph.empty_cache())
    prefix_output = graph(*prefill_inputs)
    suffix_batch = engine._pad([s[prefix:] for s in prepared.sequences])
    branch_inputs = (suffix_batch['input_ids'], suffix_batch['attention_mask'], sizes, *prefix_output[1:])
    branches = graph(*branch_inputs)[0]
    print('Portable cache logit delta', (actual-branches).abs().max().item(), flush=True)
    torch.testing.assert_close(branches, actual, atol=0.1 if precision == 'fp16' else 2e-4, rtol=1e-3)
    names = ['input_ids', 'attention_mask', 'sizes', *graph.state_names]
    outputs = ['logits', *['present_' + name for name in graph.state_names]]
    dynamic = {'input_ids': {0: 'batch', 1: 'sequence'}, 'attention_mask': {0: 'batch', 1: 'sequence'},
               'sizes': {0: 'batch'}, 'logits': {0: 'batch'}}
    for name in graph.state_names:
        dynamic[name] = {0: 'cache_batch'}
        dynamic['present_' + name] = {0: 'batch'}
        if not name.startswith('conv_'):
            dynamic[name][2] = 'past_plus_dummy'
            dynamic['present_' + name][2] = 'total_plus_dummy'
    print('Exporting ONNX', flush=True)
    path = output / 'decision.onnx'
    torch.onnx.export(graph, inputs, str(path), input_names=names, output_names=outputs,
                      dynamic_axes=dynamic, opset_version=17, dynamo=False)
    import onnx
    proto = onnx.load(path)
    onnx.checker.check_model(proto)
    # Keep the graph small and load weights as a separate asset in the browser.
    weights = output / 'decision.weights'
    if weights.exists(): weights.unlink()
    onnx.save_model(proto, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=weights.name, size_threshold=1024)
    del proto
    import onnxruntime as ort
    options = ort.SessionOptions(); options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    def run(args):
        return session.run(None, dict(zip(names, [t.cpu().numpy() for t in args])))
    full = run(inputs)
    pref = run(prefill_inputs)
    cached_inputs = (*branch_inputs[:3], *[torch.from_numpy(x) for x in pref[1:]])
    cached = run(cached_inputs)
    delta = float(np.abs(full[0]-actual.numpy()).max())
    cache_delta = float(np.abs(cached[0]-full[0]).max())
    print('ONNX full / cached delta', delta, cache_delta, flush=True)
    np.testing.assert_allclose(full[0], actual.numpy(), atol=0.12 if precision == 'fp16' else 3e-4, rtol=1e-3)
    np.testing.assert_allclose(cached[0], full[0], atol=0.12 if precision == 'fp16' else 3e-4, rtol=1e-3)
    assert np.array_equal(full[0].argmax(-1), expected.numpy().argmax(-1))
    metadata = {'format': 1, 'precision': precision, 'temperature': model.temperature,
                'max_length': model.cfg.max_length, 'pad_token_id': model.tokenizer.pad_token_id,
                'state_names': graph.state_names, 'state_shapes': graph.state_shapes,
                'weights_bytes': weights.stat().st_size, 'graph_bytes': path.stat().st_size,
                'model': getattr(model.cfg,'model','LiquidAI/LFM2.5-230M'),
                'model_name':getattr(model,'model_name','decision-lab-v2'),
                'max_choices':model.choice.out_features,'checkpoint':str(checkpoint),
                'base_revision':getattr(model.cfg,'revision',V2_REVISION),
                'checkpoint_sha256':hashlib.sha256((Path(checkpoint)/'best.pt').read_bytes()).hexdigest(),
                'validation': {'onnx_full_logit_delta': delta, 'onnx_cached_logit_delta': cache_delta}}
    (output / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    model.tokenizer.save_pretrained(output / 'tokenizer')
    from huggingface_hub import hf_hub_download
    base_id=metadata['model']
    for source,destination in [('LICENSE','BASE_MODEL_LICENSE'),('README.md','BASE_MODEL_README.md')]:
        shutil.copy2(hf_hub_download(base_id,source,revision=metadata['base_revision']),output/destination)
    (output/'MODIFICATIONS.md').write_text(
        '# Modified model artifact\n\nThis is Decision Lab, a derivative of '+base_id+'.\n'
        'The checkpoint adds supervised LoRA updates and a small trained answer-token head.\n'
        'LoRA updates are merged for inference. The original full vocabulary output is replaced\n'
        'with constrained choice scores. Weights are converted to '+precision+' and an ONNX graph\n'
        'with explicit attention and convolution caches. Calibration changes the probability readout.\n'
        'It is not an unmodified Liquid AI release or the Jev model.\n'
        'See BASE_MODEL_LICENSE and BASE_MODEL_README.md for upstream terms and attribution.\n')
    fixture = {'rows': rows, 'sequences': prepared.sequences, 'sizes': prepared.sizes,
               'fields': prepared.fields, 'orders': prepared.orders, 'prefix_length': prefix,
               'logits': full[0].tolist(),
               'probabilities': aligned_probabilities(torch.from_numpy(full[0]), prepared, model.temperature).tolist()}
    (output / 'fixture.json').write_text(json.dumps(fixture, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='browser/public/model')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp16')
    parser.add_argument('--checkpoint',default='runs/v2')
    args = parser.parse_args()
    export(args.output, args.precision,args.checkpoint)
