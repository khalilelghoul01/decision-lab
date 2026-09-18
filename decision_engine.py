"""Typed decisions with one shared prefill and independent field branches.

Inference for frozen V2/V3 checkpoints; this module does not train weights.
The legacy schema interface supports a required, flat object of booleans and
2-4-value enums. The typed API in jev_protocol supports each model's choice cap.
General JSON Schema generation is not implemented by this engine.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers.cache_utils import DynamicLayer, LinearAttentionLayer

from decision_v2 import load_v2


def fork_cache(cache, batch_size):
    """Share read-only attention tensors, clone mutable convolution buffers."""
    result = copy.copy(cache)
    result.layers = []
    for source in cache.layers:
        layer = copy.copy(source)
        if type(source) is DynamicLayer:
            for name in ('keys', 'values'):
                tensor = getattr(source, name)
                setattr(layer, name, tensor.expand(batch_size, *tensor.shape[1:]))
        elif type(source) is LinearAttentionLayer:
            for name, value in vars(source).items():
                if isinstance(value, dict):
                    setattr(layer, name, {
                        k: v.expand(batch_size, *v.shape[1:]).clone()
                        if isinstance(v, torch.Tensor) else v for k, v in value.items()
                    })
        else:
            raise TypeError(f'Unsupported cache layer: {type(source).__name__}')
        result.layers.append(layer)
    return result


@dataclass(frozen=True)
class Prepared:
    sequences: list[list[int]]
    sizes: list[int]
    fields: int
    orders: int
    prefix_length: int


def prepare_rows(model, rows, orders=2):
    if not rows or orders not in (1, 2):
        raise ValueError('Provide at least one question and one or two orders')
    mapped = []
    for reverse in range(orders):
        for row in rows:
            limit=getattr(model.cfg,'max_choices',4)
            if not 2 <= len(row['options']) <= limit:
                raise ValueError(f'Each question needs 2-{limit} choices')
            item = {**row, 'target': 0}
            if reverse:
                item['options'] = list(reversed(row['options']))
            mapped.append(item)
    encoded = model.encode(mapped)
    sequences = [ids[:int(mask.sum())].tolist()
                 for ids, mask in zip(encoded['input_ids'], encoded['attention_mask'])]
    # Compare actual tokens. BPE boundaries and V2 context truncation can vary.
    prefix = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        prefix += 1
    prefix = min(prefix, min(map(len, sequences)) - 1)
    return Prepared(sequences, encoded['sizes'].tolist(), len(rows), orders, prefix)


def aligned_probabilities(logits, prepared, temperature):
    probabilities = (logits.float() / temperature).softmax(-1)
    if prepared.orders == 1:
        return probabilities
    reverse = probabilities[prepared.fields:].clone()
    for i, size in enumerate(prepared.sizes[:prepared.fields]):
        reverse[i, :size] = reverse[i, :size].flip(0)
    return (probabilities[:prepared.fields] + reverse) / 2


class DecisionEngine:
    def __init__(self, model, field_batch_size=8):
        if field_batch_size < 1:
            raise ValueError('field_batch_size must be positive')
        self.model = model.eval()
        self.field_batch_size = field_batch_size
        self.device = next(model.parameters()).device

    @classmethod
    def load(cls, path='runs/v2', device='auto', precision='fp32', field_batch_size=8):
        if precision not in ('fp32', 'fp16'):
            raise ValueError('precision must be fp32 or fp16')
        config=json.loads((Path(path)/'config.json').read_text())
        if config.get('model'):
            from decision_v3 import load_v3
            from decision_lab import device_for
            model=load_v3(path,str(device_for(device)))
        else:model = load_v2(path, device)
        model.backbone = model.backbone.merge_and_unload(safe_merge=True)
        model.backbone.to(dtype=torch.float16 if precision == 'fp16' else torch.float32)
        return cls(model, field_batch_size)

    def prepare(self, rows, orders=2):
        return prepare_rows(self.model, rows, orders)

    def _pad(self, sequences):
        batch = self.model.tokenizer.pad({'input_ids': sequences}, padding=True, return_tensors='pt')
        return {k: v.to(self.device) for k, v in batch.items()}

    @torch.inference_mode()
    def logits(self, prepared, shared=True):
        cache = None
        prefix = prepared.prefix_length if shared and len(prepared.sequences) > 1 else 0
        if prefix:
            ids = torch.tensor([prepared.sequences[0][:prefix]], device=self.device)
            cache = self.model.backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        use_cache=True).past_key_values
        outputs = []
        for start in range(0, len(prepared.sequences), self.field_batch_size):
            sequences = prepared.sequences[start:start + self.field_batch_size]
            batch = self._pad([x[prefix:] for x in sequences])
            lengths = batch['attention_mask'].sum(-1)
            branch = fork_cache(cache, len(sequences)) if cache is not None else None
            if prefix:
                batch['attention_mask'] = torch.cat((torch.ones((len(sequences), prefix),
                    device=self.device, dtype=torch.long), batch['attention_mask']), dim=1)
            hidden = self.model.backbone(**batch, past_key_values=branch,
                                         use_cache=branch is not None).last_hidden_state
            pooled = hidden[torch.arange(len(sequences), device=self.device), lengths - 1].float()
            logits = self.model.choice(pooled)
            sizes = torch.tensor(prepared.sizes[start:start + len(sequences)], device=self.device)
            outputs.append(logits.masked_fill(torch.arange(logits.shape[-1], device=self.device)[None] >= sizes[:, None], -1e4))
        return torch.cat(outputs)

    def score(self, rows, orders=2, shared=True):
        prepared = self.prepare(rows, orders)
        return aligned_probabilities(self.logits(prepared, shared), prepared, self.model.temperature)

    def decide(self, context, schema, fast=False, include_schema=None):
        if include_schema is None:
            include_schema=getattr(self.model,'model_name','decision-lab-v2')=='decision-lab-v2'
        compiled = compile_schema(schema)
        rows, truncated = schema_rows(self.model, context, compiled, include_schema)
        probabilities = self.score(rows, orders=1 if fast else 2).cpu().tolist()
        return assemble(compiled, probabilities, truncated)


@dataclass(frozen=True)
class CompiledSchema:
    schema: dict
    names: list[str]
    questions: list[str]
    choices: list[list[str]]
    values: list[list]


def compile_schema(schema):
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(schema)
    schema = copy.deepcopy(schema)
    allowed = {'type', 'properties', 'required', 'additionalProperties', 'title', 'description', '$schema'}
    if set(schema) - allowed or schema.get('type') != 'object':
        raise ValueError('Only a flat object schema is supported')
    props = schema.get('properties', {})
    if not props or len(props) > 32:
        raise ValueError('Use 1-32 fields')
    if set(schema.get('required', [])) != set(props) or schema.get('additionalProperties') is not False:
        raise ValueError('Require every field and set additionalProperties to false')
    names, questions, choices, values = [], [], [], []
    for name, spec in props.items():
        if not isinstance(spec, dict) or set(spec) - {'type', 'enum', 'title', 'description'}:
            raise ValueError(f'{name}: unsupported field constraints')
        kind = spec.get('type')
        if kind == 'boolean' and 'enum' not in spec:
            field_values, labels = [False, True], ['No', 'Yes']
        elif kind in ('string', 'number', 'integer') and 'enum' in spec:
            field_values = spec['enum']
            if not 2 <= len(field_values) <= 4:
                raise ValueError(f'{name}: use 2-4 enum values')
            validator = Draft202012Validator({'type': kind})
            for v in field_values:
                validator.validate(v)
                if isinstance(v, float) and not math.isfinite(v):
                    raise ValueError('Enum numbers must be finite')
            labels = [str(v) for v in field_values]
            if len(set(labels)) != len(labels):
                raise ValueError(f'{name}: ambiguous labels')
        else:
            raise ValueError(f'{name}: use a boolean or a typed enum')
        names.append(name)
        questions.append(spec.get('description') or spec.get('title') or name.replace('_', ' ') + '?')
        choices.append(labels)
        values.append(field_values)
    return CompiledSchema(schema, names, questions, choices, values)


def schema_rows(model, context, compiled, include_schema=True):
    """Reserve the schema and every complete question; truncate only the document."""
    schema_text = '\n\nOutput schema:\n' + json.dumps(compiled.schema, ensure_ascii=False, separators=(',', ':')) if include_schema else ''
    context_ids = model.tokenizer.encode(context, add_special_tokens=False)
    original_length = len(context_ids)
    def rows_for(text):
        return [{'context': text + schema_text, 'question': q, 'options': options}
                for q, options in zip(compiled.questions, compiled.choices)]
    # V2 reserves four extra tokens. Find a common budget before V2 encoding so
    # every field sees the same document and the complete schema.
    overhead = max(len(model.tokenizer.apply_chat_template([
        {'role': 'system', 'content': 'You answer multiple-choice questions.'},
        {'role': 'user', 'content': 'Context:\n\nQuestion: ' + r['question'] + '\n' +
         '\n'.join(f'{chr(65+i)}. {s}' for i, s in enumerate(r['options'])) +
         '\nReply with only the letter of the correct choice.'}
    ], add_generation_prompt=True, tokenize=True, return_dict=False))
        for base in rows_for('')
        for r in (base, {**base, 'options': list(reversed(base['options']))}))
    budget = model.cfg.max_length - overhead - 4
    while True:
        document = model.tokenizer.decode(context_ids, skip_special_tokens=True)
        length = len(model.tokenizer.encode(document + schema_text, add_special_tokens=False))
        if length <= budget:
            break
        if not context_ids:
            raise ValueError(f'Schema/questions exceed the {model.cfg.max_length}-token model budget')
        context_ids = context_ids[:max(0, len(context_ids) - max(1, length - budget))]
    return rows_for(document), len(context_ids) < original_length


def assemble(compiled, probabilities, truncated=False):
    from jsonschema import Draft202012Validator
    value, fields = {}, {}
    for name, candidates, p in zip(compiled.names, compiled.values, probabilities, strict=True):
        p = p[:len(candidates)]
        index = max(range(len(p)), key=p.__getitem__)
        value[name] = candidates[index]
        fields[name] = {'confidence': p[index], 'probabilities': [
            {'value': v, 'probability': prob} for v, prob in zip(candidates, p)]}
    Draft202012Validator(compiled.schema).validate(value)
    return {'value': value, 'fields': fields, 'document_truncated': truncated}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema', required=True)
    parser.add_argument('--document', required=True)
    parser.add_argument('--checkpoint', default='runs/v2')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()
    from pathlib import Path
    engine = DecisionEngine.load(args.checkpoint, precision=args.precision)
    print(json.dumps(engine.decide(Path(args.document).read_text(), json.loads(Path(args.schema).read_text()), args.fast), indent=2))
