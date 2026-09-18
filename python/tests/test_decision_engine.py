"""Cache isolation, padding and portable-graph equivalence on a tiny LFM2."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Lfm2Config, Lfm2Model

from decision_lab.engine import fork_cache, compile_schema, assemble
from decision_lab.portable import PortableDecision


@pytest.fixture(params=[4,8])
def tiny(request):
    torch.manual_seed(73)
    cfg = Lfm2Config(vocab_size=100, hidden_size=32, intermediate_size=64,
                     num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                     layer_types=['conv', 'full_attention', 'conv'], conv_L_cache=3,
                     block_auto_adjust_ff_dim=False)
    cfg._attn_implementation = 'eager'
    model = SimpleNamespace(backbone=Lfm2Model(cfg).eval(), choice=nn.Linear(32, request.param, bias=False).eval())
    return model


@torch.no_grad()
def test_hybrid_cache_forks_are_independent(tiny):
    ids = torch.randint(1, 100, (1, 12))
    prefix = tiny.backbone(input_ids=ids[:, :8], use_cache=True).past_key_values
    original = {i: x.conv_states[0].clone() for i, x in enumerate(prefix.layers) if hasattr(x, 'conv_states')}
    for suffix in (ids[:, 8:], ids[:, 8:9]):
        branch = fork_cache(prefix, 2)
        actual = tiny.backbone(input_ids=suffix.expand(2, -1), past_key_values=branch, use_cache=True).last_hidden_state
        expected = tiny.backbone(input_ids=torch.cat((ids[:, :8], suffix), 1), use_cache=False).last_hidden_state[:, -suffix.shape[1]:]
        torch.testing.assert_close(actual, expected.expand(2, -1, -1), atol=1e-5, rtol=1e-5)
        for i, old in original.items():
            torch.testing.assert_close(prefix.layers[i].conv_states[0], old, atol=0, rtol=0)
        assert prefix.get_seq_length() == 8


@torch.no_grad()
def test_portable_full_cached_and_right_padded(tiny):
    graph = PortableDecision(tiny).eval()
    ids = torch.randint(1, 100, (2, 12)); ids[1, :8] = ids[0, :8]
    mask = torch.ones_like(ids); mask[1, -2:] = 0
    sizes = torch.tensor([2, tiny.choice.out_features])
    hidden = tiny.backbone(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    expected = tiny.choice(hidden[torch.arange(2), mask.sum(-1) - 1])
    expected[0, 2:] = -1e4
    full = graph(ids, mask, sizes, *graph.empty_cache())
    pref = graph(ids[:1, :8], mask[:1, :8], sizes[:1], *graph.empty_cache())
    cached = graph(ids[:, 8:], mask[:, 8:], sizes, *pref[1:])
    torch.testing.assert_close(full[0], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(cached[0], expected, atol=1e-5, rtol=1e-5)


def test_schema_returns_typed_values():
    schema = {'type': 'object', 'properties': {'review': {'type': 'boolean'}, 'level': {'type': 'integer', 'enum': [1, 2, 3]}},
              'required': ['review', 'level'], 'additionalProperties': False}
    compiled = compile_schema(schema)
    assert assemble(compiled, [[0.1, 0.9], [0.2, 0.7, 0.1]])['value'] == {'review': True, 'level': 2}


@pytest.mark.parametrize('field', [
    {'type': 'string'}, {'type': 'object'}, {'type': 'number', 'minimum': 0},
    {'type': 'string', 'enum': ['a', 'b'], 'pattern': '^a'},
])
def test_unsupported_schema_rejected(field):
    with pytest.raises(ValueError):
        compile_schema({'type': 'object', 'properties': {'x': field}, 'required': ['x'], 'additionalProperties': False})
