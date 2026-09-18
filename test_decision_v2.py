import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from decision_lab import example_key
from decision_v2 import predict_rows


class PositionBiasedMock(nn.Module):
    """A deliberately position-biased policy to test semantic remapping."""
    def __init__(self):
        super().__init__()
        self.placeholder=nn.Parameter(torch.tensor(0.))
        self.bins=torch.linspace(0,1,21)
        self.temperature=1.
    def encode(self,rows,rng=None):
        # semantic signal: positive option has score 1; first slot has bias +2
        values=[]
        for row in rows:
            values.append([float(s=='Positive')+(2 if i==0 else 0) for i,s in enumerate(row['options'])]+[-1e4]*(4-len(row['options'])))
        return {'scores':torch.tensor(values)}
    def forward(self,scores):
        return scores,torch.zeros(len(scores),4,21)


def test_binary_order_averaging_preserves_semantics():
    model=PositionBiasedMock()
    row={'context':'good','question':'sentiment','options':['Negative','Positive']}
    reversed_row={**row,'options':['Positive','Negative']}
    p,_=predict_rows(model,[row],orders=2)
    reverse,_=predict_rows(model,[reversed_row],orders=2)
    torch.testing.assert_close(p[0,:2],reverse[0,:2].flip(0))
    assert p[0,1]>p[0,0]
    assert p[0,2:].sum()==0
    torch.testing.assert_close(p.sum(-1),torch.ones(1))


def test_temperature_softens_confidence_without_missing_options():
    model=PositionBiasedMock()
    row={'context':'good','question':'sentiment','options':['Negative','Positive']}
    p,_=predict_rows(model,[row],orders=1,temperature=1)
    soft,_=predict_rows(model,[row],orders=1,temperature=2)
    assert soft.max()<p.max()
    assert soft[0,2:].sum()==0


def test_actual_v2_splits_are_disjoint_and_final_is_fresh():
    path=Path('runs/v2/data.json')
    if not path.exists():pytest.skip('Run prepare_v2 first for real-data integrity check')
    data=json.loads(path.read_text())
    old=json.loads(Path('runs/decision-lab-first/data.json').read_text())
    keys={s:set(map(example_key,data[s])) for s in ['train','dev','calibration','test']}
    for a in keys:
        assert len(keys[a])==len(data[a])
        for b in keys:
            if a!=b:assert not keys[a]&keys[b]
    old_seen={example_key(x) for s in ['train','dev','test'] for x in old[s]}
    assert not keys['test']&old_seen
    assert not keys['calibration']&old_seen


def test_finalize_resumes_remaining_checks_without_refitting_or_retesting(tmp_path, monkeypatch):
    import decision_v2 as v2
    from decision_lab import SOURCES
    cfg = v2.ConfigV2(output=str(tmp_path))
    torch.save({'stage': 'complete'}, tmp_path / 'latest.pt')
    rows = [{'task': task, 'target': 0, 'options': ['A', 'B']} for task in SOURCES]
    (tmp_path / 'data.json').write_text(json.dumps({'train': rows, 'test': rows, 'calibration': []}))
    (tmp_path / 'calibration.json').write_text(json.dumps({'temperature': 1.2}))
    fixed = {'overall': {'accuracy': .75}}
    (tmp_path / 'final-test.json').write_text(json.dumps(fixed))
    model = SimpleNamespace(backbone=SimpleNamespace(save_pretrained=lambda *a, **kw: None),
                            tokenizer=SimpleNamespace(save_pretrained=lambda *a, **kw: None))
    monkeypatch.setattr(v2, 'load_v2', lambda *a: model)
    monkeypatch.setattr(v2, 'calibrate', lambda *a: pytest.fail('Do not refit frozen calibration'))
    calls = []
    monkeypatch.setattr(v2, 'evaluate_v2', lambda model, rows, cfg, tag, **kw: calls.append(tag))
    assert v2.finalize(cfg) == fixed
    assert calls == ['final-fast-test', 'final-permuted-test']
