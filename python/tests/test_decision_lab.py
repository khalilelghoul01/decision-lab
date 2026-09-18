import numpy as np
import torch
from decision_lab.legacy.v1 import rlcr_reward, sampled_rl_loss, metrics, reliability


def test_reward_penalizes_confident_errors():
    actual = rlcr_reward(torch.tensor([1, 1, 0, 0]), torch.tensor([1., 0., 1., 0.]))
    torch.testing.assert_close(actual, torch.tensor([1., 0., -1., 0.]))


def test_policy_gradient_matches_exact_expected_reward():
    # Independent enumerated derivative checks the sign and joint-policy gradient.
    torch.manual_seed(7)
    logits = torch.tensor([[0.3, -0.2]], requires_grad=True)
    reports = torch.tensor([[[0.2, -0.1, 0.1], [-0.2, 0.1, 0.4]]], requires_grad=True)
    bins = torch.tensor([0., 0.5, 1.])
    correct = torch.tensor([1., 0.])[:, None]
    rewards = rlcr_reward(correct, bins[None])
    exact = -(logits.softmax(-1)[..., None] * reports.softmax(-1) * rewards).sum()
    expected = torch.autograd.grad(exact, (logits, reports))
    estimate, _ = sampled_rl_loss(logits, reports, torch.tensor([0]), bins, 150000, 0.)
    actual = torch.autograd.grad(estimate, (logits, reports))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=0.003, rtol=0.05)


def test_masked_choice_never_sampled():
    logits = torch.tensor([[0., 0., -1e4, -1e4]], requires_grad=True)
    reports = torch.zeros(1, 4, 21, requires_grad=True)
    loss, stats = sampled_rl_loss(logits, reports, torch.tensor([0]), torch.linspace(0, 1, 21))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(logits.grad[0, 2:]) == 0
    assert torch.count_nonzero(reports.grad[0, 2:]) == 0


def test_reliability_includes_confidence_one():
    rows, ece = reliability([1, 0], [1., 0.])
    assert sum(r['n'] for r in rows) == 2
    assert ece == 0


def test_metrics_known_values():
    rows = [{"correct": c, "confidence": q, "policy_probability": q, "choice_nll": 0.5}
            for c, q in [(1, 0.75), (0, 0.25)]]
    actual = metrics(rows)
    assert actual['accuracy'] == 0.5
    assert actual['confidence_brier'] == 0.0625
    assert actual['ece_10bins'] == 0.25


def test_brier_report_optimum_is_underlying_accuracy():
    q = torch.linspace(0, 1, 21)
    p = 0.7
    expected_reward = p * rlcr_reward(torch.ones_like(q), q) + (1-p) * rlcr_reward(torch.zeros_like(q), q)
    assert np.isclose(q[expected_reward.argmax()].item(), p)
