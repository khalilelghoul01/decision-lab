"""Small typed decision policies with genuine sampled calibration-aware RL.

An educational Jev-inspired experiment, not TypeSafe's unpublished RLCD recipe.
The published RLCR reward is adapted to direct numeric heads (no generated text).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

MODEL = "LiquidAI/LFM2.5-230M"
REVISION = "40cb2ad3b3044d5a41eee083a6103c8b523afa45"
SOURCES = {
    "boolq": ("google/boolq", "35b264d03638db9f4ce671b711558bf7ff0f80d5"),
    "sst2": ("stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"),
    "ag_news": ("fancyzhx/ag_news", "eb185aade064a813bc0b7f42de02595523103ca4"),
    "snli": ("stanfordnlp/snli", "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"),
}
LETTERS = "ABCD"
N_BINS = 21


@dataclass
class Config:
    output: str = "runs/first"
    seed: int = 42
    max_length: int = 256
    batch_size: int = 4
    accumulate: int = 4
    train_per_task: int = 1500
    eval_per_task: int = 200
    warmup_steps: int = 200
    rl_steps: int = 300
    group_size: int = 32
    adapter_lr: float = 5e-5
    head_lr: float = 3e-4
    rl_lr_scale: float = 0.3
    entropy_weight: float = 0.001
    lora_rank: int = 8
    checkpoint_every: int = 25
    log_every: int = 10
    max_minutes: float = 180.0
    device: str = "auto"


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize(task, row, index):
    if task == "boolq":
        context, question = row["passage"], row["question"]
        options, target = ["No", "Yes"], int(row["answer"])
    elif task == "sst2":
        context, question = row["sentence"], "What is the sentiment of this movie review?"
        options, target = ["Negative", "Positive"], int(row["label"])
    elif task == "ag_news":
        context, question = row["text"], "Which topic best describes this news item?"
        options, target = ["World", "Sports", "Business", "Science and technology"], int(row["label"])
    elif task == "snli":
        context = row["premise"]
        question = "Given the context, is this statement entailed, undetermined, or contradicted? " + row["hypothesis"]
        options, target = ["Entailed", "Undetermined", "Contradicted"], int(row["label"])
    else:
        raise ValueError(task)
    return {"id": f"{task}:{index}", "task": task, "context": context.strip(),
            "question": question.strip(), "options": options, "target": target}


def example_key(row):
    # Include the question: SNLI legitimately repeats premises across hypotheses.
    text = row["context"].strip().lower() + "\n" + row["question"].strip().lower()
    return hashlib.sha256(text.encode()).hexdigest()


def prepare_data(cfg):
    """Train/dev from official train; sealed evaluation from labeled held-out split."""
    from datasets import load_dataset
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    data_file = output / "data.json"
    if data_file.exists():
        result = json.loads(data_file.read_text())
        assert result["settings"] == [cfg.seed, cfg.train_per_task, cfg.eval_per_task], "Dataset config changed; use a new run folder."
        return result
    result = {"train": [], "dev": [], "test": [], "sources": SOURCES,
              "settings": [cfg.seed, cfg.train_per_task, cfg.eval_per_task]}
    for task, (name, revision) in SOURCES.items():
        print(f"Loading {name} @ {revision[:8]}", flush=True)
        ds = load_dataset(name, revision=revision)
        heldout = "test" if task in ("ag_news", "snli") else "validation"
        # Select a fixed test sample first, then remove exact overlap from training.
        candidates = ds[heldout].shuffle(seed=cfg.seed)
        seen = set()
        for i, row in enumerate(candidates):
            item = normalize(task, row, f"{heldout}:{i}")
            if item["target"] < 0 or example_key(item) in seen:
                continue
            seen.add(example_key(item))
            result["test"].append(item)
            if sum(x["task"] == task for x in result["test"]) == cfg.eval_per_task:
                break
        count = 0
        for i, row in enumerate(ds["train"].shuffle(seed=cfg.seed)):
            item = normalize(task, row, f"train:{i}")
            if item["target"] < 0 or example_key(item) in seen:
                continue
            seen.add(example_key(item))
            split = "dev" if count < cfg.eval_per_task else "train"
            result[split].append(item)
            count += 1
            if count >= cfg.eval_per_task + cfg.train_per_task:
                break
        assert count == cfg.eval_per_task + cfg.train_per_task
    for split in ("train", "dev", "test"):
        random.Random(cfg.seed).shuffle(result[split])
    sets = [set(map(example_key, result[s])) for s in ("train", "dev", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "Split overlap"
    data_file.write_text(json.dumps(result, ensure_ascii=False))
    return result


class DecisionPolicy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # FP32 avoids unsupported T4 BF16 kernels and keeps this small run stable.
        backbone = AutoModel.from_pretrained(MODEL, revision=REVISION,
                                             dtype=torch.float32, attn_implementation="eager")
        backbone.config.use_cache = False
        self.backbone = get_peft_model(backbone, LoraConfig(
            r=cfg.lora_rank, lora_alpha=2 * cfg.lora_rank, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], bias="none"))
        dim = backbone.config.hidden_size
        self.choice = nn.Linear(dim, 4)
        # One confidence-report policy per possible selected answer.
        self.confidence = nn.Linear(dim, 4 * N_BINS)
        nn.init.normal_(self.choice.weight, std=0.01)
        nn.init.zeros_(self.choice.bias)
        nn.init.zeros_(self.confidence.weight)
        nn.init.zeros_(self.confidence.bias)
        self.register_buffer("bins", torch.linspace(0, 1, N_BINS))

    def encode(self, rows, rng=None):
        sequences, targets, sizes = [], [], []
        tok = self.tokenizer
        for row in rows:
            order = list(range(len(row["options"])))
            if rng is not None:
                rng.shuffle(order)
            options = [row["options"][i] for i in order]
            targets.append(order.index(row.get("target", 0)))
            sizes.append(len(options))
            # Reserve the question/options; truncate only the context when necessary.
            suffix = "\nQuestion: " + row["question"] + "\nChoices:\n" + "\n".join(
                f"{LETTERS[i]}. {value}" for i, value in enumerate(options)) + "\nDecision:"
            prefix_ids = tok.encode("Context:\n", add_special_tokens=False)
            suffix_ids = tok.encode(suffix, add_special_tokens=False)
            budget = self.cfg.max_length - len(prefix_ids) - len(suffix_ids)
            if budget < 8:
                raise ValueError("Question and choices too long for max_length; shorten them or increase max_length.")
            context_ids = tok.encode(row["context"], add_special_tokens=False)[:budget]
            sequences.append(prefix_ids + context_ids + suffix_ids)
        padded = tok.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        return {**padded, "targets": torch.tensor(targets), "sizes": torch.tensor(sizes)}

    def forward(self, input_ids, attention_mask, sizes, **ignored):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False).last_hidden_state
        last = attention_mask.sum(-1) - 1
        pooled = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
        # Heads and categorical distributions stay in FP32, even under autocast.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = self.choice(pooled)
            logits = logits.masked_fill(torch.arange(4, device=logits.device)[None] >= sizes[:, None], -1e4)
            reports = self.confidence(pooled).reshape(-1, 4, N_BINS)
        return logits, reports


def rlcr_reward(correct, confidence):
    return correct.float() - (confidence - correct.float()).square()


def sampled_rl_loss(logits, reports, targets, bins, group_size=32, entropy_weight=0.001):
    """On-policy REINFORCE with an unbiased leave-one-out group baseline.

    Samples JOINT actions (answer, confidence report). The reward is detached;
    gradients flow through log pi(answer) + log pi(report | answer), not Brier loss.
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    answer_policy = Categorical(logits=logits.float())
    actions = answer_policy.sample((group_size,))  # G x B
    batch = torch.arange(len(targets), device=targets.device)[None].expand(group_size, -1)
    report_policy = Categorical(logits=reports.float()[batch, actions])
    report_actions = report_policy.sample()
    q = bins[report_actions]
    correct = actions.eq(targets[None]).float()
    rewards = rlcr_reward(correct, q).detach()
    baseline = (rewards.sum(0, keepdim=True) - rewards) / (group_size - 1)
    advantage = (rewards - baseline).detach()
    log_prob = answer_policy.log_prob(actions) + report_policy.log_prob(report_actions)
    # Exact joint entropy; includes the gradient through the answer mixture weights.
    report_entropy = Categorical(logits=reports.float()).entropy()
    entropy = answer_policy.entropy() + (answer_policy.probs * report_entropy).sum(-1)
    loss = -(advantage * log_prob).mean() - entropy_weight * entropy.mean()
    return loss, {"reward": rewards.mean().item(), "sample_accuracy": correct.mean().item(),
                  "sample_confidence": q.mean().item(), "entropy": entropy.mean().item()}


def warmup_loss(logits, reports, targets, bins):
    # Explicitly supervised warm-up. RL starts only after this phase finishes.
    target_correctness = F.one_hot(targets, 4).float()
    expected_brier = (reports.softmax(-1) * (bins[None, None] - target_correctness[..., None]).square()).sum(-1)
    valid = logits > -100
    calibration = expected_brier[valid].mean()
    return F.cross_entropy(logits, targets) + 0.2 * calibration


def reliability(correct, confidence, n_bins=10):
    correct, confidence = np.asarray(correct), np.asarray(confidence)
    assignments = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        mask = assignments == i
        n = int(mask.sum())
        if n:
            acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
            ece += n / len(correct) * abs(acc - conf)
            rows.append({"bin": i, "n": n, "accuracy": acc, "confidence": conf})
    return rows, ece


def metrics(records):
    y = np.array([r["correct"] for r in records], dtype=float)
    q = np.array([r["confidence"] for r in records], dtype=float)
    policy_q = np.array([r["policy_probability"] for r in records])
    rows, ece = reliability(y, q)
    clamped = q.clip(1e-6, 1 - 1e-6)
    n = len(y)
    # Wilson interval for accuracy; ECE is descriptive, especially on small samples.
    p, z = y.mean(), 1.96
    mid = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return {"n": n, "accuracy": float(p), "accuracy_95ci": [mid-half, mid+half],
            "confidence_brier": float(((q-y)**2).mean()), "ece_10bins": float(ece),
            "confidence_nll": float(-(y*np.log(clamped)+(1-y)*np.log(1-clamped)).mean()),
            "policy_top1_brier": float(((policy_q-y)**2).mean()),
            "choice_nll": float(np.mean([r["choice_nll"] for r in records])),
            "reliability": rows}


@torch.no_grad()
def evaluate(model, rows, cfg, tag, split="dev", permute_seed=None):
    model.eval()
    device = next(model.parameters()).device
    output, start = [], time.monotonic()
    rng = random.Random(permute_seed) if permute_seed is not None else None
    for offset in range(0, len(rows), cfg.batch_size):
        chunk = rows[offset:offset+cfg.batch_size]
        batch = {k: v.to(device) for k, v in model.encode(chunk, rng).items()}
        logits, reports = model(**batch)
        probs = logits.softmax(-1)
        selected = probs.argmax(-1)
        conf = (reports.softmax(-1) * model.bins).sum(-1)
        for i, item in enumerate(chunk):
            a = selected[i].item()
            output.append({"id": item["id"], "task": item["task"],
                           "correct": int(a == batch["targets"][i].item()),
                           "confidence": conf[i, a].item(),
                           "policy_probability": probs[i, a].item(),
                           "choice_nll": -probs[i, batch["targets"][i]].clamp_min(1e-9).log().item()})
    summary = {"overall": metrics(output), "tasks": {t: metrics([r for r in output if r["task"] == t]) for t in SOURCES},
               "seconds": time.monotonic()-start, "confidence_readout": "mean of sampled-report policy; evaluate this deployed readout separately"}
    path = Path(cfg.output) / f"{tag}-{split}"
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    path.with_suffix(".jsonl").write_text("\n".join(map(json.dumps, output)))
    print(f"{tag} {split}: " + json.dumps({k: v for k, v in summary["overall"].items() if k != "reliability"}), flush=True)
    return summary


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    named = dict(model.named_parameters())
    expected = {n for n, p in named.items() if p.requires_grad}
    if expected != set(state):
        raise ValueError("Checkpoint and model trainable parameter names differ")
    with torch.no_grad():
        for n, value in state.items():
            named[n].copy_(value.to(named[n]))


def save_checkpoint(model, optimizer, scaler, cfg, stage, step, rng, name="latest.pt"):
    state = {"format_version": 1, "model_id": MODEL, "model_revision": REVISION,
             "config": dataclasses.asdict(cfg), "stage": stage, "step": step,
             "trainable": trainable_state(model), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "python_rng": random.getstate(),
             "batch_rng": rng.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(),
             "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    destination = Path(cfg.output) / name
    temporary = destination.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)
    return destination


def run(cfg, resume=True):
    """Returns a model after completion or a clean, resumable time-budget stop."""
    seed_all(cfg.seed)
    root = Path(cfg.output)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        current = dataclasses.asdict(cfg)
        for key in current.keys() - {"output", "device", "max_minutes"}:
            if previous[key] != current[key]:
                raise ValueError(f"Config field {key} changed; use a new output directory to avoid mixing runs.")
    config_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    data = prepare_data(cfg)
    model = DecisionPolicy(cfg).to(device_for(cfg.device))
    device = next(model.parameters()).device
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}; trainable parameters: {trainable_count:,}", flush=True)
    import importlib.metadata
    versions = {k: importlib.metadata.version(k) for k in ("torch", "transformers", "peft", "datasets", "accelerate", "numpy")}
    (root / "environment.json").write_text(json.dumps({"python": platform.python_version(), "packages": versions,
        "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "model": MODEL, "revision": REVISION, "trainable_parameters": trainable_count}, indent=2))
    adapter = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    heads = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": adapter, "lr": cfg.adapter_lr}, {"params": heads, "lr": cfg.head_lr}], weight_decay=0.01)
    # Keep training in FP32 for stability on a T4; the small model fits comfortably.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    rng, stage, done = random.Random(cfg.seed + 1), "warmup", 0
    latest = root / "latest.pt"
    if resume and latest.exists():
        # Only load checkpoints produced by this notebook; pickle is not safe for untrusted files.
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        assert saved["model_revision"] == REVISION
        load_trainable(model, saved["trainable"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        stage, done = saved["stage"], saved["step"]
        rng.setstate(saved["batch_rng"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved.get("mps_rng") is not None and device.type == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        if saved["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        print(f"Resuming {stage} at optimizer step {done}", flush=True)
    started = time.monotonic()
    if stage == "complete":
        print("Run already complete. Use a new output folder for a new experiment.", flush=True)
        return model
    for phase, total in (("warmup", cfg.warmup_steps), ("rl", cfg.rl_steps)):
        if phase == "warmup" and stage == "rl":
            continue
        start_step = done if phase == stage else 0
        factor = cfg.rl_lr_scale if phase == "rl" else 1.0
        for group, base_lr in zip(optimizer.param_groups, (cfg.adapter_lr, cfg.head_lr)):
            group["lr"] = base_lr * factor
        model.train()
        for step in range(start_step, total):
            optimizer.zero_grad(set_to_none=True)
            stats, loss_sum = {}, 0.0
            for _ in range(cfg.accumulate):
                rows = [data["train"][rng.randrange(len(data["train"]))] for _ in range(cfg.batch_size)]
                batch = {k: v.to(device) for k, v in model.encode(rows, rng).items()}
                logits, reports = model(**batch)
                if phase == "warmup":
                    loss = warmup_loss(logits, reports, batch["targets"], model.bins)
                else:
                    loss, stats = sampled_rl_loss(logits, reports, batch["targets"], model.bins,
                                                 cfg.group_size, cfg.entropy_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; latest.pt is the last safe checkpoint")
                (loss / cfg.accumulate).backward()
                loss_sum += loss.item() / cfg.accumulate
            norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            completed = step + 1
            if completed % cfg.log_every == 0 or completed == 1:
                log = {"phase": phase, "step": completed, "total": total, "loss": loss_sum,
                       "grad_norm": float(norm), "minutes": (time.monotonic()-started)/60, **stats}
                print(json.dumps(log), flush=True)
                with (root / "training.jsonl").open("a") as f:
                    f.write(json.dumps(log) + "\n")
            if completed % cfg.checkpoint_every == 0 or completed == total:
                save_checkpoint(model, optimizer, scaler, cfg, phase, completed, rng)
            if (time.monotonic()-started)/60 >= cfg.max_minutes:
                save_checkpoint(model, optimizer, scaler, cfg, phase, completed, rng)
                print("Time budget reached. Saved latest.pt with optimizer and RNG states; rerun to resume.", flush=True)
                return model
        if phase == "warmup":
            save_checkpoint(model, optimizer, scaler, cfg, "warmup", total, rng, "warmup.pt")
            evaluate(model, data["dev"], cfg, "warmup")
            # Record boundary so resuming won't repeat warm-up.
            save_checkpoint(model, optimizer, scaler, cfg, "rl", 0, rng)
            stage, done = "rl", 0
        else:
            save_checkpoint(model, optimizer, scaler, cfg, "rl", total, rng, "rl.pt")
            evaluate(model, data["dev"], cfg, "rl")
    # A fixed run, no checkpoint selection on test. Compare both stages only now.
    final_weights = trainable_state(model)
    warmup = torch.load(root / "warmup.pt", map_location="cpu", weights_only=False)
    load_trainable(model, warmup["trainable"])
    evaluate(model, data["test"], cfg, "warmup", "test")
    load_trainable(model, final_weights)
    evaluate(model, data["test"], cfg, "rl", "test")
    evaluate(model, data["test"], cfg, "rl-permuted", "test", permute_seed=cfg.seed+99)
    save_checkpoint(model, optimizer, scaler, cfg, "complete", cfg.rl_steps, rng)
    model.backbone.save_pretrained(root / "adapter", safe_serialization=True)
    model.tokenizer.save_pretrained(root / "adapter")
    torch.save({"choice": model.choice.state_dict(), "confidence": model.confidence.state_dict()}, root / "heads.pt")
    print(f"Complete. Results and resumable weights: {root}", flush=True)
    return model


@torch.no_grad()
def decide(model, context, questions):
    """Batch a shared context with arbitrary questions and 2-4 named choices.

    Questions are independent. This prototype repeats context per question; it
    does not implement Jev's alleged shared-context branch attention architecture.
    """
    for q in questions:
        if not 2 <= len(q["choices"]) <= 4 or len(set(q["choices"])) != len(q["choices"]):
            raise ValueError("Each question needs 2-4 distinct choices")
    model.eval()
    device = next(model.parameters()).device
    rows = [{"context": context, "question": q["question"], "options": q["choices"]} for q in questions]
    batch = {k: v.to(device) for k, v in model.encode(rows).items()}
    logits, reports = model(**batch)
    probabilities = logits.softmax(-1)
    confidence = (reports.softmax(-1) * model.bins).sum(-1)
    answer = []
    for i, question in enumerate(questions):
        a = probabilities[i].argmax().item()
        answer.append({"type": "choice", "question": question["question"],
                       "choice": question["choices"][a], "confidence": confidence[i, a].item(),
                       "policy_probabilities": dict(zip(question["choices"], probabilities[i, :len(question["choices"])].tolist()))})
    return answer


def load_run(path, device="auto"):
    root = Path(path)
    cfg = Config(**json.loads((root / "config.json").read_text()))
    model = DecisionPolicy(cfg).to(device_for(device))
    saved = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
    load_trainable(model, saved["trainable"])
    return model.eval()


def report(path):
    """Display fixed held-out accuracy and confidence calibration comparisons."""
    import matplotlib.pyplot as plt
    root = Path(path)
    files = [root / "warmup-test.json", root / "rl-test.json"]
    if not all(p.exists() for p in files):
        raise RuntimeError("Training is not complete; rerun the training cell to resume before reporting.")
    summaries = [json.loads(p.read_text()) for p in files]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = ["#64748b", "#1565c0"]
    for label, summary, color in zip(("Supervised warm-up", "After calibration RL"), summaries, colors):
        m = summary["overall"]
        print(f"{label}: n={m['n']}, accuracy={m['accuracy']:.3f}, confidence Brier={m['confidence_brier']:.3f}, ECE={m['ece_10bins']:.3f}")
        bins = m["reliability"]
        axes[0].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", label=label, color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8", label="Perfect calibration")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean reported confidence", ylabel="Observed correctness", title="Held-out reliability · 10 bins")
    axes[0].legend(fontsize=8)
    x = np.arange(len(SOURCES))
    for j, (summary, color) in enumerate(zip(summaries, colors)):
        vals = [summary["tasks"][t]["accuracy"] for t in SOURCES]
        axes[1].bar(x + (j-0.5)*0.35, vals, 0.35, color=color)
    axes[1].set(xticks=x, xticklabels=list(SOURCES), ylim=(0, 1), ylabel="Accuracy", title="Accuracy by task · held-out examples")
    fig.savefig(root / "evaluation.png", dpi=160)
    plt.show()
    print("Lower Brier/ECE is better. RL can fail to improve either metric; inspect each task and choice-order sensitivity.")
    return summaries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/first")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = Config(output=args.output)
    if args.smoke:
        config = Config(output=args.output, train_per_task=8, eval_per_task=4,
                        warmup_steps=2, rl_steps=2, batch_size=2, accumulate=1,
                        max_length=128, checkpoint_every=1, log_every=1)
    run(config)
