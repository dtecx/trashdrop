"""Which classifier sends an UNSEEN physical object to the right bin?

The number that matters at the venue is not accuracy on frames: frames of one
object are near-duplicates, so any split that puts the same bottle in training
and test flatters the model. What matters is an object nobody photographed.
So every local score here is leave-one-object-out: the model never saw any
frame of the object it is judged on.

The team's criterion is zero wrong-bin decisions. Anything the model is unsure
of -- or calls "other" -- goes to the mixed bin, which is a safe outcome, not
an error. So each variant is reported as correct / WRONG / mixed across
confidence thresholds, and the headline is the best coverage it reaches with
no wrong bins at all.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embed import CACHE, MODEL, PRETRAINED, ROUTES, cache_path, load_model, pick_device

TEMPLATES = (
    "a photo of {}.",
    "a photo of {} lying on a table, seen from above.",
    "a close-up photo of {}.",
)
PROMPTS = {
    "plastic": (
        "a plastic bottle", "a crushed plastic bottle", "a clear plastic bottle",
        "a plastic cup", "a plastic container", "plastic packaging", "a plastic wrapper",
    ),
    "paper": (
        "a piece of paper", "crumpled paper", "a piece of cardboard", "a paper cup",
        "a paper flyer", "a cardboard box", "a paper bag", "a newspaper",
    ),
    "metal": (
        "an aluminium can", "a crushed aluminium can", "a metal drink can",
        "a beer can", "a tin can", "aluminium foil",
    ),
    "other": (
        "a glass bottle", "food waste", "a banana peel", "a piece of fabric",
        "an empty table", "a battery", "a styrofoam tray",
    ),
}
THRESHOLDS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
LOCAL_WEIGHT = 10.0  # our crops are ~3 % of the combined set; let them count


def load(name: str):
    with np.load(cache_path(name), allow_pickle=False) as stored:
        return (
            stored["embeddings"].astype(np.float32),
            stored["routes"].astype(str),
            stored["groups"].astype(str),
        )


def text_weights() -> np.ndarray:
    """One unit vector per route: the mean of its prompt embeddings."""

    import torch

    target = CACHE / f"{MODEL}_{PRETRAINED}_text.npy"
    model, _, tokenizer = load_model(pick_device())
    rows = []
    for route in ROUTES:
        texts = [t.format(p) for p in PROMPTS[route] for t in TEMPLATES]
        with torch.no_grad():
            features = model.encode_text(tokenizer(texts).to(next(model.parameters()).device))
        features = features / features.norm(dim=-1, keepdim=True)
        mean = features.mean(dim=0)
        rows.append((mean / mean.norm()).float().cpu().numpy())
    weights = np.stack(rows)
    np.save(target, weights)
    return weights


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def zero_shot(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    return softmax(100.0 * X @ W.T)  # CLIP's own logit scale


def fit_probe(X, y, weights=None):
    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    probe.fit(X, y, sample_weight=weights)
    return probe


def probe_probs(probe, X) -> np.ndarray:
    """Probabilities in ROUTES order, zeros for routes the probe never saw."""

    out = np.zeros((len(X), len(ROUTES)), np.float32)
    probs = probe.predict_proba(X)
    for column, label in enumerate(probe.classes_):
        out[:, ROUTES.index(label)] = probs[:, column]
    return out


def decide(probs: np.ndarray, threshold: float) -> np.ndarray:
    best = probs.argmax(axis=1)
    routes = np.array(ROUTES)[best]
    unsure = probs.max(axis=1) < threshold
    return np.where(unsure | (routes == "other"), "mixed", routes)


def score(decisions, truth, groups) -> dict:
    correct = decisions == truth
    wrong = (decisions != truth) & (decisions != "mixed")
    objects = sorted(set(groups))
    per_object = [correct[groups == g].mean() for g in objects]
    return {
        "correct": int(correct.sum()),
        "wrong": int(wrong.sum()),
        "mixed": int((decisions == "mixed").sum()),
        "object_mean_correct": float(np.mean(per_object)),
        "objects_with_a_wrong_bin": [g for g in objects if wrong[groups == g].any()],
    }


def leave_one_object_out(X_local, y_local, g_local, make_probs) -> np.ndarray:
    """Probabilities for each local frame from a model that never saw its object."""

    probs = np.zeros((len(X_local), len(ROUTES)), np.float32)
    for obj in sorted(set(g_local)):
        held = g_local == obj
        probs[held] = make_probs(~held, held)
    return probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="test_1")
    args = parser.parse_args()

    public = [load(name) for name in ("trashnet", "realwaste", "drinking_waste")]
    X_pub = np.concatenate([p[0] for p in public])
    y_pub = np.concatenate([p[1] for p in public])
    X_loc, y_loc, g_loc = load(f"local_{args.session}")
    print(f"public images: {len(X_pub)}   local frames: {len(X_loc)} from {len(set(g_loc))} objects\n")

    W = text_weights()
    variants: dict[str, np.ndarray] = {}

    variants["CLIP zero-shot (no training)"] = zero_shot(X_loc, W)

    public_probe = fit_probe(X_pub, y_pub)
    variants["CLIP + head on public only"] = probe_probs(public_probe, X_loc)

    def local_only(train, test):
        return probe_probs(fit_probe(X_loc[train], y_loc[train]), X_loc[test])

    variants["CLIP + head on OUR crops only (LOO)"] = leave_one_object_out(X_loc, y_loc, g_loc, local_only)

    def combined(train, test):
        X = np.concatenate([X_pub, X_loc[train]])
        y = np.concatenate([y_pub, y_loc[train]])
        w = np.concatenate([np.ones(len(X_pub)), np.full(train.sum(), LOCAL_WEIGHT)])
        return probe_probs(fit_probe(X, y, w), X_loc[test])

    variants["CLIP + head on public + ours (LOO)"] = leave_one_object_out(X_loc, y_loc, g_loc, combined)
    variants["ensemble: zero-shot + public+ours"] = (
        variants["CLIP zero-shot (no training)"] + variants["CLIP + head on public + ours (LOO)"]
    ) / 2

    total = len(X_loc)
    print(f"{'variant':38s} {'thr':>5s} {'correct':>8s} {'WRONG':>6s} {'mixed':>6s} {'per-object':>10s}  objects with a wrong bin")
    headline = []
    for name, probs in variants.items():
        best_zero_wrong = None
        for threshold in THRESHOLDS:
            result = score(decide(probs, threshold), y_loc, g_loc)
            if result["wrong"] == 0 and best_zero_wrong is None:
                best_zero_wrong = (threshold, result)
            bad = ", ".join(result["objects_with_a_wrong_bin"]) or "-"
            print(
                f"{name:38s} {threshold:5.2f} {result['correct']:8d} {result['wrong']:6d} "
                f"{result['mixed']:6d} {result['object_mean_correct']:9.0%}  {bad}"
            )
        print()
        headline.append((name, best_zero_wrong))

    print("HEADLINE -- with ZERO wrong bins, how much does each variant sort correctly?")
    for name, found in headline:
        if found is None:
            print(f"  {name:38s} never reaches zero wrong bins")
        else:
            threshold, result = found
            print(
                f"  {name:38s} at threshold {threshold:.2f}: {result['correct']}/{total} frames correct "
                f"({result['correct'] / total:.0%}), per-object {result['object_mean_correct']:.0%}, "
                f"rest to mixed"
            )


if __name__ == "__main__":
    main()
