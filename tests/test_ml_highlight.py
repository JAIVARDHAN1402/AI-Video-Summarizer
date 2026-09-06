"""M7 highlight model: KTS, knapsack F1, the scorer, and the runtime no-op."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml import evaluate as mev  # noqa: E402
from ml import infer as minfer  # noqa: E402
from ml import kts  # noqa: E402
from ml.model import ImportanceScorer, ModelConfig, masked_ranking_loss  # noqa: E402


def test_kts_finds_the_obvious_boundary():
    rng = np.random.default_rng(0)
    feats = np.vstack([
        rng.normal(0.0, 0.25, (50, 12)),
        rng.normal(5.0, 0.25, (50, 12)),
    ]).astype(np.float32)
    cps = kts.cpd_auto(feats, max_ncp=8)
    assert len(cps) >= 1
    assert min(abs(int(c) - 50) for c in cps) <= 3
    bounds = kts.segment_bounds(len(feats), cps)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(feats)
    assert all(b[0] < b[1] for b in bounds)


def test_cpd_fixed_returns_exactly_the_requested_change_points():
    """The DP counts segments, backtrack counts change points — an off-by-one
    here silently halves the shot granularity instead of failing."""
    rng = np.random.default_rng(1)
    feats = np.vstack([
        rng.normal(0.0, 0.25, (100, 16)),
        rng.normal(6.0, 0.25, (100, 16)),
    ]).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)

    one = kts.cpd_fixed(feats, 1)
    assert len(one) == 1
    assert abs(int(one[0]) - 100) <= 3          # lands on the real boundary

    for k in (1, 5, 17):
        assert len(kts.cpd_fixed(feats, k)) == k
        assert len(kts.segment_bounds(len(feats), kts.cpd_fixed(feats, k))) == k + 1


def test_knapsack_respects_budget_and_picks_high_scoring_shots():
    shots = [(i * 5, (i + 1) * 5) for i in range(20)]      # 100 frames
    scores = np.zeros(100)
    scores[10:15] = 1.0
    scores[55:60] = 0.9
    mask = mev.knapsack_summary(scores, shots, ratio=0.15)
    assert mask.sum() <= 15
    assert mask[10:15].all() and mask[55:60].all()


def test_summary_f1_perfect_prediction():
    shots_cp = np.arange(5, 100, 5)
    scores = np.zeros(100)
    scores[10:15] = 1.0
    scores[55:60] = 1.0
    # both annotators pick exactly the two hot shots -> a prediction that also
    # picks them scores a perfect F1.
    users = np.zeros((100, 2))
    users[10:15, :] = 1
    users[55:60, :] = 1
    f1 = mev.summary_f1(scores, shots_cp, users, dataset="summe")
    assert f1 == pytest.approx(1.0, abs=1e-6)


def test_model_forward_backward_and_roundtrip(tmp_path):
    m = ImportanceScorer(ModelConfig(input_dim=64, hidden_dim=32))
    x = torch.randn(3, 40, 64)
    lengths = torch.tensor([40, 25, 12])
    y = torch.rand(3, 40)
    mask = torch.zeros(3, 40)
    for k, L in enumerate(lengths):
        mask[k, :L] = 1
    out = m(x, lengths)
    assert out.shape == (3, 40)
    assert (out >= 0).all() and (out <= 1).all()
    loss = torch.nn.functional.mse_loss(out, y) + masked_ranking_loss(out, y, mask)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())

    ckpt = tmp_path / "m.pt"
    m.save(ckpt)
    m2 = ImportanceScorer.load(ckpt)          # loads in eval() mode
    m.eval()                                  # match it — no dropout on either side
    with torch.no_grad():
        assert torch.allclose(m(x, lengths), m2(x, lengths), atol=1e-5)


def test_infer_is_noop_without_checkpoint(tmp_path):
    missing = tmp_path / "nope.pt"
    assert not minfer.available(missing)
    assert minfer.score_scene_embeddings(np.zeros((5, 512), np.float32), missing) is None
    data = {"scenes": [{"id": 0, "clip_embedding_idx": 0}], "embeddings_file": str(tmp_path / "e.npy")}
    assert minfer.attach_importance(data, tmp_path, missing) is data
