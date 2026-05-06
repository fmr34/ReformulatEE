"""Tests for epistemic effectiveness reward computation."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestEEBasics:
    """Test basic EE scoring logic."""

    def test_ee_bounds(self):
        """EE scores should be in [0, 1]."""
        from src.ee.reward import _NullIndex
        from src.ee.reward import compute_ee

        # Use null index for testing (doesn't need corpus)
        null_idx = _NullIndex()

        result = compute_ee("What is consciousness?", "What is consciousness?", null_idx)

        assert 0.0 <= result.ee <= 1.0, f"EE out of bounds: {result.ee}"
        assert 0.0 <= result.respondibilidade <= 1.0
        assert 0.0 <= result.tratabilidade <= 1.0
        assert 0.0 <= result.nao_trivialidade <= 1.0

    def test_ee_components_sum(self):
        """EE should be weighted sum of components."""
        from src.ee.reward import BETA1
        from src.ee.reward import BETA2
        from src.ee.reward import BETA3
        from src.ee.reward import _NullIndex
        from src.ee.reward import compute_ee

        null_idx = _NullIndex()
        result = compute_ee("Test question", "Test question", null_idx)

        expected_ee = (
            BETA1 * result.respondibilidade
            + BETA2 * result.tratabilidade
            + BETA3 * result.nao_trivialidade
        )

        assert abs(result.ee - expected_ee) < 1e-5, "EE computation mismatch"

    def test_identical_questions_have_low_proximity(self):
        """Identical questions should have 0 proximity (low reward for triviality)."""
        from src.ee.reward import _NullIndex
        from src.ee.reward import compute_ee

        null_idx = _NullIndex()
        result = compute_ee("Question", "Question", null_idx)

        # Identical question should have 0 proximity (no improvement)
        assert result.prox == 0.0, f"Identical questions should have 0 proximity, got {result.prox}"

    def test_ee_different_prompts(self):
        """Different questions should yield different EE scores."""
        from src.ee.reward import _NullIndex
        from src.ee.reward import compute_ee

        null_idx = _NullIndex()

        q_vague = "What is life?"
        q_specific = "What biological processes differentiate living from non-living systems?"

        result_vague = compute_ee(q_vague, q_vague, null_idx)
        result_specific = compute_ee(q_specific, q_vague, null_idx)

        # Specific question should have different (likely higher) non-triviality
        # than comparing the same vague question to itself
        assert result_vague.nao_trivialidade != result_specific.nao_trivialidade


class TestStage1Filter:
    """Test Stage 1 epistemic filter."""

    def test_stage1_pass_condition(self):
        """Stage 1 filter should pass if EE(cand) > EE(bad) + epsilon."""
        from src.ee.reward import _EPSILON
        from src.ee.reward import passes_stage1_filter

        ee_bad = 0.50
        ee_cand_pass = ee_bad + _EPSILON + 0.01  # Slightly above threshold
        ee_cand_fail = ee_bad + _EPSILON - 0.01  # Slightly below threshold

        assert passes_stage1_filter(ee_bad, ee_cand_pass), "Should pass above threshold"
        assert not passes_stage1_filter(ee_bad, ee_cand_fail), "Should fail below threshold"

    def test_stage1_at_boundary(self):
        """Test exact boundary conditions."""
        from src.ee.reward import _EPSILON
        from src.ee.reward import passes_stage1_filter

        ee_bad = 0.50

        # Exactly at threshold (should fail)
        ee_at_threshold = ee_bad + _EPSILON
        assert not passes_stage1_filter(ee_bad, ee_at_threshold), "Should fail at exact threshold"

        # Just above threshold (should pass)
        ee_above = ee_bad + _EPSILON + 1e-6
        assert passes_stage1_filter(ee_bad, ee_above), "Should pass just above threshold"


class TestComputeScore:
    """Test score weighting with alpha parameter."""

    def test_score_monotonic_in_ee(self):
        """Score should increase monotonically with EE when proximity is fixed."""
        from src.ee.reward import compute_score

        # Higher EE should yield higher score
        score_low_ee = compute_score(
            type("Result", (), {"ee": 0.3, "prox": 0.5})(),  # Mock result
            alpha=0.5,
        )
        score_high_ee = compute_score(
            type("Result", (), {"ee": 0.8, "prox": 0.5})(),  # Mock result
            alpha=0.5,
        )

        assert score_high_ee > score_low_ee, "Higher EE should yield higher score"

    def test_score_alpha_zero(self):
        """When alpha=0, score should depend only on proximity."""
        from src.ee.reward import compute_score

        result = type("Result", (), {"ee": 0.5, "prox": 0.7})()
        score_alpha_zero = compute_score(result, alpha=0.0)

        # With alpha=0, score should be just proximity
        assert abs(score_alpha_zero - 0.7) < 1e-5


# Markers for different test categories
@pytest.mark.slow
class TestIntegration:
    """Integration tests (require corpus, may be slow)."""

    def test_corpus_available(self):
        """Check if corpus index is available for integration tests."""
        from pathlib import Path

        corpus_idx = Path("data/corpus/bm25_index.pkl")

        if not corpus_idx.exists():
            pytest.skip("Corpus index not available")
