"""Tests for epistemic effectiveness reward computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestEEConstants:
    """Test that public constants are exported and have expected types/ranges."""

    def test_beta_values_are_floats(self):
        from src.ee.reward import BETA1
        from src.ee.reward import BETA2
        from src.ee.reward import BETA3

        assert isinstance(BETA1, float)
        assert isinstance(BETA2, float)
        assert isinstance(BETA3, float)

    def test_betas_sum_to_one(self):
        from src.ee.reward import BETA1
        from src.ee.reward import BETA2
        from src.ee.reward import BETA3

        assert abs(BETA1 + BETA2 + BETA3 - 1.0) < 1e-5, "Betas should sum to 1.0"

    def test_epsilon_positive(self):
        from src.ee.reward import EPSILON

        assert EPSILON > 0.0, "Epsilon must be positive"

    def test_beta3_dominates(self):
        """NT (nao_trivialidade) should have the largest weight."""
        from src.ee.reward import BETA1
        from src.ee.reward import BETA2
        from src.ee.reward import BETA3

        assert BETA3 > BETA1, "BETA3 (NT) should dominate BETA1 (R)"
        assert BETA3 > BETA2, "BETA3 (NT) should dominate BETA2 (T)"


class TestEEResult:
    """Test EEResult dataclass structure."""

    def test_ee_result_fields(self):
        from src.ee.reward import EEResult

        r = EEResult(
            query="test",
            respondibilidade=0.5,
            tratabilidade=0.6,
            tratabilidade_confidence=0.8,
            nao_trivialidade=0.7,
            ee=0.65,
        )
        assert r.query == "test"
        assert r.ee == 0.65
        assert r.prox is None
        assert r.score is None

    def test_ee_result_betas_default(self):
        from src.ee.reward import BETA1
        from src.ee.reward import BETA2
        from src.ee.reward import BETA3
        from src.ee.reward import EEResult

        r = EEResult(
            query="q",
            respondibilidade=0.0,
            tratabilidade=0.0,
            tratabilidade_confidence=1.0,
            nao_trivialidade=0.0,
            ee=0.0,
        )
        assert r.betas == (BETA1, BETA2, BETA3)


class TestStage1Filter:
    """Test Stage 1 epistemic filter with EEResult objects."""

    def _make_result(self, ee_val: float):
        from src.ee.reward import EEResult

        return EEResult(
            query="test",
            respondibilidade=0.0,
            tratabilidade=0.0,
            tratabilidade_confidence=1.0,
            nao_trivialidade=ee_val,
            ee=ee_val,
        )

    def test_stage1_pass_condition(self):
        from src.ee.reward import EPSILON
        from src.ee.reward import passes_stage1_filter

        ee_bad = 0.50
        r_bad = self._make_result(ee_bad)
        r_pass = self._make_result(ee_bad + EPSILON + 0.01)
        r_fail = self._make_result(ee_bad + EPSILON - 0.01)

        assert passes_stage1_filter(r_pass, r_bad), "Should pass above threshold"
        assert not passes_stage1_filter(r_fail, r_bad), "Should fail below threshold"

    def test_stage1_at_boundary(self):
        from src.ee.reward import EPSILON
        from src.ee.reward import passes_stage1_filter

        ee_bad = 0.50
        r_bad = self._make_result(ee_bad)
        r_exact = self._make_result(ee_bad + EPSILON)
        r_above = self._make_result(ee_bad + EPSILON + 1e-6)

        assert not passes_stage1_filter(r_exact, r_bad), "Should fail at exact threshold"
        assert passes_stage1_filter(r_above, r_bad), "Should pass just above threshold"

    def test_stage1_custom_epsilon(self):
        from src.ee.reward import passes_stage1_filter

        r_bad = self._make_result(0.50)
        r_cand = self._make_result(0.60)

        assert passes_stage1_filter(r_cand, r_bad, epsilon=0.05)
        assert not passes_stage1_filter(r_cand, r_bad, epsilon=0.15)


class TestComputeScore:
    """Test score weighting with alpha parameter."""

    def _make_result(self, ee: float, prox: float):
        from src.ee.reward import EEResult

        return EEResult(
            query="test",
            respondibilidade=0.0,
            tratabilidade=0.0,
            tratabilidade_confidence=1.0,
            nao_trivialidade=ee,
            ee=ee,
            prox=prox,
        )

    def test_score_monotonic_in_ee(self):
        from src.ee.reward import compute_score

        r_low = self._make_result(ee=0.3, prox=0.5)
        r_high = self._make_result(ee=0.8, prox=0.5)

        assert compute_score(r_high, alpha=0.5) > compute_score(r_low, alpha=0.5)

    def test_score_alpha_zero(self):
        from src.ee.reward import compute_score

        r = self._make_result(ee=0.5, prox=0.7)
        assert abs(compute_score(r, alpha=0.0) - 0.7) < 1e-5

    def test_score_alpha_one(self):
        from src.ee.reward import compute_score

        r = self._make_result(ee=0.5, prox=0.7)
        assert abs(compute_score(r, alpha=1.0) - 0.5) < 1e-5

    def test_score_stored_on_result(self):
        from src.ee.reward import compute_score

        r = self._make_result(ee=0.5, prox=0.5)
        s = compute_score(r, alpha=0.5)
        assert r.score == s

    def test_score_raises_without_prox(self):
        from src.ee.reward import EEResult
        from src.ee.reward import compute_score

        r = EEResult(
            query="q",
            respondibilidade=0.0,
            tratabilidade=0.0,
            tratabilidade_confidence=1.0,
            nao_trivialidade=0.5,
            ee=0.5,
            prox=None,
        )
        with pytest.raises(ValueError):
            compute_score(r)


@pytest.mark.slow
class TestIntegration:
    """Integration tests (require corpus, may be slow)."""

    def test_corpus_available(self):
        corpus_idx = Path("data/corpus/bm25_index.pkl")
        if not corpus_idx.exists():
            pytest.skip("Corpus index not available")
        assert corpus_idx.stat().st_size > 0
