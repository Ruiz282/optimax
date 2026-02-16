"""
Unit Tests for Case 2: ChatGPT Launch 2022
MGMT 69000: Mastering AI for Finance

Tests validate:
1. Sector entropy calculations
2. Concentration metrics (HHI, CR)
3. Sample space expansion claims
4. Creative destruction measurements
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from starter_template import (
    sector_entropy,
    calculate_concentration_ratio,
    entropy_change_analysis,
    calculate_returns,
    DISRUPTION_CASCADE,
)

from validate_thesis import (
    sector_entropy as validate_sector_entropy,
    max_entropy,
    normalized_entropy,
    herfindahl_index,
    WEIGHTS_NOV_2022,
    WEIGHTS_NOV_2024,
    MAG7_WEIGHT_NOV_2022,
    MAG7_WEIGHT_NOV_2024,
)


# ============================================================
# ENTROPY CALCULATION TESTS
# ============================================================

class TestSectorEntropy:
    """Test sector entropy calculations."""

    def test_uniform_distribution(self):
        """Uniform distribution should give maximum entropy."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = sector_entropy(weights)
        expected = 2.0  # log2(4) = 2 bits
        assert abs(result - expected) < 0.001

    def test_concentrated_distribution(self):
        """Highly concentrated distribution should give low entropy."""
        weights = np.array([0.97, 0.01, 0.01, 0.01])
        result = sector_entropy(weights)
        # Should be much less than uniform
        assert result < 0.5

    def test_single_sector(self):
        """Single sector (100%) should give entropy = 0."""
        weights = np.array([1.0])
        result = sector_entropy(weights)
        assert result == 0.0

    def test_handles_zeros(self):
        """Should handle zero weights correctly."""
        weights = np.array([0.5, 0.5, 0.0, 0.0])
        result = sector_entropy(weights)
        expected = 1.0  # log2(2) = 1 bit for two 50% weights
        assert abs(result - expected) < 0.001

    def test_normalizes_weights(self):
        """Should normalize weights that don't sum to 1."""
        weights = np.array([1, 1, 1, 1])  # Not normalized
        result = sector_entropy(weights)
        expected = 2.0  # Should be same as [0.25, 0.25, 0.25, 0.25]
        assert abs(result - expected) < 0.001

    def test_empty_array(self):
        """Should handle empty arrays."""
        weights = np.array([])
        result = sector_entropy(weights)
        assert result == 0.0


class TestMaxEntropy:
    """Test maximum entropy calculations."""

    def test_max_entropy_4_items(self):
        """Max entropy for 4 items should be log2(4) = 2."""
        result = max_entropy(4)
        assert abs(result - 2.0) < 0.001

    def test_max_entropy_10_items(self):
        """Max entropy for 10 items should be log2(10)."""
        result = max_entropy(10)
        assert abs(result - np.log2(10)) < 0.001


class TestNormalizedEntropy:
    """Test normalized entropy (0-1 scale)."""

    def test_uniform_gives_1(self):
        """Uniform distribution should give normalized entropy = 1."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = normalized_entropy(weights)
        assert abs(result - 1.0) < 0.001

    def test_concentrated_gives_low(self):
        """Concentrated distribution should give low normalized entropy."""
        weights = np.array([0.97, 0.01, 0.01, 0.01])
        result = normalized_entropy(weights)
        assert result < 0.3

    def test_range_0_to_1(self):
        """Normalized entropy should always be between 0 and 1."""
        test_weights = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.33, 0.33, 0.34]),
            np.array([0.9, 0.05, 0.05]),
        ]
        for weights in test_weights:
            result = normalized_entropy(weights)
            assert 0.0 <= result <= 1.0


# ============================================================
# CONCENTRATION METRIC TESTS
# ============================================================

class TestHerfindahlIndex:
    """Test HHI calculation."""

    def test_uniform_4_firms(self):
        """4 equal firms should give HHI = 0.25."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        result = herfindahl_index(weights)
        assert abs(result - 0.25) < 0.001

    def test_monopoly(self):
        """Single firm should give HHI = 1.0."""
        weights = np.array([1.0])
        result = herfindahl_index(weights)
        assert abs(result - 1.0) < 0.001

    def test_concentrated_high_hhi(self):
        """Concentrated market should have high HHI."""
        weights = np.array([0.9, 0.05, 0.05])
        result = herfindahl_index(weights)
        # 0.9^2 + 0.05^2 + 0.05^2 = 0.81 + 0.0025 + 0.0025 = 0.815
        assert abs(result - 0.815) < 0.001


class TestConcentrationRatio:
    """Test concentration ratio (CR_n)."""

    def test_cr7_all_equal(self):
        """CR7 of 10 equal weights should be 0.7."""
        weights = np.array([0.1] * 10)
        result = calculate_concentration_ratio(weights, top_n=7)
        assert abs(result - 0.7) < 0.001

    def test_cr7_concentrated(self):
        """CR7 should capture top 7 weights."""
        weights = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01])
        result = calculate_concentration_ratio(weights, top_n=7)
        # Top 7: 0.3 + 0.2 + 0.15 + 0.1 + 0.1 + 0.05 + 0.05 = 0.95
        assert abs(result - 0.95) < 0.001


# ============================================================
# ENTROPY CHANGE ANALYSIS TESTS
# ============================================================

class TestEntropyChangeAnalysis:
    """Test entropy change analysis function."""

    def test_concentration_increase(self):
        """Entropy should decrease when market concentrates."""
        weights_before = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform
        weights_after = [0.5, 0.2, 0.15, 0.1, 0.05]  # Concentrated

        result = entropy_change_analysis(weights_before, weights_after)

        assert result["entropy_before"] > result["entropy_after"]
        assert result["entropy_change"] < 0  # Negative change

    def test_diversification_increase(self):
        """Entropy should increase when market diversifies."""
        weights_before = [0.5, 0.2, 0.15, 0.1, 0.05]  # Concentrated
        weights_after = [0.2, 0.2, 0.2, 0.2, 0.2]  # Uniform

        result = entropy_change_analysis(weights_before, weights_after)

        assert result["entropy_before"] < result["entropy_after"]
        assert result["entropy_change"] > 0  # Positive change


# ============================================================
# MARKET WEIGHT TESTS (CASE-SPECIFIC)
# ============================================================

class TestMarketWeights:
    """Test that market weight data is reasonable."""

    def test_weights_sum_to_1(self):
        """Sector weights should sum to approximately 1."""
        sum_2022 = sum(WEIGHTS_NOV_2022.values())
        sum_2024 = sum(WEIGHTS_NOV_2024.values())

        assert abs(sum_2022 - 1.0) < 0.05  # Allow 5% tolerance
        assert abs(sum_2024 - 1.0) < 0.05

    def test_tech_weight_increased(self):
        """Technology weight should have increased after ChatGPT."""
        tech_2022 = WEIGHTS_NOV_2022["Technology"]
        tech_2024 = WEIGHTS_NOV_2024["Technology"]

        assert tech_2024 > tech_2022

    def test_mag7_weight_increased(self):
        """Mag 7 weight should have increased after ChatGPT."""
        assert MAG7_WEIGHT_NOV_2024 > MAG7_WEIGHT_NOV_2022

    def test_mag7_weight_reasonable(self):
        """Mag 7 weights should be reasonable percentages."""
        assert 0.15 <= MAG7_WEIGHT_NOV_2022 <= 0.30
        assert 0.25 <= MAG7_WEIGHT_NOV_2024 <= 0.40


# ============================================================
# DISRUPTION CASCADE TESTS
# ============================================================

class TestDisruptionCascade:
    """Test disruption cascade structure."""

    def test_has_required_keys(self):
        """Cascade should have all required structure."""
        assert "trigger" in DISRUPTION_CASCADE
        assert "direct_effects" in DISRUPTION_CASCADE
        assert "second_order" in DISRUPTION_CASCADE

    def test_trigger_has_chatgpt_date(self):
        """Trigger should reference ChatGPT launch date."""
        trigger_date = DISRUPTION_CASCADE["trigger"]["date"]
        assert trigger_date == "2022-11-30"

    def test_direct_effects_has_winners_and_losers(self):
        """Direct effects should have winners and losers."""
        direct = DISRUPTION_CASCADE["direct_effects"]
        assert "winners" in direct
        assert "losers" in direct
        assert len(direct["winners"]) > 0
        assert len(direct["losers"]) > 0

    def test_nvidia_is_winner(self):
        """Nvidia should be listed as a winner."""
        winners = DISRUPTION_CASCADE["direct_effects"]["winners"]
        nvidia_found = any(w["ticker"] == "NVDA" for w in winners)
        assert nvidia_found

    def test_chegg_is_loser(self):
        """Chegg should be listed as a loser."""
        losers = DISRUPTION_CASCADE["direct_effects"]["losers"]
        chegg_found = any(l["ticker"] == "CHGG" for l in losers)
        assert chegg_found


# ============================================================
# RETURNS CALCULATION TESTS
# ============================================================

class TestReturnsCalculation:
    """Test return calculations."""

    def test_positive_return(self):
        """Should correctly calculate positive return."""
        dates = pd.date_range("2022-11-30", periods=100, freq="D")
        data = pd.DataFrame({
            "Close": np.linspace(100, 200, 100)  # 100% return
        }, index=dates)

        result = calculate_returns(data)
        assert abs(result["total_return"] - 100.0) < 1.0  # ~100% return

    def test_negative_return(self):
        """Should correctly calculate negative return."""
        dates = pd.date_range("2022-11-30", periods=100, freq="D")
        data = pd.DataFrame({
            "Close": np.linspace(100, 50, 100)  # -50% return
        }, index=dates)

        result = calculate_returns(data)
        assert abs(result["total_return"] - (-50.0)) < 1.0  # ~-50% return

    def test_volatility_positive(self):
        """Volatility should always be positive."""
        dates = pd.date_range("2022-11-30", periods=100, freq="D")
        data = pd.DataFrame({
            "Close": 100 + np.random.randn(100) * 5
        }, index=dates)

        result = calculate_returns(data)
        assert result["volatility"] > 0


# ============================================================
# THESIS CLAIM TESTS
# ============================================================

class TestThesisClaims:
    """Test the core thesis claims of the case."""

    def test_entropy_decreased(self):
        """Entropy should have decreased (concentration increased)."""
        weights_before = list(WEIGHTS_NOV_2022.values())
        weights_after = list(WEIGHTS_NOV_2024.values())

        entropy_before = validate_sector_entropy(np.array(weights_before))
        entropy_after = validate_sector_entropy(np.array(weights_after))

        assert entropy_after < entropy_before, "Entropy should decrease with concentration"

    def test_sample_space_expansion_vs_regime_shift(self):
        """
        Sample space expansion is different from regime shift.

        Regime shift: P changed (transition probabilities)
        Sample space expansion: X changed (universe itself)
        """
        # This is a conceptual test - verify the distinction exists
        # Week 1 (Tariff): Policy probabilities changed
        # Week 3 (ChatGPT): Investment universe changed

        # Evidence of sample space expansion:
        # 1. New allocation category ("AI infrastructure")
        # 2. New ETFs launched
        # 3. New risk factors

        # The key insight is that X changed, not just P
        # This is validated by the emergence of a new asset class
        assert MAG7_WEIGHT_NOV_2024 > MAG7_WEIGHT_NOV_2022
        assert WEIGHTS_NOV_2024["Technology"] > WEIGHTS_NOV_2022["Technology"]


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestIntegration:
    """Integration tests requiring network access."""

    @pytest.mark.skipif(
        True,  # Skip by default; set to False to run live tests
        reason="Requires network access for live data"
    )
    def test_fetch_nvidia_data(self):
        """Test fetching Nvidia data."""
        from starter_template import fetch_stock_data

        data = fetch_stock_data("NVDA", "2022-11-30", "2023-06-01")

        assert len(data) > 100
        assert "Close" in data.columns

    @pytest.mark.skipif(
        True,  # Skip by default
        reason="Requires network access for live data"
    )
    def test_fetch_chegg_data(self):
        """Test fetching Chegg data."""
        from starter_template import fetch_stock_data

        data = fetch_stock_data("CHGG", "2022-11-30", "2023-06-01")

        assert len(data) > 100
        assert "Close" in data.columns


# ============================================================
# RUN TESTS
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
