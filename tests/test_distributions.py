"""
Tests for core/distributions.py and RNG isolation across the 5 simulation engines.

Verifies: triplet validation, PERT edge cases (symmetric / skewed / degenerate),
deterministic seeding, and that engine construction/simulation NEVER mutates
numpy's process-wide random state.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.distributions import (  # noqa: E402
    generate_pert_samples,
    make_rng,
    validate_triplet,
)
from core.energy_simulation import EnergyRiskEngine          # noqa: E402
from core.reliability_engine import ReliabilityEngine        # noqa: E402
from core.supply_chain_engine import SupplyChainEngine       # noqa: E402
from core.carbon_tax_engine import CarbonTaxEngine           # noqa: E402
from core.monte_carlo import MonteCarloEngine                # noqa: E402

ALL_ENGINES = [MonteCarloEngine, EnergyRiskEngine, ReliabilityEngine,
               SupplyChainEngine, CarbonTaxEngine]


# --------------------------------------------------------------------------- #
# Triplet validation
# --------------------------------------------------------------------------- #
def test_validate_triplet_accepts_valid_ordering():
    assert validate_triplet(1, 2, 3) == (1.0, 2.0, 3.0)
    assert validate_triplet(5, 5, 5) == (5.0, 5.0, 5.0)


@pytest.mark.parametrize("triplet", [(3, 2, 1), (2, 5, 3), (1, 0, 3)])
def test_validate_triplet_raises_on_inverted_ranges(triplet):
    with pytest.raises(ValueError):
        validate_triplet(*triplet)


def test_validate_triplet_rejects_non_finite():
    with pytest.raises(ValueError):
        validate_triplet(0, float("nan"), 1)
    with pytest.raises(ValueError):
        validate_triplet(0, 1, float("inf"))


# --------------------------------------------------------------------------- #
# PERT edge cases
# --------------------------------------------------------------------------- #
def test_pert_symmetric_triplet_stays_in_bounds():
    samples = generate_pert_samples(10, 20, 30, size=20_000)
    assert samples.shape == (20_000,)
    assert samples.min() >= 10.0
    assert samples.max() <= 30.0


def test_pert_heavily_skewed_mean_is_pulled_toward_mode():
    # Mode close to 'low' -> mean should sit well below the midpoint.
    samples = generate_pert_samples(0, 2, 100, size=200_000)
    assert samples.mean() < 25.0  # midpoint would be 50


def test_pert_degenerate_identical_endpoints():
    samples = generate_pert_samples(7.5, 7.5, 7.5, size=1000)
    assert np.all(samples == 7.5)


def test_pert_inverted_range_raises_before_sampling():
    with pytest.raises(ValueError):
        generate_pert_samples(10, 25, 5, size=100)


def test_pert_invalid_size_raises():
    with pytest.raises(ValueError):
        generate_pert_samples(0, 1, 2, size=0)


def test_pert_rejects_non_generator_rng():
    with pytest.raises(TypeError):
        generate_pert_samples(0, 1, 2, size=10, rng="not-a-generator")


# --------------------------------------------------------------------------- #
# Determinism & RNG isolation
# --------------------------------------------------------------------------- #
def test_explicit_seed_is_deterministic_without_touching_global_state():
    np.random.seed(1234)                     # set a KNOWN global state
    global_before = np.random.get_state()[1].copy()

    a = generate_pert_samples(0, 5, 10, size=500, rng=make_rng(42))
    b = generate_pert_samples(0, 5, 10, size=500, rng=make_rng(42))

    assert np.array_equal(a, b)              # same seed -> identical stream

    global_after = np.random.get_state()[1]
    assert np.array_equal(global_before, global_after)  # global state untouched


def test_unseeded_instances_produce_independent_streams():
    a = generate_pert_samples(0, 5, 10, size=64)
    b = generate_pert_samples(0, 5, 10, size=64)
    assert not np.array_equal(a, b)          # fresh entropy per call


def test_engines_do_not_mutate_global_numpy_random_state():
    np.random.seed(2024)
    snapshot = np.random.get_state()[1].copy()

    engines = [E(num_simulations=200) for E in ALL_ENGINES]  # __init__ side-effect check
    assert np.array_equal(np.random.get_state()[1], snapshot)

    # Exercise the sampling paths of each engine too.
    engines[0].run_roi_simulation((100, 200, 300), (200, 300, 400))
    engines[1].simulate_energy_impact(5000.0, (1, 2, 3), (1, 2, 3), 10.0)
    engines[2].simulate_downtime_risk(5000.0, (100, 200, 300), (1, 2, 3), 5.0)
    engines[3].simulate_lead_time_risk(30.0, (20, 40, 60), 5.0)
    engines[4].simulate_iran_tax_credit((100, 200, 300), (0.5, 0.7, 0.9))

    assert np.array_equal(np.random.get_state()[1], snapshot)  # still untouched


def test_engines_with_explicit_seed_are_reproducible():
    """Two engines constructed with the same seed produce identical statistics."""
    kwargs = dict(num_simulations=2000, seed=7)
    r1 = MonteCarloEngine(**kwargs).run_roi_simulation(
        (100, 200, 300), (200, 300, 400), years=2)
    r2 = MonteCarloEngine(**kwargs).run_roi_simulation(
        (100, 200, 300), (200, 300, 400), years=2)
    assert r1["mean_roi"] == r2["mean_roi"]
    assert r1["var_95"] == r2["var_95"]


def test_engines_accept_dependency_injected_rng():
    """A shared injected Generator keeps every engine in the same reproducible stream."""
    shared = make_rng(99)
    e1 = MonteCarloEngine(num_simulations=500, rng=shared)
    e2 = EnergyRiskEngine(num_simulations=500, rng=shared)
    assert isinstance(e1.rng, np.random.Generator) and e1.rng is shared
    assert e2.rng is shared


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
