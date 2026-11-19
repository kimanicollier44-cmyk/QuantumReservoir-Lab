"""Tests for QuantumReservoir-Lab: PRNG, reservoir, protocol, readout,
memory + parity tasks, baseline, classifier, gates, seats, determinism."""
import pytest

import main as m


# ---- PRNG (the determinism receipt)
def test_prng_deterministic():
    a = m.SplitMix64(m.SEED)
    b = m.SplitMix64(m.SEED)
    assert [a.next_u64() for _ in range(100)] == [b.next_u64() for _ in range(100)]


def test_prng_uniform_range():
    rng = m.SplitMix64(m.SEED)
    assert all(0.0 <= rng.uniform() < 1.0 for _ in range(500))


def test_prng_seed_changes_stream():
    assert m.SplitMix64(m.SEED).next_u64() != m.SplitMix64(m.SEED + 1).next_u64()


# ---- linear algebra kit
def test_kron_zz_diagonal():
    zz = m.kron(m.Z2, m.Z2)
    assert [zz[i][i] for i in range(4)] == [1, -1, -1, 1]


def test_matmul_identity():
    A = [[1 + 1j, 2], [3, 4 - 1j]]
    assert m.mat_mul(A, m.mat_identity(2)) == A


def test_exp_zero_is_identity():
    dev = m.frobenius(m.mat_add(m.mat_exp(m.mat_zero(3)),
                                m.mat_scale(m.mat_identity(3), -1.0)))
    assert dev < 1e-12


def test_exp_x_rotation_formula():
    import math
    theta = 0.7
    rot = m.mat_exp(m.mat_scale(m.X2, -0.5j * theta))
    assert abs(rot[0][0] - math.cos(theta / 2)) < 1e-12
    assert abs(rot[0][1] - (-1j * math.sin(theta / 2))) < 1e-12


def test_solve_linear_known():
    x = m.solve_linear([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    assert abs(x[0] - 1.0) < 1e-12 and abs(x[1] - 3.0) < 1e-12


# ---- reservoir construction
def test_coupling_counts_and_ranges():
    res = m.build_reservoir(m.SplitMix64(m.SEED))
    assert len(res["J"]) == len(m.PAIRS) == 10
    assert len(res["h"]) == m.N_QUBITS
    assert all(m.J_LO <= v <= m.J_HI for v in res["J"])
    assert all(m.H_LO <= v <= m.H_HI for v in res["h"])


def test_hamiltonian_hermitian_traceless():
    res = m.build_reservoir(m.SplitMix64(m.SEED))
    dev = m.frobenius(m.mat_add(res["H"],
                                m.mat_scale(m.mat_dagger(res["H"]), -1.0)))
    assert dev < 1e-12
    assert abs(m.mat_trace(res["H"])) < 1e-9


def test_propagator_unitary():
    res = m.build_reservoir(m.SplitMix64(m.SEED))
    assert m.unitarity_deviation(res["U"]) < 1e-9


# ---- input protocol
def test_partial_trace_recovers_rest():
    ground = m.mat_zero(m.DIM // 2)
    ground[0][0] = 1.0 + 0j
    prod = m.kron([[0.7 + 0j, 0j], [0j, 0.3 + 0j]], ground)
    dev = m.frobenius(m.mat_add(m.partial_trace_qubit0(prod),
                                m.mat_scale(ground, -1.0)))
    assert dev < 1e-12


def test_reset_sets_qubit0_expectation():
    rho0 = m.mat_zero(m.DIM)
    rho0[0][0] = 1.0 + 0j
    for s in (0.0, 0.25, 0.7, 1.0):
        feats = m.features_from_rho(m.reset_qubit0(rho0, s))
        assert abs(feats[1] - (1.0 - 2.0 * s)) < 1e-12


def test_density_trace_and_hermiticity_preserved():
    res = m.build_reservoir(m.SplitMix64(m.SEED))
    rho = m.mat_zero(m.DIM)
    rho[0][0] = 1.0 + 0j
    for s in (0.3, 0.8, 0.1, 0.55, 0.9):
        rho = m.mat_mul(m.mat_mul(res["U"], m.reset_qubit0(rho, s)),
                        res["U_dagger"])
    assert abs(m.mat_trace(rho) - 1.0) < 1e-9
    dev = m.frobenius(m.mat_add(rho, m.mat_scale(m.mat_dagger(rho), -1.0)))
    assert dev < 1e-9


# ---- readout machinery
def test_ridge_recovers_line():
    X = [[1.0, float(x)] for x in range(-4, 5)]
    y = [2.0 * row[1] + 1.0 for row in X]
    w = m.ridge_fit(X, y, lam=1e-8)
    assert abs(w[0] - 1.0) < 1e-5 and abs(w[1] - 2.0) < 1e-5


def test_squared_correlation():
    assert abs(m.squared_correlation([1.0, 2.0, 3.0, 4.0],
                                     [3.0, 5.0, 7.0, 9.0]) - 1.0) < 1e-12
    assert m.squared_correlation([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 0.0


# ---- the experiment (cached in main)
def test_feature_width_matches_baseline():
    exp = m.experiment()
    assert exp["n_features"] == m.N_FEATURES == 1 + m.N_UNITS == 10


def test_reservoir_run_deterministic():
    exp = m.experiment()
    res = m.build_reservoir(m.SplitMix64(m.SEED))
    rows1 = m.run_quantum_reservoir(exp["series"][:60], res)
    rows2 = m.run_quantum_reservoir(exp["series"][:60], res)
    assert rows1 == rows2


def test_expectations_bounded():
    exp = m.experiment()
    res = m.build_reservoir(m.SplitMix64(m.SEED))
    rows = m.run_quantum_reservoir(exp["series"][:60], res)
    assert all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for row in rows for v in row[1:])


def test_memory_capacity_floor_and_fading():
    exp = m.experiment()
    assert exp["quantum_mc"] > m.MC_FLOOR
    assert exp["quantum_mc"] <= m.MAX_DELAY
    assert exp["quantum_memory"][1] > exp["quantum_memory"][m.MAX_DELAY]
    assert exp["quantum_memory"][1] > 0.5
    assert all(0.0 <= v <= 1.0 for v in exp["quantum_memory"].values())


def test_parity_beats_trivial():
    exp = m.experiment()
    q = exp["quantum_parity"]
    assert q["accuracy"] > q["trivial_accuracy"] + m.PARITY_MARGIN


def test_baseline_runs_and_scales():
    exp = m.experiment()
    assert exp["classical_mc"] > 0.5
    assert abs(exp["baseline_radius_after"] - m.BASELINE_RADIUS) < 0.05


def test_features_alive():
    assert m.experiment()["feature_variance"] > 0.005


def test_honesty_no_advantage_claim():
    exp = m.experiment()
    assert exp["advantage_claim"] is False
    assert "too small to conclude anything" in exp["honesty_note"]


# ---- classifier
def test_classifier_scenarios():
    rep = m.classifier_report()
    assert rep["deep_memory_and_parity"] == "nonlinear-capable"
    assert rep["deep_memory_weak_parity"] == "high-memory"
    assert rep["short_fading_memory"] == "fading-memory"
    assert rep["almost_no_memory"] == "washed-out"
    assert rep["dead_features_veto"] == "washed-out"


def test_classifier_measured_matches_policy():
    exp = m.experiment()
    assert (m.classify_reservoir(*exp["verdict_features"])
            == m.model_label(*exp["verdict_features"]))


def test_classifier_grid_leave_one_out():
    grid = m.TRAINING_GRID
    for i, (feats, label) in enumerate(grid):
        rest = grid[:i] + grid[i + 1:]
        assert m.classify_reservoir(*feats, grid=rest) == label


# ---- gates + seats + CLI
def test_hold_gates_14():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


def test_specific_gates():
    assert m.gate("quantum_advantage_claim") == "HOLD"
    assert m.gate("production_ml_deployment") == "HOLD"
    assert m.gate("real_quantum_hardware_control") == "HOLD"
    assert m.gate("scaling_claim") == "HOLD"
    assert m.gate("run_simulation") == "PROCEED"


def test_seats_never_override():
    brains = m.brains_report()
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["model_id"] == "gemini-argon-4"
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False and brains["flow"]["live"] == 0


def test_hold_demo_exit_code():
    assert m.main(["hold-demo"]) == 2


def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks)


def test_demo_runs():
    assert m.demo() == 0
