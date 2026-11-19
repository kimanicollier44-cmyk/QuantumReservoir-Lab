#!/usr/bin/env python3
"""QuantumReservoir-Lab - educational quantum reservoir computing lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * This lab simulates the quantum reservoir computing (QRC) protocol
    on FIVE qubits (a 32x32 density matrix) using a classical laptop
    CPU. The "quantum reservoir" is a fixed random Ising-style
    unitary; the only trained part is a classical ridge-regression
    readout over expectation-value features, exactly as in the QRC
    literature (Fujii & Nakajima style input injection).
  * It measures two honest things: how far back the reservoir can
    recall its input stream (memory capacity, MC = sum of squared
    correlations over delays 1..12) and whether the readout can learn
    a nonlinear task (parity of the last two inputs).
  * It prints a classical echo-state baseline with the SAME readout
    width, trained the SAME way, beside every quantum number. The
    toy is too small to conclude anything about quantum advantage -
    in either direction - and this lab says so wherever the numbers
    appear. A 5-qubit simulation is a teaching model of the protocol,
    not a quantum computer and not an advantage demonstration.

Every random draw comes from an in-file splitmix64 PRNG seeded with
SEED = 20261118, so the whole lab - couplings, input series, baseline
weights, and every reported number - is byte-identical across runs
and machines. Real quantum-advantage claims, production ML
deployment, hardware control, and vendor claims are permanent HOLD
gates - with Gemini Argon 4 Extended Thinking as DECIDER
(names-only, tighten-only), Muse as advisor (never overrides
HOLD), and Google Flow plan-only seats.
"""
import hashlib
import json
import math
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"

# The one seed for every random draw in the lab (splitmix64 stream,
# drawn in a fixed order: reservoir couplings, input series, baseline
# weights). Named here and in the README, as the determinism receipt.
SEED = 20261118

# Reservoir + protocol constants. All labelled teaching values.
N_QUBITS = 5
DIM = 1 << N_QUBITS            # 32
TAU = 5.0                      # evolution time per input step
J_LO, J_HI = 0.3, 1.0          # ZZ couplings drawn from [J_LO, J_HI]
H_LO, H_HI = 0.5, 1.5          # X fields drawn from [H_LO, H_HI]
PAIRS = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]
WASHOUT = 50                   # steps discarded before training
TRAIN = 300                    # readout training steps
TEST = 200                     # held-out test steps
TOTAL_STEPS = WASHOUT + TRAIN + TEST   # 550
MAX_DELAY = 12                 # memory probed at delays 1..12
RIDGE_LAMBDA = 1e-6            # readout regularisation (labelled)
N_UNITS = 9                    # classical baseline reservoir units
BASELINE_RADIUS = 0.9          # baseline spectral radius target
MC_FLOOR = 3.0                 # asserted floor for quantum MC
PARITY_MARGIN = 0.10           # asserted margin over trivial accuracy

FEATURE_NAMES = (
    ["bias"]
    + [f"Z{i}" for i in range(N_QUBITS)]
    + [f"Z{i}Z{i + 1}" for i in range(N_QUBITS - 1)]
)
N_FEATURES = len(FEATURE_NAMES)   # 10, bias included

HONESTY_NOTE = (
    "The toy is too small to conclude anything about quantum "
    "advantage in either direction: 5 simulated qubits, one seeded "
    "input series, one readout. The classical baseline is printed "
    "for scale, not as a verdict."
)

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "quantum_advantage_claim": "HOLD - no quantum-advantage claim; a 5-qubit classical simulation demonstrates a protocol, not an advantage",
    "production_ml_deployment": "HOLD - no deploying this teaching readout as a production ML model",
    "real_quantum_hardware_control": "HOLD - no control of real quantum hardware; nothing here touches a device",
    "hardware_spec_claim": "HOLD - no claiming real-device specs (qubit counts, coherence times, gate fidelities) from a simulation",
    "benchmark_superiority_claim": "HOLD - no claiming superiority over production classical ML from toy numbers",
    "financial_forecasting_use": "HOLD - no financial forecasting or trading use of the reservoir readout",
    "medical_prediction_use": "HOLD - no medical diagnosis or prognosis predictions from this lab",
    "surveillance_prediction_use": "HOLD - no surveillance or behaviour-prediction deployment",
    "vendor_claim": "HOLD - no vendor or product claims (any hardware or cloud provider) from this lab",
    "training_data_truth_claim": "HOLD - the seeded teaching series is synthetic; it must never be presented as real-world data",
    "autonomous_decision_use": "HOLD - no wiring the readout into autonomous decisions that affect people",
    "security_critical_use": "HOLD - no security-critical prediction (intrusion, fraud, access) deployment",
    "scaling_claim": "HOLD - no claiming these toy results scale to larger reservoirs without new evidence",
    "publication_result_claim": "HOLD - no presenting toy memory-capacity numbers as publishable research results",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Seeded PRNG
# splitmix64, written out in full. Exact 64-bit integer arithmetic, so
# the stream is identical on every machine and every run - the lab's
# determinism receipt for a topic (reservoir computing) that is
# usually presented with unseeded randomness.

class SplitMix64:
    """splitmix64 PRNG: one 64-bit state, Stafford's mix13 finaliser."""

    _MASK = (1 << 64) - 1

    def __init__(self, seed):
        self.state = seed & self._MASK

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & self._MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self._MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self._MASK
        return (z ^ (z >> 31)) & self._MASK

    def uniform(self, lo=0.0, hi=1.0):
        """Uniform float in [lo, hi) from the top 53 bits."""
        return lo + (hi - lo) * ((self.next_u64() >> 11) * (1.0 / (1 << 53)))


# ================================================================ 2. Linear algebra
# A small complex matrix kit on plain lists - enough for Kronecker
# products, the matrix exponential, and density-matrix evolution at
# dim 32. No library linear algebra anywhere in the lab.

def mat_zero(n, m=None):
    m = n if m is None else m
    return [[0j] * m for _ in range(n)]


def mat_identity(n):
    out = mat_zero(n)
    for i in range(n):
        out[i][i] = 1.0 + 0j
    return out


def mat_add(A, B):
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def mat_scale(A, s):
    return [[s * a for a in row] for row in A]


def mat_mul(A, B):
    cols = list(zip(*B))
    return [[sum(a * b for a, b in zip(row, col)) for col in cols]
            for row in A]


def mat_dagger(A):
    return [[A[j][i].conjugate() for j in range(len(A))]
            for i in range(len(A[0]))]


def mat_trace(A):
    return sum(A[i][i] for i in range(len(A)))


def frobenius(A):
    return math.sqrt(sum(abs(a) ** 2 for row in A for a in row))


def kron(A, B):
    out = []
    for ra in A:
        for rb in B:
            row = []
            for a in ra:
                row.extend(a * b for b in rb)
            out.append(row)
    return out


def kron_all(ops):
    out = [[1.0 + 0j]]
    for op in ops:
        out = kron(out, op)
    return out


def mat_exp(A):
    """Matrix exponential by scaling-and-squaring + Taylor series.

    The propagator method for the whole lab, stated openly: scale A
    down until its Frobenius norm is below 0.25, sum the Taylor
    series to machine precision, then square back up.
    """
    n = len(A)
    norm = frobenius(A)
    s = 0
    if norm > 0.25:
        s = int(math.ceil(math.log2(norm / 0.25)))
    scaled = mat_scale(A, 1.0 / (2.0 ** s))
    result = mat_identity(n)
    term = mat_identity(n)
    for k in range(1, 60):
        term = mat_scale(mat_mul(term, scaled), 1.0 / k)
        result = mat_add(result, term)
        if frobenius(term) < 1e-18:
            break
    for _ in range(s):
        result = mat_mul(result, result)
    return result


def solve_linear(A, b):
    """Solve A x = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    aug = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-300:
            raise ValueError("singular matrix")
        aug[col], aug[piv] = aug[piv], aug[col]
        pivot = aug[col][col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col] / pivot
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


# Pauli matrices (complex, plain lists).
I2 = [[1.0 + 0j, 0j], [0j, 1.0 + 0j]]
X2 = [[0j, 1.0 + 0j], [1.0 + 0j, 0j]]
Z2 = [[1.0 + 0j, 0j], [0j, -1.0 + 0j]]


# ================================================================ 3. Quantum reservoir
# Fixed random Ising-style Hamiltonian on 5 qubits, in the
# Fujii-Nakajima form - every pair coupled, plus a transverse field:
#   H = sum_{i<j} J_ij Z_i Z_j  +  sum_i h_i X_i
# with every J_ij, h_i drawn from the seeded stream (labelled ranges
# above; the all-pairs coupling is what lets one injected qubit write
# into the whole register). The reservoir unitary U = exp(-i H TAU)
# is computed once and never trained - in reservoir computing only
# the readout learns.

def build_reservoir(rng):
    """Draw couplings from rng and build H and U = exp(-i H TAU)."""
    J = [rng.uniform(J_LO, J_HI) for _ in PAIRS]
    h = [rng.uniform(H_LO, H_HI) for _ in range(N_QUBITS)]
    H = mat_zero(DIM)
    for (i, j), c in zip(PAIRS, J):
        ops = [I2] * N_QUBITS
        ops[i] = Z2
        ops[j] = Z2
        H = mat_add(H, mat_scale(kron_all(ops), c))
    for i in range(N_QUBITS):
        ops = [I2] * N_QUBITS
        ops[i] = X2
        H = mat_add(H, mat_scale(kron_all(ops), h[i]))
    U = mat_exp(mat_scale(H, -1j * TAU))
    return {"J": J, "h": h, "H": H, "U": U, "U_dagger": mat_dagger(U)}


def unitarity_deviation(U):
    """||U^dagger U - I||_F - the propagator receipt."""
    prod = mat_mul(mat_dagger(U), U)
    ident = mat_identity(DIM)
    return frobenius([[prod[i][j] - ident[i][j] for j in range(DIM)]
                      for i in range(DIM)])


# ---------------------------------------------------------------- input protocol
# Stated exactly, one step per input value s_t in [0, 1]:
#   1. RESET: qubit 0 is replaced by the pure state
#      |s><s| with |s> = sqrt(1-s)|0> + sqrt(s)|1>, tensored with the
#      partial trace of the current state over qubit 0:
#          rho -> |s><s| (x) Tr_0[rho]
#      (qubit 0 is the most significant bit of the basis index).
#   2. EVOLVE: rho -> U rho U^dagger with the fixed reservoir U.
#   3. READ: features are the expectation values <Z_i> (all qubits)
#      and <Z_i Z_{i+1}> (all bonds) of the evolved state, plus a
#      constant bias feature for the readout.

def partial_trace_qubit0(rho):
    """Trace out qubit 0 (most significant bit): returns a 16x16 matrix."""
    half = DIM // 2
    return [[rho[i][j] + rho[i + half][j + half] for j in range(half)]
            for i in range(half)]


def reset_qubit0(rho, s):
    """rho -> |s><s| (x) Tr_0[rho] for input value s in [0, 1]."""
    rest = partial_trace_qubit0(rho)
    half = DIM // 2
    c = math.sqrt(s * (1.0 - s))
    out = mat_zero(DIM)
    for i in range(half):
        for j in range(half):
            v = rest[i][j]
            out[i][j] = (1.0 - s) * v
            out[i][j + half] = c * v
            out[i + half][j] = c * v
            out[i + half][j + half] = s * v
    return out


def features_from_rho(rho):
    """[bias] + [<Z_i>] + [<Z_i Z_{i+1}>] read off the diagonal of rho."""
    diag = [rho[x][x].real for x in range(DIM)]
    feats = [1.0]
    for q in range(N_QUBITS):
        bit = N_QUBITS - 1 - q
        feats.append(sum(d if ((x >> bit) & 1) == 0 else -d
                         for x, d in enumerate(diag)))
    for q in range(N_QUBITS - 1):
        b1 = N_QUBITS - 1 - q
        b2 = N_QUBITS - 2 - q
        feats.append(sum(d if (((x >> b1) & 1) ^ ((x >> b2) & 1)) == 0 else -d
                         for x, d in enumerate(diag)))
    return feats


def run_quantum_reservoir(series, reservoir):
    """Drive the reservoir with the input series; return feature rows."""
    U = reservoir["U"]
    Ud = reservoir["U_dagger"]
    rho = mat_zero(DIM)
    rho[0][0] = 1.0 + 0j            # start in |00000><00000|
    rows = []
    for s in series:
        rho = reset_qubit0(rho, s)
        rho = mat_mul(mat_mul(U, rho), Ud)
        rows.append(features_from_rho(rho))
    return rows


# ================================================================ 4. Classical baseline
# An echo-state-style tanh reservoir with N_UNITS = 9 units, so its
# readout ([bias] + 9 units) has exactly the quantum readout's width
# (10). Weights come from the SAME splitmix64 stream, drawn after the
# quantum couplings and the input series. It exists for scale only.

def spectral_radius_estimate(W, iters=200):
    """Largest |eigenvalue| of W via power iteration on W^T W."""
    n = len(W)
    v = [1.0 / math.sqrt(n)] * n
    for _ in range(iters):
        Wv = [sum(W[i][j] * v[j] for j in range(n)) for i in range(n)]
        w = [sum(W[j][i] * Wv[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x * x for x in w))
        if norm == 0.0:
            return 0.0
        v = [x / norm for x in w]
    Wv = [sum(W[i][j] * v[j] for j in range(n)) for i in range(n)]
    return math.sqrt(sum(x * x for x in Wv))


def build_baseline(rng):
    """Draw the ESN weights from rng; scale W to the labelled radius."""
    W = [[rng.uniform(-1.0, 1.0) for _ in range(N_UNITS)]
         for _ in range(N_UNITS)]
    radius = spectral_radius_estimate(W)
    scale = BASELINE_RADIUS / radius if radius > 0 else 1.0
    W = [[scale * w for w in row] for row in W]
    w_in = [rng.uniform(-1.0, 1.0) for _ in range(N_UNITS)]
    return {"W": W, "w_in": w_in, "radius_before": radius,
            "radius_after": spectral_radius_estimate(W)}


def run_baseline(series, baseline):
    """x <- tanh(W x + w_in * s_t); features are [bias] + x."""
    W = baseline["W"]
    w_in = baseline["w_in"]
    x = [0.0] * N_UNITS
    rows = []
    for s in series:
        x = [math.tanh(sum(W[i][j] * x[j] for j in range(N_UNITS))
                       + w_in[i] * s)
             for i in range(N_UNITS)]
        rows.append([1.0] + x[:])
    return rows


# ================================================================ 5. Ridge readout
# The only trained object in the lab: ridge regression solved from
# scratch - normal equations (X^T X + lam I) w = X^T y by Gaussian
# elimination with partial pivoting.

def ridge_fit(X, y, lam=RIDGE_LAMBDA):
    p = len(X[0])
    XtX = [[sum(row[i] * row[j] for row in X) for j in range(p)]
           for i in range(p)]
    for i in range(p):
        XtX[i][i] += lam
    Xty = [sum(row[i] * t for row, t in zip(X, y)) for i in range(p)]
    return solve_linear(XtX, Xty)


def predict(X, w):
    return [sum(a * b for a, b in zip(row, w)) for row in X]


def squared_correlation(y_true, y_pred):
    """Squared Pearson correlation - the memory-task R^2, as defined
    in the reservoir-computing literature (Jaeger's MC measure)."""
    n = len(y_true)
    mt = sum(y_true) / n
    mp = sum(y_pred) / n
    cov = sum((a - mt) * (b - mp) for a, b in zip(y_true, y_pred))
    vt = sum((a - mt) ** 2 for a in y_true)
    vp = sum((b - mp) ** 2 for b in y_pred)
    if vt <= 0.0 or vp <= 0.0:
        return 0.0
    r = cov / math.sqrt(vt * vp)
    return r * r


def nmse(y_true, y_pred):
    """Normalised mean squared error: MSE / variance of the target."""
    n = len(y_true)
    mt = sum(y_true) / n
    var = sum((a - mt) ** 2 for a in y_true) / n
    if var <= 0.0:
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n / var


# ================================================================ 6. Tasks + experiment
# One seeded run feeds every task. Splits are stated once and used
# everywhere: rows [0, WASHOUT) discarded, next TRAIN rows train the
# readout, final TEST rows are held out for every reported score.

def _split(rows, series, target_fn):
    """(X_train, y_train, X_test, y_test) for a per-step target_fn."""
    lo = WASHOUT
    mid = WASHOUT + TRAIN
    X_train = rows[lo:mid]
    y_train = [target_fn(series, t) for t in range(lo, mid)]
    X_test = rows[mid:]
    y_test = [target_fn(series, t) for t in range(mid, len(series))]
    return X_train, y_train, X_test, y_test


def memory_table(rows, series):
    """Per-delay test R^2 for delays 1..MAX_DELAY + capacity MC."""
    table = {}
    for k in range(1, MAX_DELAY + 1):
        Xtr, ytr, Xte, yte = _split(
            rows, series, lambda s, t, k=k: s[t - k])
        w = ridge_fit(Xtr, ytr)
        table[k] = squared_correlation(yte, predict(Xte, w))
    return table


def parity_report(rows, series):
    """Parity of the last two input bits: b_t XOR b_{t-1}, b = s >= 0.5."""
    def target(s, t):
        return float((1 if s[t] >= 0.5 else 0) ^ (1 if s[t - 1] >= 0.5 else 0))
    Xtr, ytr, Xte, yte = _split(rows, series, target)
    w = ridge_fit(Xtr, ytr)
    pred = predict(Xte, w)
    hits = sum(1 for p, t in zip(pred, yte)
               if (1.0 if p >= 0.5 else 0.0) == t)
    ones = sum(yte)
    trivial = max(ones, len(yte) - ones) / len(yte)
    return {"accuracy": hits / len(yte), "trivial_accuracy": trivial,
            "nmse": nmse(yte, pred), "test_ones": int(ones),
            "test_n": len(yte)}


def feature_stats(rows):
    """Per-feature mean/variance/min/max over the held-out test rows."""
    test_rows = rows[WASHOUT + TRAIN:]
    stats = []
    for j, name in enumerate(FEATURE_NAMES):
        col = [r[j] for r in test_rows]
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / len(col)
        stats.append({"name": name, "mean": mean, "variance": var,
                      "min": min(col), "max": max(col)})
    signal = [s for s in stats if s["name"] != "bias"]
    mean_var = sum(s["variance"] for s in signal) / len(signal)
    return {"per_feature": stats, "mean_signal_variance": mean_var}


_EXPERIMENT_CACHE = {}


def build_experiment():
    """The full seeded experiment, drawn in the stated stream order:
    reservoir couplings -> input series -> baseline weights."""
    rng = SplitMix64(SEED)
    reservoir = build_reservoir(rng)
    series = [rng.uniform(0.0, 1.0) for _ in range(TOTAL_STEPS)]
    baseline = build_baseline(rng)
    q_rows = run_quantum_reservoir(series, reservoir)
    c_rows = run_baseline(series, baseline)
    q_mem = memory_table(q_rows, series)
    c_mem = memory_table(c_rows, series)
    q_par = parity_report(q_rows, series)
    c_par = parity_report(c_rows, series)
    stats = feature_stats(q_rows)
    exp = {
        "seed": SEED,
        "J": reservoir["J"],
        "h": reservoir["h"],
        "unitarity_deviation": unitarity_deviation(reservoir["U"]),
        "series": series,
        "baseline_radius_before": baseline["radius_before"],
        "baseline_radius_after": baseline["radius_after"],
        "quantum_memory": q_mem,
        "classical_memory": c_mem,
        "quantum_mc": sum(q_mem.values()),
        "classical_mc": sum(c_mem.values()),
        "quantum_parity": q_par,
        "classical_parity": c_par,
        "feature_stats": stats,
        "feature_variance": stats["mean_signal_variance"],
        "n_features": N_FEATURES,
        "advantage_claim": False,
        "honesty_note": HONESTY_NOTE,
    }
    exp["verdict_features"] = (
        exp["quantum_mc"], exp["quantum_memory"][1],
        exp["quantum_parity"]["accuracy"], exp["feature_variance"])
    return exp


def experiment():
    if "exp" not in _EXPERIMENT_CACHE:
        _EXPERIMENT_CACHE["exp"] = build_experiment()
    return _EXPERIMENT_CACHE["exp"]


def reservoir_report():
    exp = experiment()
    return {
        "n_qubits": N_QUBITS, "dim": DIM, "tau": TAU, "seed": SEED,
        "prng": "splitmix64 (in-file)",
        "J_zz_couplings": {f"{i}{j}": round(v, 6)
                           for (i, j), v in zip(PAIRS, exp["J"])},
        "h_x_fields": [round(v, 6) for v in exp["h"]],
        "J_range": [J_LO, J_HI], "h_range": [H_LO, H_HI],
        "connectivity": "all pairs (Fujii-Nakajima random Ising form)",
        "propagator": "exp(-i H TAU) by scaling-and-squaring Taylor",
        "unitarity_deviation": exp["unitarity_deviation"],
        "protocol": ("per step: reset qubit 0 to |s><s| (x) Tr_0[rho], "
                     "evolve rho -> U rho U^dagger, read features"),
        "encoding": "|s> = sqrt(1-s)|0> + sqrt(s)|1> on qubit 0",
        "features": FEATURE_NAMES, "n_features": N_FEATURES,
        "washout": WASHOUT, "train": TRAIN, "test": TEST,
        "ridge_lambda": RIDGE_LAMBDA,
    }


def memory_report():
    exp = experiment()
    return {
        "delays": list(range(1, MAX_DELAY + 1)),
        "quantum_r2": {k: round(v, 6)
                       for k, v in exp["quantum_memory"].items()},
        "classical_r2": {k: round(v, 6)
                         for k, v in exp["classical_memory"].items()},
        "quantum_mc": round(exp["quantum_mc"], 6),
        "classical_mc": round(exp["classical_mc"], 6),
        "mc_floor_asserted": MC_FLOOR,
        "fading_memory": exp["quantum_memory"][1] > exp["quantum_memory"][MAX_DELAY],
        "advantage_claim": False,
        "honesty_note": HONESTY_NOTE,
    }


def parity_task_report():
    exp = experiment()
    q = exp["quantum_parity"]
    c = exp["classical_parity"]
    return {
        "task": "predict b_t XOR b_{t-1} from reservoir features, b = (s >= 0.5)",
        "quantum": {"accuracy": round(q["accuracy"], 6),
                    "trivial_accuracy": round(q["trivial_accuracy"], 6),
                    "margin_over_trivial": round(
                        q["accuracy"] - q["trivial_accuracy"], 6),
                    "nmse": round(q["nmse"], 6)},
        "classical": {"accuracy": round(c["accuracy"], 6),
                      "trivial_accuracy": round(c["trivial_accuracy"], 6),
                      "nmse": round(c["nmse"], 6)},
        "margin_asserted": PARITY_MARGIN,
        "advantage_claim": False,
        "honesty_note": HONESTY_NOTE,
    }


def baseline_report():
    exp = experiment()
    return {
        "model": f"echo-state tanh reservoir, {N_UNITS} units, "
                 f"readout width {N_FEATURES} (bias + units) - "
                 f"the same width as the quantum readout",
        "weights": "same splitmix64 stream, drawn after the quantum "
                   "couplings and the input series",
        "spectral_radius_before": round(exp["baseline_radius_before"], 6),
        "spectral_radius_after": round(exp["baseline_radius_after"], 6),
        "classical_mc": round(exp["classical_mc"], 6),
        "quantum_mc": round(exp["quantum_mc"], 6),
        "classical_parity_accuracy": round(
            exp["classical_parity"]["accuracy"], 6),
        "quantum_parity_accuracy": round(
            exp["quantum_parity"]["accuracy"], 6),
        "advantage_claim": False,
        "honesty_note": HONESTY_NOTE,
    }


def features_report():
    exp = experiment()
    return {
        "window": f"held-out test rows ({TEST} steps)",
        "per_feature": [
            {"name": s["name"], "mean": round(s["mean"], 6),
             "variance": round(s["variance"], 6),
             "min": round(s["min"], 6), "max": round(s["max"], 6)}
            for s in exp["feature_stats"]["per_feature"]],
        "mean_signal_variance": round(exp["feature_variance"], 6),
    }


# ================================================================ 7. kNN verdict classifier
# Features: (memory capacity, delay-1 R^2, parity accuracy, mean
# feature variance). The training grid is labelled by the lab's own
# policy (model_label below) - the classifier interpolates the
# policy, it does not replace the measured task reports.

def model_label(mc, r2_delay1, parity_accuracy, feature_variance):
    """The lab's verdict policy, distilled to four labels."""
    if feature_variance < 0.001 or mc < 0.5:
        return "washed-out"
    if mc >= 4.0 and parity_accuracy >= 0.85:
        return "nonlinear-capable"
    if mc >= 4.0:
        return "high-memory"
    if mc >= 1.5 and r2_delay1 >= 0.5:
        return "fading-memory"
    return "washed-out"


def _training_grid():
    rows = []
    for mc in (0.0, 2.5, 5.0, 7.5, 10.0):
        for r2 in (0.1, 0.35, 0.6, 0.85):
            for parity in (0.5, 0.7, 0.95):
                for var in (0.0005, 0.05):
                    feats = (mc, r2, parity, var)
                    rows.append((feats, model_label(*feats)))
    return rows


TRAINING_GRID = _training_grid()
_LABEL_PRIORITY = {"nonlinear-capable": 0, "high-memory": 1,
                   "fading-memory": 2, "washed-out": 3}
_VAR_LOG_LO = math.log10(0.0005)
_VAR_LOG_HI = math.log10(0.05)


def _scaled(features):
    """Feature scaling for the kNN: MC and the probabilities linearly,
    the feature variance on a log scale (it spans orders of magnitude
    and the policy's dead-feature veto sits at its low end)."""
    mc, r2, parity, var = features
    var_scaled = (math.log10(max(var, 1e-12)) - _VAR_LOG_LO) / (
        _VAR_LOG_HI - _VAR_LOG_LO)
    return (mc / 8.0, r2, parity, var_scaled)


def classify_reservoir(mc, r2_delay1, parity_accuracy, feature_variance,
                       k=3, grid=None):
    """Distance-weighted kNN over the policy-labelled grid."""
    grid = TRAINING_GRID if grid is None else grid
    target = _scaled((mc, r2_delay1, parity_accuracy, feature_variance))
    ranked = []
    for features, label in grid:
        point = _scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    weights = {}
    for dist, _, label in ranked[:k]:
        weights[label] = weights.get(label, 0.0) + 1.0 / (dist + 1e-9)
    return sorted(weights.items(),
                  key=lambda kv: (-kv[1], _LABEL_PRIORITY[kv[0]]))[0][0]


def classifier_report():
    exp = experiment()
    scenarios = {
        "deep_memory_and_parity": (6.0, 0.95, 0.92, 0.05),
        "deep_memory_weak_parity": (5.0, 0.90, 0.60, 0.05),
        "short_fading_memory": (2.5, 0.70, 0.55, 0.03),
        "almost_no_memory": (0.3, 0.10, 0.50, 0.01),
        "dead_features_veto": (8.0, 0.99, 0.99, 0.0005),
    }
    out = {name: classify_reservoir(*feats)
           for name, feats in scenarios.items()}
    out["this_lab_measured"] = classify_reservoir(*exp["verdict_features"])
    return out


# ================================================================ 8. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantumreservoir-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "model_id": "gemini-argon-4",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all advantage and deployment gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantumreservoir-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Print the classical baseline beside the quantum memory capacity every time, and keep the sentence that the toy is too small to conclude anything - on a five-qubit simulation the honest comparison is the lesson, not a winner.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["a stream of numbers raining into a ring of five qubits",
                  "the reservoir state blooming into expectation-value features",
                  "a ridge readout learning to recall inputs twelve steps back",
                  "quantum and classical memory curves falling side by side",
                  "the HOLD stamp landing on every advantage claim"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. PRNG - the determinism receipt
    s1 = SplitMix64(SEED)
    s2 = SplitMix64(SEED)
    chk("prng_deterministic",
        [s1.next_u64() for _ in range(100)]
        == [s2.next_u64() for _ in range(100)])
    s3 = SplitMix64(SEED)
    draws = [s3.uniform() for _ in range(1000)]
    chk("prng_uniform_range", all(0.0 <= v < 1.0 for v in draws))
    chk("prng_seed_changes_stream",
        SplitMix64(SEED).next_u64() != SplitMix64(SEED + 1).next_u64())

    # 2. linear algebra kit
    zz = kron(Z2, Z2)
    chk("kron_zz_diagonal",
        [zz[i][i] for i in range(4)] == [1, -1, -1, 1])
    A = [[1 + 1j, 2], [3, 4 - 1j]]
    chk("matmul_identity",
        mat_mul(A, mat_identity(2)) == A)
    chk("exp_zero_is_identity",
        frobenius(mat_add(mat_exp(mat_zero(3)),
                          mat_scale(mat_identity(3), -1.0))) < 1e-12)
    theta = 0.7
    rot = mat_exp(mat_scale(X2, -0.5j * theta))
    co, si = math.cos(theta / 2), math.sin(theta / 2)
    chk("exp_x_rotation_formula",
        abs(rot[0][0] - co) < 1e-12 and abs(rot[0][1] - (-1j * si)) < 1e-12
        and abs(rot[1][0] - (-1j * si)) < 1e-12
        and abs(rot[1][1] - co) < 1e-12)
    xs = solve_linear([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0])
    chk("solve_linear_known", abs(xs[0] - 1.0) < 1e-12
        and abs(xs[1] - 3.0) < 1e-12)

    # 3. reservoir construction
    rng = SplitMix64(SEED)
    res = build_reservoir(rng)
    chk("coupling_counts", len(res["J"]) == len(PAIRS) == 10
        and len(res["h"]) == N_QUBITS)
    chk("couplings_in_labelled_range",
        all(J_LO <= v <= J_HI for v in res["J"])
        and all(H_LO <= v <= H_HI for v in res["h"]))
    H = res["H"]
    chk("hamiltonian_hermitian",
        frobenius(mat_add(H, mat_scale(mat_dagger(H), -1.0))) < 1e-12)
    chk("hamiltonian_traceless", abs(mat_trace(H)) < 1e-9)
    chk("propagator_unitary", unitarity_deviation(res["U"]) < 1e-9)

    # 4. input protocol pieces
    ground16 = mat_zero(DIM // 2)
    ground16[0][0] = 1.0 + 0j
    prod = kron([[0.7 + 0j, 0j], [0j, 0.3 + 0j]], ground16)
    chk("partial_trace_recovers_rest",
        frobenius(mat_add(partial_trace_qubit0(prod),
                          mat_scale(ground16, -1.0))) < 1e-12)
    rho0 = mat_zero(DIM)
    rho0[0][0] = 1.0 + 0j
    ok_reset = True
    for s in (0.0, 0.25, 0.7, 1.0):
        feats = features_from_rho(reset_qubit0(rho0, s))
        ok_reset = ok_reset and abs(feats[1] - (1.0 - 2.0 * s)) < 1e-12
    chk("reset_sets_qubit0_expectation", ok_reset)
    rho = rho0
    for s in (0.3, 0.8, 0.1, 0.55, 0.9):
        rho = mat_mul(mat_mul(res["U"], reset_qubit0(rho, s)),
                      res["U_dagger"])
    chk("density_trace_preserved", abs(mat_trace(rho) - 1.0) < 1e-9)
    chk("density_hermitian",
        frobenius(mat_add(rho, mat_scale(mat_dagger(rho), -1.0))) < 1e-9)

    # 5. readout machinery
    X_line = [[1.0, float(x)] for x in range(-4, 5)]
    y_line = [2.0 * row[1] + 1.0 for row in X_line]
    w_line = ridge_fit(X_line, y_line, lam=1e-8)
    chk("ridge_recovers_line", abs(w_line[0] - 1.0) < 1e-5
        and abs(w_line[1] - 2.0) < 1e-5)
    chk("squared_correlation_perfect",
        abs(squared_correlation([1.0, 2.0, 3.0, 4.0],
                                [3.0, 5.0, 7.0, 9.0]) - 1.0) < 1e-12)
    chk("squared_correlation_constant_is_zero",
        squared_correlation([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 0.0)

    # 6. the experiment itself
    exp = experiment()
    chk("stream_order_reproducible",
        exp["J"] == res["J"] and exp["h"] == res["h"])
    chk("feature_width_matches_baseline",
        exp["n_features"] == N_FEATURES == 1 + N_UNITS)
    chk("feature_names_complete", len(FEATURE_NAMES) == 10
        and FEATURE_NAMES[0] == "bias")
    q_rows = run_quantum_reservoir(exp["series"][:60], res)
    q_rows2 = run_quantum_reservoir(exp["series"][:60], res)
    chk("reservoir_run_deterministic", q_rows == q_rows2)
    chk("expectations_bounded",
        all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9
            for row in q_rows for v in row[1:]))
    chk("features_alive", exp["feature_variance"] > 0.005)
    qm = exp["quantum_memory"]
    chk("memory_r2_in_range", all(0.0 <= v <= 1.0 for v in qm.values()))
    chk("memory_capacity_floor", exp["quantum_mc"] > MC_FLOOR)
    chk("memory_capacity_bounded", exp["quantum_mc"] <= MAX_DELAY)
    chk("fading_memory", qm[1] > qm[MAX_DELAY] and qm[1] > 0.5)
    qp = exp["quantum_parity"]
    chk("parity_beats_trivial",
        qp["accuracy"] > qp["trivial_accuracy"] + PARITY_MARGIN)
    chk("parity_accuracy_range", 0.5 <= qp["accuracy"] <= 1.0)
    chk("baseline_mc_positive", exp["classical_mc"] > 0.5)
    chk("baseline_radius_scaled",
        abs(exp["baseline_radius_after"] - BASELINE_RADIUS) < 0.05)
    chk("honesty_no_advantage_claim", exp["advantage_claim"] is False
        and "too small to conclude anything" in exp["honesty_note"])

    # 7. classifier - labels come from the lab's own policy
    rep = classifier_report()
    chk("classifier_deep_nonlinear",
        rep["deep_memory_and_parity"] == "nonlinear-capable")
    chk("classifier_deep_memory",
        rep["deep_memory_weak_parity"] == "high-memory")
    chk("classifier_fading", rep["short_fading_memory"] == "fading-memory")
    chk("classifier_no_memory", rep["almost_no_memory"] == "washed-out")
    chk("classifier_dead_features_veto",
        rep["dead_features_veto"] == "washed-out")
    chk("classifier_measured_matches_policy",
        rep["this_lab_measured"] == model_label(*exp["verdict_features"]))
    loo_ok = True
    for i, (feats, label) in enumerate(TRAINING_GRID):
        rest = TRAINING_GRID[:i] + TRAINING_GRID[i + 1:]
        if classify_reservoir(*feats, grid=rest) != label:
            loo_ok = False
            break
    chk("classifier_grid_leave_one_out", loo_ok)

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_advantage_hold", gate("quantum_advantage_claim") == "HOLD")
    chk("gate_deployment_hold", gate("production_ml_deployment") == "HOLD")
    chk("gate_hardware_hold", gate("real_quantum_hardware_control") == "HOLD")
    chk("gate_scaling_hold", gate("scaling_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking"
        and brains["decider"]["model_id"] == "gemini-argon-4")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    exp = experiment()
    print("=" * 68)
    print("QuantumReservoir-Lab - quantum reservoir computing lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - what this is and is not")
    print("    A 5-qubit reservoir (32x32 density matrix) simulated on a")
    print("    classical machine. The reservoir is a FIXED random unitary;")
    print("    only the classical ridge readout is trained. This is a")
    print("    teaching model of the QRC protocol - not a quantum")
    print("    computer, and not an advantage demonstration.")
    print(f"    Seed: splitmix64 SEED = {SEED} (couplings -> series ->")
    print("    baseline weights, one stream, byte-identical reruns).")
    print("\n[2] Reservoir - fixed random Ising-style unitary, all pairs")
    print(f"    J (ZZ pairs): {[round(v, 4) for v in exp['J']]}")
    print(f"    h (X fields): {[round(v, 4) for v in exp['h']]}")
    print(f"    U = exp(-i H tau), tau = {TAU}; unitarity deviation "
          f"{exp['unitarity_deviation']:.3e}")
    print("    Protocol per step: reset qubit 0 to |s><s| (x) Tr_0[rho],")
    print("    evolve rho -> U rho U+, read <Z_i> and <Z_i Z_i+1> features")
    print(f"    ({N_FEATURES} features incl. bias; washout {WASHOUT}, "
          f"train {TRAIN}, test {TEST}; ridge lambda {RIDGE_LAMBDA})")
    print("\n[3] Memory task - recall the input k steps back (test R^2)")
    print("    delay      quantum      classical")
    for k in range(1, MAX_DELAY + 1):
        print(f"    {k:5d}      {exp['quantum_memory'][k]:.6f}     "
              f"{exp['classical_memory'][k]:.6f}")
    print(f"    MC (sum of R^2): quantum {exp['quantum_mc']:.6f} | "
          f"classical {exp['classical_mc']:.6f}")
    print(f"    Fading memory: R^2(1) = {exp['quantum_memory'][1]:.6f} > "
          f"R^2(12) = {exp['quantum_memory'][MAX_DELAY]:.6f}")
    print(f"    {HONESTY_NOTE}")
    print("\n[4] Nonlinear task - parity of the last two input bits")
    qp = exp["quantum_parity"]
    cp = exp["classical_parity"]
    print(f"    quantum:   accuracy {qp['accuracy']:.6f} vs trivial "
          f"{qp['trivial_accuracy']:.6f} (NMSE {qp['nmse']:.6f})")
    print(f"    classical: accuracy {cp['accuracy']:.6f} vs trivial "
          f"{cp['trivial_accuracy']:.6f} (NMSE {cp['nmse']:.6f})")
    print("\n[5] Regime classifier (TinyML kNN, labels from the lab policy)")
    for name, label in classifier_report().items():
        print(f"    {name:36s} -> {label}")
    print("\n[6] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[7] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "reservoir":
        print(json.dumps(reservoir_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "memory":
        print(json.dumps(memory_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "parity":
        print(json.dumps(parity_task_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "baseline":
        print(json.dumps(baseline_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "features":
        print(json.dumps(features_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(classifier_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "quantum_advantage_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
