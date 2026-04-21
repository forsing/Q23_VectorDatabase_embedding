#!/usr/bin/env python3
"""
Q23 Vector Database (embeddings) — tehnika: Quantum Vector Database (QVDB)
(čisto kvantno: entangled index⊗embedding DB + jedan SWAP-test + post-selekcija anc=0).

Koncept (kvantni analog vector-DB-a / kNN-lookup-a):
  1) Korpus: CEO CSV podeljen u D = 2^m „dokumenata" (chunk-ovi ~ N/D redova).
     Svaki dokument d je amplitude-encoding svog freq_vector-a → |ψ_d⟩_emb.
  2) Entanglovan DB state u jednom kolu (registri idx + emb):
        H^⊗m na idx → uniformna superpozicija D indeksa.
        Za d = 0..D-1: multi-ctrl StatePreparation(|ψ_d⟩) sa ctrl_state=d.
        Rezultat: |DB⟩ = (1/√D) Σ_d |d⟩_idx ⊗ |ψ_d⟩_emb.
  3) Query: |ψ_q⟩ = amplitude-encoding freq_vector-a poslednjih L redova, pripremljen
     na zasebnom query-registru.
  4) Jedan SWAP-test između emb i query registra (1 ancilla):
        H(anc), cswap(anc, emb_i, query_i) za i = 0..nq-1, H(anc).
     Kvantna interferencija simultano „pretraži" sve dokumente u superpoziciji:
        P(anc=0, idx=d) ∝ (1 + |⟨ψ_d|ψ_q⟩|²) / (2D).
  5) Post-selekcija anc=0 + marginalizacija (idx, query) → marginala emb-registra =
     kvantno-ponderisana mešavina dokumenata po similarity-interferenciji.
  6) bias_39 → TOP-7 = NEXT.

Razlika u odnosu na Q22 (QRAG):
  Q22: D ZASEBNIH SWAP-test kola + KLASIČAN top-K ranking + aux-weighted StatePreparation
       (two-stage: izmeri → selektuj → rekonstruiši).
  QVDB: JEDNO kolo sa entanglovanim DB-om i JEDAN SWAP-test (single-stage kvantni
        NN-lookup kroz interferenciju). Težine se pojavljuju prirodno iz
        post-selekcije, bez eksplicitnog ranking koraka.

Sve deterministički: seed=39; ceo CSV je podeljen u D chunk-ova (pravilo 10).
Deterministička grid-optimizacija (nq, m, L) po cos(bias_39, freq_csv).

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1 (vidi README.md).
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# =========================
# Seed
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/data/loto7hh_4600_k31.csv")
N_NUMBERS = 7
N_MAX = 39

GRID_NQ = (5, 6)
GRID_M = (1, 2, 3)
GRID_L = (50, 200, 1000)


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


def freq_vector(H: np.ndarray) -> np.ndarray:
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def amp_from_freq(f: np.ndarray, nq: int) -> np.ndarray:
    dim = 2 ** nq
    edges = np.linspace(0, N_MAX, dim + 1, dtype=int)
    amp = np.array(
        [float(f[edges[i] : edges[i + 1]].mean()) if edges[i + 1] > edges[i] else 0.0 for i in range(dim)],
        dtype=np.float64,
    )
    amp = np.maximum(amp, 0.0)
    n2 = float(np.linalg.norm(amp))
    if n2 < 1e-18:
        amp = np.ones(dim, dtype=np.float64) / np.sqrt(dim)
    else:
        amp = amp / n2
    return amp


# =========================
# Korpus embedding-a i query
# =========================
def document_amps(H: np.ndarray, nq: int, D: int) -> List[np.ndarray]:
    n = H.shape[0]
    edges = np.linspace(0, n, int(D) + 1, dtype=int)
    amps: List[np.ndarray] = []
    for d in range(int(D)):
        lo, hi = int(edges[d]), int(edges[d + 1])
        if hi <= lo:
            amps.append(amp_from_freq(np.zeros(N_MAX), nq))
        else:
            amps.append(amp_from_freq(freq_vector(H[lo:hi]), nq))
    return amps


def query_amp(H: np.ndarray, nq: int, L: int) -> np.ndarray:
    L_eff = max(N_NUMBERS, min(H.shape[0], int(L)))
    return amp_from_freq(freq_vector(H[-L_eff:]), nq)


# =========================
# QVDB kolo: idx + emb + query + anc, jedan SWAP-test, post-selekcija anc=0
# Registri: emb (nq), query (nq), idx (m), anc (1)
# Qiskit little-endian: emb = najniži bitovi, anc = najviši bit.
# =========================
def build_qvdb_state(H: np.ndarray, nq: int, m: int, L: int) -> Statevector:
    D = 2 ** m
    amps_d = document_amps(H, nq, D)
    amp_q = query_amp(H, nq, L)

    emb = QuantumRegister(nq, name="e")
    qry = QuantumRegister(nq, name="q")
    idx = QuantumRegister(m, name="i")
    anc = QuantumRegister(1, name="a")
    qc = QuantumCircuit(emb, qry, idx, anc)

    qc.h(idx)

    for d in range(D):
        sp = StatePreparation(amps_d[d].tolist())
        sp_ctrl = sp.control(num_ctrl_qubits=m, ctrl_state=d)
        qc.append(sp_ctrl, list(idx) + list(emb))

    qc.append(StatePreparation(amp_q.tolist()), qry)

    qc.h(anc[0])
    for i in range(nq):
        qc.cswap(anc[0], emb[i], qry[i])
    qc.h(anc[0])

    return Statevector(qc)


def qvdb_state_probs(H: np.ndarray, nq: int, m: int, L: int) -> Tuple[np.ndarray, float]:
    """Vraća (marginala nad emb post-select anc=0 [normalizovana], P(anc=0))."""
    sv = build_qvdb_state(H, nq, m, L)
    p = np.abs(sv.data) ** 2
    dim_emb = 2 ** nq
    dim_qry = 2 ** nq
    dim_idx = 2 ** m
    dim_anc = 2

    mat = p.reshape(dim_anc, dim_idx, dim_qry, dim_emb)
    p_anc0_all = float(mat[0].sum())
    if p_anc0_all < 1e-18:
        return np.zeros(dim_emb, dtype=np.float64), 0.0

    p_emb_post = mat[0].sum(axis=(0, 1))
    p_emb_post = p_emb_post / p_anc0_all
    return p_emb_post, p_anc0_all


def qvdb_posterior_idx(H: np.ndarray, nq: int, m: int, L: int) -> Tuple[np.ndarray, float]:
    """Posterior nad idx-registrom nakon post-select anc=0 (za dijagnostiku)."""
    sv = build_qvdb_state(H, nq, m, L)
    p = np.abs(sv.data) ** 2
    dim_emb = 2 ** nq
    dim_qry = 2 ** nq
    dim_idx = 2 ** m
    dim_anc = 2

    mat = p.reshape(dim_anc, dim_idx, dim_qry, dim_emb)
    p_anc0_all = float(mat[0].sum())
    if p_anc0_all < 1e-18:
        return np.zeros(dim_idx, dtype=np.float64), 0.0

    p_idx_post = mat[0].sum(axis=(1, 2))
    p_idx_post = p_idx_post / p_anc0_all
    return p_idx_post, p_anc0_all


# =========================
# Readout
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    return tuple(sorted(int(o + 1) for o in order[:k]))


# =========================
# Determ. grid-optimizacija (nq, m, L)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    s_tot = float(f_csv.sum())
    f_csv_n = f_csv / s_tot if s_tot > 0 else np.ones(N_MAX) / N_MAX
    best = None
    for nq in GRID_NQ:
        for m in GRID_M:
            for L in GRID_L:
                try:
                    p_emb, p_anc0 = qvdb_state_probs(H, nq, int(m), int(L))
                    bi = bias_39(p_emb)
                    score = cosine(bi, f_csv_n)
                except Exception:
                    continue
                key = (score, nq, int(m), -int(L))
                if best is None or key > best[0]:
                    best = (
                        key,
                        dict(
                            nq=nq, m=int(m), L=int(L),
                            score=float(score), p_anc0=float(p_anc0),
                        ),
                    )
    return best[1] if best else None


def main() -> int:
    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q23 Vector Database (QVDB — entangled DB + jedan SWAP-test + post-select): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    D_best = 2 ** best["m"]
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| m (idx aux):", best["m"],
        "| D = 2^m (dokumenata):", D_best,
        "| L (query window):", best["L"],
        "| P(anc=0):", round(float(best["p_anc0"]), 6),
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    nq_best = int(best["nq"])
    m_best = int(best["m"])
    L_best = int(best["L"])

    p_idx, p_anc0 = qvdb_posterior_idx(H, nq_best, m_best, L_best)
    print("--- posterior nad idx-registrom (anc=0) ---")
    for d in range(D_best):
        print(f"  d={d:2d}  P(idx=d | anc=0) = {float(p_idx[d]):.6f}")

    p_emb, _ = qvdb_state_probs(H, nq_best, m_best, L_best)
    pred = pick_next_combination(p_emb)
    print("--- glavna predikcija (QVDB post-select anc=0, marginala emb) ---")
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q23 Vector Database (QVDB — entangled DB + jedan SWAP-test + post-select): CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 5 | m (idx aux): 1 | D = 2^m (dokumenata): 2 | L (query window): 1000 | P(anc=0): 0.997545 | cos(bias, freq_csv): 0.89716
--- posterior nad idx-registrom (anc=0) ---
  d= 0  P(idx=d | anc=0) = 0.499445
  d= 1  P(idx=d | anc=0) = 0.500555
--- glavna predikcija (QVDB post-select anc=0, marginala emb) ---
predikcija NEXT: (7, 9, x, y, z, 27, 28)
"""



"""
Q23_VectorDatabase_embedding.py — tehnika: Quantum Vector Database (QVDB).

Koncept:
Entanglovan DB state (1/√D) Σ_d |d⟩_idx ⊗ |ψ_d⟩_emb drži SVE dokument-embedding-e
u jednoj superpoziciji, indeksirane idx-registrom. Jedan SWAP-test između emb
i query registra realizuje similarity-lookup za sve dokumente ODJEDNOM preko
kvantnog paralelizma. Post-selekcija anc=0 pomera verovatnoću ka visoko-sličnim
(query-bliskim) dokumentima:
        P(anc=0, idx=d) ∝ (1 + |⟨ψ_d|ψ_q⟩|²) / (2D).

Kolo (2·nq + m + 1 qubit-a):
  H^⊗m na idx, multi-ctrl StatePreparation po d za emb (entanglovan DB).
  StatePreparation(|ψ_q⟩) na query.
  H(anc), cswap(anc, emb_i, query_i), H(anc).
Readout:
  Marginala emb registra nakon post-select anc=0 + marginalizacija idx, query.
  bias_39 → TOP-7 = NEXT.

Tehnike:
Amplitude encoding po dokumentu (StatePreparation).
Entanglovana indeks-embedding struktura (H^⊗m + multi-ctrl SP).
Interferencijski similarity-lookup kroz SWAP-test (single-stage).
Post-selekcija kao „algoritamski" mehanizam ka bližim dokumentima.
Egzaktni Statevector (bez uzorkovanja).
Deterministička grid-optimizacija (nq, m, L).

Prednosti:
Single-stage kvantni pristup (razlika od Q22 QRAG koji je two-stage).
Pravi „vector DB" pattern: indeks + embedding + NN-lookup u jednom kolu.
Težine dokumenata izlaze prirodno iz kvantne interferencije, bez eksplicitnog
klasičnog ranking koraka.
Ceo CSV je korpus (pravilo 10), query je tail-L kao „prompt".
Čisto kvantno: bez klasičnog treninga, bez softmax-a, bez hibrida.

Nedostaci:
Post-selekcija na anc=0 ima P(anc=0) ∈ [0.5, 1], ali u praksi blizu 0.5
(slabo diskriminiše low-similarity od high-similarity dokumenata).
(1+s_d) formula znači da čak i potpuno neslični dokumenti (s_d=0) imaju nenula
težinu 1/(D+Σs_d') — blago zaglađeno u poređenju sa top-K selekcijom (Q22).
Budžet qubit-a: 2·nq+m+1 → do 16 qubit-a za nq=6, m=3.
mod-39 readout meša stanja (dim 2^nq ≠ 39).
"""
