"""Scientific kernels for one independently restartable cluster task."""
from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

from gbosons import banded as bd
from gbosons import benchmarking as bm
from gbosons import bounded_n as bn
from gbosons import fock_ref
from gbosons import lattices
from gbosons import nilpotent_phase as nph
from gbosons import variational as var


def chain_hopping(n, J=1.0, periodic=False):
    h = np.zeros((n, n), dtype=complex)
    for j in range(n - 1):
        h[j, j + 1] = h[j + 1, j] = -J
    if periodic:
        h[0, -1] = h[-1, 0] = -J
    return h


def bose_hubbard(n, N, J, U, periodic=False):
    basis, index, d = bn.sector_basis(n, N)
    chi = np.zeros((n, n)); np.fill_diagonal(chi, U / 2)
    H = bn.number_conserving_hamiltonian(
        chain_hopping(n, J, periodic), chi, basis, index)
    return H, basis, index, d


def bounded(task):
    n, cutoff = task["modes"], task["Nmax"]
    t0 = perf_counter()
    basis, index, _ = bn.sector_union_basis(n, range(cutoff + 1))
    chi = np.zeros((n, n)); np.fill_diagonal(chi, task["U"] / 2)
    H = bn.number_conserving_hamiltonian(
        chain_hopping(n, task["J"]), chi, basis, index)
    build_seconds = perf_counter() - t0
    rng = np.random.default_rng(task["seed"])
    psi = rng.standard_normal(len(basis)) + 1j * rng.standard_normal(len(basis))
    psi /= np.linalg.norm(psi)
    timing = bm.timed_call(
        lambda: expm_multiply(-1j * task["time"] * H, psi),
        repeats=task["repeats"], warmups=task["warmups"], target_seconds=0.03)
    return ({"carrier_dimension": len(basis), "hamiltonian_nnz": int(H.nnz),
             "build_seconds": build_seconds, **timing}, {})


def _doublon_diag(basis):
    occ = np.asarray(basis, dtype=float)
    return 0.5 * np.sum(occ * (occ - 1), axis=1)


def doublon(task):
    n, U, J = task["modes"], task["U"], task["J"]
    H, basis, index, d = bose_hubbard(n, 2, J, U,
                                      periodic=task["kind"] == "spectrum")
    if task["kind"] == "spectrum":
        k = min(n, d - 2)
        energies = eigsh(H, k=k, which="LA", return_eigenvectors=False,
                         tol=task["eig_tol"])
        width = float(np.ptp(energies))
        return ({"carrier_dimension": d, "band_width": width,
                 "J_eff": width / 4}, {"energies": energies})
    c = n // 2
    occ = [0] * n; occ[c] = 2
    psi0 = bn.basis_state(occ, index)
    ts = np.linspace(0, task["T"], task["n_time"])
    states = expm_multiply(-1j * H, psi0, start=0, stop=ts[-1],
                           num=len(ts), endpoint=True)
    probabilities = np.abs(states) ** 2
    number = np.asarray([bn.number_diag(j, basis) for j in range(n)])
    density = probabilities @ number.T
    doublon_fraction = probabilities @ _doublon_diag(basis)
    sites = np.arange(n) - c
    rms = np.sqrt(np.maximum((density @ sites ** 2) / 2, 0))
    fit = (ts >= 0.25 * ts[-1]) & (ts <= 0.7 * ts[-1])
    velocity = float(np.polyfit(ts[fit], rms[fit], 1)[0])
    return ({"carrier_dimension": d, "velocity": velocity,
             "final_doublon_fraction": float(doublon_fraction[-1])},
            {"times": ts, "density": density, "doublon_fraction": doublon_fraction,
             "rms": rms})


def _sector_otoc_chunked(H, basis, psi, center, times, chunk, stride=1):
    n = len(basis[0])
    sites = np.arange(0, n, stride)
    ndiag = np.asarray([bn.number_diag(i, basis) for i in sites])
    nc = bn.number_diag(center, basis)
    out = np.zeros((len(times), len(sites)))
    for ti, t in enumerate(times):
        if t == 0:
            continue
        psit = expm_multiply(-1j * H * t, psi)
        at = expm_multiply(-1j * H * t, nc * psi)
        for start in range(0, len(sites), chunk):
            stop = min(start + chunk, len(sites))
            block = ndiag[start:stop]
            ea = expm_multiply(1j * H * t, (block * at).T)
            ep = expm_multiply(1j * H * t, (block * psit).T)
            comm = ea - nc[:, None] * ep
            out[ti, start:stop] = np.sum(np.abs(comm) ** 2, axis=0)
    return sites, out


def otoc(task):
    n, U = task["modes"], task["U"]
    H, basis, _, d = bose_hubbard(n, 2, task["J"], U)
    rng = np.random.default_rng(task["seed"])
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    ts = np.linspace(0, task["T"], task["n_time"])
    sites, values = _sector_otoc_chunked(H, basis, psi, n // 2, ts,
                                         task["site_chunk"],
                                         int(task.get("site_stride", 1)))
    distances = np.abs(sites - n // 2)
    velocities = {}
    for threshold in task["thresholds"]:
        fronts = np.array([distances[row >= threshold].max(initial=0) for row in values])
        mask = (fronts > 0) & (fronts < 0.4 * n)
        velocities[str(threshold)] = (float(np.polyfit(ts[mask], fronts[mask], 1)[0])
                                      if np.count_nonzero(mask) >= 3 else float("nan"))
    return ({"carrier_dimension": d, "front_velocities": velocities},
            {"times": ts, "otoc": values, "sites": sites})


def topology_twist(task):
    L, U, flux = task["L"], task["U"], task["flux"]
    grid, ix, iy = task["twist_grid"], task["ix"], task["iy"]
    theta_x, theta_y = 2 * np.pi * ix / grid, 2 * np.pi * iy / grid
    h, _ = lattices.hofstadter_torus(L, L, flux=flux,
                                     theta_x=theta_x, theta_y=theta_y)
    M = L * L
    basis, index, d = bn.sector_basis(M, 2)
    chi = np.zeros((M, M)); np.fill_diagonal(chi, U / 2)
    H = bn.number_conserving_hamiltonian(h, chi, basis, index)
    k = min(M + task["oversample"], d - 2)
    energies, vectors = eigsh(H, k=k, sigma=2 * U, which="LM",
                              tol=task["eig_tol"])
    order = np.argsort(energies)
    energies, vectors = energies[order], vectors[:, order]
    doublon_rows = [index[tuple(2 if s == site else 0 for s in range(M))]
                    for site in range(M)]
    fraction = np.sum(np.abs(vectors[doublon_rows]) ** 2, axis=0)
    candidates = np.argsort(np.abs(energies - 2 * U) - 0.25 * fraction)[:M]
    candidates = candidates[np.argsort(energies[candidates])]
    selected_e = energies[candidates]
    selected_v = vectors[:, candidates]
    split = M // 2
    gap = float(selected_e[split] - selected_e[split - 1])
    frame = selected_v[:, :split]
    return ({"carrier_dimension": d, "rank": split, "subband_gap": gap,
             "min_doublon_fraction": float(np.min(fraction[candidates]))},
            {"energies": selected_e, "frame": frame})


def _control_problem(L, geometry, flux):
    M = L * L
    h, _ = lattices.hofstadter(L, L, flux=flux)
    basis, index, d = bn.sector_basis(M, 2)
    Hhop = bn.passive_hamiltonian(h, basis, index)
    chi = np.zeros((M, M)); np.fill_diagonal(chi, 1.0)
    Hkerr = bn.cross_kerr_hamiltonian(chi, basis)
    ndiag = [bn.number_diag(k, basis) for k in range(M)]
    pots = [sp.diags(diag).tocsr() for diag in ndiag]
    occ = np.asarray(basis, dtype=float)
    weights = (0.5 * occ * (occ - 1)).T
    if geometry == 0:
        inputs, target = [0, 1], L + 1
    else:
        inputs, target = [0, L], (L // 2) * L + L // 2
    initial = [1 if site in inputs else 0 for site in range(M)]
    return M, d, Hhop, Hkerr, ndiag, pots, weights, bn.basis_state(initial, index), target


def _control_layers(depth, M, Hhop, Hkerr, ndiag, pots):
    gens, diags, kerr_mask = [], [], []
    for _ in range(depth):
        gens.extend([Hhop, Hkerr]); diags.extend([None, Hkerr.diagonal()])
        kerr_mask.extend([False, True])
        gens.extend(pots); diags.extend(ndiag); kerr_mask.extend([False] * M)
    return gens, diags, np.asarray(kerr_mask)


def control(task):
    M, d, Hhop, Hkerr, ndiag, pots, weights, psi0, target_site = _control_problem(
        task["L"], task["geometry"], task["flux"])
    gens, diags, kerr_mask = _control_layers(task["depth"], M, Hhop, Hkerr,
                                              ndiag, pots)
    target = np.eye(M)[target_site]
    if task["kind"] == "gradient":
        rng = np.random.default_rng(task["seed"])
        theta = 0.15 * rng.standard_normal(len(gens))
        lossgrad = lambda x: var.sparse_layered_loss_and_grad(
            x, gens, psi0, weights, target, diags)
        _, gradient, _ = lossgrad(theta)
        step = task["finite_difference_step"]
        finite = np.empty_like(gradient)
        for j in range(len(theta)):
            delta = np.zeros_like(theta); delta[j] = step
            finite[j] = (lossgrad(theta + delta)[0] - lossgrad(theta - delta)[0]) / (2 * step)
        return ({"carrier_dimension": d,
                 "gradient_max_absolute_error": float(np.max(np.abs(gradient - finite)))},
                {"gradient": gradient, "finite_difference": finite})
    active = np.ones(len(gens), bool) if task["model"] == "kerr" else ~kerr_mask
    lossgrad = lambda x: var.sparse_layered_loss_and_grad(
        x, gens, psi0, weights, target, diags)
    records = []
    histories, profiles = [], []
    for restart in range(task["restart_start"], task["restart_stop"]):
        rng = np.random.default_rng(task["seed"] + restart)
        theta = 0.15 * rng.standard_normal(len(gens)); theta[~active] = 0
        _, history, obs = var.adam(theta, lossgrad, steps=task["steps"],
                                   lr=task["lr"], mask=active)
        records.append({"restart": restart, "seed": task["seed"] + restart,
                        "target_probability": float(obs[target_site]),
                        "final_loss": float(history[-1])})
        histories.append(history); profiles.append(obs)
    return ({"carrier_dimension": d, "records": records},
            {"loss_histories": np.asarray(histories), "final_profiles": np.asarray(profiles)})


def _packed_phase(n, degree, scale):
    def coefficient(ordinal, modes):
        return scale * np.cos(0.37 * (ordinal + sum(modes) + degree)) / np.sqrt(n)
    return nph.homogeneous_polynomial(n, degree, coefficient)


def cubic(task):
    n, degree, depth = task["modes"], task["degree"], task["depth"]
    indices = np.arange(n, dtype=float)
    shifts = [task["shift_scale"] * np.cos((layer + 1) * (indices + 1)) / np.sqrt(n)
              for layer in range(depth)]
    if degree == 3:
        total = indices[:, None, None] + indices[None, :, None] + indices[None, None, :]
        base = task["coefficient_scale"] * np.cos(0.37 * (total + 3)) / np.sqrt(n)
        phases = [base / (layer + 1) for layer in range(depth)]
        fn = lambda: nph.effective_cubic_tensors(phases, shifts)
        output = fn()
        coefficient_count = int(sum(np.size(x) for x in output))
    else:
        base = _packed_phase(n, degree, task["coefficient_scale"])
        phases = [nph.scale(base, 1 / (layer + 1)) for layer in range(depth)]
        fn = lambda: nph.effective_packed_phases(phases, shifts)
        output = fn(); coefficient_count = len(output)
    timing = bm.timed_call(fn, repeats=task["repeats"], warmups=task["warmups"],
                           target_seconds=0.04)
    return ({"algebra_dimension": nph.algebra_dimension(n, degree),
             "coefficient_count": coefficient_count, **timing}, {})


def _squeezing_state(n, N, k, U, J, r, time):
    basis, index, _ = bd.reachable_basis(n, N, k)
    h = chain_hopping(n, J)
    chi = np.zeros((n, n)); chi[0, 1] = U
    H = (bd.hopping(h, basis, index) + bd.cross_kerr(chi, basis)
         + bd.squeeze(0, 1, r, basis, index))
    occ = [0] * n; occ[0] = occ[1] = 1
    psi0 = bd.basis_state(occ, index, len(basis))
    return basis, index, expm_multiply(-1j * time * H, psi0)


def _band_rows(states, index):
    return np.fromiter((index[state] for state in states), dtype=np.int64,
                       count=len(states))


def squeezing(task):
    args = [task[key] for key in ("modes", "N", "U", "J", "r", "time")]
    n, N, U, J, r, time = args
    ref_basis, ref_index, ref = _squeezing_state(n, N, task["reference_order"],
                                                 U, J, r, time)
    cert_basis, cert_index, cert = _squeezing_state(n, N, task["certificate_order"],
                                                    U, J, r, time)
    ref_obs = bd.nn_expect(0, 1, ref_basis, ref)
    cert_obs = bd.nn_expect(0, 1, cert_basis, cert)
    overlap = complex(np.vdot(ref, cert[_band_rows(ref_basis, cert_index)]))
    cert_gap = float(np.sqrt(max(0.0, 2 - 2 * abs(overlap))))
    ref_weight = np.abs(ref) ** 2
    records = []
    for order in task["orders"]:
        basis, _, psi = _squeezing_state(n, N, order, U, J, r, time)
        obs = bd.nn_expect(0, 1, basis, psi)
        # sum the complement directly: 1 - retained cancels catastrophically
        # once the leaked weight drops below machine epsilon
        inside = np.zeros(len(ref), dtype=bool)
        inside[_band_rows(basis, ref_index)] = True
        leakage = float(np.sqrt(ref_weight[~inside].sum()))
        records.append({"order": order, "carrier_dimension": len(basis),
                        "observable_error": abs(obs - ref_obs), "leakage_norm": leakage})
    return ({"records": records, "reference_observable_gap": abs(ref_obs - cert_obs),
             "reference_state_gap": cert_gap}, {})


RUNNERS = {
    "bounded": bounded,
    "doublon": doublon,
    "otoc": otoc,
    "topology": topology_twist,
    "control": control,
    "cubic": cubic,
    "squeezing": squeezing,
}


def run(experiment, task):
    return RUNNERS[experiment](task)
