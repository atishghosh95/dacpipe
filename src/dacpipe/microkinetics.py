"""Mean-field microkinetics for one dual-atom site.

States: 0 bare, 1 N2*, 2 N2H*, 3 HNNH*, 4 HNNH2*, 5 NH2NH2*, 6 urea*, 7 CO*, 8 H*, 9 NH2NH3*, 10 NH3NH3*
Barrier of every step: Ea = max(dG, 0) + E0, with E0 an intrinsic barrier you choose.
The steady state is solved with mpmath because coverages can differ by tens of orders of magnitude.
"""
import numpy as np
import mpmath as mp

mp.mp.dps = 60
KT = 0.025693                    # eV at 298 K
NU = KT / 4.135667696e-15        # kT/h, s^-1
NU_ADS = 1e8                     # prefactor for adsorption, s^-1 bar^-1
N_STATES = 11
REL_DEFAULT = -0.5               # eV, release of urea; set it to your own number


def rate_constants(dG, E0, nu=NU):
    Ea = max(dG, 0.0) + E0
    return nu * np.exp(-Ea / KT), nu * np.exp(-(Ea - dG) / KT)


def build_edges(P, U, pN2=1.0, pCO=0.1, E_pcet=0.3, E_ins=0.2, a_urea=1e-6, shift=None):
    """Edges as (i, j, k_fwd, k_bwd, electrons). P holds dG_N2, s1..s5, rel and optionally
    dG_CO, dG_H, n5, n6 (the ammonia branch). U is the potential in V vs RHE."""
    edges = []

    def add(i, j, dG, E0, ne=0, f=1.0, nu=NU, r=1.0):
        kf, kb = rate_constants(dG, E0, nu)
        edges.append((i, j, kf * f, kb * r, ne))

    add(0, 1, P['dG_N2'], 0.0, 0, pN2, NU_ADS)
    for a, b, key in ((1, 2, 's1'), (2, 3, 's2'), (3, 4, 's3'), (4, 5, 's4')):
        add(a, b, P[key] + U, E_pcet, 1)
    add(5, 6, P['s5'], E_ins, 0, pCO)
    add(6, 0, P['rel'], 0.0, 0, 1.0, NU, a_urea)
    if 'dG_CO' in P:
        add(0, 7, P['dG_CO'], 0.0, 0, pCO, NU_ADS)
    if 'dG_H' in P:
        add(0, 8, P['dG_H'] + U, E_pcet, 1)
        add(8, 0, -P['dG_H'] + U, E_pcet, 1)
        add(8, 1, P['dG_N2'] - P['dG_H'] - U, E_pcet, -1, pN2, NU_ADS)
    if 'n5' in P:
        add(5, 9, P['n5'] + U, E_pcet, 1)
        add(9, 10, P['n6'] + U, E_pcet, 1)
        edges.append((10, 0, NU, 0.0, 0))
    if shift:   # per-edge free-energy perturbation, used for robustness sampling
        edges = [(i, j, kf * np.exp(shift.get((i, j), 0) / KT), kb * np.exp(shift.get((i, j), 0) / KT), ne)
                 for i, j, kf, kb, ne in edges]
    return edges


def steady_state(edges):
    used = sorted({e[0] for e in edges} | {e[1] for e in edges})
    pos = {s: k for k, s in enumerate(used)}
    n = len(used)
    A = mp.zeros(n, n)
    for i, j, kf, kb, _ in edges:
        a, b = pos[i], pos[j]
        kf, kb = mp.mpf(kf), mp.mpf(kb)
        A[b, a] += kf
        A[a, a] -= kf
        A[a, b] += kb
        A[b, b] -= kb
    for t in range(n):          # replace one balance equation by sum(theta) = 1
        A[n - 1, t] = 1
    rhs = mp.zeros(n, 1)
    rhs[n - 1] = 1
    th = mp.lu_solve(A, rhs)
    out = np.zeros(N_STATES)
    for s in used:
        out[s] = float(th[pos[s]])
    return out


def run(P, U, **kw):
    edges = build_edges(P, U, **kw)
    th = steady_state(edges)
    flux = {(i, j): th[i] * kf - th[j] * kb for i, j, kf, kb, _ in edges}
    r_urea = flux[(6, 0)]
    r_nh3 = flux.get((10, 0), 0.0)
    r_h2 = flux.get((8, 0), 0.0)
    ru, rn, rh = max(r_urea, 0.0), max(r_nh3, 0.0), max(r_h2, 0.0)
    den = 4 * ru + 6 * rn + 2 * rh      # electrons per urea / NH3 / H2
    fe = (lambda x: x / den) if den > 1e-30 else (lambda x: np.nan)
    return dict(theta=th, tof=r_urea, fe_urea=fe(4 * ru), fe_nh3=fe(6 * rn), fe_h2=fe(2 * rh))


def onset(P, rate=1.0, hi=1.0, lo=-1.8, step=0.01, **kw):
    """Most positive potential on a descending grid where the urea rate reaches `rate` s^-1."""
    for U in np.arange(hi, lo - 1e-9, -step):
        if run(P, U, **kw)['tof'] >= rate:
            return round(float(U), 2)
    return None


def pairs_from_table(df, rel=REL_DEFAULT):
    """One dict per row of a DataFrame with columns pair, dG_N2, s1..s5 (+ optional extras)."""
    optional = [c for c in ('dG_CO', 'dG_H', 'n5', 'n6') if c in df.columns]
    out = {}
    for _, r in df.iterrows():
        d = {k: float(r[k]) for k in ('dG_N2', 's1', 's2', 's3', 's4', 's5')}
        d['rel'] = rel
        d.update({k: float(r[k]) for k in optional if r[k] == r[k]})
        out[r['pair']] = d
    return out
