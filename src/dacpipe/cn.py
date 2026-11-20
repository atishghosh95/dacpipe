"""Free-energy profile along a distance coordinate from constrained (blue-moon) MD.

dF/dxi = < lambda + kT * G >, where lambda is the Lagrange multiplier of the constraint and G the
metric correction printed by the code. Averages are taken in bins of xi and integrated.
"""
import numpy as np
import pandas as pd

KB = 8.617333e-5   # eV/K


def read_constraint_log(path):
    """Read a log with lines 'cc> ... xi' followed by 'b_m> lambda z^-1/2 G'.
    A line starting with '#' begins a new segment; for a repeated xi the latest segment wins."""
    seg, xi, rows = -1, None, []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0].startswith('#'):
                seg += 1
            elif t[0] == 'cc>':
                xi = float(t[3])
            elif t[0] == 'b_m>' and xi is not None:
                rows.append((seg, xi, float(t[1]), float(t[2]), float(t[3])))
                xi = None
    latest = {}
    for s, x, l, z, g in rows:
        latest.setdefault(round(x, 4), []).append((s, l, z, g))
    out = []
    for key in sorted(latest):
        top = max(v[0] for v in latest[key])
        s, l, z, g = [v for v in latest[key] if v[0] == top][-1]
        out.append((key, l, z, g))
    return np.array(out)


def profile(D, sign=1, bin_width=0.05, T=400.0, n_blocks=5):
    kT = KB * T
    xi, lam, _, g = D.T
    f = sign * (lam + kT * g)
    edges = np.arange(xi.min(), xi.max() + bin_width, bin_width)
    which = np.digitize(xi, edges) - 1
    xc, fm, fe = [], [], []
    for b in range(len(edges) - 1):
        sel = which == b
        if sel.sum() < 3:
            continue
        vals = f[sel]
        k = min(n_blocks, len(vals) // 2)
        blocks = np.array_split(vals, k) if k >= 2 else [vals]
        xc.append(xi[sel].mean())
        fm.append(vals.mean())
        fe.append(np.std([c.mean() for c in blocks], ddof=1) / np.sqrt(len(blocks)) if len(blocks) > 1 else np.nan)
    xc, fm, fe = map(np.array, (xc, fm, fe))
    fe = np.nan_to_num(fe, nan=np.nanmedian(fe))
    dx = np.diff(xc)
    F = np.r_[0, np.cumsum(0.5 * (fm[1:] + fm[:-1]) * dx)]
    dF = np.r_[0, np.sqrt(np.cumsum((0.5 * dx) ** 2 * (fe[1:] ** 2 + fe[:-1] ** 2)))]
    return pd.DataFrame(dict(xi=xc, force=fm, force_err=fe, F=F, F_err=dF))


def barrier(prof):
    """highest point of F minus the plateau at the far end (mean of the last three bins)"""
    return float(prof.F.max() - prof.F.iloc[-3:].mean())
