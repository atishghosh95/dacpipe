"""Computational hydrogen electrode bookkeeping and the volcano built from it."""
import numpy as np
import pandas as pd

STEP_COLS = ['s1', 's2', 's3', 's4', 's5']   # step free energies at 0 V (eV); s5 is the C-N step


def free_energy(dE, zpe_ts):
    """electronic energy + (ZPE - T S) correction, element-wise"""
    return np.asarray(dE) + np.asarray(zpe_ts)


def limiting_potential(steps, electrochemical=(0, 1, 2, 3)):
    """-eta in V. Only the proton-electron steps move with potential, so the chemical
    C-N step is left out unless you pass it in `electrochemical`."""
    s = np.asarray(steps)[:, list(electrochemical)]
    return -s.max(axis=1)


def add_columns(df):
    d = df.copy()
    d['G_NH2NH2'] = d.dG_N2 + d.s1 + d.s2 + d.s3 + d.s4
    d['worst_hyd'] = d[['s2', 's3', 's4']].max(axis=1)
    return d


def volcano_branches(df):
    """Linear scaling of the two competing steps against the descriptor G(*NH2NH2)."""
    d = add_columns(df)
    kL, bL = np.polyfit(d.G_NH2NH2, d.worst_hyd, 1)
    kR, bR = np.polyfit(d.G_NH2NH2, d.s5, 1)
    return dict(slope_hyd=kL, icpt_hyd=bL, slope_ins=kR, icpt_ins=bR,
                r_hyd=np.corrcoef(d.G_NH2NH2, d.worst_hyd)[0, 1],
                r_ins=np.corrcoef(d.G_NH2NH2, d.s5)[0, 1])


def apex(f):
    """where the two branches meet: (descriptor in eV, U_L in V)"""
    g = (f['icpt_ins'] - f['icpt_hyd']) / (f['slope_hyd'] - f['slope_ins'])
    return g, -(f['slope_hyd'] * g + f['icpt_hyd'])


def volcano_curve(f, g):
    """U_L along the descriptor axis for plotting"""
    return -np.maximum(f['slope_hyd'] * g + f['icpt_hyd'], f['slope_ins'] * g + f['icpt_ins'])
