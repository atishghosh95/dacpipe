"""Atomic features for a pair of metals. Nothing here needs a calculation."""
import numpy as np
import pandas as pd

# nval, electronegativity (Pauling), first ionisation energy (eV), cohesive energy (eV/atom),
# covalent radius (A), period
ELEMENTS = {
    'Sc': (3, 1.36, 6.56, 3.90, 1.70, 4), 'Ti': (4, 1.54, 6.83, 4.85, 1.60, 4),
    'V':  (5, 1.63, 6.75, 5.31, 1.53, 4), 'Cr': (6, 1.66, 6.77, 4.10, 1.39, 4),
    'Mn': (7, 1.55, 7.43, 2.92, 1.39, 4), 'Fe': (8, 1.83, 7.90, 4.28, 1.32, 4),
    'Co': (9, 1.88, 7.88, 4.39, 1.26, 4), 'Ni': (10, 1.91, 7.64, 4.44, 1.24, 4),
    'Cu': (11, 1.90, 7.73, 3.49, 1.32, 4), 'Zn': (12, 1.65, 9.39, 1.35, 1.22, 4),
    'Y':  (3, 1.22, 6.22, 4.37, 1.90, 5), 'Zr': (4, 1.33, 6.63, 6.25, 1.75, 5),
    'Nb': (5, 1.60, 6.76, 7.57, 1.64, 5), 'Mo': (6, 2.16, 7.09, 6.82, 1.54, 5),
    'Tc': (7, 1.90, 7.28, 6.85, 1.47, 5), 'Ru': (8, 2.20, 7.36, 6.74, 1.46, 5),
    'Rh': (9, 2.28, 7.46, 5.75, 1.42, 5), 'Pd': (10, 2.20, 8.34, 3.89, 1.39, 5),
    'Ag': (11, 1.93, 7.58, 2.95, 1.45, 5), 'Cd': (12, 1.69, 8.99, 1.16, 1.44, 5),
}
PROPS = ['nval', 'chi', 'IE', 'Ecoh', 'rcov', 'per']


def split_pair(names):
    """'Zn-Zr' -> ('Zn', 'Zr') for a whole column."""
    parts = pd.Series(names).str.split('-', n=1, expand=True)
    return parts[0].to_numpy(), parts[1].to_numpy()


def pair_features(m1, m2):
    """Six properties x (metal 1, metal 2, mean, |difference|) = 24 columns."""
    out = {}
    for k, p in enumerate(PROPS):
        a = np.array([ELEMENTS[m][k] for m in m1], dtype=float)
        b = np.array([ELEMENTS[m][k] for m in m2], dtype=float)
        out[p + '1'], out[p + '2'] = a, b
        out[p + 'm'], out[p + 'd'] = (a + b) / 2, np.abs(a - b)
    return pd.DataFrame(out)


def all_pairs(metals=None):
    """Every homo- and heteronuclear combination, e.g. 20 metals -> 210 pairs."""
    metals = list(metals or ELEMENTS)
    return [f"{a}-{b}" for i, a in enumerate(metals) for b in metals[i:]]
