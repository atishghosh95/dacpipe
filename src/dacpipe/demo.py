"""Synthetic stand-ins for calculation tables, so every stage can be tried and tested without data.
The numbers are random draws with plausible scaling; they mean nothing physically."""
import numpy as np
import pandas as pd

from .features import all_pairs, pair_features, split_pair


def fake_screening(seed=0, metals=None):
    rng = np.random.default_rng(seed)
    pairs = all_pairs(metals)
    m1, m2 = split_pair(pairs)
    F = pair_features(m1, m2)
    # descriptor loosely driven by electron count and electronegativity, plus noise
    g = (-0.55 * (F.nvalm - 7.5) - 1.1 * (F.chim - 1.7) * 2 - 2.6) + rng.normal(0, 0.35, len(pairs))
    n2 = 0.45 * g + rng.normal(0, 0.15, len(pairs)) - 0.6
    ef = 0.8 * (F.nvalm - 7.5) * 0.15 - 0.35 * F.Ecohm * 0.1 + rng.normal(0, 0.2, len(pairs))
    worst = 0.48 * g + 2.4 + rng.normal(0, 0.25, len(pairs))
    s5 = -0.89 * g - 2.6 + rng.normal(0, 0.25, len(pairs))
    rest = g - n2
    s1 = 0.2 * rest + rng.normal(0, 0.08, len(pairs))
    s2 = worst - np.abs(rng.normal(0, 0.15, len(pairs)))
    s3 = worst - np.abs(rng.normal(0, 0.2, len(pairs)))
    s4 = rest - s1 - s2 - s3
    s4 = np.minimum(s4, worst - 0.01)
    df = pd.DataFrame(dict(pair=pairs, dG_N2=n2, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, E_f=ef))
    return df.join(F)


def write_constraint_log(path, n=400, seed=0):
    """A fake constraint log whose force integrates to a smooth barrier."""
    rng = np.random.default_rng(seed)
    xi = np.linspace(3.2, 1.6, n)
    force = -0.9 * np.exp(-((xi - 2.4) / 0.35) ** 2) * (xi - 2.4) / 0.35 + rng.normal(0, 0.05, n)
    with open(path, 'w') as fh:
        fh.write('# segment 0\n')
        for x, f in zip(xi, force):
            fh.write(f'cc> 1 1 {x:.6f}\n')
            fh.write(f'b_m> {f:.6f} 1.0 0.0\n')
    return xi, force
