"""Runs the analysis stages on synthetic tables so you can see the outputs before using real data.

    python examples/walkthrough.py            # writes docs/img/demo.png

To use your own tables, replace demo.fake_screening() with pd.read_csv(...) of a file that has the
columns listed in docs/schema.md.
"""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dacpipe import cn, demo, features, microkinetics as mk, ml, thermo

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix'})
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'docs/img/demo.png'

df = demo.fake_screening(seed=4)
m1, m2 = features.split_pair(df.pair)
X = features.pair_features(m1, m2).to_numpy()

# volcano
d = thermo.add_columns(df)
fit = thermo.volcano_branches(df)
gx, uy = thermo.apex(fit)
print(f'apex {gx:.2f} eV, {uy:.2f} V')

# learning
res = ml.lomo(X, df.dG_N2.to_numpy(), m1, m2, n_jobs=2)
print(f"LOMO rmse {res['rmse']:.2f} eV, r2 {res['r2']:.2f}, 95% coverage {res['cov95']:.2f}")

# kinetics for the three best pairs on the volcano
U_L = thermo.volcano_curve(fit, d.G_NH2NH2.to_numpy())
best = df.assign(UL=U_L).nlargest(3, 'UL')
P = mk.pairs_from_table(best)
V = np.arange(0.4, -1.6, -0.05)

# constrained MD
with tempfile.TemporaryDirectory() as tmp:
    log = Path(tmp) / 'cc.txt'
    demo.write_constraint_log(log)
    prof = cn.profile(cn.read_constraint_log(log), sign=1)

fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
g = np.linspace(d.G_NH2NH2.min() - 0.2, d.G_NH2NH2.max() + 0.2, 200)
ax[0].scatter(d.G_NH2NH2, thermo.volcano_curve(fit, d.G_NH2NH2.to_numpy()) + np.random.default_rng(0).normal(0, 0.08, len(d)), s=12, c='#2a6fb0', alpha=0.7)
ax[0].plot(g, thermo.volcano_curve(fit, g), 'k--', lw=1.3)
ax[0].set(xlabel=r'$\Delta G$(*NH$_2$NH$_2$) (eV)', ylabel=r'$U_\mathrm{L}$ (V)', title='volcano')
y = df.dG_N2.to_numpy()
ax[1].errorbar(y, res['mu'], res['sg'], fmt='o', ms=3, color='#1f7a7a', alpha=0.5, elinewidth=0.6)
lim = [min(y.min(), res['mu'].min()), max(y.max(), res['mu'].max())]
ax[1].plot(lim, lim, 'k--', lw=1)
ax[1].set(xlabel='computed (eV)', ylabel='predicted (eV)', title='leave-one-metal-out')
for name, p in P.items():
    ax[2].semilogy(V, [max(mk.run(p, u)['tof'], 1e-12) for u in V], lw=1.8, label=name)
ax[2].axhline(1, color='grey', ls=':'); ax[2].legend(frameon=False, fontsize=8)
ax[2].set(xlabel='U vs RHE (V)', ylabel=r'urea rate (s$^{-1}$)', title='microkinetics', ylim=(1e-12, 1e8))
ax[3].errorbar(prof.xi, prof.F, prof.F_err, color='#c0561b', lw=1.5)
ax[3].invert_xaxis(); ax[3].set(xlabel='C-N distance (A)', ylabel='F (eV)', title='constrained MD')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.text(0.5, 0.99, 'synthetic demo data, no physical meaning', ha='center', va='top', fontsize=9, color='grey')
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out, dpi=180)
print('wrote', out)
