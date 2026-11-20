"""Redraws every picture used in the README. Schematics only, no calculation data.
    python docs/make_graphics.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
OUT = Path(__file__).resolve().parent / 'img'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 11})
INK, TEAL, RUST, BLUE, GREY = '#12263a', '#1f7a7a', '#c0561b', '#2a6fb0', '#8a95a3'
PALE = {'ink': '#eef2f6', 'teal': '#e3f1f1', 'rust': '#fbeee4', 'blue': '#e6eef8'}


def box(ax, x, y, w, h, text, fc=PALE['ink'], ec=INK, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.02', fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold' if bold else 'normal', linespacing=1.2)


def arrow(ax, a, b, color=INK, rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14, lw=1.5, color=color,
                                 connectionstyle=f'arc3,rad={rad}', linestyle=ls))


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.2, 1); ax.axis('off')
    cols = [(0.01, 'ink'), (0.255, 'blue'), (0.50, 'teal'), (0.745, 'rust')]
    heads = ['1  Calculations', '2  Bookkeeping', '3  Models', '4  Questions answered']
    items = [
        ['structures + INCAR\n(templates/)', 'job folders\ndacpipe make', 'submit, wait\ndacpipe submit'],
        ['collect energies\ndacpipe collect', 'small-file archive\ndacpipe pack', 'tables in data/\n(see docs/schema.md)'],
        ['free energies, volcano\nthermo.py', 'GP + leave-one-metal-out\nml.py', 'site microkinetics\nmicrokinetics.py'],
        ['which pairs bind and\nactivate N$_2$?', 'which quantities can\nbe learned, how sure?', 'onset potential,\nselectivity, C-N barrier'],
    ]
    for (x, c), head, rows in zip(cols, heads, items):
        ax.text(x + 0.105, 0.94, head, ha='center', fontsize=12, fontweight='bold', color=INK)
        for k, t in enumerate(rows):
            y = 0.68 - 0.28 * k
            box(ax, x, y, 0.21, 0.2, t, fc=PALE[c], ec={'ink': INK, 'blue': BLUE, 'teal': TEAL, 'rust': RUST}[c])
            if k < 2:
                arrow(ax, (x + 0.105, y - 0.005), (x + 0.105, y - 0.075))
    for x in (0.222, 0.467, 0.712):
        arrow(ax, (x, 0.5), (x + 0.03, 0.5))
    box(ax, 0.01, -0.17, 0.945, 0.1, 'constrained MD profile of the C-N step  (cn.py)   |   job helpers run on slurm, pbs, lsf or plain bash',
        fc='white', ec=GREY, fs=10)
    fig.savefig(OUT / 'pipeline.png', dpi=200, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- routes
def routes():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(0.0, 0.93, 'CO-route', fontweight='bold', color=BLUE, fontsize=12)
    ax.text(0.12, 0.93, 'hydrogenate first, insert CO last', color=GREY, fontsize=10.5)
    names = ['*N$_2$', '*N$_2$H', '*HNNH', '*HNNH$_2$', '*NH$_2$NH$_2$']
    for k, n in enumerate(names):
        box(ax, 0.0 + k * 0.165, 0.62, 0.125, 0.18, n, fc=PALE['blue'], ec=BLUE)
        if k < 4:
            arrow(ax, (0.127 + k * 0.165, 0.71), (0.163 + k * 0.165, 0.71))
            ax.text(0.145 + k * 0.165, 0.83, 'H$^+$/e$^-$', ha='center', fontsize=8, color=GREY)
    arrow(ax, (0.79, 0.71), (0.836, 0.71), color=RUST)
    ax.text(0.813, 0.83, '+ CO', ha='center', fontsize=9, color=RUST)
    box(ax, 0.84, 0.62, 0.155, 0.18, '*NH$_2$CONH$_2$', fc=PALE['teal'], ec=TEAL)
    ax.text(0.0, 0.40, 'NCON-route', fontweight='bold', color=RUST, fontsize=12)
    ax.text(0.14, 0.40, 'couple C and N first, hydrogenate afterwards', color=GREY, fontsize=10.5)
    n2 = ['*N$_2$', '*NCON', '*NCONH', '*HNCONH', '*HNCONH$_2$']
    for k, n in enumerate(n2):
        box(ax, 0.0 + k * 0.165, 0.09, 0.125, 0.18, n, fc=PALE['rust'], ec=RUST)
        if k < 4:
            arrow(ax, (0.127 + k * 0.165, 0.18), (0.163 + k * 0.165, 0.18), color=RUST)
    arrow(ax, (0.79, 0.18), (0.836, 0.18), color=RUST)
    box(ax, 0.84, 0.09, 0.155, 0.18, '*NH$_2$CONH$_2$', fc=PALE['teal'], ec=TEAL)
    fig.savefig(OUT / 'routes.png', dpi=200, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- learning
def learning():
    fig = plt.figure(figsize=(9, 4.25))
    ax = fig.add_axes([0, 0.5, 1, 0.5]); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.5, 1); ax.axis('off')
    bw, bh, xs = 0.215, 0.62, [0.01, 0.2625, 0.515, 0.775]
    box(ax, xs[0], 0.2, bw, bh, 'atomic features\n24 numbers per pair,\nno calculation')
    box(ax, xs[1], 0.2, bw, bh, 'DFT on a subset\n(the labels)', fc=PALE['teal'], ec=TEAL)
    box(ax, xs[2], 0.2, bw, bh, 'Gaussian process:\ntrend + its own\nuncertainty', fc=PALE['teal'], ec=TEAL)
    box(ax, xs[3], 0.2, bw, bh, 'prediction $\\pm\\sigma$\nfor the rest', fc=PALE['rust'], ec=RUST)
    for i in range(3):
        arrow(ax, (xs[i] + bw + 0.004, 0.51), (xs[i + 1] - 0.004, 0.51))
    arrow(ax, (0.88, 0.2), (0.37, 0.2), color=RUST, rad=-0.12)
    ax.text(0.625, -0.2, 'run DFT where the model is least sure, add it, refit', ha='center', fontsize=10, color=RUST, style='italic')

    rng = np.random.default_rng(3)
    f = lambda x: np.sin(2.2 * x) + 0.35 * x
    xt = np.array([0.3, 0.9, 1.5, 3.9, 4.4, 5.2]); yt = f(xt) + rng.normal(0, 0.05, xt.size)
    xg = np.linspace(0, 5.6, 300)
    k = lambda a, b: np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / 0.7 ** 2)
    K = k(xt, xt) + 0.01 * np.eye(xt.size); Ks = k(xg, xt)
    mu = Ks @ np.linalg.solve(K, yt)
    sd = np.sqrt(np.clip(1 - np.einsum('ij,ji->i', Ks, np.linalg.solve(K, Ks.T)), 0, None))
    a1 = fig.add_axes([0.07, 0.1, 0.4, 0.33])
    a1.fill_between(xg, mu - 1.5 * sd, mu + 1.5 * sd, color=TEAL, alpha=0.22, lw=0)
    a1.plot(xg, mu, color=TEAL, lw=2); a1.plot(xt, yt, 'o', color=INK, ms=6)
    i = np.argmax(sd)
    a1.annotate('compute\nnext', xy=(xg[i], mu[i]), xytext=(xg[i] - 1.3, mu[i] + 1.1), ha='center', color=RUST,
                fontsize=10, arrowprops=dict(arrowstyle='-|>', color=RUST, lw=1.4))
    a1.axvline(xg[i], color=RUST, ls=':', lw=1.2)
    a1.set(xlabel='feature space (schematic)', ylabel='target energy', xticks=[], yticks=[], ylim=(-2.2, 2.4))
    a1.set_title('where the model is unsure, compute', fontsize=11, color=INK)
    a2 = fig.add_axes([0.58, 0.1, 0.4, 0.33])
    n = np.linspace(10, 130, 60)
    a2.plot(n, 0.25 + 0.55 * np.exp(-n / 35), color=RUST, lw=2.2, label='uncertainty-picked')
    a2.plot(n, 0.25 + 0.6 * np.exp(-n / 110), color=GREY, lw=2.2, ls='--', label='random')
    a2.set(xlabel='pairs computed', ylabel='error on unseen pairs', xticks=[], yticks=[])
    a2.legend(frameon=False, fontsize=9.5); a2.set_title('shape of the learning curve (schematic)', fontsize=11, color=INK)
    clean(a1); clean(a2)
    fig.savefig(OUT / 'learning.png', dpi=200); plt.close(fig)


# ---------------------------------------------------------------- microkinetics
def kinetics():
    fig = plt.figure(figsize=(9, 4.25))
    a = fig.add_axes([0.05, 0.12, 0.27, 0.7])
    x = [0, 1, 2, 3, 4]; G = [0, -0.6, -0.1, -0.9, -0.4]
    for xx, g in zip(x, G):
        a.plot([xx - 0.3, xx + 0.3], [g, g], color=INK, lw=3)
    for i in range(4):
        a.plot([x[i] + 0.3, x[i + 1] - 0.3], [G[i], G[i + 1]], color=GREY, lw=1, ls=':')
    xa = np.linspace(1, 2, 60)
    a.plot(xa, np.interp(xa, [1, 1.5, 2], [-0.6, 0.35, -0.1]), color=RUST, lw=1.8, ls='--')
    a.annotate('', xy=(1.5, 0.35), xytext=(1.5, -0.6), arrowprops=dict(arrowstyle='<->', color=RUST, lw=1.4))
    a.text(1.5, 0.5, '$E_a=\\max(\\Delta G,0)+E_0$', color=RUST, fontsize=10, ha='center')
    a.set_xticks(x); a.set_xticklabels(['*N$_2$', '*N$_2$H', '*HNNH', '*HNNH$_2$', '*NH$_2$NH$_2$'], fontsize=8.5, rotation=25, ha='right')
    a.set(yticks=[], ylabel='free energy', ylim=(-1.2, 1.0))
    a.set_title('1  energies and barriers', fontsize=11, color=INK, loc='left'); clean(a)

    n = fig.add_axes([0.37, 0.12, 0.33, 0.7]); n.set_xlim(0, 1); n.set_ylim(0, 1); n.axis('off')
    fig.text(0.37, 0.865, '2  network of site states', fontsize=11, color=INK)
    pos = {'N2': (0.14, 0.78), 'N2H': (0.5, 0.78), 'HNNH': (0.86, 0.78), 'NH2NH2': (0.86, 0.46),
           'U': (0.5, 0.46), 'NH3': (0.14, 0.46), 'H': (0.14, 0.14)}
    lab = {'N2': '*N$_2$', 'N2H': '*N$_2$H', 'HNNH': '*HNNH...', 'NH2NH2': '*NH$_2$NH$_2$',
           'U': '*NH$_2$CONH$_2$', 'NH3': '*NH$_2$NH$_3$', 'H': '*H'}
    for key, (px, py) in pos.items():
        col = {'U': TEAL, 'NH3': RUST, 'H': RUST}.get(key, INK)
        n.add_patch(FancyBboxPatch((px - 0.13, py - 0.065), 0.26, 0.13, boxstyle='round,pad=0.005,rounding_size=0.02', fc='white', ec=col, lw=1.6))
        n.text(px, py, lab[key], ha='center', va='center', fontsize=9, color=col)

    def na(p, q, c=INK):
        n.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=11, lw=1.4, color=c))
    na((0.275, 0.78), (0.365, 0.78)); na((0.635, 0.78), (0.725, 0.78)); na((0.86, 0.715), (0.86, 0.535))
    n.text(0.5, 0.95, 'H$^+$/e$^-$ steps depend on potential', fontsize=9, ha='center', color=INK)
    na((0.725, 0.46), (0.635, 0.46), c=TEAL); n.text(0.68, 0.34, '+ CO', fontsize=9, color=TEAL, ha='center')
    na((0.14, 0.715), (0.14, 0.535), c=RUST); n.text(0.14, 0.30, 'side routes\ncompete', fontsize=8.5, color=RUST, ha='center')
    n.text(0.7, 0.14, 'coverage $\\theta_i$ per state,\nrate $k=(k_BT/h)\\,e^{-E_a/k_BT}$', fontsize=9.5, ha='center', color=INK)
    n.text(0.7, 0.0, 'steady state: $d\\theta_i/dt=0$', fontsize=10.5, ha='center', color=INK)

    o = fig.add_axes([0.76, 0.12, 0.22, 0.7])
    U = np.linspace(-1.2, 0.8, 200)
    s = lambda w: 1 / (1 + np.exp(-(U + 0.35) / w))
    o.plot(U, np.log10(1e-3 + 10 ** (7 * s(0.18) - 4)), color=TEAL, lw=2.2, label='no C-N barrier')
    o.plot(U, np.log10(1e-3 + 10 ** (3.6 * s(0.18) - 4)), color=RUST, lw=2.2, ls='--', label='barrier caps rate')
    o.axhline(0, color=GREY, lw=1, ls=':')
    o.set(xlabel='potential', ylabel='log rate', xticks=[], yticks=[])
    o.legend(frameon=False, fontsize=8.5, loc='lower right'); o.set_title('3  rate and onset', fontsize=11, color=INK, loc='left'); clean(o)
    fig.savefig(OUT / 'kinetics.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    pipeline(); routes(); learning(); kinetics()
    print('wrote', *sorted(p.name for p in OUT.glob('*.png')))
