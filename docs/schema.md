# Table layouts

All tables are plain csv, one row per pair (or per calculation). Nothing is shipped with the repository,
so these are the columns the code looks for. Energies are in eV, potentials in V vs RHE.

## screening table (thermo.py, ml.py, microkinetics.py)

| column | meaning |
|---|---|
| `pair` | metal pair as `A-B`, e.g. `Zn-Zr` |
| `dG_N2` | free energy of N2 adsorption on the pair |
| `s1` ... `s4` | free energy of the four proton-electron steps at 0 V (N2 to NH2NH2) |
| `s5` | free energy of the C-N step (CO insertion) |
| `E_f` | formation energy of the site, used as a stability target |

Optional columns for the extended microkinetic network: `dG_CO`, `dG_H`, `n5`, `n6` (ammonia branch steps).
Missing optional columns simply remove the matching states from the network.

## calculation table (written by `dacpipe collect`)

`dir, energy_eV, nsteps, converged, finished, nions, magnetization`

## constraint log (cn.py)

Text file written during a constrained MD run. A line starting with `#` begins a new segment, `cc>` lines carry the
constrained coordinate in the fourth field, and each is followed by a `b_m>` line with the multiplier, the
`z^-1/2` factor and the metric term. If a coordinate value appears in several segments the latest one is used.

## task table (`dacpipe make`)

`name, poscar, incar, kpoints` with paths relative to where you run the command. `kpoints` may be empty.
