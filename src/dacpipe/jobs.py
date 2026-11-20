"""Job folders, submission, result collection and packing. Standard library only (plus tomllib).

Works with slurm, pbs, lsf or plain bash. Nothing is tied to one machine; site specifics live in
config.toml (see templates/).
"""
import csv
import hashlib
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:     # python < 3.11
    tomllib = None

DEFAULTS = dict(scheduler='slurm', account='', partition='', nodes=1, cores=32, walltime='04:00:00',
                modules=[], vasp_cmd='mpirun -np {cores} vasp_std', potcar_dir='', pack_max_mb=50)


def load_config(path=None):
    cfg = dict(DEFAULTS)
    if path and Path(path).exists():
        if tomllib is None:
            raise RuntimeError('reading config.toml needs python 3.11 or newer')
        with open(path, 'rb') as fh:
            cfg.update(tomllib.load(fh))
    return cfg


def job_script(cfg, name):
    sch = cfg['scheduler']
    head = []
    if sch == 'slurm':
        head = [f'#SBATCH --job-name={name}', f"#SBATCH --nodes={cfg['nodes']}",
                f"#SBATCH --ntasks={cfg['cores']}", f"#SBATCH --time={cfg['walltime']}"]
        if cfg['account']:
            head.append(f"#SBATCH --account={cfg['account']}")
        if cfg['partition']:
            head.append(f"#SBATCH --partition={cfg['partition']}")
    elif sch == 'pbs':
        head = [f'#PBS -N {name}',
                f"#PBS -l select={cfg['nodes']}:ncpus={cfg['cores']}:mpiprocs={cfg['cores']}",
                f"#PBS -l walltime={cfg['walltime']}"]
        if cfg['account']:
            head.append(f"#PBS -A {cfg['account']}")
        if cfg['partition']:
            head.append(f"#PBS -q {cfg['partition']}")
    elif sch == 'lsf':
        head = [f'#BSUB -J {name}', f"#BSUB -n {cfg['cores']}", f"#BSUB -W {cfg['walltime'][:5]}"]
        if cfg['account']:
            head.append(f"#BSUB -P {cfg['account']}")
        if cfg['partition']:
            head.append(f"#BSUB -q {cfg['partition']}")
    cd = {'slurm': 'cd "$SLURM_SUBMIT_DIR"', 'pbs': 'cd "$PBS_O_WORKDIR"',
          'lsf': 'cd "$LS_SUBCWD"', 'local': 'cd "$(dirname "$0")"'}[sch]
    mods = [f'module load {m}' for m in cfg['modules']]
    run = cfg['vasp_cmd'].format(cores=cfg['cores'])
    return '\n'.join(['#!/bin/bash', *head, *mods, cd, f'{run} > vasp.out 2>&1', 'touch .done', ''])


def build_potcar(poscar, potcar_dir, out):
    """Concatenate potentials in POSCAR element order. The potentials are licensed, you supply them."""
    elements = Path(poscar).read_text().splitlines()[5].split()
    with open(out, 'wb') as fh:
        for el in elements:
            src = Path(potcar_dir) / el / 'POTCAR'
            if not src.exists():
                raise FileNotFoundError(f'no POTCAR for {el} under {potcar_dir}')
            fh.write(src.read_bytes())


def make(tasks_csv, cfg, root='jobs'):
    """tasks.csv: name, poscar, incar, kpoints (optional). One folder per row."""
    made = []
    with open(tasks_csv) as fh:
        for row in csv.DictReader(fh):
            d = Path(root) / row['name']
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(row['poscar'], d / 'POSCAR')
            shutil.copy(row['incar'], d / 'INCAR')
            if row.get('kpoints'):
                shutil.copy(row['kpoints'], d / 'KPOINTS')
            if cfg['potcar_dir']:
                build_potcar(d / 'POSCAR', cfg['potcar_dir'], d / 'POTCAR')
            (d / 'job.sh').write_text(job_script(cfg, row['name']))
            made.append(d)
    return made


def submit(root, cfg, dry_run=False):
    cmd = {'slurm': ['sbatch'], 'pbs': ['qsub'], 'local': ['bash']}.get(cfg['scheduler'])
    n = 0
    for script in sorted(Path(root).rglob('job.sh')):
        d = script.parent
        if (d / '.submitted').exists() or (d / '.done').exists():
            continue
        if cfg['scheduler'] == 'lsf':
            full = ['bash', '-c', f'bsub < {script.resolve()}']
        else:
            full = cmd + [str(script.resolve())]
        print(('would run: ' if dry_run else 'run: ') + ' '.join(full))
        if not dry_run:
            subprocess.run(full, cwd=d, check=False)
            (d / '.submitted').touch()
        n += 1
    print(n, 'job(s)', '(dry run)' if dry_run else 'submitted')
    return n


_TOTEN = re.compile(r'free  energy   TOTEN\s*=\s*(-?\d+\.\d+)')
_NIONS = re.compile(r'NIONS\s*=\s*(\d+)')
_MAG = re.compile(r'number of electron\s+\S+\s+magnetization\s+(-?\d+\.\d+)')


def read_outcar(path):
    txt = Path(path).read_text(errors='ignore')
    e = _TOTEN.findall(txt)
    mag = _MAG.findall(txt)
    n = _NIONS.search(txt)
    return dict(energy_eV=float(e[-1]) if e else None, nsteps=len(e),
                converged='reached required accuracy' in txt,
                finished='Voluntary context switches' in txt,
                nions=int(n.group(1)) if n else None,
                magnetization=float(mag[-1]) if mag else None)


def collect(root, out_csv):
    root = Path(root)
    rows = [dict(dir=str(p.parent.relative_to(root)), **read_outcar(p)) for p in sorted(root.rglob('OUTCAR'))]
    cols = ['dir', 'energy_eV', 'nsteps', 'converged', 'finished', 'nions', 'magnetization']
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(len(rows), 'calculations ->', out_csv)
    return rows


# small files only; wavefunctions, densities, potentials and the full OUTCAR stay where they are
KEEP = {'INCAR', 'KPOINTS', 'POSCAR', 'CONTCAR', 'OSZICAR', 'XDATCAR', 'ICONST', 'REPORT', 'vasp.out',
        'job.sh', 'lobsterin', 'ICOHPLIST.lobster', 'COHPCAR.lobster', 'DOSCAR', '.done', '.submitted'}
KEEP_EXT = ('.csv', '.json', '.txt', '.py', '.sh', '.toml', '.md')
NEVER = {'WAVECAR', 'CHGCAR', 'CHG', 'vasprun.xml', 'POTCAR', 'OUTCAR', 'LOCPOT', 'PROCAR',
         'AECCAR0', 'AECCAR1', 'AECCAR2'}


def pack(root, out, max_mb=50):
    root, out = Path(root), Path(out)
    n = total = 0
    with tarfile.open(out, 'w:gz', compresslevel=6) as tar:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.name in NEVER or p.stat().st_size > max_mb * 1e6:
                continue
            if p.name in KEEP or p.name.endswith(KEEP_EXT):
                tar.add(p, arcname=str(p.relative_to(root)))
                n += 1
                total += p.stat().st_size
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + '.sha256').write_text(f'{digest}  {out.name}\n')
    print(f'{n} files, {total / 1e6:.1f} MB -> {out.stat().st_size / 1e6:.1f} MB, sha256 {digest}')
    return digest
