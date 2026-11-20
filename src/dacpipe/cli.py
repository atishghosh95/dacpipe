import argparse
from pathlib import Path

from . import jobs

CONFIG_TEMPLATE = '''# site settings, edit the values only
scheduler  = "slurm"      # slurm | pbs | lsf | local
account    = ""
partition  = ""
nodes      = 1
cores      = 32
walltime   = "04:00:00"
modules    = []           # e.g. ["intel-mpi", "vasp/6.4"]
vasp_cmd   = "mpirun -np {cores} vasp_std"
potcar_dir = ""           # one sub-folder per element containing POTCAR (licensed, not shipped)
pack_max_mb = 50
'''


def main(argv=None):
    ap = argparse.ArgumentParser(prog='dacpipe')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('init', help='write a config template')
    p.add_argument('path', nargs='?', default='config.toml')
    p = sub.add_parser('make', help='job folders from a task table')
    p.add_argument('tasks')
    p.add_argument('--config', default='config.toml')
    p.add_argument('--root', default='jobs')
    p = sub.add_parser('submit', help='submit every folder that has a job.sh')
    p.add_argument('root')
    p.add_argument('--config', default='config.toml')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('collect', help='energies and convergence flags into one csv')
    p.add_argument('root')
    p.add_argument('out')
    p = sub.add_parser('pack', help='tar.gz of the small files plus a checksum')
    p.add_argument('root')
    p.add_argument('out')
    p.add_argument('--max-mb', type=float, default=50)
    a = ap.parse_args(argv)

    if a.cmd == 'init':
        Path(a.path).write_text(CONFIG_TEMPLATE)
        print('wrote', a.path)
    elif a.cmd == 'make':
        jobs.make(a.tasks, jobs.load_config(a.config), a.root)
    elif a.cmd == 'submit':
        jobs.submit(a.root, jobs.load_config(a.config), a.dry_run)
    elif a.cmd == 'collect':
        jobs.collect(a.root, a.out)
    elif a.cmd == 'pack':
        jobs.pack(a.root, a.out, a.max_mb)


if __name__ == '__main__':
    main()
