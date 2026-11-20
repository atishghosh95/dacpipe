import numpy as np
import pytest

from dacpipe import cn, demo, features, jobs, microkinetics as mk, ml, thermo


@pytest.fixture(scope='module')
def table():
    return demo.fake_screening(seed=1)


def test_pair_count_and_features():
    pairs = features.all_pairs()
    assert len(pairs) == 210
    m1, m2 = features.split_pair(pairs)
    F = features.pair_features(m1, m2)
    assert F.shape == (210, 24)
    assert (F.nvald >= 0).all()


def test_volcano_recovers_scaling(table):
    f = thermo.volcano_branches(table)
    assert 0.3 < f['slope_hyd'] < 0.7
    assert -1.1 < f['slope_ins'] < -0.7
    g, u = thermo.apex(f)
    assert abs(thermo.volcano_curve(f, g) - u) < 1e-9


def test_lomo_runs_and_calibrates(table):
    sub = table.iloc[::3].reset_index(drop=True)
    m1, m2 = features.split_pair(sub.pair)
    X = features.pair_features(m1, m2).to_numpy()
    r = ml.lomo(X, sub.dG_N2.to_numpy(), m1, m2, n_jobs=1)
    assert r['r2'] > 0.3
    assert r['z_std'] > 0
    assert len(r['mu']) == len(sub)


def test_onset_monotone_in_barrier():
    P = dict(dG_N2=-0.5, s1=0.3, s2=0.2, s3=0.1, s4=0.0, s5=-0.2, rel=-0.5)
    easy = mk.onset(P, E_pcet=0.1, E_ins=0.1)
    hard = mk.onset(P, E_pcet=0.5, E_ins=0.5)
    assert easy is not None
    assert hard is None or hard <= easy


def test_blocked_insertion_never_turns_over():
    P = dict(dG_N2=-0.5, s1=0.0, s2=0.0, s3=0.0, s4=0.0, s5=1.2, rel=-0.5)
    assert mk.onset(P) is None


def test_constraint_profile(tmp_path):
    f = tmp_path / 'cc.txt'
    demo.write_constraint_log(f)
    D = cn.read_constraint_log(f)
    p = cn.profile(D)
    assert len(p) > 10
    assert np.isfinite(p.F).all()


def test_job_roundtrip(tmp_path):
    cfg = dict(jobs.DEFAULTS, scheduler='local', cores=2, vasp_cmd='echo {cores}')
    pos, inc = tmp_path / 'POSCAR', tmp_path / 'INCAR'
    pos.write_text('x\n1.0\n1 0 0\n0 1 0\n0 0 1\nH\n1\nDirect\n0 0 0\n')
    inc.write_text('ENCUT = 400\n')
    tasks = tmp_path / 'tasks.csv'
    tasks.write_text(f'name,poscar,incar,kpoints\na,{pos},{inc},\n')
    made = jobs.make(tasks, cfg, root=tmp_path / 'jobs')
    assert (made[0] / 'job.sh').exists()
    (made[0] / 'OUTCAR').write_text('  free  energy   TOTEN  =       -12.50000000 eV\n'
                                    ' reached required accuracy - stopping\n Voluntary context switches: 1\n')
    rows = jobs.collect(tmp_path / 'jobs', tmp_path / 'c.csv')
    assert rows[0]['energy_eV'] == -12.5 and rows[0]['converged']
    assert len(jobs.pack(tmp_path / 'jobs', tmp_path / 'p.tar.gz')) == 64
