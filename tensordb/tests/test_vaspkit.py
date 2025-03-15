#!coding=utf-8

import nose
from nose.tools import assert_almost_equal, assert_equal, assert_true
from tensordb.vaspkit import VaspJob
from pathlib import Path


def test_vasp_scf_convergence():
    vjob = VaspJob(Path(__file__).parent / 'data')
    assert_true(vjob.check_vasp_job_scf_convergence())


def test_vasp_elapsed_time():
    vjob = VaspJob(Path(__file__).parent / 'data')
    assert_almost_equal(vjob.get_vasp_elapsed_time(), 59.672)


def test_get_band_occ():
    vjob = VaspJob(Path(__file__).parent / 'data')
    ispin, band_occ = vjob.get_band_occupation()
    assert band_occ.shape == (1000, 36)
    assert_almost_equal(band_occ[251, 29], 1.01613)
    assert_equal(ispin, 1)
    assert_true(vjob.check_band_occ())


if __name__ == '__main__':
    nose.runmodule()
