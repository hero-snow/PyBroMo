import tables

import pybromo as pbm


def test_NumericPSF_hash_roundtrip() -> None:
    psf = pbm.NumericPSF()
    with tables.open_file("psfntest.h5", mode="w") as h5:
        psf.to_hdf5(h5)
    with tables.open_file("psfntest.h5", mode="r") as h5:
        a = h5.get_node("/xz_realistic_z50_150_160_580nm_n1335_HR2")
        assert pbm.NumericPSF(psf_pytables=a).hash() == psf.hash()


def test_GaussianPSF_hash_roundtrip() -> None:
    psf = pbm.GaussianPSF()
    with tables.open_file("psfgtest.h5", mode="w") as h5:
        psf.to_hdf5(h5)
    with tables.open_file("psfgtest.h5", mode="r") as h5:
        a = h5.get_node("/gauss_psf_params")
        assert pbm.GaussianPSF(psf_pytables=a).hash() == psf.hash()
