# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.io import fits
from astropy import wcs as fitswcs
from astropy import units as u
from astropy import coordinates as coord
from astropy.modeling import models
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import assert_quantity_allclose
import pytest
from numpy.testing import assert_allclose

from .. import utils as gwutils
from ..utils import UnsupportedProjectionError


def test_wrong_projcode():
    with pytest.raises(UnsupportedProjectionError):
        ctype = {"CTYPE": ["RA---TAM", "DEC--TAM"]}
        gwutils.get_projcode(ctype)


def test_wrong_projcode2():
    with pytest.raises(UnsupportedProjectionError):
        gwutils.create_projection_transform("TAM")


@pytest.mark.remote_data
def test_fits_transform():
    hdr = fits.Header.fromfile(get_pkg_data_filename('data/simple_wcs2.hdr'))
    gw1 = gwutils.make_fitswcs_transform(hdr)
    w1 = fitswcs.WCS(hdr)
    assert_allclose(gw1(1, 2), w1.wcs_pix2world(1, 2, 0), atol=10 ** -8)


def test_lon_pole():
    tan = models.Pix2Sky_TAN()
    car = models.Pix2Sky_CAR()
    sky_positive_lat = coord.SkyCoord(3 * u.deg, 1 * u.deg)
    sky_negative_lat = coord.SkyCoord(3 * u.deg, -1 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole(sky_positive_lat, tan), 180 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole(sky_negative_lat, tan), 180 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole(sky_positive_lat, car), 0 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole(sky_negative_lat, car), 180 * u.deg)
    assert_quantity_allclose(gwutils._compute_lon_pole((0, 34 * u.rad), tan), 180 * u.deg)
    assert_allclose(gwutils._compute_lon_pole((1, -34), tan), 180)


def test_unknown_ctype():
    wcsinfo = {'CDELT': np.array([3.61111098e-05, 3.61111098e-05, 2.49999994e-03]),
               'CRPIX': np.array([17., 16., 1.]),
               'CRVAL': np.array([4.49999564e+01, 1.72786731e-04, 4.84631542e+00]),
               'CTYPE': np.array(['MRSAL1A', 'MRSBE1A', 'WAVE']),
               'CUNIT': np.array([u.Unit("deg"), u.Unit("deg"), u.Unit("um")], dtype=object),
               'PC': np.array([[1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]]),
               'WCSAXES': 3,
               'NAXIS': 3
               }
    wcsinfo['CD'] = (wcsinfo['CDELT'] * wcsinfo['PC'].T).T

    transform = gwutils.make_fitswcs_transform(wcsinfo)
    x = np.linspace(-5, 7, 10)
    y = np.linspace(-5, 7, 10)
    z = np.ones_like(x)
    expected = (np.array([-0.00075833, -0.00071019, -0.00066204, -0.00061389, -0.00056574,
                          -0.00051759, -0.00046944, -0.0004213 , -0.00037315, -0.000325]),
                np.array([-0.00072222, -0.00067407, -0.00062593, -0.00057778, -0.00052963,
                           -0.00048148, -0.00043333, -0.00038519, -0.00033704, -0.00028889])
                )
    a, b, c = transform(x, y, z)
    assert_allclose(a, expected[0], atol=10**-8)
    assert_allclose(b, expected[1], atol=10**-8)


def test_longslit_wcs():
    # WCS for a longslit with spectra dispersed horizontally and the
    # slit along the y-axis at PA=90
    hdr = fits.Header(dict([
        ('CRPIX1', 201.),
        ('CRPIX2', 1001.),
        ('CTYPE1', 'WAVE'),
        ('CRVAL1', 575.),
        ('CUNIT1', 'nm'),
        ('CTYPE2', 'RA---TAN'),
        ('CRVAL2', 270.),
        ('CTYPE3', 'DEC--TAN'),
        ('CRVAL3', 20.),
        ('CD1_1', -0.01),
        ('CD2_2', -0.0001),
        ('CD3_2', 0),
        ('RADECSYS', 'FK5'),
    ]))

    transform = gwutils.make_fitswcs_transform(hdr)
    x = np.linspace(0, 400, 5)
    y = np.linspace(600, 1400, 5)

    expected = (
        np.array([577, 576, 575, 574, 573]),
        np.array([270.04256711, 270.02128356, 270., 269.97871644, 269.95743289]),
        np.array([19.99999492, 19.99999873, 20., 19.99999873, 19.99999492])
    )

    # Have to do this iteratively until the mdoels.fix_inputs() bug is fixed
    for i in range(len(x)):
        a, b, c = transform(x[i], y[i])
        assert_allclose(a, expected[0][i], atol=10**-8)
        assert_allclose(b, expected[1][i], atol=10**-8)
        assert_allclose(c, expected[2][i], atol=10**-8)

        x2, y2 = transform.inverse(a, b, c)
        assert_allclose(x2, x[i])
        assert_allclose(y2, y[i])


def test_get_axes():
    wcsinfo = {'CTYPE': np.array(['MRSAL1A', 'MRSBE1A', 'WAVE'])}
    cel, spec, other = gwutils.get_axes(wcsinfo)
    assert not cel
    assert spec == [2]
    assert other == [0, 1]
    wcsinfo = {'CTYPE': np.array(['RA---TAN', 'WAVE', 'DEC--TAN'])}
    cel, spec, other = gwutils.get_axes(wcsinfo)
    assert cel == [0, 2]
    assert spec == [1]
    assert not other


def test_isnumerical():
    sky = coord.SkyCoord(1 * u.deg, 2 * u.deg)
    assert not gwutils.isnumerical(sky)

    assert not gwutils.isnumerical(2 * u.m)

    assert gwutils.isnumerical(np.float(0))
    assert gwutils.isnumerical(np.array(0))

    assert not gwutils.isnumerical(np.array(['s200', '234']))

    assert gwutils.isnumerical(np.array(0, dtype='>f8'))
    assert gwutils.isnumerical(np.array(0, dtype='>i4'))


def test_get_values():
    args = 2 * u.cm
    units=(u.m, )
    res = gwutils.get_values(units, args)
    assert res == [.02]

    res = gwutils.get_values(None, args)
    assert res == [2]
