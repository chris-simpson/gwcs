# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utility function for WCS

"""
import re
import functools
import numpy as np
from astropy.modeling import models as astmodels
from astropy.modeling import core, projections
from astropy.io import fits
from astropy import coordinates as coords
from astropy import units as u


# these ctype values do not include yzLN and yzLT pairs
sky_pairs = {"equatorial": ["RA", "DEC"],
             "ecliptic": ["ELON", "ELAT"],
             "galactic": ["GLON", "GLAT"],
             "helioecliptic": ["HLON", "HLAT"],
             "supergalactic": ["SLON", "SLAT"],
             # "spec": specsystems
            }

radesys = ['ICRS', 'FK5', 'FK4', 'FK4-NO-E', 'GAPPT', 'GALACTIC']


class UnsupportedTransformError(Exception):

    def __init__(self, message):
        super(UnsupportedTransformError, self).__init__(message)


class UnsupportedProjectionError(Exception):
    def __init__(self, code):
        message = "Unsupported projection: {0}".format(code)
        super(UnsupportedProjectionError, self).__init__(message)


class RegionError(Exception):

    def __init__(self, message):
        super(RegionError, self).__init__(message)


class CoordinateFrameError(Exception):

    def __init__(self, message):
        super(CoordinateFrameError, self).__init__(message)


def _toindex(value):
    """
    Convert value to an int or an int array.

    Input coordinates converted to integers
    corresponding to the center of the pixel.
    The convention is that the center of the pixel is
    (0, 0), while the lower left corner is (-0.5, -0.5).
    The outputs are used to index the mask.

    Examples
    --------
    >>> _toindex(np.array([-0.5, 0.49999]))
    array([0, 0])
    >>> _toindex(np.array([0.5, 1.49999]))
    array([1, 1])
    >>> _toindex(np.array([1.5, 2.49999]))
    array([2, 2])
    """
    indx = np.asarray(np.floor(np.asarray(value) + 0.5), dtype=np.int)
    return indx


def get_values(units, *args):
    """
    Return the values of Quantity objects after optionally converting to units.

    Parameters
    ----------
    units : str or `~astropy.units.Unit` or None
        Units to convert to. The input values are converted to ``units``
        before the values are returned.
    args : `~astropy.units.Quantity`
        Quantity inputs.
    """
    if units is not None:
        result = [a.to_value(unit) for a, unit in zip(args, units)]
    else:
        result = [a.value for a in args]
    return result

def _get_contributing_axes(wcs_info, world_axes):
    """
    Returns a tuple indicating which axes in the pixel frame make a
    contribution to an axis or axes in the output frame.

    Parameters
    ----------
    wcs_info: dict
        dict of WCS information
    world_axes: int/iterable of ints
        axes in the world coordinate system

    Returns
    -------
    axes: list
        axes whose pixel coordinates affect the output axis/axes
    """
    cd = wcs_info['CD']
    try:
        return sorted(set(np.nonzero(cd[tuple(world_axes), :wcs_info['NAXIS']])[1]))
    except TypeError:  # world_axes is an int
        return sorted(np.nonzero(cd[world_axes, :wcs_info['NAXIS']])[0])
    #return sorted(set(j for j in range(wcs_info['NAXIS'])
    #                    for i in world_axes if cd[i, j] != 0))

def _compute_lon_pole(skycoord, projection):
    """
    Compute the longitude of the celestial pole of a standard frame in the
    native frame.

    This angle then can be used as one of the Euler angles (the other two being skyccord)
    to rotate the native frame into the standard frame ``skycoord.frame``.

    Parameters
    ----------
    skycoord : `astropy.coordinates.SkyCoord`, or
               sequence of floats or `~astropy.units.Quantity` of length 2
        The fiducial point of the native coordinate system.
        If tuple, its length is 2
    projection : `astropy.modeling.projections.Projection`
        A Projection instance.

    Returns
    -------
    lon_pole : float or `~astropy/units.Quantity`
        Native longitude of the celestial pole [deg].

    TODO: Implement all projections
        Currently this only supports Zenithal and Cylindrical.
    """
    if isinstance(skycoord, coords.SkyCoord):
        lat = skycoord.spherical.lat
        unit = u.deg
    else:
        lon, lat = skycoord
        if isinstance(lat, u.Quantity):
            unit = u.deg
        else:
            unit = None
    if isinstance(projection, projections.Zenithal):
        lon_pole = 180
    elif isinstance(projection, projections.Cylindrical):
        if lat >= 0:
            lon_pole = 0
        else:
            lon_pole = 180
    else:
        raise UnsupportedProjectionError("Projection {0} is not supported.".format(projection))
    if unit is not None:
        lon_pole = lon_pole * unit
    return lon_pole


def get_projcode(wcs_info):
    # CTYPE here is only the imaging CTYPE keywords
    sky_axes, _, _ = get_axes(wcs_info)
    if not sky_axes:
        return None
    projcode = wcs_info['CTYPE'][sky_axes[0]][5:8].upper()
    if projcode not in projections.projcodes:
        raise UnsupportedProjectionError('Projection code %s, not recognized' % projcode)
    return projcode


def read_wcs_from_header(header):
    """
    Extract basic FITS WCS keywords from a FITS Header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS Header with WCS information.

    Returns
    -------
    wcs_info : dict
        A dictionary with WCS keywords.
    """
    wcs_info = {}

    try:
        wcs_info['WCSAXES'] = header['WCSAXES']
    except KeyError:
        p = re.compile(r'ctype[\d]*', re.IGNORECASE)
        ctypes = header['CTYPE*']
        keys = list(ctypes.keys())
        for key in keys[::-1]:
            if p.split(key)[-1] != "":
                keys.remove(key)
        wcs_info['WCSAXES'] = len(keys)
    wcsaxes = wcs_info['WCSAXES']
    # if not present call get_csystem
    wcs_info['RADESYS'] = header.get('RADESYS', 'ICRS')
    wcs_info['VAFACTOR'] = header.get('VAFACTOR', 1)
    wcs_info['NAXIS'] = header.get('NAXIS', max(int(k[5:]) for k in header['CRPIX*'].keys()))
    # date keyword?
    # wcs_info['DATEOBS'] = header.get('DATE-OBS', 'DATEOBS')
    wcs_info['EQUINOX'] = header.get("EQUINOX", None)
    wcs_info['EPOCH'] = header.get("EPOCH", None)
    wcs_info['DATEOBS'] = header.get("MJD-OBS", header.get("DATE-OBS", None))

    ctype = []
    cunit = []
    crpix = []
    crval = []
    cdelt = []
    for i in range(1, wcsaxes + 1):
        ctype.append(header['CTYPE{0}'.format(i)])
        cunit.append(header.get('CUNIT{0}'.format(i), None))
        crpix.append(header.get('CRPIX{0}'.format(i), 0.0))
        crval.append(header.get('CRVAL{0}'.format(i), 0.0))
        cdelt.append(header.get('CDELT{0}'.format(i), 1.0))

    if 'CD1_1' in header:
        wcs_info['has_cd'] = True
    else:
        wcs_info['has_cd'] = False
    cd = np.zeros((wcsaxes, wcsaxes))
    for i in range(1, wcsaxes + 1):
        for j in range(1, wcsaxes + 1):
            try:
                if wcs_info['has_cd']:
                    cd[i - 1, j - 1] = header['CD{0}_{1}'.format(i, j)]
                else:
                    cd[i - 1, j - 1] = cdelt[i - 1] * header['PC{0}_{1}'.format(i, j)]
            except KeyError:
                if i == j:
                    cd[i - 1, j - 1] = cdelt[i - 1]
                else:
                    cd[i - 1, j - 1] = 0.
    wcs_info['CTYPE'] = ctype
    wcs_info['CUNIT'] = cunit
    wcs_info['CRPIX'] = crpix
    wcs_info['CRVAL'] = crval
    wcs_info['CD'] = cd
    return wcs_info


def get_axes(header):
    """
    Matches input with spectral and sky coordinate axes.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header (or dict) with basic WCS information.

    Returns
    -------
    sky_inmap, spectral_inmap, unknown : lists
        indices in the output representing sky and spectral coordinates.

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    # Split each CTYPE value at "-" and take the first part.
    # This should represent the coordinate system.
    ctype = [ax.split('-')[0].upper() for ax in wcs_info['CTYPE']]
    sky_inmap = []
    spec_inmap = []
    unknown = []
    skysystems = np.array(list(sky_pairs.values())).flatten()
    for ax in ctype:
        ind = ctype.index(ax)
        if ax in specsystems:
            spec_inmap.append(ind)
        elif ax in skysystems:
            sky_inmap.append(ind)
        else:
            unknown.append(ind)

    if sky_inmap:
        _is_skysys_consistent(ctype, sky_inmap)

    return sky_inmap, spec_inmap, unknown


def _is_skysys_consistent(ctype, sky_inmap):
    """ Determine if the sky axes in CTYPE match to form a standard celestial system."""
    if len(sky_inmap) != 2:
        raise ValueError("{} sky coordinate axes found. "
                         "There must be exactly 2".format(len(sky_inmap)))

    for item in sky_pairs.values():
        if ctype[sky_inmap[0]] == item[0]:
            if ctype[sky_inmap[1]] != item[1]:
                raise ValueError(
                    "Inconsistent ctype for sky coordinates {0} and {1}".format(*ctype))
            break
        elif ctype[sky_inmap[1]] == item[0]:
            if ctype[sky_inmap[0]] != item[1]:
                raise ValueError(
                    "Inconsistent ctype for sky coordinates {0} and {1}".format(*ctype))
            sky_inmap.reverse()
            break


specsystems = ["WAVE", "FREQ", "ENER", "WAVEN", "AWAV",
               "VRAD", "VOPT", "ZOPT", "BETA", "VELO"]

sky_systems_map = {'ICRS': coords.ICRS,
                   'FK5': coords.FK5,
                   'FK4': coords.FK4,
                   'FK4NOE': coords.FK4NoETerms,
                   'GAL': coords.Galactic,
                   'HOR': coords.AltAz
                  }


def make_fitswcs_transform(header):
    """
    Create a basic FITS WCS transform.
    It does not include distortions.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header (or dict) with basic WCS information

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    # CRPIX shift is always first and always in the pixel frame
    crpix = wcs_info['CRPIX'][:wcs_info['NAXIS']]
    translation_models = [astmodels.Shift(-(shift - 1), name='crpix' + str(i + 1))
                          for i, shift in enumerate(crpix)]
    translation = functools.reduce(lambda x, y: x & y, translation_models)
    transforms = [translation]

    # The tricky stuff!
    sky_model = fitswcs_image(wcs_info)
    linear_models = fitswcs_linear(wcs_info)
    all_models = linear_models
    if sky_model:
        all_models.append(sky_model)

    # Now arrange the models so the inputs and outputs are in the right places
    all_models.sort(key=lambda m: m.meta['output_axes'][0])
    input_axes = [ax for m in all_models for ax in m.meta['input_axes']]
    output_axes = [ax for m in all_models for ax in m.meta['output_axes']]
    if input_axes != list(range(len(input_axes))):
        input_mapping = astmodels.Mapping(input_axes)
        transforms.append(input_mapping)

    transforms.append(functools.reduce(core._model_oper('&'), all_models))

    if output_axes != list(range(len(output_axes))):
        output_mapping = astmodels.Mapping(output_axes)
        transforms.append(output_mapping)

    return functools.reduce(core._model_oper('|'), transforms)

def fitswcs_image(header):
    """
    Make a complete transform from CRPIX-shifted pixels to
    sky coordinates from FITS WCS keywords. A Mapping is inserted
    at the beginning, which may be removed later

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header or dict with basic FITS WCS keywords.

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    cd = wcs_info['CD']
    # get the part of the PC matrix corresponding to the imaging axes
    sky_axes, spec_axes, unknown = get_axes(wcs_info)
    if not sky_axes:
        if len(unknown) == 2:
            sky_axes = unknown
        else:  # No sky here
            return
    pixel_axes = _get_contributing_axes(wcs_info, sky_axes)
    if len(pixel_axes) > 2:
        raise ValueError("More than 2 pixel axes contribute to the sky coordinates")

    # If only one axis is contributing to the sky (e.g., slit spectrum)
    # then it must be that there's an extra axis in the CD matrix, so we
    # create a "ghost" orthogonal axis here so an inverse can be defined
    # Modify the CD matrix in case we have to use a backup Matrix Model later
    if len(pixel_axes) == 1:
        cd[sky_axes[0], -1] = -cd[sky_axes[1], pixel_axes[0]]
        cd[sky_axes[1], -1] = cd[sky_axes[0], pixel_axes[0]]
        sky_cd = cd[np.ix_(sky_axes, pixel_axes + [-1])]
        affine = astmodels.AffineTransformation2D(matrix=sky_cd, name='cd_matrix')
        rotation = astmodels.fix_inputs(affine, {'y': 0})
        rotation.inverse = affine.inverse | astmodels.Mapping((0,), n_inputs=2)
    else:
        sky_cd = cd[np.ix_(sky_axes, pixel_axes)]
        rotation = astmodels.AffineTransformation2D(matrix=sky_cd, name='cd_matrix')

    projection = fitswcs_nonlinear(wcs_info)
    if projection:
        sky_model = rotation | projection
    else:
        sky_model = rotation
    sky_model.meta.update({'input_axes': pixel_axes,
                           'output_axes': sky_axes})
    return sky_model

def fitswcs_linear(header):
    """
    Create WCS linear transforms for any axes not associated with
    celestial coordinates. We require that each world axis aligns
    precisely with only a single pixel axis.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header or dict with basic FITS WCS keywords.

    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    cd = wcs_info['CD']
    # get the part of the CD matrix corresponding to the imaging axes
    sky_axes, spec_axes, unknown = get_axes(wcs_info)
    if not sky_axes and len(unknown) == 2:
        unknown = []

    linear_models = []
    for ax in spec_axes + unknown:
        pixel_axes = _get_contributing_axes(wcs_info, ax)
        if len(pixel_axes) == 1:
            linear_model = (astmodels.Scale(cd[ax, pixel_axes[0]]) |
                            astmodels.Shift(wcs_info['CRVAL'][ax]))
            linear_model.meta.update({'input_axes': pixel_axes,
                                      'output_axes': [ax]})
            linear_models.append(linear_model)
        else:
            raise ValueError(f"Axis {ax} depends on more than one input axis")

    return linear_models


def fitswcs_nonlinear(header):
    """
    Create a WCS linear transform from a FITS header.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict
        FITS Header or dict with basic FITS WCS keywords.
    """
    if isinstance(header, fits.Header):
        wcs_info = read_wcs_from_header(header)
    elif isinstance(header, dict):
        wcs_info = header
    else:
        raise TypeError("Expected a FITS Header or a dict.")

    transforms = []
    projcode = get_projcode(wcs_info)
    if projcode is not None:
        projection = create_projection_transform(projcode).rename(projcode)
        transforms.append(projection)
    # Create the sky rotation transform
    sky_axes, _, _ = get_axes(wcs_info)
    if sky_axes:
        phip, lonp = [wcs_info['CRVAL'][i] for i in sky_axes]
        # TODO: write "def compute_lonpole(projcode, l)"
        # Set a default value for now
        thetap = 180
        n2c = astmodels.RotateNative2Celestial(phip, lonp, thetap, name="crval")
        transforms.append(n2c)
    if transforms:
        return functools.reduce(core._model_oper('|'), transforms)
    return None


def create_projection_transform(projcode):
    """
    Create the non-linear projection transform.

    Parameters
    ----------
    projcode : str
        FITS WCS projection code.

    Returns
    -------
    transform : astropy.modeling.Model
        Projection transform.
    """

    projklassname = 'Pix2Sky_' + projcode
    try:
        projklass = getattr(projections, projklassname)
    except AttributeError:
        raise UnsupportedProjectionError(projcode)

    projparams = {}
    return projklass(**projparams)


def isnumerical(val):
    """
    Determine if a value is numerical (number or np.array of numbers).
    """
    isnum = True
    if isinstance(val, coords.SkyCoord):
        isnum = False
    elif isinstance(val, u.Quantity):
        isnum = False
    elif (isinstance(val, np.ndarray)
          and not np.issubdtype(val.dtype, np.floating)
          and not np.issubdtype(val.dtype, np.integer)):
        isnum = False
    return isnum
