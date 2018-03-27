#!/usr/bin/env python

import numpy as np

from scipy import constants
from functools import reduce

from astropy import time
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5

from PyAstronomy import pyasl

def doppler_correct(ra, dec, mean_time, obs_lat=38.433056, obs_lon=-79.839722):
    """
    computes the projected velocity of the telescope wrt four coordinate systems: 
    geo, helio, bary, lsr.
    
    :param ra: right ascension in degrees, J2000 frame.
    :param dec: declination in degrees, J2000 frame.
    :param mean_time: mean time of observation in isot format. Example "2017-01-15T01:59:58.99"
    :param obs_lon: East-longitude of observatory in degrees.
    :param obs_lat: Latitude of observatory in degrees.
    """

    # Initialize ra and dec of source
    src = SkyCoord(ra, dec, frame='icrs', unit=u.deg)
    mytime = time.Time(mean_time, format='isot', scale='utc')

    # Orbital velocity of Earth with respect to the Sun.
    # helio = for source projected velocity of earth orbit with respect to the Sun center.
    # bary = for source projected velocity of earth + moon orbit with respect to the Sun center.
    v_orbit_helio, v_orbit_bary = pyasl.baryCorr(mytime.jd, src.ra.deg, src.dec.deg, deq=2000.0)
        
    ## Earth rotational velocity
    # TO DO: determine LOFAR latitude and longitude from station positions
    # Taken from chdoppler.pro, "Spherical Astronomy" R. Green p.270 
    lst = mytime.sidereal_time('apparent', obs_lon)
    obs_lat = obs_lat * u.deg
    obs_lon = obs_lon * u.deg
    hour_angle = lst - src.ra
    v_spin = -0.465 * np.cos( obs_lat ) * np.cos( src.dec ) * np.sin( hour_angle )

    # LSR defined as: Sun moves at 20.0 km/s toward RA=18.0h and dec=30.0d in 1900J coords
    # WRT objects near to us in Milky Way (not sun's rotation WRT to galactic center!)
    lsr_coord = SkyCoord( '18h', '30d', frame='fk5', equinox='J1900')
    lsr_coord = lsr_coord.transform_to(FK5(equinox='J2000'))

    lsr_comp = np.array([ np.cos(lsr_coord.dec.rad) * np.cos(lsr_coord.ra.rad), \
                          np.cos(lsr_coord.dec.rad) * np.sin(lsr_coord.ra.rad), \
                          np.sin(lsr_coord.dec.rad) ])

    src_comp = np.array([ np.cos(src.dec.rad) * np.cos(src.ra.rad), \
                          np.cos(src.dec.rad) * np.sin(src.ra.rad), \
                          np.sin(src.dec.rad) ])

    k = np.array( [lsr_comp[0]*src_comp[0], lsr_comp[1]*src_comp[1], lsr_comp[2]*src_comp[2]] )
    v_lsr = 20. * np.sum( k )
    
    geo = - v_spin
    helio = - v_spin - v_orbit_helio
    bary = - v_spin - v_orbit_bary
    lsr = - v_spin - v_orbit_bary - v_lsr
    vtotal = [geo, helio, bary, lsr]
    
    return vtotal

def factors(n):
    """
    Decomposes a number into its factors.
    :param n: Number to decompose.
    :type n: int
    :return: List of values into which n can be decomposed.
    :rtype: list
    """

    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def get_freq_axis(data, chstart=0, chstop=-1, apply_doppler=True):
    """
    """

    # Copied from GBT gridder: 
    # https://github.com/nrao/gbtgridder/blob/master/src/get_data.py
    
    if chstop == -1:
        chstop = data.field('data').shape[1]
        
    freq = np.zeros(data.field('data').shape)

    crv1 = data.field('crval1')
    cd1 = data.field('cdelt1')
    crp1 = data.field('crpix1')
    vframe = data.field('vframe')
    #frest = data.field('restfreq')
    
    # Observatory redshift.
    beta = vframe/constants.c
    
    # Doppler correction.
    doppler = 1.
    if apply_doppler:
        doppler = np.sqrt((1.0 + beta)/(1.0 - beta))
    
    # Full frequency axis in doppler tracked frame from first row.
    # FITS counts from 1, this indx refers to the original axis, before chan selection.
    indx = np.arange(chstop - chstart) + chstart
    
    for ch in indx:
        freq[:,int(ch)] = (crv1 + cd1*(ch - crp1))*doppler

    return freq

def mask_hi(freq, volt, line_freq, line_fwhm):
    """
    """

    #hifrq = 1420.4e6
    #fwhm = 0.71e6

    if np.sort(freq)[0] <= line_freq <= np.sort(freq)[-1]:
        mfreq, mvolt = mask_line(freq, volt, line_freq, line_fwhm)
    else:
        mfreq, mvolt = freq, volt

    return mfreq, mvolt

def mask_line(freq, spec, line, fwhm):
    """
    """

    ch0 = np.argmin(abs(freq - (line - fwhm/2.)))
    chf = np.argmin(abs(freq - (line + fwhm/2.)))

    if ch0 > chf: ch0,chf = chf,ch0

    #for i,s in enumerate(spec):

    spec.mask[:,ch0:chf] = True
    freq.mask[ch0:chf] = True

    return freq, spec
