#!/usr/bin/env python

import numpy as np

from scipy import constants
from functools import reduce

from astropy import time
from astropy import units as u
from astropy.coordinates import SkyCoord, FK5


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


def mask_line(freq, spec, line_freq, fwhm):
    """
    Masks a line in a spectrum given its frequency and FWHM.
    
    :param freq: Frequency axis of the spectrum.
    :param spec: Amplitude axis of the spectrum.
    :param line_freq: Frequency at which the line is centered.
    :param fwhm: FWHM of the line.
    """

    ch0 = np.argmin(abs(freq - (line_freq - fwhm/2.)))
    chf = np.argmin(abs(freq - (line_freq + fwhm/2.)))

    if ch0 > chf: ch0,chf = chf,ch0

    spec.mask[:,ch0:chf] = True
    freq.mask[ch0:chf] = True

    return freq, spec


