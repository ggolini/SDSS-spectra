# Code to download sdss spectra


# Original author:
# Giulia Golini <giulia.golini@gmail.com>
# Contributing author(s)
# Copyright (C) 2020, Giulia Golini.
#
# This Python script is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This Python script is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details. See <http://www.gnu.org/licenses/>.

# System imports

import os
import pdb
import sys
import warnings


from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.io import fits
import matplotlib.pyplot   as plt
import numpy as np
import pandas as pd


# read the list
filename = sys.argv[1]
print(filename)


# Load only the RA and Dec columns
ra_dec_data = pd.read_csv(filename, usecols=['ra', 'dec','sn_median_r'])


for index, row in ra_dec_data.iterrows():
    ra = str(row['ra'])  # Access RA
    dec = str(row['dec'])  # Access Dec
    sn = float(row['sn_median_r'])
    if sn > 3 :         # cut in signal to noise
        outfile = 'spectra/spectra_' + ra + '-' + dec + '.fits'
        pos = coords.SkyCoord(ra + ' ' + dec, unit="deg", frame='icrs')
        print(outfile, pos)
        
        # Do the actual query
        xid = SDSS.query_region(pos, spectro=True)
        sp = SDSS.get_spectra(matches=xid)
        
        # Save file
        sp[0].writeto(outfile, overwrite=True)
