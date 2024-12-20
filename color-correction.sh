# bin/bash
#
# Code to download stellar spectra from SDSS and analyze
# color dependence of stars for calibration.
#
# Usage:
# Optional: first go to Skyviever, then you find the correct plate
# around the galaxy (this is useful if you only want to download
# stellar spectra on the field of the galaxy you are studying).
#
# Save a list with ra, dec and sn of all the stars
# at
# https://dr16.sdss.org/optical/spectrum/search
# Download the cvs file on the directory you are working on.
#
#
# This code is also based on M.Montes codes to get the
# convolution to get magnitudes and fluxes.

# To run the code you need the filter response saved in the folder
# filters and the ancillary files plot_spectra.py and ret_spectra.py
#
#
#
# Copyright (C) 2021 Golini Giulia <giulia.golini@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.











# Stop the execution if something crashes
set -e





# Save the current system language, and then change it to English to avoid
# problems in some parts of the code (AWK with `,' instead of `.' for
# decimal separator).
export LANG=C







# file containting ra, dec etc of the stars you want to download the spectra
file="optical_search_617515.csv"

# Output Directories
# General directory for that galaxy
bdir=build
if ! [ -d $bdir ]; then mkdir $bdir; fi

plots=$bdir/plots
if ! [ -d $plots ]; then mkdir $plots; fi





###########################################################################

#                           DOwnload spectra                            #

###########################################################################
# for each ra and dec in the list I run the python code to download the table
# directory to save spectras
spectra=spectra
if ! [ -d $spectra ]; then mkdir $spectra; fi
ssaved=$spectra/done_sed.txt
if [ -f $ssaved ]; then
    echo " "
    echo "Spectra already saved"
    echo " "
else
    # code to download spectra in the list
    python3.11 ret_spectra.py $file
    echo done > $ssaved
fi



###########################################################################

#                            Save    SED                                 #

###########################################################################
# directory to save SEDs
seds=seds
if ! [ -d $seds ]; then mkdir $seds; fi
# save table with only the flux and the wavelength from the fits file
sedsaved=$seds/done_sed.txt
if [ -f $sedsaved ]; then
    echo " "
    echo "SED already saved"
    echo " "
else
    for i in $(ls $spectra/*.fits); do
        base=$(basename $i)
        asttable $i -h1 -c1,2 -o $seds/$base.txt
    done
    
    echo done > $sedsaved
fi



###########################################################################

#                                   Plots                                 #

###########################################################################

python3.11 plot_spectra.py
