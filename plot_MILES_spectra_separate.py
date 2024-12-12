import matplotlib.cm as cm
from scipy.interpolate import interp1d
from astropy.table import Table
import os, glob, sys, warnings, array, re, math, time, copy
import numpy               as np
import matplotlib.pyplot   as plt
from   astropy.io          import fits, ascii
import gc


def read_vazdekis(path, imf = 1.3, redshift = 0):
       nmet          = np.array([-2.32, -1.71, -1.31, -0.71, -0.40, 0.00, +0.22])
       ages          = np.array([0.063, 0.071, 0.079, 0.089, 0.10, 0.11, 0.13, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 0.35, 0.40, 0.45, 0.50, 0.56, 0.63, 0.71, 0.79, 0.89, 1.00, 1.12, 1.26, 1.41, 1.58, 1.78, 2.00, 2.24, 2.51, 2.82, 3.16, 3.55, 3.98, 4.47, 5.01, 5.62, 6.31, 7.08, 7.94, 8.91, 10.00, 11.22, 12.59, 14.13, 15.85, 17.78])
       nage          = len(ages)
       files         = glob.glob(path + '*.fits')
       print("...Reading " + files[0] + " files...")
       im, hdr       = fits.getdata(files[0], header = True)
       spectra_miles = np.zeros([len(nmet), nage, len(im)])
       npix          = len(im)
       crpix         = hdr['CRPIX1']
       crval         = hdr['CRVAL1']
       cdelt         = hdr['CDELT1']
       wave          = ((np.arange(npix) + 1.0) - crpix) * cdelt + crval
       constant      = 1.

       if redshift > 0.0:
          wave *= (1.0 + redshift)
          constant = (1.0 + redshift)
       for tt in np.arange(len(files[0:50])):
           flux = fits.getdata(files[tt])
           spectra_miles[0,tt,:] = flux/constant
       for tt in np.arange(len(files[50:100])):
           flux = fits.getdata(files[tt+50])
           spectra_miles[1,tt,:] = flux/constant
       for tt in np.arange(len(files[100:150])):
           flux = fits.getdata(files[tt+100])
           spectra_miles[2,tt,:] = flux/constant
       for tt in np.arange(len(files[150:200])):
           flux = fits.getdata(files[tt+150])
           spectra_miles[3,tt,:] = flux/constant
       for tt in np.arange(len(files[200:250])):
           flux = fits.getdata(files[tt+200])
           spectra_miles[4,tt,:] = flux/constant
       for tt in np.arange(len(files[250:300])):
           flux = fits.getdata(files[tt+250])
           spectra_miles[5,tt,:] = flux/constant
       for tt in np.arange(len(files[300:350])):
           flux = fits.getdata(files[tt+300])
           spectra_miles[6,tt,:] = flux/constant

       return (wave, spectra_miles, nmet, ages)
       

# Constants
C_LIGHT = 2.99792458e18  # Speed of light in Angstrom/s
DL_10PC = 1E-5  # 10 pc in Mpc, z=0
CFACT = 5.0 * np.log10(1.7684E8 * DL_10PC) - 48.6 + 2.5 * np.log10(C_LIGHT)  # Magnitude factor



def read_filter(filter_file):
    """Reads filter data from a file."""
    data = ascii.read(filter_file)
    wavefilt = data['col1']
    fluxfilt = data['col2']
    fluxfilt = fluxfilt / np.amax(fluxfilt)
    return wavefilt, fluxfilt


# Function to compute magnitudes
def compute_magnitudes(wave_mod, flux_mod, filter_files):
    n_filters = len(filter_files)
    magnitudes = np.zeros(n_filters)
    effective_wavelengths = np.zeros(n_filters)
    vega_flux = np.zeros(n_filters)

    for i, filter_file in enumerate(filter_files):
        wave_filter, flux_filter = read_filter(filter_file)
        effective_wavelengths[i] = np.sqrt(
            np.trapz(flux_filter, wave_filter) / np.trapz(flux_filter / wave_filter**2, wave_filter)
        )
        w_filter_range = (wave_filter > 0)
        w_low, w_high = np.amin(wave_filter[w_filter_range]), np.amax(wave_filter[w_filter_range])

        # Select relevant spectrum range
        spectrum_mask = (wave_mod >= w_low) & (wave_mod <= w_high)
        if not np.any(spectrum_mask):
            continue

        wave_mod_filtered = wave_mod[spectrum_mask]
        flux_mod_filtered = flux_mod[spectrum_mask]
        response = interp1d(wave_filter[w_filter_range], flux_filter[w_filter_range], bounds_error=False, fill_value=0)(
            wave_mod_filtered)

        # Magnitude calculations
        flux = np.trapz(flux_mod_filtered * response, wave_mod_filtered)
        vega_flux[i] = np.trapz(1.0 / wave_mod_filtered**2 * response, wave_mod_filtered)
        magnitudes[i] = -2.5 * np.log10(flux / vega_flux[i]) + CFACT

    return effective_wavelengths, magnitudes, vega_flux










# --- MAIN SCRIPT ---
# read filters

path_filters = 'filters/'

# read filters
# Load the data from the first file
file1 = path_filters + "LBT_LBCB.sdss-g_1.dat"  # Replace with your actual filename
file2 = path_filters + "CTIO_DECam.g_filter.dat"  # Replace with your actual filename
file3 = path_filters + "interpolated_data_TWFC_g.dat"

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)


# Extract x and y from both files
x1g, y1g = data1[:, 0], data1[:, 1]
x2g, y2g = data2[:, 0], data2[:, 1]
x3g, y3g = data3[:, 0], data3[:, 1]

##################
#################
###################



# Load the data from the first file
file1 = path_filters + "LBT_LBCR.sdss-r.dat"  # Replace with your actual filename
file2 = path_filters + "CTIO_DECam.r_filter.dat"  # Replace with your actual filename
file3 = path_filters + "interpolated_data_TWFC_r.dat"

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)


# Extract x and y from both files
x1r, y1r = data1[:, 0], data1[:, 1]
x2r, y2r = data2[:, 0], data2[:, 1]
x3r, y3r = data3[:, 0], data3[:, 1]


# Filters
filters = [
    f"{path_filters}/interpolated_data_TWFC_r.dat",
    f"{path_filters}/interpolated_data_TWFC_g.dat",
    f"{path_filters}/CTIO_DECam.r_filter.dat",
    f"{path_filters}/CTIO_DECam.g_filter.dat",
    f"{path_filters}/LBT_LBCR.sdss-r.dat",
    f"{path_filters}/LBT_LBCB.sdss-g_1.dat"
]







folder_path = "EMILES_PADOVA00_BASE_KU_FITS/"
wave, spectra_miles, nmet, ages = read_vazdekis(folder_path)



# Plot with a unique color
print(len(wave), len(spectra_miles), len(nmet), len(ages))
magnitudes_data = []
vega_flux_data = []
colors_g_r = []




# Generate a colormap for different colors
colors = plt.cm.viridis(np.linspace(0, 1, len(os.listdir(folder_path))))


for j in range(0, len(ages), 10):  # Loop over ages
    # Initialize a figure for the plot
    plt.figure(figsize=(10, 6))
    
    for i in range(len(nmet)):  # Loop over metallicities
        # Plot the spectra
        plt.plot(
            wave,
            np.log10(spectra_miles[i, j, :]),
            alpha=0.7,
            label=f"Metallicity = {nmet[i]}, Age = {ages[j]}"
        )
        
        # Compute magnitudes and save results
        wave_eff, mags, veg_flux = compute_magnitudes(wave, spectra_miles[i, j, :], filters)
        magnitudes_data.append(mags)
        vega_flux_data.append(veg_flux)
        colors_g_r.append(mags[3] - mags[2])  # g-r color DECALS
    
    # Customize the plot
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Flux [arbitrary units]")
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Ensure unique filename for saving
    filename = f"spectra_age_{ages[j]:.2f}.png"  # Use age as part of the filename
    plt.savefig("build/plots/" + filename, dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory
    print(f"Saved: {filename}")


for j in range(len(nmet)):  # Loop over ages
    # Initialize a figure for the plot
    plt.figure(figsize=(10, 6))
    
    for i in range(0, len(ages), 10):  # Loop over metallicities
        # Plot the spectra
        plt.plot(
            wave,
            np.log10(spectra_miles[j, i, :]),
            alpha=0.7,
            label=f"Metallicity = {nmet[j]}, Age = {ages[i]}"
        )
        
        # Compute magnitudes and save results
        wave_eff, mags, veg_flux = compute_magnitudes(wave, spectra_miles[j, i, :], filters)
        magnitudes_data.append(mags)
        vega_flux_data.append(veg_flux)
        colors_g_r.append(mags[3] - mags[2])  # g-r color DECALS
    
    # Customize the plot
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Flux [arbitrary units]")
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Ensure unique filename for saving
    filename = f"spectra_nmet_{nmet[j]:.2f}.png"  # Use age as part of the filename
    plt.savefig("build/plots/" + filename, dpi=300)  # Save the figure
    plt.close()  # Close the figure to free memory
    print(f"Saved: {filename}")


# Prepare results for saving
magnitudes_data = np.array(magnitudes_data)
vega_flux_data = np.array(vega_flux_data)
colors_g_r = np.array(colors_g_r)

table_to_save = Table(
    [ magnitudes_data[:, 0], magnitudes_data[:, 1], magnitudes_data[:, 2],
     magnitudes_data[:, 3], magnitudes_data[:, 4], magnitudes_data[:, 5], colors_g_r],
    names=[ 'mag_TWFC_r', 'mag_TWFC_g', 'mag_DEcam_r', 'mag_DEcam_g',
           'mag_LBT_r', 'mag_LBT_g', 'g-r Decals']
)
table_to_save.write(f'build/table_mag_reference.txt', format='ascii.commented_header', overwrite=True)








for j in range(len(nmet)):  # Loop over ages
    # Initialize a figure for the plot
    plt.figure(figsize=(10, 6))
    magnitudes_data = []
    vega_flux_data = []
    colors_g_r = []
    for i in range(len(ages)):  # Loop over metallicities
        # Plot the spectra
        plt.plot(
            wave,
            np.log10(spectra_miles[j, i, :]),
            alpha=0.7,
            label=f"Metallicity = {nmet[j]}, Age = {ages[i]}"
        )
        
        # Compute magnitudes and save results
        wave_eff, mags, veg_flux = compute_magnitudes(wave, spectra_miles[j, i, :], filters)
        magnitudes_data.append(mags)
        vega_flux_data.append(veg_flux)
        colors_g_r.append(mags[3] - mags[2])  # g-r color DECALS
    
    
    magnitudes_data = np.array(magnitudes_data)
    vega_flux_data = np.array(vega_flux_data)
    colors_g_r = np.array(colors_g_r)

    # Plot Results
    twfc_g_flux = 10**(-0.4 * magnitudes_data[:, 1])
    decals_g_flux = 10**(-0.4 * magnitudes_data[:, 3])
    flux_ratio_g = twfc_g_flux / decals_g_flux

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = cm.tab20(np.linspace(0, 1, len(colors_g_r)))

    # Scatter plot of fluxes
    axes[0].scatter(decals_g_flux, twfc_g_flux, c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'$F_g$ DECaLS', fontsize=16)
    axes[0].set_ylabel(r'$F_g$ TWFC', fontsize=16)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)

    # Flux ratio vs color
    axes[1].scatter(colors_g_r, flux_ratio_g, c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
    axes[1].set_xlabel(r'$g-r$ (DECaLS)', fontsize=16)
    axes[1].set_ylabel(r'$F_g$ TWFC / $F_g$ DECaLS', fontsize=16)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.6)

    axes[1].tick_params(axis='both', labelsize=15)
    axes[0].tick_params(axis='both', labelsize=15)
    axes[1].set_ylim(0.75, 1.1)
    axes[1].set_xlim(-1, 2.5)

    # Final touches
    plt.tight_layout()
    filename = f"flux_comparison_TWFC_DECALS_g.png_{nmet[j]:.2f}.png"  # Use age as part of the filename
    plt.savefig("build/plots/" + filename, dpi=300)
    plt.close()
    
    
    
    
    # Plot Results
    twfc_g_flux = 10**(-0.4 * magnitudes_data[:, 0])
    decals_g_flux = 10**(-0.4 * magnitudes_data[:, 2])
    flux_ratio_g = twfc_g_flux / decals_g_flux

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = cm.tab20(np.linspace(0, 1, len(colors_g_r)))

    # Scatter plot of fluxes
    axes[0].scatter(decals_g_flux, twfc_g_flux, c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'$F_r$ DECaLS', fontsize=16)
    axes[0].set_ylabel(r'$F_r$ TWFC', fontsize=16)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)

    # Flux ratio vs color
    axes[1].scatter(colors_g_r, flux_ratio_g, c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
    axes[1].set_xlabel(r'$g-r$ (DECaLS)', fontsize=16)
    axes[1].set_ylabel(r'$F_r$ TWFC / $F_r$ DECaLS', fontsize=16)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.6)


    axes[1].tick_params(axis='both', labelsize=15)
    axes[1].set_ylim(0.75, 1.1)
    axes[1].set_xlim(-1, 2.5)


    axes[0].tick_params(axis='both', labelsize=15)
    # Final touches
    plt.tight_layout()

    filename = f"flux_comparison_TWFC_DECALS_r.png_{nmet[j]:.2f}.png"  # Use age as part of the filename
    plt.savefig("build/plots/" + filename, dpi=300)
    plt.close()










    ######### IN THE G BAND LBT


    # Plot Results
    twfc_g_flux = 10**(-0.4 * magnitudes_data[:, 5])
    decals_g_flux = 10**(-0.4 * magnitudes_data[:, 3])
    flux_ratio_g = twfc_g_flux / decals_g_flux

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = cm.tab20(np.linspace(0, 1, len(colors_g_r)))

    # Scatter plot of fluxes
    axes[0].scatter(decals_g_flux, twfc_g_flux, c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'$F_g$ DECaLS', fontsize=16)
    axes[0].set_ylabel(r'$F_g$ LBT', fontsize=16)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)

    # Flux ratio vs color
    axes[1].scatter(colors_g_r, flux_ratio_g, c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
    axes[1].set_xlabel(r'$g-r$ (DECaLS)', fontsize=16)
    axes[1].set_ylabel(r'$F_g$ LBT / $F_g$ DECaLS', fontsize=16)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.6)


    axes[1].tick_params(axis='both', labelsize=15)
    axes[0].tick_params(axis='both', labelsize=15)

    # Final touches
    plt.tight_layout()
    filename = f"flux_comparison_LBT_DECALS_g.png_{nmet[j]:.2f}.png"  # Use age as part of the filename
    plt.savefig("build/plots/" + filename, dpi=300)
    
    plt.close()




    ######### IN THE R BAND


    # Plot Results
    twfc_g_flux = 10**(-0.4 * magnitudes_data[:, 4])
    decals_g_flux = 10**(-0.4 * magnitudes_data[:, 2])
    flux_ratio_g = twfc_g_flux / decals_g_flux

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = cm.tab20(np.linspace(0, 1, len(colors_g_r)))

    # Scatter plot of fluxes
    axes[0].scatter(decals_g_flux, twfc_g_flux, c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'$F_r$ DECaLS', fontsize=16)
    axes[0].set_ylabel(r'$F_r$ LBT', fontsize=16)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)

    # Flux ratio vs color
    axes[1].scatter(colors_g_r, flux_ratio_g, c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
    axes[1].set_xlabel(r'$g-r$ (DECaLS)', fontsize=16)
    axes[1].set_ylabel(r'$F_r$ LBT / $F_r$ DECaLS', fontsize=16)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.6)


    axes[1].tick_params(axis='both', labelsize=15)


    axes[0].tick_params(axis='both', labelsize=15)
    # Final touches
    plt.tight_layout()
    filename = f"flux_comparison_LBT_DECALS_R.png_{nmet[j]:.2f}.png"  # Use age as part of the filename
    plt.savefig("build/plots/" + filename, dpi=300)

    plt.close()







######### IN THE colors


# Plot Results

colors_g_r_TWFC = magnitudes_data[:, 1] - magnitudes_data[:, 0]
colors_g_r_LBT = magnitudes_data[:, 5] - magnitudes_data[:, 4]
# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
colors = cm.tab20(np.linspace(0, 1, len(colors_g_r)))


# Flux ratio vs color
axes[0].scatter(colors_g_r, colors_g_r_LBT , c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
axes[0].set_xlabel(r'$g-r$ (DECaLS)', fontsize=16)
axes[0].set_ylabel(r'$g-r$ (LBT)', fontsize=16)
axes[0].grid(True, which="both", linestyle="--", alpha=0.6)

x = np.arange(-0.4, 1, 0.1)
print(x)
axes[0].plot(x, x)



# Flux ratio vs color
axes[1].scatter(colors_g_r, colors_g_r_TWFC , c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
axes[1].set_xlabel(r'$g-r$ (DECaLS)', fontsize=16)
axes[1].set_ylabel(r'$g-r$ (TWFC)', fontsize=16)
axes[1].grid(True, which="both", linestyle="--", alpha=0.6)
axes[1].plot(x, x)

axes[1].tick_params(axis='both', labelsize=15)
axes[0].tick_params(axis='both', labelsize=15)

# Final touches
plt.tight_layout()

plt.savefig(f'build/plots/colors.png', dpi=300)
plt.close()
    
    
    
    
    

exit()





