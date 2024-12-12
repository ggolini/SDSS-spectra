import matplotlib.cm as cm
from scipy.interpolate import interp1d
from astropy.table import Table
import os, glob, sys, warnings, array, re, math, time, copy
import numpy               as np
import matplotlib.pyplot   as plt
from   astropy.io          import fits, ascii
import gc
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress




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
file4 = path_filters + "SLOAN_SDSS.gprime_filter.dat"

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)
data4 = np.genfromtxt(file4)

# Extract x and y from both files
x1g, y1g = data1[:, 0], data1[:, 1]
x2g, y2g = data2[:, 0], data2[:, 1]
x3g, y3g = data3[:, 0], data3[:, 1]
x4g, y4g = data4[:, 0], data4[:, 1]
##################
#################
###################



# Load the data from the first file
file1 = path_filters + "LBT_LBCR.sdss-r.dat"  # Replace with your actual filename
file2 = path_filters + "CTIO_DECam.r_filter.dat"  # Replace with your actual filename
file3 = path_filters + "interpolated_data_TWFC_r.dat"
file4 = path_filters + "SLOAN_SDSS.rprime_filter.dat"

data1 = np.genfromtxt(file1)
data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)
data4 = np.genfromtxt(file4)

# Extract x and y from both files
x1r, y1r = data1[:, 0], data1[:, 1]
x2r, y2r = data2[:, 0], data2[:, 1]
x3r, y3r = data3[:, 0], data3[:, 1]
x4r, y4r = data4[:, 0], data4[:, 1]


# Filters
filters = [
    f"{path_filters}/interpolated_data_TWFC_r.dat",
    f"{path_filters}/interpolated_data_TWFC_g.dat",
    f"{path_filters}/CTIO_DECam.r_filter.dat",
    f"{path_filters}/CTIO_DECam.g_filter.dat",
    f"{path_filters}/LBT_LBCR.sdss-r.dat",
    f"{path_filters}/LBT_LBCB.sdss-g_1.dat",
    f"{path_filters}/SLOAN_SDSS.rprime_filter.dat",
    f"{path_filters}/SLOAN_SDSS.gprime_filter.dat",
    f"{path_filters}/interpolated_data_TWFC_i.dat",
    f"{path_filters}/CTIO_DECam.i_filter.dat",
    f"{path_filters}/SLOAN_SDSS.iprime_filter.dat"]





# Initialize a figure for the plot


folder_path = "EMILES_PADOVA00_BASE_KU_FITS/"
wave, spectra_miles, nmet, ages = read_vazdekis(folder_path)

magnitudes_data = []
vega_flux_data = []
colors_g_r = []

# Generate a colormap for different colors
colors = plt.cm.viridis(np.linspace(0, 1, len(os.listdir(folder_path))))

plt.figure(figsize=(10, 6))

for j in np.arange(0, len(ages), 1):
    for i in np.arange(0, len(nmet), 1):
        plt.plot(wave,np.log10(spectra_miles[i, j, :]), alpha=0.7, label = "metallicity = " +  str(nmet[i]) + "age" + str(ages[j]))
        # save things
        wave_eff, mags, veg_flux = compute_magnitudes(wave, spectra_miles[i, j, :], filters)
        magnitudes_data.append(mags)
        vega_flux_data.append(veg_flux)
        colors_g_r.append(mags[3] - mags[2])  # g-r color DECALS
    

# Customize the plot
plt.xlabel(r"$\lambda$")
plt.ylabel("Flux [arbitrary units]")
plt.grid(True)
# Show the plot
plt.savefig("spectras.png")
plt.close()






# Prepare results for saving
magnitudes_data = np.array(magnitudes_data)
vega_flux_data = np.array(vega_flux_data)
colors_g_r = np.array(colors_g_r)

table_to_save = Table(
    [ magnitudes_data[:, 0], magnitudes_data[:, 1], magnitudes_data[:, 2],
     magnitudes_data[:, 3], magnitudes_data[:, 4], magnitudes_data[:, 5], colors_g_r, magnitudes_data[:, 6], magnitudes_data[:, 7],],
    names=[ 'mag_TWFC_r', 'mag_TWFC_g', 'mag_DEcam_r', 'mag_DEcam_g',
           'mag_LBT_r', 'mag_LBT_g', 'g-r Decals', 'mag_SDSS_r', 'mag_SDSS_g']
)
table_to_save.write(f'build/table_mag_reference.txt', format='ascii.commented_header', overwrite=True)





# Define a function to compute flux and flux ratios
def compute_flux(magnitudes, indices):
    """Compute fluxes and flux ratios for the given indices."""
    flux_1 = 10**(-0.4 * magnitudes[:, indices[0]])
    flux_2 = 10**(-0.4 * magnitudes[:, indices[1]])
    flux_ratio = flux_1 / flux_2
    return flux_1, flux_2, flux_ratio

# Define a function to create scatter and fit plots
def plot_flux_comparison(
    x_flux, y_flux, flux_ratio, colors, xlabel, ylabel, ratio_label, save_path
):
    """Plots the flux comparison and flux ratio with spline and linear fits."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Scatter plot of fluxes
    axes[0].scatter(x_flux, y_flux, c=colors, cmap='viridis', s=50, edgecolor='k')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(xlabel, fontsize=16)
    axes[0].set_ylabel(ylabel, fontsize=16)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.6)

    # Sort data for fitting
    sorted_indices = np.argsort(colors)
    colors_sorted = np.array(colors)[sorted_indices]
    flux_ratio_sorted = np.array(flux_ratio)[sorted_indices]

    # Fit a spline
    spline_fit = UnivariateSpline(colors_sorted, flux_ratio_sorted, s=0.5)

    # Linear fit
    slope, intercept, _, _, _ = linregress(colors_sorted, flux_ratio_sorted)
    flux_ratio_predicted = slope * colors_sorted + intercept

    # Spline plot
    x_smooth = np.linspace(min(colors_sorted), max(colors_sorted), 500)
    y_smooth = spline_fit(x_smooth)
    axes[1].scatter(colors, flux_ratio, c=colors, cmap='viridis', s=50, edgecolor='k')
    axes[1].plot(x_smooth, y_smooth, color='red', lw=2, label='Spline Fit')
    axes[1].plot(colors_sorted, flux_ratio_predicted, color='pink', label=f'Fit: y = {slope:.2f}x + {intercept:.2f}')
    axes[1].set_xlabel(r'$g-r$ (DECaLS)', fontsize=16)
    axes[1].set_ylabel(ratio_label, fontsize=16)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.6)
    axes[1].legend()

    # Styling
    for ax in axes:
        ax.tick_params(axis='both', labelsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Band combinations and labels for comparison
band_combinations = [
    {"indices": [1, 3], "xlabel": r"$F_g$ DECaLS", "ylabel": r"$F_g$ TWFC", "ratio_label": r"$F_g$ TWFC / $F_g$ DECaLS", "save_name": "flux_comparison_TWFC_DECALS_g.png"},
    {"indices": [0, 2], "xlabel": r"$F_r$ DECaLS", "ylabel": r"$F_r$ TWFC", "ratio_label": r"$F_r$ TWFC / $F_r$ DECaLS", "save_name": "flux_comparison_TWFC_DECALS_r.png"},
    {"indices": [5, 3], "xlabel": r"$F_g$ DECaLS", "ylabel": r"$F_g$ LBT", "ratio_label": r"$F_g$ LBT / $F_g$ DECaLS", "save_name": "flux_comparison_LBT_DECALS_g.png"},
    {"indices": [4, 2], "xlabel": r"$F_r$ DECaLS", "ylabel": r"$F_r$ LBT", "ratio_label": r"$F_r$ LBT / $F_r$ DECaLS", "save_name": "flux_comparison_LBT_DECALS_r.png"},
    {"indices": [5, 1], "xlabel": r"$F_g$ WHT", "ylabel": r"$F_g$ LBT", "ratio_label": r"$F_g$ LBT / $F_g$ WHT", "save_name": "flux_comparison_LBT_WHT_g.png"},
    {"indices": [4, 0], "xlabel": r"$F_r$ WHT", "ylabel": r"$F_r$ LBT", "ratio_label": r"$F_r$ LBT / $F_r$ WHT", "save_name": "flux_comparison_LBT_WHT_r.png"},
]

# Main loop
for combination in band_combinations:
    flux_1, flux_2, flux_ratio = compute_flux(magnitudes_data, combination["indices"])
    plot_flux_comparison(
        flux_1, flux_2, flux_ratio, colors_g_r,
        combination["xlabel"], combination["ylabel"], combination["ratio_label"],
        f"build/plots/{combination['save_name']}"
    )






######### IN THE colors


# Plot Results

colors_g_r_TWFC = magnitudes_data[:, 1] - magnitudes_data[:, 0]
colors_g_r_LBT = magnitudes_data[:, 5] - magnitudes_data[:, 4]
colors_g_r_SDSS = magnitudes_data[:, 7] - magnitudes_data[:, 6]

colors_g_i_TWFC = magnitudes_data[:, 1] - magnitudes_data[:, 8]
colors_g_i_SDSS = magnitudes_data[:, 7] - magnitudes_data[:, 10]
colors_g_i = magnitudes_data[:, 3] - magnitudes_data[:, 9]

colors_r_i_TWFC = magnitudes_data[:, 0] - magnitudes_data[:, 8]
colors_r_i_SDSS = magnitudes_data[:, 6] - magnitudes_data[:, 10]
colors_r_i = magnitudes_data[:, 2] - magnitudes_data[:, 9]




# Data arrays and subplot specifications
data_pairs = [
    {"x": colors_g_r, "y": colors_g_r_LBT, "xlabel": r"$g-r$ (DECaLS)", "ylabel": r"$g-r$ (LBT)"},
    {"x": colors_g_r, "y": colors_g_r_TWFC, "xlabel": r"$g-r$ (DECaLS)", "ylabel": r"$g-r$ (TWFC)"},
    {"x": colors_g_r, "y": colors_g_r_SDSS, "xlabel": r"$g-r$ (DECaLS)", "ylabel": r"$g-r$ (SDSS)"},
    {"x": colors_g_i, "y": colors_g_i_SDSS, "xlabel": r"$g-i$ (DECaLS)", "ylabel": r"$g-i$ (SDSS)"},
    {"x": colors_g_i, "y": colors_g_i_TWFC, "xlabel": r"$g-i$ (DECaLS)", "ylabel": r"$g-i$ (TWFC)"},
    {"x": colors_g_r_TWFC, "y": colors_g_r_LBT, "xlabel": r"$g-r$ (TWFC)", "ylabel": r"$g-r$ (LBT)"},
    {"x": colors_r_i, "y": colors_r_i_TWFC, "xlabel": r"$r-i$ (DECaLS)", "ylabel": r"$r-i$ (TWFC)"},
    {"x": colors_r_i, "y": colors_r_i_SDSS, "xlabel": r"$r-i$ (DECaLS)", "ylabel": r"$r-i$ (SDSS)"}
]

# Create figure and axes
fig, axes = plt.subplots(3, 3, figsize=(14, 14))
x_identity = np.linspace(-0.5, 1.5, 100)
colors = cm.tab20(np.linspace(0, 1, len(data_pairs)))

# Plot each data pair
for idx, pair in enumerate(data_pairs):
    row, col = divmod(idx, 3)
    x, y = pair["x"], pair["y"]

    # Sort for fitting and plotting
    sorted_indices = np.argsort(x)
    x_sorted, y_sorted = np.array(x)[sorted_indices], np.array(y)[sorted_indices]

    # Perform linear regression
    slope, intercept, _, _, _ = linregress(x_sorted, y_sorted)
    y_predicted = slope * x_sorted + intercept

    # Scatter plot
    sc = axes[row, col].scatter(x, y, c=x, cmap='viridis', s=50, edgecolor='k', label='Data')
    # Fit plot
    axes[row, col].plot(x_sorted, y_predicted, color='pink', label=f'Fit: y={slope:.2f}x + {intercept:.2f}')
    # Identity line
    axes[row, col].plot(x_identity, x_identity, linestyle='--', color='gray')

    # Labels, legend, and grid
    axes[row, col].set_xlabel(pair["xlabel"], fontsize=16)
    axes[row, col].set_ylabel(pair["ylabel"], fontsize=16)
    axes[row, col].grid(True, which="both", linestyle="--", alpha=0.6)
    axes[row, col].legend(fontsize=12)
    axes[row, col].tick_params(axis='both', labelsize=15)


fig.tight_layout()
plt.savefig("build/plots/colors.png", dpi=300)
plt.close()

