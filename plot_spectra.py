import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from astropy.io import ascii
from astropy.table import Table



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
# Define paths


# --- MAIN SCRIPT ---
# Define paths
spectra_folder = 'seds/'
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





# Define the folder containing the files
folder_path = 'seds/'

# Initialize a figure for the plot
plt.figure(figsize=(10, 6))

# Generate a colormap for different colors
colors = plt.cm.viridis(np.linspace(0, 1, len(os.listdir(folder_path))))


magnitudes_data = []
vega_flux_data = []
colors_g_r = []
# Iterate through the files in the folder
for i, filename in enumerate(sorted(os.listdir(folder_path))):
    if i % 2 == 0 and filename.endswith('.txt'):  # Adjust extension if needed
        file_path = os.path.join(folder_path, filename)
        try:
            # Load the data (assuming whitespace-separated values)
            data = np.loadtxt(file_path)
            
            # Extract columns
            x = 10**data[:, 1]  # Second column for x
            y = data[:, 0]  # First column for y
            
            
            # Plot with a unique color
            plt.plot(x,y, color=colors[i], alpha=0.7)
            # Compute magnitudes and color ratios
            
            wave_eff, mags, veg_flux = compute_magnitudes(x, y, filters)
            magnitudes_data.append(mags)
            vega_flux_data.append(veg_flux)
            colors_g_r.append(mags[3] - mags[2])  # g-r color DECALS
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")



factor1=800

plt.plot(x1g, y1g*factor1, color="black", label="LBT g", alpha=0.9, ls = 'dashed')
plt.plot(x2g, y2g*factor1, color="darkgreen", label="DECALS g", alpha=0.9, ls = 'dashed')
plt.plot(x3g, y3g*factor1, color="blue", label="TWFC g", alpha=0.9, ls = 'dashed')


plt.plot(x1r, y1r*factor1, color="brown", label="LBT r", alpha=0.9, ls = 'dashed')
plt.plot(x2r, y2r*factor1, color="rebeccapurple", label="DECALS r", alpha=0.9, ls = 'dashed')
plt.plot(x3r, y3r*factor1, color="pink", label="TWFC r", alpha=0.9, ls = 'dashed')



plt.legend(loc = 1, fontsize = 10)
# Customize the plot
plt.xlabel(r"$\lambda$")
plt.ylabel("Flux [arbitrary units]")
plt.grid(True)
plt.title("Stellar spectras")

# Show the plot
plt.savefig("spectras.png")
plt.close()








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






# seleziono color range between 0 and 1.6
colors_g_r[colors_g_r < 0 ] = np.nan
colors_g_r[colors_g_r > 1.6] = np.nan



######### IN THE G BAND TWFC


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

x = np.arange(np.nanmin(decals_g_flux), np.nanmax(decals_g_flux), 1)

axes[0].plot(x, x)
axes[1].tick_params(axis='both', labelsize=15)
axes[0].tick_params(axis='both', labelsize=15)
axes[1].set_ylim(0.75, 1.1)
axes[1].set_xlim(-1, 2.5)

# Final touches
plt.tight_layout()

plt.savefig(f'build/plots/flux_comparison_TWFC_DECALS_g.png', dpi=300)
plt.close()



######### IN THE R BAND


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

x = np.arange(np.nanmin(decals_g_flux), np.nanmax(decals_g_flux), 1)

axes[0].plot(x, x)
axes[1].tick_params(axis='both', labelsize=15)
axes[1].set_ylim(0.75, 1.1)
axes[1].set_xlim(-1, 2.5)


axes[0].tick_params(axis='both', labelsize=15)
# Final touches
plt.tight_layout()

plt.savefig(f'build/plots/flux_comparison_TWFC_DECALS_r.png', dpi=300)
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

x = np.arange(np.nanmin(decals_g_flux), np.nanmax(decals_g_flux), 1)

axes[0].plot(x, x)
axes[1].tick_params(axis='both', labelsize=15)
axes[0].tick_params(axis='both', labelsize=15)

# Final touches
plt.tight_layout()

plt.savefig(f'build/plots/flux_comparison_LBT_DECALS_g.png', dpi=300)
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

x = np.arange(np.nanmin(decals_g_flux), np.nanmax(decals_g_flux), 1)

axes[0].plot(x, x)
axes[1].tick_params(axis='both', labelsize=15)


axes[0].tick_params(axis='both', labelsize=15)
# Final touches
plt.tight_layout()

plt.savefig(f'build/plots/flux_comparison_LBT_DECALS_r.png', dpi=300)
plt.close()




######### IN THE colors


# Plot Results

colors_g_r_TWFC = magnitudes_data[:, 1] - magnitudes_data[:, 0]
colors_g_r_LBT = magnitudes_data[:, 5] - magnitudes_data[:, 4]
# Plotting
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
colors = cm.tab20(np.linspace(0, 1, len(colors_g_r)))

x = np.arange(0, 1.6, 1)


# Flux ratio vs color
axes[0].scatter(colors_g_r, colors_g_r_LBT , c=colors_g_r, cmap='viridis', s=50, edgecolor='k')
axes[0].set_xlabel(r'$g-r$ (DECaLS)', fontsize=16)
axes[0].set_ylabel(r'$g-r$ (LBT)', fontsize=16)
axes[1].grid(True, which="both", linestyle="--", alpha=0.6)
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
