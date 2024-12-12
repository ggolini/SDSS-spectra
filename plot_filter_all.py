import numpy as np
import matplotlib.pyplot as plt

# Load the data from the first file
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



# Load the data from the first file

file2 = path_filters + "CTIO_DECam.i_filter.dat"  # Replace with your actual filename
file3 = path_filters + "interpolated_data_TWFC_i.dat"
file4 = path_filters + "SLOAN_SDSS.iprime_filter.dat"


data2 = np.genfromtxt(file2)
data3 = np.genfromtxt(file3)
data4 = np.genfromtxt(file4)

# Extract x and y from both files

x2i, y2i = data2[:, 0], data2[:, 1]
x3i, y3i = data3[:, 0], data3[:, 1]
x4i, y4i = data4[:, 0], data4[:, 1]









# Create the plot
plt.figure(figsize=(10, 6))

plt.scatter(x1g, y1g, color="blue", label="LBT", alpha=0.7)
plt.scatter(x2g, y2g, color="green", label="DECALS", alpha=0.7)
plt.scatter(x3g, y3g, color="orange", label="TWFC", alpha=0.7)
plt.scatter(x4g, y4g, color="pink", label="SDSS", alpha=0.7)

plt.scatter(x1r, y1r, color="blue", alpha=0.7)
plt.scatter(x2r, y2r, color="green", alpha=0.7)
plt.scatter(x3r, y3r, color="orange", alpha=0.7)
plt.scatter(x4r, y4r, color="pink",  alpha=0.7)


plt.scatter(x2i, y2i, color="green", alpha=0.7)
plt.scatter(x3i, y3i, color="orange", alpha=0.7)
plt.scatter(x4i, y4i, color="pink",  alpha=0.7)



# Customize the plot
plt.xlabel("Wavelength A", fontsize=14)
plt.ylabel("Transmission", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Save and display the plot
plt.tight_layout()
plt.savefig("filters.png", dpi=300)

