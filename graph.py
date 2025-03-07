import scipy.io
import matplotlib.pyplot as plt

# graphing E00004
mat_data = scipy.io.loadmat('Data/E00004.mat')
data_to_plot = mat_data['val'][:, 0]

plt.plot(data_to_plot)  # For a line plot
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('E00004.mat')
plt.show()

# graphing E00001
mat_data = scipy.io.loadmat('Data/E00001.mat')
data_to_plot = mat_data['val'][:, 0]

plt.plot(data_to_plot)  # For a line plot
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('E00001.mat')
plt.show()
