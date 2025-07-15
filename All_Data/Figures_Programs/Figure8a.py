import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.lines import Line2D
import string

#to assess how 
path = "../Data"
rate = loadmat(f"{path}/Total_Rate_IDs.mat")['Rate'].T
inh = loadmat(f"{path}/Total_Rate_IDs.mat")['Inhib_ID']
colors = []
for i in range(700):
    if i+1 in inh:
        colors.append('blue')
    else:
        colors.append('red')

#Plotting
fig, ax = plt.subplots(1, 1, layout = 'constrained')
ax.scatter(np.linspace(1, 700, 700), rate, color = colors, s = 5)

#Layout
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize = 14)
ax.set_xlabel("Neuron index", fontsize = 14)
ax.set_ylabel("Rate [Hz]", fontsize = 14)
leg_elements = [Line2D([0], [0], color = 'red', marker = 'o', label = 'Excitatory'), 
                Line2D([0], [0], color = 'blue', marker = 'o', label = 'Inhibitory')]
ax.legend(handles=leg_elements, fontsize = 14)
ax.text(-0.05, 1.05, s = string.ascii_uppercase[0], transform = ax.transAxes, size = 20, weight = 'bold') #Only for sublabel A

plt.show()

#Saving the figure
fig.savefig("Figure8a.pdf", bbox_inches = 'tight')
fig.savefig("Figure8a.eps", bbox_inches = 'tight')