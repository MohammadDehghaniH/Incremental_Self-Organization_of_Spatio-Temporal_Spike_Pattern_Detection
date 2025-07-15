import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import string

#To assess which neurons were inhibitory
factorr = []
for j in range(6):
    factor = []
    for i in range(700):
        if i not in loadmat("../Data/Total_Rate_IDs.mat")['Inhib_ID'][0]-1:
            factor.append(1)
        else:
            factor.append(-1)
    factorr.append(factor)
factorr = np.array(factorr)

#load the data
path = "../Data/Figure8"
w = np.load(f"{path}/Wa_Zero_One_id5.npy") * np.load(f"{path}/Wb_Zero_One_id5.npy") * factorr.T

#plotting
fig, ax = plt.subplots(layout = 'constrained')
for i in range(6):
    ax.plot(w[:, i]-3*i, '.')
    ax.hlines(-3*i, 0, 715, color = 'black', linewidth = 1)
    ax.vlines(0, -3*i+1, -3*i - 1, color = 'black', linewidth = 1)


#layout 
ax.set_xlim(0, 700)
ax.set_ylim(-16.1, 1.1)
ax.set_yticks([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(labelsize = 14)
ax.set_xlabel("Neuron index", fontsize = 14)
ax.set_ylabel(r"Synaptic efficacy $w$", fontsize = 14)
ax.text(-0.05, 1.05, s = string.ascii_uppercase[2], transform = ax.transAxes, size = 20, weight = 'bold')   #Only for sublabel C

plt.show()

#saving the figure
fig.savefig("Figure8c.pdf", bbox_inches = 'tight')
fig.savefig("Figure8c.eps", bbox_inches = 'tight')