import numpy as np
import matplotlib.pyplot as plt

#Loading the files
pshsp_random = np.load("../Data/Figure3/PSHSP_random.npy")
pshsp_fixed = np.load("../Data/Figure3/PSHSP_fixed.npy")
nopshsp_random = np.load("../Data/Figure3/noPSHSP_random.npy")
nopshsp_fixed = np.load("../Data/Figure3/noPSHSP_fixed.npy")

learningcycles = np.linspace(0, 10000, 200)

fig, ax = plt.subplots(layout = 'constrained')

#Plotting
ax.plot(learningcycles, pshsp_random, label = 'with PSHSP, random', color = 'red', linewidth = 3)
ax.plot(learningcycles, pshsp_fixed, label = 'with PSHSP, fixed', color = 'green', linewidth = 3)
ax.plot(learningcycles, nopshsp_fixed, label = 'without PSHSP, fixed', color = 'blue', linewidth = 3)
ax.plot(learningcycles, nopshsp_random, label = 'without PSHSP, random', color = 'grey', linewidth = 3)


#Layout
ax.hlines(1, 0, 1e4, color = 'black', linestyle = '--')
ax.set_xlim(0, 10000)
ax.set_ylim(0, 1.02)
ax.legend(fontsize = 14, loc = 'upper right', bbox_to_anchor = (1, 0.85))
ax.tick_params(labelsize = 20)
ax.set_ylabel(r"$\Omega$", fontsize = 20)
ax.set_xlabel(r"Learning cycle", fontsize = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.show()

#Saving the figures
fig.savefig("Figure3.pdf", bbox_inches = 'tight')
fig.savefig("Figure3.eps", bbox_inches = 'tight')