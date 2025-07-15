import numpy as np
import matplotlib.pyplot as plt
import string
import pickle

#Loading the data
path = '../Data/Figure5'
with open(f'{path}/Incremental.pkl', 'rb') as f:
    Incremental = pickle.load(f)
with open(f'{path}/Incremental_std.pkl', 'rb') as f:
    Incremental_std = pickle.load(f)

Onoise, Onoise_std = np.load(f'{path}/Onoise.npy'), np.load(f'{path}/Onoise_std.npy')
Ononoise, Ononoise_std = np.load(f'{path}/Ononoise.npy'), np.load(f'{path}/Ononoise_std.npy')

noise7, n7err = np.load(f"{path}/7Noise.npy"), np.load(f"{path}/7Noise_std.npy")
nonoise7, nn7err = np.load(f"{path}/7NoNoise.npy"), np.load(f"{path}/7NoNoise_std.npy")

#Plotting
fig, ax = plt.subplots(1, 3, layout = 'constrained', figsize = (18, 5))

ax[0].bar(np.linspace(1, 7, 4)-0.25, Onoise, width = 0.5, color = 'purple', label = 'noisy pattern')
ax[0].errorbar(np.linspace(1, 7, 4)-0.25, Onoise, yerr = Onoise_std/10, color = 'black', fmt = '.')
ax[0].bar(np.linspace(1, 7, 4)+0.25, Ononoise, width = 0.5, color = 'green', label = 'frozen pattern')
ax[0].errorbar(np.linspace(1, 7, 4)+0.25, Ononoise, yerr = Ononoise_std/10, color = 'black', fmt = '.')
ax[0].set_ylim(0, 1.02)
ax[0].set_xlim(0, 8)
ax[0].set_xticks([1, 3, 5, 7])
ax[0].set_xticklabels(['0+2', '2+1', '3+1', '4+1'])
ax[0].tick_params(labelsize = 14)
ax[0].set_ylabel(r"Orthogonality", fontsize  = 14)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].legend(fontsize = 14, loc = 'lower left')
ax[0].set_title("7 Postsynaptic neurons", fontsize = 14)


ax[1].set_title("7 Postsynaptic neurons", fontsize = 14)
ax[1].plot(np.linspace(2, 7, 6), np.linspace(2, 7, 6), color = 'black', linestyle = '--', label = 'ideal')
ax[1].errorbar(np.linspace(2, 7, 6), noise7, yerr = n7err/10, fmt = '.--', label = 'noisy pattern', elinewidth=1.2, capsize=2, color = 'purple')
ax[1].errorbar(np.linspace(2, 7, 6), nonoise7, yerr = nn7err/10, fmt = '.--', label = 'frozen pattern', elinewidth=1.2, capsize=2, color = 'green')
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].tick_params(labelsize = 14)
ax[1].set_xticklabels(['0+2', '0+2', '2+1', '3+1', '4+1', '5+1', '6+1'])
ax[1].set_ylabel(r"Rank($A_{ij}$)", fontsize = 14)
ax[1].legend(fontsize = 14)

for i in range(5):
    ax[2].errorbar(np.linspace(1, i+3, i+3), Incremental[i], yerr = np.array(Incremental_std[i]) / 10, fmt ='o', label = r'$N_{psn}$' + f'= {i+4}', elinewidth= 1, capsize=2)
ax[2].legend(fontsize = 14, loc = 'lower left')
ax[2].set_ylim(0.0, 1.02)
ax[2].hlines(1, 0.5, 7.5, linestyle = '--', color = 'black')
ax[2].set_xlim(0.5, 7.5)
ax[2].set_xticks(np.linspace(1, 7, 7))
ax[2].set_xticklabels(['0+2', '2+1', '3+1', '4+1', '5+1', '6+1', '7+1'])
ax[2].set_ylabel(r"$\left< \Omega \right>$", fontsize = 14)
ax[2].tick_params(labelsize = 14)
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].set_title("Incremental learning task", fontsize = 14)

for i, axs in enumerate(ax.flat):
    axs.text(-0.05, 1.05, s = string.ascii_lowercase[i], transform = axs.transAxes, size = 14, weight = 'bold')

plt.show()

#fig.savefig("Figure5.pdf", bbox_inches = 'tight')
fig.savefig("Figure5.eps", bbox_inches = 'tight')