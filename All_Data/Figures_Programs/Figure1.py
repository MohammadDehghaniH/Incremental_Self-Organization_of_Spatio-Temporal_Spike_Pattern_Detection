import numpy as np
import matplotlib.pyplot as plt
import string

#Load the files
V_vec = np.load("../Data/Figure1/VoltageTrace.npy")
InputSpikes = np.load("../Data/Figure1/InputSpikes.npy")


fig, ax = plt.subplot_mosaic([[0, 0, 1], [2, 2, 1]], layout = 'constrained')

#Note that this looks different from the plot in the paper, we used here another simulation
ax[2].plot(V_vec, color = 'black')
ax[2].axvspan(2000, 2000 + 500, -0.5, 1.2, alpha = 0.2, color = 'blue')
ax[2].set_xlim(0, 5000)
ax[2].set_ylim(-0.5, 1.02)
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].hlines(0, 0, 5000, color = 'green', linestyle = '--')
ax[2].hlines(1, 0, 5000, color = 'red', linestyle = '--')
ax[2].set_yticks([-0.5, 0, 0.5, 1])
ax[2].tick_params(labelsize = 10)
ax[2].set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax[2].set_xticklabels([0, 100, 200, 300, 400, 500])
ax[2].set_xlabel("Time [ms]", fontsize = 14)
ax[2].set_ylabel("V(t)", fontsize = 14)

ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].spines['bottom'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_ylim(0, 1.2)
ax[1].set_xlim(0, 0.5)

ax[1].plot([0.085, 0.1], [1, 0.65], color = 'black')
ax[1].plot([0.1625, 0.175], [1, 0.65], color = 'lightgrey')
ax[1].plot([0.25, 0.25], [1, 0.65], color = 'black')
ax[1].plot([0.3375, 0.325], [1, 0.65], color = 'lightgrey')
ax[1].plot([0.415, 0.4], [1, 0.65], color = 'black')

ax[1].plot([0.175, 0.125], [0.55, 0.15], color = 'lightgrey')
ax[1].plot([0.325, 0.375], [0.55, 0.15], color = 'lightgrey')
for i in range(3):
    ax[1].plot([(i)*0.15 + 0.1, 0.25], [0.55, 0.15], color = 'black')


ax[1].add_patch(plt.Circle((0.25, 0.15), 0.04, color = 'black', zorder = 2))
ax[1].add_patch(plt.Circle((0.125, 0.15), 0.04, facecolor = 'lightgrey', edgecolor = 'darkgrey', zorder = 2))
ax[1].add_patch(plt.Circle((0.375, 0.15), 0.04, facecolor = 'lightgrey', edgecolor = 'darkgrey', zorder = 2))

acolor = ['green', 'lightgrey', 'green', 'lightgrey', 'green']
edgec  = ['None', 'darkgrey', 'None', 'darkgrey', 'None']
bcolor = ['red', 'lightgrey', 'red', 'lightgrey', 'red']
for i in range(5):
    ax[1].add_patch(plt.Circle((i*0.075+0.1, 0.65), 0.02, facecolor = acolor[i], edgecolor = edgec[i], zorder = 2))
    ax[1].add_patch(plt.Circle((i*0.075+0.1, 0.55), 0.02, facecolor = bcolor[i], edgecolor = edgec[i], zorder = 2))

ax[1].arrow(0.05, 0.8, 0, -0.5, linewidth = 1, head_length = 0.1, head_width = 0.05, color = 'black')

for i in range(500):
    ax[0].vlines(np.where(InputSpikes[i]), i, i+2, color = 'black')
ax[0].set_xlim(0, 5000)
ax[0].axvspan(2000, 2500, 0, 500, alpha = 0.2, color = 'blue')
ax[0].set_ylim(75, 200) #We only show a fourth of the actual afferents
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].spines['bottom'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_ylabel("Input afferent", fontsize = 14, labelpad=34)


#You may want to label the subplots
# for i, axs in enumerate(ax):
#     ax[i].text(-0.05, 1.05, s = string.ascii_lowercase[i], transform = ax[i].transAxes, size = 14, weight = 'bold')


plt.show()

#Save the figure

fig.savefig("Figure1.pdf", bbox_inches = 'tight')
fig.savefig("Figure1.eps", bbox_inches = 'tight')