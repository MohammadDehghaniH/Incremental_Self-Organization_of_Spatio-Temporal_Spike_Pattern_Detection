import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import string
import soundfile as sf

path = "../Data/Figure7"
fig, ax = plt.subplots(2, 2, layout = 'constrained', figsize = (2*6.4, 2*4.8))
y, x = np.where(loadmat(f"{path}/Speaker_3_Label_0_N_4.mat")['spike_data'])
span = len(loadmat(f"{path}/Speaker_3_Label_0_N_4.mat")['spike_data'][0])/10000
audio = sf.read(f"{path}/lang-english_speaker-03_trial-4_digit-0.flac")[0]
ax[1, 0].plot(x/10000, -1*y, '.', color = 'cornflowerblue', markersize = 1)
ax[0, 0].plot(np.linspace(0, span, len(audio)), audio, color = 'royalblue', linewidth = 1)

y, x = np.where(loadmat(f"{path}/Speaker_4_Label_1_N_9.mat")['spike_data'])
span = len(loadmat(f"{path}/Speaker_4_Label_1_N_9.mat")['spike_data'][0])/10000
audio = sf.read(f"{path}/lang-english_speaker-04_trial-9_digit-1.flac")[0]
ax[0, 1].plot(np.linspace(0, span, len(audio)), audio, color = 'darkorange', linewidth = 1)
ax[1, 1].plot(x/10000, -1*y, '.', color = 'orange', markersize = 1)

for i in range(2):
    ax[1, i].set_yticks([0, -100, -200, -300, -400, -500, -600, -700])
    ax[1, i].set_yticklabels(np.linspace(0, 700, 8, dtype = int))
    for j in range(2):
        ax[i, j].spines['top'].set_visible(False)
        ax[i, j].spines['right'].set_visible(False)
        ax[i, j].tick_params(labelsize = 14)
for i, axs in enumerate(ax.flat):
    axs.text(-0.05, 1.05, s = string.ascii_uppercase[i], transform = axs.transAxes, size = 20, weight = 'bold')

ax[0, 0].set_ylabel("Amplitude", fontsize = 14)
ax[1, 0].set_ylabel("Neuron index", fontsize = 14)
ax[1, 0].set_xlabel("Time [s]", fontsize = 14)
ax[1, 1].set_xlabel("Time [s]", fontsize = 14)

plt.show()

fig.savefig("Figure7.pdf", bbox_inches = 'tight')
#fig.savefig("Figure7.eps", bbox_inches = 'tight')