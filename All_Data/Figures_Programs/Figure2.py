import numpy as np
import matplotlib.pyplot as plt
import string

#Loading the files
path = "../Data/Figure2"
cycle = np.linspace(1, 10000, 10000)
ISO_R_noise = np.load(f"{path}/ISO_R_noise.npy")
ISO_R_nonoise = np.load(f"{path}/ISO_R_nonoise.npy")
ISO_Sens_noise = np.load(f"{path}/ISO_sens_noise.npy")
ISO_Spec_noise = np.load(f"{path}/ISO_spec_noise.npy")
ISO_Sens_clear = np.load(f"{path}/ISO_sens_clear.npy")
ISO_Spec_clear = np.load(f"{path}/ISO_spec_clear.npy")
CBL_R_noise = np.load(f"{path}/CBL_R_noise.npy")
CBL_R_nonoise = np.load(f"{path}/CBL_R_nonoise.npy")
CBL_Sens_noise = np.load(f"{path}/CBL_Sens_noise.npy")
CBL_Spec_noise = np.load(f"{path}/CBL_spec_noise.npy")
CBL_Sens_clear = np.load(f"{path}/MST_sens_clear.npy")
CBL_Spec_clear = np.load(f"{path}/MST_spec_clear.npy")
MST_R_noise = np.load(f"{path}/MST_R_noise.npy")
MST_R_nonoise = np.load(f"{path}/MST_R_nonoise.npy")
MST_Sens_noise = np.load(f"{path}/MST_Sens_noise.npy")
MST_Spec_noise = np.load(f"{path}/MST_Spec_noise.npy")
MST_Sens_clear = np.load(f"{path}/CBL_sens_clear.npy")
MST_Spec_clear = np.load(f"{path}/CBL_spec_clear.npy")


#Plotting & Layout
fig, ax = plt.subplots(2, 3, layout = 'constrained', figsize = (12, 5))


ax[0, 0].plot(cycle, ISO_R_nonoise, color = 'green', label = "unsupervised self-organization")
ax[0, 0].plot(cycle, MST_R_nonoise, color = 'purple', label = "multispike-tempotron")
ax[0, 0].plot(cycle, CBL_R_nonoise, color = 'orange', label = "correlation-based learning")
ax[0, 0].set_xlim(0, 10000)
ax[0, 0].set_ylim(0, 1.02)
ax[0, 0].hlines(1, 1, 1e4, linestyle = '--', color = 'black')
ax[0, 0].spines['top'].set_visible(False)
ax[0, 0].spines['right'].set_visible(False)

ax[1, 0].plot(cycle, ISO_R_noise, color = 'green')
ax[1, 0].plot(cycle, MST_R_noise, color = 'purple')
ax[1, 0].plot(cycle, CBL_R_noise, color = 'orange')
ax[1, 0].set_xlim(0, 10000)
ax[1, 0].set_ylim(0, 1.02)
ax[1, 0].hlines(1, 1, 1e4, linestyle = '--', color = 'black')
ax[1, 0].spines['top'].set_visible(False)
ax[1, 0].spines['right'].set_visible(False)

ax[0, 1].set_xlim(0, 50)
ax[0, 2].set_xlim(0, 50)
ax[1, 1].set_xlim(0, 50)
ax[1, 2].set_xlim(0, 50)
ax[0, 1].hlines(1, 1, 1e4, linestyle = '--', color = 'black')
ax[0, 2].hlines(1, 1, 1e4, linestyle = '--', color = 'black')
ax[1, 1].hlines(1, 1, 1e4, linestyle = '--', color = 'black')
ax[1, 2].hlines(1, 1, 1e4, linestyle = '--', color = 'black')


ax[0, 1].plot(ISO_Spec_clear, color = 'green')
ax[0, 1].plot(MST_Spec_clear, color = 'purple')
ax[0, 1].plot(CBL_Spec_clear, color = 'orange')

ax[1, 1].plot(ISO_Spec_noise, color = 'green')
ax[1, 1].plot(MST_Spec_noise, color = 'purple')
ax[1, 1].plot(CBL_Spec_noise, color = 'orange')

ax[0, 2].plot(ISO_Sens_clear, color = 'green')
ax[0, 2].plot(MST_Sens_clear, color = 'purple')
ax[0, 2].plot(CBL_Sens_clear, color  = 'orange')

ax[1, 2].plot(ISO_Sens_noise, color = 'green')
ax[1, 2].plot(MST_Sens_noise, color = 'purple')
ax[1, 2].plot(CBL_Sens_noise, color = 'orange')


ax[0, 1].spines['top'].set_visible(False)
ax[0, 1].spines['right'].set_visible(False)
ax[0, 2].spines['top'].set_visible(False)
ax[0, 2].spines['right'].set_visible(False)
ax[1, 1].spines['top'].set_visible(False)
ax[1, 1].spines['right'].set_visible(False)
ax[1, 2].spines['top'].set_visible(False)
ax[1, 2].spines['right'].set_visible(False)
ax[0, 1].set_ylim(0, 1.02)
ax[0, 2].set_ylim(0, 1.02)
ax[1, 1].set_ylim(0, 1.02)
ax[1, 2].set_ylim(0, 1.02)

ax[0, 0].set_ylabel("Frozen", fontsize = 14)
ax[1, 0].set_ylabel("Noisy", fontsize = 14)

ax[1, 0].set_xlabel("Learning cycle", fontsize = 14)
ax[1, 1].set_xlabel(r"Evaluation $\sigma$ [ms]", fontsize = 14)
ax[1, 2].set_xlabel(r"Evaluation $\sigma$ [ms]", fontsize = 14)
ax[0, 0].set_title(r"$R$", fontsize = 14)
ax[0, 1].set_title(r"Specificity", fontsize = 14)
ax[0, 2].set_title(r"Sensitivity", fontsize = 14)

ax[0, 0].legend()

for i, axs in enumerate(ax.flat):
    axs.text(-0.05, 1.05, s = string.ascii_lowercase[i], transform = axs.transAxes, size = 14, weight = 'bold')


plt.show()

#Saving the figure
fig.savefig("Figure2.pdf", bbox_inches = 'tight')
fig.savefig("Figure2.eps", bbox_inches = 'tight')