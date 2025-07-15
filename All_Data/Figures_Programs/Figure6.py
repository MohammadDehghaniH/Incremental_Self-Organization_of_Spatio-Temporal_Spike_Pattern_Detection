import numpy as np
import matplotlib.pyplot as plt


#Load the file
total = np.zeros((7, 6*600))
arr = np.linspace(0, 29999, 30000, dtype = int)
ind = np.where(np.mod(arr, 50) == 0)
path = "../Data/Figure6"
x = np.load(f"{path}/final.npy")
x = x[ind][:, 100:]
w = np.load(f"{path}/0+2.npy")
w = w[ind][:, 100:]
one = []
for i in range(7):
    two = []
    for j in range(600):
        two.append(np.dot(w[j, :, i], x[-1, :, i]) / (np.linalg.norm(w[j, :, i]) * np.linalg.norm(x[-1, :, i])))
    one.append(two)
total[:, :600] = one
for _ in range(2, 7):
    w = np.load(f"{path}/{_}+1.npy")
    w = w[ind][:, 100:]
    one = []
    for i in range(7):
        two = []
        for j in range(600):
            two.append(np.dot(w[j, :, i], x[-1, :, i]) / (np.linalg.norm(w[j, :, i]) * np.linalg.norm(x[-1, :, i])))
        one.append(two)
    total[:, (_-1)*600:_*600] = one

fig, ax = plt.subplots(layout = 'constrained', figsize = (12, 5))

#Plotting in nicer order to make it according to matplotlibs standard color palette
ax.plot(np.linspace(0, 6*30000, 6*600), total[4], linewidth = 2)
ax.plot(np.linspace(0, 6*30000, 6*600), total[0], linewidth = 2)
ax.plot(np.linspace(0, 6*30000, 6*600), total[5], linewidth = 2)
ax.plot(np.linspace(0, 6*30000, 6*600), total[6], linewidth = 2)
ax.plot(np.linspace(0, 6*30000, 6*600), total[2], linewidth = 2)
ax.plot(np.linspace(0, 6*30000, 6*600), total[1], linewidth = 2)
ax.plot(np.linspace(0, 6*30000, 6*600), total[3], linewidth = 2)

#Layout
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize = 14)
ax.set_xlabel("Learning cycle (1e4)", fontsize = 14)
ax.set_xticks(np.linspace(0, 6*3e4, 10))
ax.set_xticklabels(np.linspace(0, 6*3, 10, dtype = int))
ax.set_xlim(0, 6*3e4)
ax.set_ylim(0, 1.05)
ax.hlines(1, 0, 6*3e4, color = 'black', linestyle = '-')
ax.text(15000, 0.5, '0+2', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 20)
ax.text(45000, 0.5, '2+1', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 20)
ax.text(75000, 0.5, '3+1', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 20)
ax.text(105000, 0.5, '4+1', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 20)
ax.text(135000, 0.5, '5+1', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 20)
ax.text(165000, 0.5, '6+1', horizontalalignment = 'center', verticalalignment = 'center', fontsize = 20)
ax.vlines(np.linspace(3e4, 5*3e4, 5), 0, 1.02, color = 'black', linestyle = '--')
ax.set_ylabel(r"$\cos(w, w_{final})$", fontsize = 14)

plt.show()

#Saving the figure
fig.savefig("Figure6.pdf", bbox_inches = 'tight')
fig.savefig("Figure6.eps", bbox_inches = 'tight')