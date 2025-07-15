import numpy as np
import matplotlib.pyplot as plt

#Load the activity matrix for all 500 simulations over the whole training 
values = np.load("../Data/Figure4/IncrementalLearningTask_7Neurons_Noise2ms.npy")

#Map the activity matrix to Omega
value = np.zeros((500, 3600))
for i in range(500):
    value[i] = np.linalg.matrix_rank(values[i])
for i in range(6):
    value[:, i*600 : (i+1)*600] *= 1/(i+2)
value = np.mean(value, axis = 0)
ind = np.linspace(1, 30000*6, 6*600)

#Plotting
fig, ax = plt.subplots(layout = 'constrained', figsize = (12, 4))
ax.plot(ind[:1800], value[:1800], linewidth = 3, color = 'green')

#Layout
ax.set_xlim(0, 3*30000)
ax.set_xticks(np.linspace(0, 3*30000, 3*3+1))
ax.set_xticklabels(np.linspace(0, 3*3, 3*3+1, dtype = int))
ax.set_ylim(0, 1.05)
ax.tick_params(labelsize = 20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel("Learning cycle [1e4]", fontsize = 20)
ax.set_ylabel(r"$\left<\Omega\right>$", fontsize = 20)
ax.hlines(1, 0, 3*30000, color = 'black', linestyle = '--')
ax.text(15000, 0.5, '0+2', fontsize = 20, verticalalignment = 'center', horizontalalignment = 'center')
for i in range(1, 3*1):
    ax.text(i*30000+15000, 0.5, f'{i+1}+1', fontsize = 20, verticalalignment = 'center', horizontalalignment = 'center')


plt.show()

#Save the figure
fig.savefig("Figure4.pdf", bbox_inches = 'tight')
fig.savefig("Figure4.eps", bbox_inches = 'tight')