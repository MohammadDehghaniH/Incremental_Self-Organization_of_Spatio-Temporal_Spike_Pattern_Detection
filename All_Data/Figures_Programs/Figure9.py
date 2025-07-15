import numpy as np
import matplotlib.pyplot as plt

#Loading the Data
words = [['Zero', 'One'], ['Two', 'Three'], ['Four', 'Five'], ['Six', 'Seven'], ['Eight', 'Nine']]
path = "../Data/Figure9/TrainingPerformance"
perf = []
for i in range(5):
    perff = []
    for j in range(1, 6):
        perff.append(np.mean(np.load(f"{path}/Omega_{words[i][0]}_{words[i][1]}_id{j}.npy")))
    perf.append(np.array(perff))
Training = np.mean(perf, axis = 1), np.std(perf, axis = 1)

path = "../Data/Figure9/TestingPerformance"
perf = []
for i in range(5):
    perff = []
    for j in range(1, 6):
        perff.append(np.mean(np.load(f"{path}/Omega_{words[i][0]}_{words[i][1]}_id{j}.npy")))
    perf.append(np.array(perff))
Testing = np.mean(perf, axis = 1), np.std(perf, axis = 1)

#Plotting
fig, ax = plt.subplots(layout = 'constrained')
Arr = [Testing, Training]
colors = ['royalblue',  'darkorange', ]
labels = ['Testing', 'Training']
for i in range(2):
    ax.bar((i*0.5) - 0.25 + np.linspace(0, 8, 5), Arr[i][0], width = 0.5, yerr = Arr[i][1], color = colors[i], label = f'{labels[i]}', capsize = 2, error_kw = {'elinewidth' : 1})


#Layout
ax.hlines(1, -2, 10, linestyle = '--', color = 'black')
ax.legend(loc = 'lower right', fontsize = 14)
ax.set_xticks(np.linspace(0, 8, 5))
ax.set_xticklabels([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
ax.set_xticklabels(['0 & 1', '2 & 3', '4 & 5', '6 & 7', '8 & 9'])
ax.set_xlabel("Word pairs", fontsize = 14)
ax.set_ylabel(r"$\left< \Omega \right>$", fontsize = 14)
ax.tick_params(labelsize = 14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, 1.02)
ax.set_xlim(-1, 9)
plt.show()

#Saving the figure
fig.savefig("/home/lenny/Desktop/AllPlots+Code/Plotting/Figure9.pdf", bbox_inches = 'tight')
fig.savefig("/home/lenny/Desktop/AllPlots+Code/Plotting/Figure9.eps", bbox_inches = 'tight')