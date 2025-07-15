import numpy as np
import matplotlib.pyplot as plt
import os

#Loading the Data
os.chdir("../Data/Figure10/Test")
x = ['0-1', '2-3', '4-5', '6-7', '8-9']
test = []
for dir in x:
    os.chdir(dir)
    subdir = os.listdir()
    subtest = []
    for file in subdir:
        subtest.append(np.mean(np.load(file)))
    test.append([np.mean(subtest), np.std(subtest)])
    os.chdir("../")
os.chdir("../../../")
os.chdir("Data/Figure10/Train")
x = ['0-1', '2-3', '4-5', '6-7', '8-9']
train = []
for dir in x:
    os.chdir(dir)
    subdir = os.listdir()
    subtrain = []
    for file in subdir:
        subtrain.append(np.mean(np.load(file)))
    train.append([np.mean(subtrain), np.std(subtrain)])
    os.chdir("../")
os.chdir("../../../Plotting")
test = np.array(test)
train = np.array(train)

#Plotting
fig, ax = plt.subplots(layout = 'constrained')
Arr = [train, test]
colors = ['royalblue',  'darkorange', ]
labels = ['Training', 'Testing']
for i in range(2):
    ax.bar((i*0.5) - 0.25 + np.linspace(0, 8, 5), Arr[i][:, 0], width = 0.5, yerr = Arr[i][:, 1], color = colors[i], label = f'{labels[i]}', capsize = 2, error_kw = {'elinewidth' : 1})
ax.hlines(1, -2, 10, linestyle = '--', color = 'black')


#Layout
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
fig.savefig("Figure10.pdf", bbox_inches = 'tight')
fig.savefig("Figure10.eps", bbox_inches = 'tight')