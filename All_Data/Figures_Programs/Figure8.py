import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.lines import Line2D
import string

fig, ax = plt.subplots(1, 3, layout = 'constrained', figsize = (19.2, 4.8))
#to assess how 
path = "../Data"
rate = loadmat(f"{path}/Total_Rate_IDs.mat")['Rate'].T
inh = loadmat(f"{path}/Total_Rate_IDs.mat")['Inhib_ID']
colors = []
for i in range(700):
    if i+1 in inh:
        colors.append('blue')
    else:
        colors.append('red')

#Plotting
ax[0].scatter(np.linspace(1, 700, 700), rate, color = colors, s = 5)

#Layout
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].tick_params(labelsize = 14)
ax[0].set_xlabel("Neuron index", fontsize = 14)
ax[0].set_ylabel("Rate [Hz]", fontsize = 14)
leg_elements = [Line2D([0], [0], color = 'red', marker = 'o', label = 'Excitatory'), 
                Line2D([0], [0], color = 'blue', marker = 'o', label = 'Inhibitory')]
ax[0].legend(handles=leg_elements, fontsize = 14)
ax[0].text(-0.05, 1.05, s = string.ascii_lowercase[0], transform = ax[0].transAxes, size = 20, weight = 'bold') #Only for sublabel A




#Loading the weights and generate membrane potential 

device = 'cpu'
N_Post = 6
N_Pres = 700
taumem = 15.0
tautraceE = 3.0
tautraceI = 5.0
tauriseE = 0.5
tauriseI = 1.0
tr = 200
delt = 0.1
dta = delt / taumem

T = 25000
EpL = T
times = np.arange(1, EpL + 1) * delt    

# Berechnung der Spike-Traces (Exzitatorisch und inhibitorisch)
traceE_np = np.exp(-times / tautraceE) - np.exp(-times / tauriseE)
ettaE = tautraceE / tauriseE
VnormE = (ettaE ** (ettaE / (ettaE - 1))) / (ettaE - 1)
traceE_np = traceE_np * VnormE
traceE = torch.tensor(traceE_np, dtype=torch.float32, device=device)
traceI_np = np.exp(-times / tautraceI) - np.exp(-times / tauriseI)    
ettaI = tautraceI / tauriseI
VnormI = (ettaI ** (ettaI / (ettaI - 1))) / (ettaI - 1)
traceI_np = traceI_np * VnormI
traceI = torch.tensor(traceI_np, dtype=torch.float32, device=device)

path = "../Data/Figure8"
W_veca = torch.tensor(np.load(f"{path}/Wa_Zero_One_id5.npy"), dtype=torch.float32, device=device)
W_vecb = torch.tensor(np.load(f"{path}/Wb_Zero_One_id5.npy"), dtype=torch.float32, device=device)

# Vorbereitung für die Faltung: inall hat die Größe (N_Pres, T + len(trace) - 1)
inall = torch.zeros(N_Pres, T + len(traceE) - 1, device=device)
resy = 0.0
threshy = 1.0

# Laden von Total_Rate_IDs.mat (enthält Exc_ID, Inhib_ID, Rate, u. a.)
total_rate = loadmat('../Data/Total_Rate_IDs.mat')
Exc_ID = total_rate['Exc_ID'].flatten().astype(int) - 1  
# Anpassung auf 0-indexierung
Inhib_ID = total_rate['Inhib_ID'].flatten().astype(int) - 1
Rate_np = total_rate['Rate'].flatten()
Rate = torch.tensor(Rate_np, dtype=torch.float32, device=device)
N_E = Exc_ID
N_I = Inhib_ID
NI = len(N_I)

L = traceE.shape[0]  # assuming traceE and traceI have same length
n = T + L -1     # length of full convolution result

kernelE_fft = torch.fft.rfft(traceE, n=n)
kernelI_fft = torch.fft.rfft(traceI, n=n)

# If N_E and N_I are not tensors, convert them (if needed):
N_E = torch.tensor(N_E, dtype=torch.long, device=device)
N_I = torch.tensor(N_I, dtype=torch.long, device=device)
intimeP_vec = [2000, 13000]

mat_data = loadmat(f'{path}/Speaker_3_Label_0_N_4.mat')
PATP1 = torch.tensor(mat_data['spike_data'], dtype=torch.float32, device=device)

mat_data = loadmat(f'{path}/Speaker_4_Label_1_N_9.mat')
PATP2 = torch.tensor(mat_data['spike_data'], dtype=torch.float32, device=device)

Pat_L = [PATP1.shape[1], PATP2.shape[1]]
PAT1 = torch.rand(N_Pres, T, device=device)
PAT1[N_E, :] = (PAT1[N_E, :] < (Rate[N_E].unsqueeze(1) * delt / 1000)).float()
PAT1[N_I, :] = (PAT1[N_I, :] < (Rate[N_I].unsqueeze(1) * delt / 1000)).float()

# Einfügen der eingebetteten Muster
PAT1[:, intimeP_vec[0]:intimeP_vec[0] + Pat_L[0]] = PATP1
PAT1[:, intimeP_vec[1]:intimeP_vec[1] + Pat_L[1]] = PATP2

WW_vec = W_veca * W_vecb
WW_vec[N_I, :] = -WW_vec[N_I, :]

# Compute the FFT of the input signals along the time dimension
PAT1_fft = torch.fft.rfft(PAT1, n=n, dim=1)

# Preallocate the output tensor for the convolution results.
conv_result = torch.empty((N_Pres, n), device=device)

# For excitatory neurons, multiply by the excitatory kernel FFT and invert.
conv_result[N_E] = torch.fft.irfft(PAT1_fft[N_E] * kernelE_fft, n=n, dim=1)

# For inhibitory neurons, multiply by the inhibitory kernel FFT and invert.
conv_result[N_I] = torch.fft.irfft(PAT1_fft[N_I] * kernelI_fft, n=n, dim=1)

inall = conv_result[:, :T]

I_ext = torch.zeros(EpL, N_Post, device=device)
INP = inall[:, :EpL]
for iw in range(N_Post):
    I_ext[:, iw] = torch.matmul(WW_vec[:, iw].unsqueeze(0), INP).squeeze()
INP = torch.abs(INP)
num_spikes = torch.zeros(N_Post, EpL, device=device)
V_vec = torch.zeros(EpL, N_Post, device=device)
for it in range(1, EpL):
    V_vec[it, :] = (1 - dta) * V_vec[it - 1, :] + I_ext[it - 1, :] * dta
    spike_idx = (V_vec[it - 1, :] >= threshy).nonzero(as_tuple=True)[0]
    if spike_idx.numel() > 0:
        V_vec[it, spike_idx] = resy
        num_spikes[spike_idx, it] += 1




#Plotting (Potential outside of the embedded patterns will deviate from the Figure8b in the paper)


y, x = torch.where(num_spikes)
for i in range(6):
    ax[1].plot(V_vec[:, i]- 2*i)
ax[1].axvspan(intimeP_vec[0], intimeP_vec[0] + Pat_L[0], 0, 12, alpha = 0.2, color = 'lightsalmon')
ax[1].axvspan(intimeP_vec[1], intimeP_vec[1] + Pat_L[1], 0, 12, alpha = 0.2, color = 'lightsteelblue')

for i, val in enumerate(x):
    ax[1].vlines(val, -y[i]*2 + 1, -y[i]*2 + 1.3, color = 'black')
ax[1].set_xlim(0, 25000)
ax[1].set_ylim(-11.5, 1.5)
ax[1].set_yticks([0, -2, -4, -6, -8, -10])
ax[1].set_yticklabels([])
ax[1].set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax[1].set_xticklabels([0, 0.5, 1, 1.5, 2, 2.5])
ax[1].tick_params(labelsize = 14)
ax[1].set_xlabel("Time [s]", fontsize = 14)
ax[1].set_ylabel("Postsynaptic potential V(t)", fontsize = 14)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].text(-0.05, 1.05, s = string.ascii_lowercase[1], transform = ax[1].transAxes, size = 20, weight = 'bold')   #Only for sublabel 

#ax[1].set_rasterized(True)



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
for i in range(6):
    ax[2].plot(w[:, i]-3*i, '.')
    ax[2].hlines(-3*i, 0, 715, color = 'black', linewidth = 1)
    ax[2].vlines(0, -3*i+1, -3*i - 1, color = 'black', linewidth = 1)


#layout 
ax[2].set_xlim(0, 700)
ax[2].set_ylim(-16.1, 1.1)
ax[2].set_yticks([])
ax[2].set_yticklabels([])
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)
ax[2].spines['left'].set_visible(False)
ax[2].tick_params(labelsize = 14)
ax[2].set_xlabel("Neuron index", fontsize = 14)
ax[2].set_ylabel(r"Synaptic efficacy $w$", fontsize = 14)
ax[2].text(-0.05, 1.05, s = string.ascii_lowercase[2], transform = ax[2].transAxes, size = 20, weight = 'bold')   #Only for sublabel C

plt.show()

#saving the figure
fig.savefig("Figure8.pdf", bbox_inches = 'tight')
fig.savefig("Figure8.eps", bbox_inches = 'tight', dpi = 600)