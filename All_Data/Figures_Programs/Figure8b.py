import numpy as np
import matplotlib.pyplot as plt
import string
import torch
from scipy.io import loadmat



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


fig, ax = plt.subplots(layout = 'constrained')
y, x = torch.where(num_spikes)
for i in range(6):
    ax.plot(V_vec[:, i]- 2*i)
ax.axvspan(intimeP_vec[0], intimeP_vec[0] + Pat_L[0], 0, 12, alpha = 0.2, color = 'red')
ax.axvspan(intimeP_vec[1], intimeP_vec[1] + Pat_L[1], 0, 12, alpha = 0.2, color = 'blue')

for i, val in enumerate(x):
    ax.vlines(val, -y[i]*2 + 1, -y[i]*2 + 1.3, color = 'black')
ax.set_xlim(0, 25000)
ax.set_ylim(-11.5, 1.5)
ax.set_yticks([0, -2, -4, -6, -8, -10])
ax.set_yticklabels([])
ax.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax.set_xticklabels([0, 0.5, 1, 1.5, 2, 2.5])
ax.tick_params(labelsize = 14)
ax.set_xlabel("Time [s]", fontsize = 14)
ax.set_ylabel("Postsynaptic potential V(t)", fontsize = 14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(-0.05, 1.05, s = string.ascii_uppercase[1], transform = ax.transAxes, size = 20, weight = 'bold')   #Only for sublabel B

plt.show()

#saving the figure
fig.savefig("Figure8b.pdf", bbox_inches = 'tight')
fig.savefig("Figure8b.eps", bbox_inches = 'tight')