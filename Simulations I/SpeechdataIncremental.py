import os
import numpy as np
import torch
from scipy.io import loadmat

def Model_Train(firstword, secondword):
    # Parameterinitialisation
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    p1 = int(os.getenv('SGE_TASK_ID'))  #Seed
    Speaker = int(os.getenv('SGE_TASK_ID')) - 1 #SpeakerID
    SDF = 100 + (p1 - 1)
    np.random.seed(SDF)
    torch.manual_seed(SDF)
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

    # learning rates and other parameters
    de_f = -1e-4
    alpha_1 = 4e-2
    C_E = 0.9e-4
    C_I = 1e-4
    w_max_a = 1.0
    w_max_b = 1.0
    Gamma_star = 0.9
    Gamma_starM1 = 1 - Gamma_star
    Gamma_E = 0.99
    Gamma_EM1 = 1 - Gamma_E
    PSI_par = 12

    # Simulationtime: T is number of timesteps    
    T = 25000
    EpL = T
    times = np.arange(1, EpL + 1) * delt    

    # Spike-Traces (Excitatory und inhibitory)
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

    # Initialize synaptic weights
    W_veca = 0.1 + 1e-2 * torch.randn(N_Pres, N_Post, device=device)
    W_veca = torch.clamp(W_veca, min=0)
    W_vecb = 0.1 + 1e-2 * torch.randn(N_Pres, N_Post, device=device)
    W_vecb = torch.clamp(W_vecb, min=0)

    # Preparation for convolution and LIF threshold and resting potential
    inall = torch.zeros(N_Pres, T + len(traceE) - 1, device=device)
    Filterd_Spikes = torch.zeros(N_Post, device=device)
    desired_S = 2.0
    resy = 0.0
    threshy = 1.0

    # load Total_Rate_IDs.mat (contains Exc_ID, Inhib_ID, Rate, u. a.)
    total_rate = loadmat('Total_Rate_IDs.mat')
    Exc_ID = total_rate['Exc_ID'].flatten().astype(int) - 1  
    # Adjust for 0-indexing
    Inhib_ID = total_rate['Inhib_ID'].flatten().astype(int) - 1
    Rate_np = total_rate['Rate'].flatten()
    Rate = torch.tensor(Rate_np, dtype=torch.float32, device=device)
    N_E = Exc_ID
    N_I = Inhib_ID
    NI = len(N_I)

    # Initialize plasticity variables
    del_b_E = torch.zeros((N_Pres - NI, N_Post), device=device)
    del_a_E = torch.zeros((N_Pres - NI, N_Post), device=device)
    del_b_I = torch.zeros((NI, N_Post), device=device)

    # Load the embedded pattern i.e. the spoken words
    PATP1 = []
    PATP2 = []
    PATP1.append(torch.tensor(loadmat(f"PathToSpeechData/Speaker_{Speaker}_Label_{firstword}_N_1.mat")['spike_data'], device=device))
    PATP2.append(torch.tensor(loadmat(f"PathToSpeechData/Speaker_{Speaker}_Label_{secondword}_N_1.mat")['spike_data'], device=device))      

    L = traceE.shape[0]  # assuming traceE and traceI have same length
    n = T + L -1     # length of full convolution result

    KernelI = -VnormI * ((taumem - tautraceI) * tauriseI * np.exp(times/taumem + times/tautraceI) + ((tauriseI*tautraceI - taumem * tautraceI)* np.exp(times/taumem) + (taumem*tautraceI - taumem * tauriseI)*np.exp(times/tautraceI))*np.exp(times/tauriseI)) * np.exp(-times/tauriseI -times/taumem - times/tautraceI) / ((taumem - tautraceI) * (tauriseI - taumem)) # * taumem for correctness?
    KernelE = -VnormE * ((taumem - tautraceE) * tauriseE * np.exp(times/taumem + times/tautraceE) + ((tauriseE*tautraceE - taumem * tautraceE)* np.exp(times/taumem) + (taumem*tautraceE - taumem * tauriseE)*np.exp(times/tautraceE))*np.exp(times/tauriseE)) * np.exp(-times/tauriseE -times/taumem - times/tautraceE) / ((taumem - tautraceE) * (tauriseE - taumem)) # * taumem for correctness?
    eligE = np.zeros(T)
    eligE[:3000] = KernelE[:3000]
    eligI = np.zeros(T)
    eligI[:3000] = KernelI[:3000]

    eligE = torch.tensor(eligE, dtype=torch.float32, device=device)
    eligI = torch.tensor(eligI, dtype=torch.float32, device=device)

    kernelE_fft = torch.fft.rfft(traceE, n=n)
    kernelI_fft = torch.fft.rfft(traceI, n=n)

    eligE_fft = torch.fft.rfft(eligE, n = n)
    eligI_fft = torch.fft.rfft(eligI, n = n)
    # If N_E and N_I are not tensors, convert them (if needed):
    N_E = torch.tensor(N_E, dtype=torch.long, device=device)
    N_I = torch.tensor(N_I, dtype=torch.long, device=device)

    for itrial in range(500):

        # Erzeugen eines zufälligen Eingangsmusters (Poisson-Spikes)
        PAT1 = torch.rand(N_Pres, T, device=device)
        PAT1[N_E, :] = (PAT1[N_E, :] < (Rate[N_E].unsqueeze(1) * delt / 1000)).float()
        PAT1[N_I, :] = (PAT1[N_I, :] < (Rate[N_I].unsqueeze(1) * delt / 1000)).float()

        # Actual weight with negative values for the inhibitory weights
        WW_vec = W_veca * W_vecb        
        WW_vec[N_I, :] = -WW_vec[N_I, :]
        

        # Compute the FFT of the input signals along the time dimension
        PAT1_fft = torch.fft.rfft(PAT1, n=n, dim=1)

        # Preallocate the output tensor for the convolution results.
        conv_result = torch.empty((N_Pres, n), device=device)
        elig_conv = torch.empty((N_Pres, n), device=device)

        # For excitatory neurons, multiply by the excitatory kernel FFT and invert.
        conv_result[N_E] = torch.fft.irfft(PAT1_fft[N_E] * kernelE_fft, n=n, dim=1)
        elig_conv[N_E] = torch.fft.irfft(PAT1_fft[N_E] * eligE_fft, n=n, dim=1)

        # For inhibitory neurons, multiply by the inhibitory kernel FFT and invert.
        conv_result[N_I] = torch.fft.irfft(PAT1_fft[N_I] * kernelI_fft, n=n, dim=1)
        elig_conv[N_I] = torch.fft.irfft(PAT1_fft[N_I] * eligI_fft, n=n, dim=1)


        inall = conv_result[:, :T]
        
        # Computing the external input I_ext     
        I_ext = torch.zeros(EpL, N_Post, device=device)        
        INP = inall[:, :EpL]
        eligINP = torch.abs(elig_conv[:, :EpL])

        for iw in range(N_Post):
            I_ext[:, iw] = torch.matmul(WW_vec[:, iw].unsqueeze(0), INP).squeeze()
            INP = torch.abs(INP)

        # Computing the membrane potential
        num_spikes = torch.zeros(N_Post, EpL, device=device)
        V_vec = torch.zeros(EpL, N_Post, device=device)
        
        for it in range(1, EpL):
            V_vec[it, :] = (1 - dta) * V_vec[it - 1, :] + I_ext[it - 1, :] * dta
            spike_idx = (V_vec[it - 1, :] >= threshy).nonzero(as_tuple=True)[0]
            if spike_idx.numel() > 0:
                V_vec[it, spike_idx] = resy                
                num_spikes[spike_idx, it] += 1
        
        Filterd_Spikes = Gamma_star * Filterd_Spikes + Gamma_starM1 * torch.sum(num_spikes[:, tr:EpL], dim=1)
        
        # Plasticity rule      
        del_b_I = Gamma_E * del_b_I + Gamma_EM1 * torch.matmul(eligINP[N_I, tr:EpL], V_vec[tr:EpL, :])        
        Filterd_Dw_E = torch.matmul(eligINP[N_E, tr:EpL], (V_vec[tr:EpL, :] * (V_vec[tr:EpL, :] > 0).float()))
        Filterd_Dw_E = Filterd_Dw_E - torch.mean(Filterd_Dw_E, dim=0)
        del_b_E = Gamma_E * del_b_E + Gamma_EM1 * Filterd_Dw_E        
        W_vecb[N_I, :] = W_vecb[N_I, :] + C_I * del_b_I        
        W_vecb[N_E, :] = (1 + de_f) * W_vecb[N_E, :] + alpha_1 * (W_vecb[N_E, :] * torch.tanh(desired_S - Filterd_Spikes)) + C_E * del_b_E
        SDF = torch.sum((Filterd_Dw_E > 0).float(), dim=1)
        Filterd_Dw_E_n = Filterd_Dw_E * (Filterd_Dw_E > 0).float()
        temp = ((SDF > 1).float() *(torch.sum((Filterd_Dw_E_n.double() ** PSI_par) * (Filterd_Dw_E_n > 0).float(), dim=1) / (SDF + 1e-16))) ** (1 / PSI_par)
        del_a_E = Gamma_E * del_a_E + Gamma_EM1 * (Filterd_Dw_E_n - temp.unsqueeze(1)).float()
        W_veca[N_I, :] = W_veca[N_I, :] + C_I * del_b_I        
        W_veca[N_E, :] = (1 + de_f) * W_veca[N_E, :] + alpha_1 * (W_veca[N_E, :] * torch.tanh(desired_S - Filterd_Spikes)) + C_E * del_a_E
        W_veca = W_veca * (W_veca > 0).float()
        W_vecb = W_vecb * (W_vecb > 0).float()
        W_veca[N_E, :] = torch.clamp(W_veca[N_E, :], max=w_max_a)
        W_vecb[N_E, :] = torch.clamp(W_vecb[N_E, :], max=w_max_b)
        
    # Saving the weights after initialization
    save_dir = f'Init/'
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"SaveDirectory/Init/Wa_{firstword}_{secondword}_Speaker{Speaker}_id{p1}.npy", W_veca.cpu().numpy())
    np.save(f"SaveDirectory/Init/Wb_{firstword}_{secondword}_Speaker{Speaker}_id{p1}.npy", W_vecb.cpu().numpy())
    
    em_id = 2
    Omega = []
    intimeP_vec = [2000, 13000]
    
    #Training for the first pattern
    for itrial in range(6000):
        om = []
        for id in range(1):
            PAT1 = torch.rand(N_Pres, T, device=device)
            PAT1[N_E, :] = (PAT1[N_E, :] < (Rate[N_E].unsqueeze(1) * delt / 1000)).float()
            PAT1[N_I, :] = (PAT1[N_I, :] < (Rate[N_I].unsqueeze(1) * delt / 1000)).float()

            # Einfügen der eingebetteten Muster
            Pat_L = [PATP1[id].shape[1], PATP2[id].shape[1]]
            PAT1[:, intimeP_vec[0]:intimeP_vec[0] + Pat_L[0]] = PATP1[id]
            #PAT1[:, intimeP_vec[1]:intimeP_vec[1] + Pat_L[1]] = PATP2[id]
            
            WW_vec = W_veca * W_vecb
            WW_vec[N_I, :] = -WW_vec[N_I, :]

            # Compute the FFT of the input signals along the time dimension
            PAT1_fft = torch.fft.rfft(PAT1, n=n, dim=1)

            # Preallocate the output tensor for the convolution results.
            conv_result = torch.empty((N_Pres, n), device=device)
            elig_conv = torch.empty((N_Pres, n), device=device)


            # For excitatory neurons, multiply by the excitatory kernel FFT and invert.
            conv_result[N_E] = torch.fft.irfft(PAT1_fft[N_E] * kernelE_fft, n=n, dim=1)
            elig_conv[N_E] = torch.fft.irfft(PAT1_fft[N_E] * eligE_fft, n=n, dim=1)
            
            # For inhibitory neurons, multiply by the inhibitory kernel FFT and invert.
            conv_result[N_I] = torch.fft.irfft(PAT1_fft[N_I] * kernelI_fft, n=n, dim=1)
            elig_conv[N_I] = torch.fft.irfft(PAT1_fft[N_I] * eligI_fft, n=n, dim=1)
            
            inall = conv_result[:, :T]


            I_ext = torch.zeros(EpL, N_Post, device=device)
            INP = inall[:, :EpL]
            eligINP = torch.abs(elig_conv[:, :EpL])

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
            
            Filterd_Spikes = Gamma_star * Filterd_Spikes + Gamma_starM1 * torch.sum(num_spikes[:, tr:EpL], dim=1)

            del_b_I = Gamma_E * del_b_I + Gamma_EM1 * torch.matmul(eligINP[N_I, tr:EpL], V_vec[tr:EpL, :])
            Filterd_Dw_E = torch.matmul(eligINP[N_E, tr:EpL], (V_vec[tr:EpL, :] * (V_vec[tr:EpL, :] > 0).float()))
            Filterd_Dw_E = Filterd_Dw_E - torch.mean(Filterd_Dw_E, dim=0)
            del_b_E = Gamma_E * del_b_E + Gamma_EM1 * Filterd_Dw_E        
            W_vecb[N_I, :] = W_vecb[N_I, :] + C_I * del_b_I
            W_vecb[N_E, :] = (1 + de_f) * W_vecb[N_E, :] + alpha_1 * (W_vecb[N_E, :] * torch.tanh(desired_S - Filterd_Spikes)) + C_E * del_b_E        
            SDF = torch.sum((Filterd_Dw_E > 0).float(), dim=1)
            Filterd_Dw_E_n = Filterd_Dw_E * (Filterd_Dw_E > 0).float()        
            temp = ((SDF > 1).float() * (torch.sum((Filterd_Dw_E_n.double() ** PSI_par) * (Filterd_Dw_E_n > 0).float(), dim=1) / (SDF + 1e-16))) ** (1 / PSI_par)        
            del_a_E = Gamma_E * del_a_E + Gamma_EM1 * (Filterd_Dw_E_n - temp.unsqueeze(1)).float()   
            W_veca[N_I, :] = W_veca[N_I, :] + C_I * del_b_I        
            W_veca[N_E, :] = (1 + de_f) * W_veca[N_E, :] + alpha_1 * (W_veca[N_E, :] * torch.tanh(desired_S - Filterd_Spikes)) + C_E * del_a_E
            W_veca = W_veca * (W_veca > 0).float()
            W_vecb = W_vecb * (W_vecb > 0).float()
            W_veca[N_E, :] = torch.clamp(W_veca[N_E, :], max=w_max_a)
            W_vecb[N_E, :] = torch.clamp(W_vecb[N_E, :], max=w_max_b)
            
            #Evaluating on the first pattern
            if np.mod(itrial, 50) == 0:      
                V_vec_test2 = (V_vec >= 1).float()
                V_vec_test1 = torch.zeros(em_id, N_Post, device=device)
                for io in range(1):
                        V_vec_test1[io, :] = torch.sum(V_vec_test2[intimeP_vec[io]:intimeP_vec[io] + Pat_L[io] + 150, :], dim=0)

                rank_val = torch.linalg.matrix_rank((V_vec_test1 > 0).float())   
                om.append(rank_val)
                Omega.append(om)

    #Training with the second pattern
    for itrial in range(6000):
        om = []
        for id in range(1):
            PAT1 = torch.rand(N_Pres, T, device=device)
            PAT1[N_E, :] = (PAT1[N_E, :] < (Rate[N_E].unsqueeze(1) * delt / 1000)).float()
            PAT1[N_I, :] = (PAT1[N_I, :] < (Rate[N_I].unsqueeze(1) * delt / 1000)).float()

            # Inserting the embedded pattern
            Pat_L = [PATP1[id].shape[1], PATP2[id].shape[1]]
            #PAT1[:, intimeP_vec[0]:intimeP_vec[0] + Pat_L[0]] = PATP1[id]
            PAT1[:, intimeP_vec[1]:intimeP_vec[1] + Pat_L[1]] = PATP2[id]
            
            WW_vec = W_veca * W_vecb
            WW_vec[N_I, :] = -WW_vec[N_I, :]

            # Compute the FFT of the input signals along the time dimension
            PAT1_fft = torch.fft.rfft(PAT1, n=n, dim=1)

            # Preallocate the output tensor for the convolution results.
            conv_result = torch.empty((N_Pres, n), device=device)
            elig_conv = torch.empty((N_Pres, n), device=device)


            # For excitatory neurons, multiply by the excitatory kernel FFT and invert.
            conv_result[N_E] = torch.fft.irfft(PAT1_fft[N_E] * kernelE_fft, n=n, dim=1)
            elig_conv[N_E] = torch.fft.irfft(PAT1_fft[N_E] * eligE_fft, n=n, dim=1)
            
            # For inhibitory neurons, multiply by the inhibitory kernel FFT and invert.
            conv_result[N_I] = torch.fft.irfft(PAT1_fft[N_I] * kernelI_fft, n=n, dim=1)
            elig_conv[N_I] = torch.fft.irfft(PAT1_fft[N_I] * eligI_fft, n=n, dim=1)
            
            inall = conv_result[:, :T]


            I_ext = torch.zeros(EpL, N_Post, device=device)
            INP = inall[:, :EpL]
            eligINP = torch.abs(elig_conv[:, :EpL])

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
            
            Filterd_Spikes = Gamma_star * Filterd_Spikes + Gamma_starM1 * torch.sum(num_spikes[:, tr:EpL], dim=1)

            del_b_I = Gamma_E * del_b_I + Gamma_EM1 * torch.matmul(eligINP[N_I, tr:EpL], V_vec[tr:EpL, :])
            Filterd_Dw_E = torch.matmul(eligINP[N_E, tr:EpL], (V_vec[tr:EpL, :] * (V_vec[tr:EpL, :] > 0).float()))
            Filterd_Dw_E = Filterd_Dw_E - torch.mean(Filterd_Dw_E, dim=0)
            del_b_E = Gamma_E * del_b_E + Gamma_EM1 * Filterd_Dw_E        
            W_vecb[N_I, :] = W_vecb[N_I, :] + C_I * del_b_I
            W_vecb[N_E, :] = (1 + de_f) * W_vecb[N_E, :] + alpha_1 * (W_vecb[N_E, :] * torch.tanh(desired_S - Filterd_Spikes)) + C_E * del_b_E        
            SDF = torch.sum((Filterd_Dw_E > 0).float(), dim=1)
            Filterd_Dw_E_n = Filterd_Dw_E * (Filterd_Dw_E > 0).float()        
            temp = ((SDF > 1).float() * (torch.sum((Filterd_Dw_E_n.double() ** PSI_par) * (Filterd_Dw_E_n > 0).float(), dim=1) / (SDF + 1e-16))) ** (1 / PSI_par)        
            del_a_E = Gamma_E * del_a_E + Gamma_EM1 * (Filterd_Dw_E_n - temp.unsqueeze(1)).float()   
            W_veca[N_I, :] = W_veca[N_I, :] + C_I * del_b_I        
            W_veca[N_E, :] = (1 + de_f) * W_veca[N_E, :] + alpha_1 * (W_veca[N_E, :] * torch.tanh(desired_S - Filterd_Spikes)) + C_E * del_a_E
            W_veca = W_veca * (W_veca > 0).float()
            W_vecb = W_vecb * (W_vecb > 0).float()
            W_veca[N_E, :] = torch.clamp(W_veca[N_E, :], max=w_max_a)
            W_vecb[N_E, :] = torch.clamp(W_vecb[N_E, :], max=w_max_b)
            
            #Evaluate on both patterns
            if np.mod(itrial, 50) == 0: 
                PAT1 = torch.rand(N_Pres, T, device=device)
                PAT1[N_E, :] = (PAT1[N_E, :] < (Rate[N_E].unsqueeze(1) * delt / 1000)).float()
                PAT1[N_I, :] = (PAT1[N_I, :] < (Rate[N_I].unsqueeze(1) * delt / 1000)).float()

                # Einfügen der eingebetteten Muster
                Pat_L = [PATP1[id].shape[1], PATP2[id].shape[1]]
                PAT1[:, intimeP_vec[0]:intimeP_vec[0] + Pat_L[0]] = PATP1[id]
                PAT1[:, intimeP_vec[1]:intimeP_vec[1] + Pat_L[1]] = PATP2[id]
                
                WW_vec = W_veca * W_vecb
                WW_vec[N_I, :] = -WW_vec[N_I, :]

                # Compute the FFT of the input signals along the time dimension
                PAT1_fft = torch.fft.rfft(PAT1, n=n, dim=1)

                # Preallocate the output tensor for the convolution results.
                conv_result = torch.empty((N_Pres, n), device=device)
                elig_conv = torch.empty((N_Pres, n), device=device)


                # For excitatory neurons, multiply by the excitatory kernel FFT and invert.
                conv_result[N_E] = torch.fft.irfft(PAT1_fft[N_E] * kernelE_fft, n=n, dim=1)
                elig_conv[N_E] = torch.fft.irfft(PAT1_fft[N_E] * eligE_fft, n=n, dim=1)
                
                # For inhibitory neurons, multiply by the inhibitory kernel FFT and invert.
                conv_result[N_I] = torch.fft.irfft(PAT1_fft[N_I] * kernelI_fft, n=n, dim=1)
                elig_conv[N_I] = torch.fft.irfft(PAT1_fft[N_I] * eligI_fft, n=n, dim=1)
                
                inall = conv_result[:, :T]


                I_ext = torch.zeros(EpL, N_Post, device=device)
                INP = inall[:, :EpL]
                eligINP = torch.abs(elig_conv[:, :EpL])

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


                V_vec_test2 = (V_vec >= 1).float()
                V_vec_test1 = torch.zeros(em_id, N_Post, device=device)
                for io in range(2):
                        V_vec_test1[io, :] = torch.sum(V_vec_test2[intimeP_vec[io]:intimeP_vec[io] + Pat_L[io] + 150, :], dim=0)

                rank_val = torch.linalg.matrix_rank((V_vec_test1 > 0).float())   
                om.append(rank_val / em_id)
                Omega.append(om)
    np.save(f"SaveDirectory/Omega_{firstword}_{secondword}_Speaker{Speaker}.npy", torch.tensor(Omega).cpu().numpy())
    np.save(f"SaveDirectory/Wa_{firstword}_{secondword}_Speaker{Speaker}.npy", W_veca.cpu().numpy())
    np.save(f"SaveDirectory/Wb_{firstword}_{secondword}_Speaker{Speaker}.npy", W_vecb.cpu().numpy())
    np.save(f"SaveDirectory/Spikes_{firstword}_{secondword}_Speaker{Speaker}.npy", Filterd_Spikes.cpu().numpy())
