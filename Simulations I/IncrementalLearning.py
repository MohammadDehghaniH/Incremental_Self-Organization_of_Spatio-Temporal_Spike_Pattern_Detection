# Written by: Lenny Müller (muellele@uni-bremen.de)
import torch
import time
import os
import numpy as np
from scipy import signal
def TestFunc():
    directory_name = 'NameforDirectory'
    path = 'PathToSaveDirectory'
    try:
        os.mkdir(f"{path}/{directory_name}")
    except:
        None
    os.chdir(f"{path}/{directory_name}")
    task_id = int(os.getenv('SGE_TASK_ID')) #Setting the value used to set the seed
    #task_id = 1
    torch.manual_seed(task_id * 1000)
    if torch.cuda.is_available():
        devicce = 'cuda:0'
    else:
        devicce = 'cpu'
    NN = 500    #Number of total neurons
    NE = 400    #Number of excitatory neurons
    NI = 100    #Number of inhibitory neurons
    sigma = 2   #Width of gaussian used for noise
    T = 1000    #Epoch length (ms)
    dta = 0.1   #Time discretization
    em = 7      #Number of embedded patterns
    N_psn = 7   #Number of postsynaptic neurons
    V_thr = 1   #Threshold for membrane potential
    V0 = 0      #Resting potential 
    tau_m = 15  #Membrane time constant
    R_m = 1     #Resistance
    tau_r_e = 0.5   #excitatory rise time
    tau_r_i = 1     #inhibitory rise time
    tau_d_e = 3     #excitatory decay time
    tau_d_i = 5     #inhibitory decay time
    rateE = 5       #firing rate excitatory neurons
    rateI = 20      #firing rate inhibitory neurons
    c_I = 1 * 1e-4  #inhibitory learning rate (scaled with discretization)
    c_E = 0.9 * 1e-4    #inhibitory learning rate (scaled with discretization)
    alpha = 1e-2    #synaptic scaling constant
    chi = 1e-4      #decay constant
    w_max = 1       #maximal weight strength
    desired_S = 2   #desired number of spikes per epoch

    #Generating the kernel K^tilde
    eta_e = tau_d_e/tau_r_e
    eta_i = tau_d_i/tau_r_i
    Inorm_e = eta_e**(eta_e/(eta_e-1))/(eta_e-1)
    Inorm_i = eta_i**(eta_i/(eta_i-1))/(eta_i-1)
    t = np.linspace(0, 200, int(200/dta))
    KernelI = -Inorm_i * ((tau_m - tau_d_i) * tau_r_i * np.exp(t/tau_m + t/tau_d_i) + ((tau_r_i*tau_d_i - tau_m * tau_d_i)* np.exp(t/tau_m) + (tau_m*tau_d_i - tau_m * tau_r_i)*np.exp(t/tau_d_i))*np.exp(t/tau_r_i)) * np.exp(-t/tau_r_i -t/tau_m - t/tau_d_i) / ((tau_m - tau_d_i) * (tau_r_i - tau_m)) # * tau_m for correctness?
    KernelE = -Inorm_e * ((tau_m - tau_d_e) * tau_r_e * np.exp(t/tau_m + t/tau_d_e) + ((tau_r_e*tau_d_e - tau_m * tau_d_e)* np.exp(t/tau_m) + (tau_m*tau_d_e - tau_m * tau_r_e)*np.exp(t/tau_d_e))*np.exp(t/tau_r_e)) * np.exp(-t/tau_r_e -t/tau_m - t/tau_d_e) / ((tau_m - tau_d_e) * (tau_r_e - tau_m)) # * tau_m for correctness?
    kernelI = torch.zeros((int(200/dta)), device=devicce)
    kernelI[:] = torch.from_numpy(KernelI).flip(0)
    kernelE = torch.zeros((int(200/dta)), device=devicce)
    kernelE[:] = torch.from_numpy(KernelE).flip(0)
    
    #reversal potential 
    revpot = torch.zeros((int(200/dta)), device = devicce)
    revpot[:] = -1*torch.exp(- torch.from_numpy(t) / tau_m)
    t = torch.linspace(0, T-dta, int(T/dta), device = devicce)
    
    #Other parameters
    trials = 30000
    patlength = int(50/dta)
    pattime = torch.empty(N_psn, dtype = int, device = devicce)#int(500/dta)
    pattime[:] = torch.linspace(int(patlength), int((T-sigma-tau_m)/dta - int(patlength)), N_psn, dtype=int)
    patlength = int(50/dta)
    length = int(T/dta) #number of timesteps

    #Gaussian used for noise
    gaussian = dta/(sigma*np.sqrt(2*torch.pi))*torch.exp(-1/2*((t-(T/2))/sigma)**2) 
    gauss = torch.empty((1, length), device = devicce)
    gauss[:] = gaussian
    kernel = gauss[:, gauss.shape[1]//2-100:gauss.shape[1]//2+100]

    traceE = torch.empty((1, length), device = devicce)
    traceE[:] = Inorm_e * (torch.exp(-(t)/tau_d_e)-torch.exp(-(t)/tau_r_e))
    traceE = traceE[:, :500].flip(1)
    #traceI = torch.array(traceE) # Experimental 
    traceI = torch.empty((1, length), device = devicce)
    traceI[:] = Inorm_i * (torch.exp(-(t)/tau_d_i)-torch.exp(-(t)/tau_r_i))
    traceI = traceI[:, :500].flip(1)

    #Generating the embedded pattern
    embedded_pattern = torch.empty((NN, patlength, em), device = devicce)
    embedded_pattern = embedded_pattern.uniform_(0, 1)
    embedded_pattern[:NI, :] = torch.where(embedded_pattern[:NI, :] <= rateI*dta/1000, 1, 0)
    embedded_pattern[NI:, :] = torch.where(embedded_pattern[NI:, :] <= rateE*dta/1000, 1, 0)

    eulerkernel = torch.empty(length, device = devicce)
    eulerkernel[:] = torch.Tensor(dta *(1 / tau_m * (1 - 1/ tau_m)**( t.flip(0))))
    zeros = torch.zeros((length - 1, N_psn), device = devicce)
    null = torch.zeros(1, device = devicce)

    #Initializing the weights
    a = torch.ones((NN, N_psn), device = devicce)
    a = torch.normal(1e-1, 1e-2, (NN, N_psn), device = devicce)
    b = torch.normal(1e-1, 1e-2, (NN, N_psn), device = devicce)
    a[torch.where(a < 0)] = 0
    b[torch.where(b < 0)] = 0

    #Initialize the variables needed
    filtered_elig = torch.zeros((NN, N_psn), device = devicce)
    filtered_spikes = torch.zeros((1, N_psn), device = devicce)

    spike_vec = torch.zeros((length, N_psn), device = devicce)
    spikes = []
    da = torch.empty((NN, N_psn), device = devicce)
    db = torch.empty((NN, N_psn), device = devicce)
    del_b_I = torch.zeros((NI, N_psn),device = devicce)
    del_a_E = torch.zeros((NE, N_psn), device = devicce)
    del_b_E = torch.zeros((NE, N_psn), device = devicce)
    num_spikes = torch.sum(spike_vec, axis = 0)
    filtered_spikes = 0.9 * filtered_spikes + 0.1 * num_spikes
    poisson = False #Should noise be used (Yes/No)

    #Initialization phase without embedded pattern
    for init in range(2000):
        input = torch.empty((NN, length), device = devicce)
        input = input.uniform_(0, 1, )
        input[:NI, :] = torch.where(input[:NI, :] <= rateI*dta/1000, 1, 0)
        input[NI:, :] = torch.where(input[NI:, :] <= rateE*dta/1000, 1, 0)
        #input[:, pattime[0] : pattime[0] + patlength] = embedded_pattern[:, :, 0]
        input = torch.concatenate((torch.zeros(NN, 2000-1, device = devicce), input), axis = 1)
        Input = torch.empty(1, NN, length, device = devicce)
        Input[:, NI:] = torch.nn.functional.conv1d(input[NI:].unsqueeze(0), kernelE.unsqueeze(0).repeat(NE, 1, 1), bias= None, groups=NE)
        Input[:, :NI] = torch.nn.functional.conv1d(input[:NI].unsqueeze(0), kernelI.unsqueeze(0).repeat(NI, 1, 1), bias= None, groups=NI)
        weights = (a * b)
        weights[:NI] *= -1
        I_ext = torch.empty((length, N_psn), device = devicce)
        for i in range(N_psn):
            I_ext[:, i] = (weights[:, i].unsqueeze(1) * Input[0]).sum(axis =  0).unsqueeze(0)
        V_vec = torch.zeros((length, N_psn), device = devicce)
        spike_vec = torch.zeros((length, N_psn), device = devicce)
        # for it in range(1, length):
        #     V_vec[it, :] = (1 - dta/tau_m) * V_vec[it-1, :] + I_ext[it, :] * dta/tau_m
        #     if (V_vec[it-1, :] >= V_thr).any():
        #         for jj in (torch.where(V_vec[it-1, :] >= V_thr)[0]):
        #             spike_vec[it-1, int(jj)] = 1
        #             V_vec[it, int(jj)] = 0
        #Here we use a faster way of computing the membrane potential than with the euler method
        V_vec[:, :] = I_ext
        while len(torch.where(V_vec > V_thr)[0]) != 0:
            tn, tspike = torch.where(V_vec.T > V_thr)
            V_vec[tspike[0], tn[0]] = 1
            spike_vec[tspike[0], tn[0]] = 1
            V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]] += revpot[:len(V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]])]

        #Computing the plasticity updates
        num_spikes = torch.sum(spike_vec, axis = 0)
        filtered_spikes = 0.9 * filtered_spikes + 0.1 * num_spikes
        Bmean = torch.empty((NE, N_psn), device = devicce)
        del_b_I = 0.99 * del_b_I + 0.01 *  (Input[0, :NI].unsqueeze(1).repeat(1, N_psn, 1) * V_vec.T).sum(axis = 2)
        filtered_dW_E = (Input[0, NI:].unsqueeze(1).repeat(1, N_psn, 1) * torch.where(V_vec.T > 0, V_vec.T, 0)).sum(axis = 2)
        filtered_dW_E = filtered_dW_E - torch.mean(filtered_dW_E, axis = 0)
        del_b_E = 0.99 * del_b_E + 0.01 * filtered_dW_E
        Bmean[:, :] = ((torch.heaviside(torch.sum((torch.heaviside(torch.where(filtered_dW_E> 0, filtered_dW_E, 0), null)), axis = 1)- 1 - 1e-13, null) / (torch.sum((torch.heaviside(torch.where(filtered_dW_E > 0, filtered_dW_E, 0), null)), axis = 1)+1e-13)).unsqueeze(1) * ((torch.sum(torch.where(filtered_dW_E > 0, filtered_dW_E, 0), axis = 1)).unsqueeze(1).repeat(1, N_psn)))
        B = torch.where(filtered_dW_E > 0, filtered_dW_E, 0) - Bmean
        del_a_E = 0.99 * del_a_E + 0.01 * B
        db[:NI] =  c_I * del_b_I# - chi_I * b[:NI]
        db[NI:] =  c_E * del_b_E + alpha * b[NI:] * torch.tanh(desired_S - filtered_spikes) - chi*b[NI:]
        da[:NI] = c_I * del_b_I
        da[NI:] = c_E * del_a_E + alpha * a[NI:] * torch.tanh(desired_S - filtered_spikes) - chi*a[NI:]

        #Apply plasticity and update the weights
        a += da
        a = torch.where(a > 0, a, 0)
        a = torch.where(a > w_max, w_max, a)
        b += db
        b = torch.where(b > 0, b, 0)
        b = torch.where(b > w_max, w_max, b)


    #First stage with one embedded pattern
    for patterns in range(2, 3):
        omega = torch.zeros((trials, patterns, N_psn), device = devicce) #Tracing Omega (every 50 learning cycles) 
        wvec = torch.empty((trials, NN, N_psn), device = devicce) #Tracing weights (every 50 learning cycles) 
        for cycle in range(trials):
            #print(cycle)
            input = torch.empty((NN, length), device = devicce)
            input = input.uniform_(0, 1, )
            input[:NI, :] = torch.where(input[:NI, :] <= rateI*dta/1000, 1, 0)
            input[NI:, :] = torch.where(input[NI:, :] <= rateE*dta/1000, 1, 0)
            for pat in range(patterns):
                input[:, pattime[pat] : pattime[pat] + patlength] = embedded_pattern[:, :, pat]
            if poisson == True:
                stochastic = torch.nn.functional.conv1d(input.unsqueeze(0), kernel.unsqueeze(0).repeat(NN, 1, 1), bias= None, padding="same", groups=NN)
                newinp = torch.empty((NN, length), device = devicce)
                newinp = newinp.uniform_(0, 1, )
                input = torch.where(newinp <= stochastic, 1., 0.)[0]
            input = torch.concatenate((torch.zeros(NN, 2000-1, device = devicce), input), axis = 1)
            Input = torch.empty(1, NN, length, device = devicce)
            Input[:, NI:] = torch.nn.functional.conv1d(input[NI:].unsqueeze(0), kernelE.unsqueeze(0).repeat(NE, 1, 1), bias= None, groups=NE)
            Input[:, :NI] = torch.nn.functional.conv1d(input[:NI].unsqueeze(0), kernelI.unsqueeze(0).repeat(NI, 1, 1), bias= None, groups=NI)
            weights = (a * b)
            weights[:NI] *= -1
            I_ext = torch.empty((length, N_psn), device = devicce)
            for i in range(N_psn):
                I_ext[:, i] = (weights[:, i].unsqueeze(1) * Input[0]).sum(axis =  0).unsqueeze(0)
            V_vec = torch.zeros((length, N_psn), device = devicce)
            spike_vec = torch.zeros((length, N_psn), device = devicce)
            # for it in range(1, length):
            #     V_vec[it, :] = (1 - dta/tau_m) * V_vec[it-1, :] + I_ext[it, :] * dta/tau_m
            #     if (V_vec[it-1, :] >= V_thr).any():
            #         for jj in (torch.where(V_vec[it-1, :] >= V_thr)[0]):
            #             spike_vec[it-1, int(jj)] = 1
            #             V_vec[it, int(jj)] = 0
            V_vec[:, :] = I_ext
            while len(torch.where(V_vec > V_thr)[0]) != 0:
                tn, tspike = torch.where(V_vec.T > V_thr)
                V_vec[tspike[0], tn[0]] = 1
                spike_vec[tspike[0], tn[0]] = 1
                V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]] += revpot[:len(V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]])]
                
            num_spikes = torch.sum(spike_vec, axis = 0)
            filtered_spikes = 0.9 * filtered_spikes + 0.1 * num_spikes
            Bmean = torch.empty((NE, N_psn), device = devicce)
            del_b_I = 0.99 * del_b_I + 0.01 *  (Input[0, :NI].unsqueeze(1).repeat(1, N_psn, 1) * V_vec.T).sum(axis = 2)
            filtered_dW_E = (Input[0, NI:].unsqueeze(1).repeat(1, N_psn, 1) * torch.where(V_vec.T > 0, V_vec.T, 0)).sum(axis = 2)
            filtered_dW_E = filtered_dW_E - torch.mean(filtered_dW_E, axis = 0)
            del_b_E = 0.99 * del_b_E + 0.01 * filtered_dW_E
            Bmean[:, :] = ((torch.heaviside(torch.sum((torch.heaviside(torch.where(filtered_dW_E> 0, filtered_dW_E, 0), null)), axis = 1)- 1 - 1e-13, null) / (torch.sum((torch.heaviside(torch.where(filtered_dW_E > 0, filtered_dW_E, 0), null)), axis = 1)+1e-13)).unsqueeze(1) * ((torch.sum(torch.where(filtered_dW_E > 0, filtered_dW_E, 0), axis = 1)).unsqueeze(1).repeat(1, N_psn)))
            B = torch.where(filtered_dW_E > 0, filtered_dW_E, 0) - Bmean
            del_a_E = 0.99 * del_a_E + 0.01 * B
            db[:NI] =  c_I * del_b_I
            db[NI:] =  c_E * del_b_E + alpha * b[NI:] * torch.tanh(desired_S - filtered_spikes) - chi * b[NI:]
            da[:NI] = c_I * del_b_I
            da[NI:] = c_E * del_a_E + alpha * a[NI:] * torch.tanh(desired_S - filtered_spikes) - chi * a[NI:]
            a += da
            a = torch.where(a > 0, a, 0)
            a = torch.where(a > w_max, w_max, a)
            b += db
            b = torch.where(b > 0, b, 0)
            b = torch.where(b > w_max, w_max, b)


            spikes.append(num_spikes.cpu().numpy())
            #Every 50 epochs we evaluate the networks performance on all so far embedded patterns
            if np.mod(cycle, 50) == 0:
                wvec[cycle] = weights
                lat_input = torch.zeros((N_psn, N_psn, length), device=devicce)
                input = torch.empty((NN, length), device = devicce)
                input = input.uniform_(0, 1, )
                input[:NI, :] = torch.where(input[:NI, :] <= rateI*dta/1000, 1, 0)
                input[NI:, :] = torch.where(input[NI:, :] <= rateE*dta/1000, 1, 0)
                for pat in range(patterns):
                    input[:, pattime[pat] : pattime[pat] + patlength] = embedded_pattern[:, :, pat]
                # if poisson == True:
                #     stochastic = torch.nn.functional.conv1d(input.unsqueeze(0), kernel.unsqueeze(0).repeat(NN, 1, 1), bias= None, padding="same", groups=NN)
                #     newinp = torch.empty((NN, length), device = devicce)
                #     newinp = newinp.uniform_(0, 1, )
                #     input = torch.where(newinp <= stochastic, 1., 0.)[0]
                input = torch.concatenate((torch.zeros(NN, 2000-1, device = devicce), input), axis = 1)
                Input = torch.empty(1, NN, length, device = devicce)
                Input[:, NI:] = torch.nn.functional.conv1d(input[NI:].unsqueeze(0), kernelE.unsqueeze(0).repeat(NE, 1, 1), bias= None, groups=NE)
                Input[:, :NI] = torch.nn.functional.conv1d(input[:NI].unsqueeze(0), kernelI.unsqueeze(0).repeat(NI, 1, 1), bias= None, groups=NI)
                #Input = torch.concatenate((torch.zeros(1, NN, 250, device = devicce), Input), axis = 2)[:, :, :length]
                weights = (a * b)
                weights[:NI] *= -1
                I_ext = torch.empty((length, N_psn), device = devicce)
                for i in range(N_psn):
                    I_ext[:, i] = (weights[:, i].unsqueeze(1) * Input[0]).sum(axis =  0).unsqueeze(0)
                V_vec = torch.zeros((length, N_psn), device = devicce)
                spike_vec = torch.zeros((length, N_psn), device = devicce)
                # for it in range(1, length):
                #     V_vec[it, :] = (1 - dta/tau_m) * V_vec[it-1, :] + I_ext[it, :] * dta/tau_m
                #     if (V_vec[it-1, :] >= V_thr).any():
                #         for jj in (torch.where(V_vec[it-1, :] >= V_thr)[0]):
                #             spike_vec[it-1, int(jj)] = 1
                #             V_vec[it, int(jj)] = 0
                V_vec[:, :] = I_ext
                while len(torch.where(V_vec > V_thr)[0]) != 0:
                    tn, tspike = torch.where(V_vec.T > V_thr)
                    V_vec[tspike[0], tn[0]] = 1
                    spike_vec[tspike[0], tn[0]] = 1
                    V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]] += revpot[:len(V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]])]
                om = torch.empty((patterns, N_psn), device = devicce)
                for pat in range(patterns):
                    om[pat] = (torch.where(torch.sum(spike_vec[pattime[pat] - int(sigma/dta): pattime[pat] + patlength + int(tau_m/dta) + int(sigma/dta)], axis = 0) != 0, 1, 0))
                omega[cycle] = om
    
        if patterns == 2:
            try:
                os.mkdir(f"{path}/{directory_name}/0+{patterns}/")
            except:
                None
            np.save(f"{path}/{directory_name}/0+{patterns}/omega_{task_id}.npy", omega.cpu().numpy())
            np.save(f"{path}/{directory_name}/0+{patterns}/spikes_{task_id}.npy", spikes)
            np.save(f"{path}/{directory_name}/0+{patterns}/b_{task_id}.npy", b.cpu().numpy())
            np.save(f"{path}/{directory_name}/0+{patterns}/a_{task_id}.npy", a.cpu().numpy())
            np.save(f"{path}/{directory_name}/0+{patterns}/wvec_{task_id}.npy", wvec.cpu().numpy())
            np.save(f"{path}/{directory_name}/0+{patterns}/embedded_patterns_{task_id}.npy", embedded_pattern.cpu().numpy())

        else:
            try:
                os.mkdir(f"{path}/{directory_name}/{patterns - 1}+1/")
            except:
                None
            np.save(f"{path}/{directory_name}/{patterns - 1}+1/omega_{task_id}.npy", omega.cpu().numpy())
            np.save(f"{path}/{directory_name}/{patterns - 1}+1/spikes_{task_id}.npy", spikes)
            np.save(f"{path}/{directory_name}/{patterns - 1}+1/b_{task_id}.npy", b.cpu().numpy())
            np.save(f"{path}/{directory_name}/{patterns - 1}+1/a_{task_id}.npy", a.cpu().numpy())
            np.save(f"{path}/{directory_name}/{patterns - 1}+1/wvec_{task_id}.npy", wvec.cpu().numpy())

    #used to compute the cosine between the weight vectors
    a02 = torch.zeros((NN, N_psn), device = devicce)
    a02[:, :] = torch.from_numpy(np.load(f"{path}/{directory_name}/0+2/a_{task_id}.npy"))
    b02 = torch.zeros((NN, N_psn), device = devicce)
    b02[:, :] = torch.from_numpy(np.load(f"{path}/{directory_name}/0+2/b_{task_id}.npy"))
    w02 = a02*b02

    #Next stages of one more embedded pattern without showing the previous ones
    for patterns in range(2, em):
        wvec = torch.empty((trials, NN, N_psn), device=devicce)
        omega = torch.zeros((trials, patterns+1, N_psn), device = devicce)
        cosine = torch.empty((trials, N_psn), device = devicce)
        for cycle in range(trials):
            #print(cycle)
            lat_input = torch.zeros((N_psn, N_psn, length), device=devicce)
            input = torch.empty((NN, length), device = devicce)
            input = input.uniform_(0, 1, )
            input[:NI, :] = torch.where(input[:NI, :] <= rateI*dta/1000, 1, 0)
            input[NI:, :] = torch.where(input[NI:, :] <= rateE*dta/1000, 1, 0)
            input[:, pattime[patterns] : pattime[patterns] + patlength] = embedded_pattern[:, :, patterns]
            if poisson == True:
                stochastic = torch.nn.functional.conv1d(input.unsqueeze(0), kernel.unsqueeze(0).repeat(NN, 1, 1), bias= None, padding="same", groups=NN)
                newinp = torch.empty((NN, length), device = devicce)
                newinp = newinp.uniform_(0, 1, )
                input = torch.where(newinp <= stochastic, 1., 0.)[0]
            input = torch.concatenate((torch.zeros(NN, 2000-1, device = devicce), input), axis = 1)
            Input = torch.empty(1, NN, length, device = devicce)
            Input[:, NI:] = torch.nn.functional.conv1d(input[NI:].unsqueeze(0), kernelE.unsqueeze(0).repeat(NE, 1, 1), bias= None, groups=NE)
            Input[:, :NI] = torch.nn.functional.conv1d(input[:NI].unsqueeze(0), kernelI.unsqueeze(0).repeat(NI, 1, 1), bias= None, groups=NI)
            weights = (a * b)
            weights[:NI] *= -1
            I_ext = torch.empty((length, N_psn), device = devicce)
            for i in range(N_psn):
                I_ext[:, i] = (weights[:, i].unsqueeze(1) * Input[0]).sum(axis =  0).unsqueeze(0)
            V_vec = torch.zeros((length, N_psn), device = devicce)
            spike_vec = torch.zeros((length, N_psn), device = devicce)
            # for it in range(1, length):
            #     V_vec[it, :] = (1 - dta/tau_m) * V_vec[it-1, :] + I_ext[it, :] * dta/tau_m
            #     if (V_vec[it-1, :] >= V_thr).any():
            #         for jj in (torch.where(V_vec[it-1, :] >= V_thr)[0]):
            #             spike_vec[it-1, int(jj)] = 1
            #             V_vec[it, int(jj)] = 0
            V_vec[:, :] = I_ext
            while len(torch.where(V_vec > V_thr)[0]) != 0:
                tn, tspike = torch.where(V_vec.T > V_thr)
                V_vec[tspike[0], tn[0]] = 1
                spike_vec[tspike[0], tn[0]] = 1
                V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]] += revpot[:len(V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]])]
                
            num_spikes = torch.sum(spike_vec, axis = 0)
            filtered_spikes = 0.9 * filtered_spikes + 0.1 * num_spikes
            Bmean = torch.empty((NE, N_psn), device = devicce)
            del_b_I = 0.99 * del_b_I + 0.01 *  (Input[0, :NI].unsqueeze(1).repeat(1, N_psn, 1) * V_vec.T).sum(axis = 2)
            filtered_dW_E = (Input[0, NI:].unsqueeze(1).repeat(1, N_psn, 1) * torch.where(V_vec.T > 0, V_vec.T, 0)).sum(axis = 2)
            filtered_dW_E = filtered_dW_E - torch.mean(filtered_dW_E, axis = 0)
            del_b_E = 0.99 * del_b_E + 0.01 * filtered_dW_E
            Bmean[:, :] = ((torch.heaviside(torch.sum((torch.heaviside(torch.where(filtered_dW_E> 0, filtered_dW_E, 0), null)), axis = 1)- 1 - 1e-13, null) / (torch.sum((torch.heaviside(torch.where(filtered_dW_E > 0, filtered_dW_E, 0), null)), axis = 1)+1e-13)).unsqueeze(1) * ((torch.sum(torch.where(filtered_dW_E > 0, filtered_dW_E, 0), axis = 1)).unsqueeze(1).repeat(1, N_psn)))
            B = torch.where(filtered_dW_E > 0, filtered_dW_E, 0) - Bmean
            del_a_E = 0.99 * del_a_E + 0.01 * B
            db[:NI] =  c_I * del_b_I
            db[NI:] =  c_E * del_b_E + alpha * b[NI:] * torch.tanh(desired_S - filtered_spikes) - chi * b[NI:]
            da[:NI] = c_I * del_b_I
            da[NI:] = c_E * del_a_E + alpha * a[NI:] * torch.tanh(desired_S - filtered_spikes) - chi * a[NI:]
            a += da
            a = torch.where(a > 0, a, 0)
            a = torch.where(a > w_max, w_max, a)
            b += db
            b = torch.where(b > 0, b, 0)
            b = torch.where(b > w_max, w_max, b)

            spikes.append(num_spikes.cpu().numpy())

            #Evaluation on all patterns
            if np.mod(cycle, 50) == 0:
                wvec[cycle] = weights
                lat_input = torch.zeros((N_psn, N_psn, length), device=devicce)
                input = torch.empty((NN, length), device = devicce)
                input = input.uniform_(0, 1, )
                input[:NI, :] = torch.where(input[:NI, :] <= rateI*dta/1000, 1, 0)
                input[NI:, :] = torch.where(input[NI:, :] <= rateE*dta/1000, 1, 0)
                for pat in range(patterns+1):
                    input[:, pattime[pat] : pattime[pat] + patlength] = embedded_pattern[:, :, pat]
                # if poisson == True:
                #     stochastic = torch.nn.functional.conv1d(input.unsqueeze(0), kernel.unsqueeze(0).repeat(NN, 1, 1), bias= None, padding="same", groups=NN)
                #     newinp = torch.empty((NN, length), device = devicce)
                #     newinp = newinp.uniform_(0, 1, )
                #     input = torch.where(newinp <= stochastic, 1., 0.)[0]
                input = torch.concatenate((torch.zeros(NN, 2000-1, device = devicce), input), axis = 1)
                Input = torch.empty(1, NN, length, device = devicce)
                Input[:, NI:] = torch.nn.functional.conv1d(input[NI:].unsqueeze(0), kernelE.unsqueeze(0).repeat(NE, 1, 1), bias= None, groups=NE)
                Input[:, :NI] = torch.nn.functional.conv1d(input[:NI].unsqueeze(0), kernelI.unsqueeze(0).repeat(NI, 1, 1), bias= None, groups=NI)
                weights = (a * b)
                weights[:NI] *= -1
                I_ext = torch.empty((length, N_psn), device = devicce)
                for i in range(N_psn):
                    I_ext[:, i] = (weights[:, i].unsqueeze(1) * Input[0]).sum(axis =  0).unsqueeze(0)
                V_vec = torch.zeros((length, N_psn), device = devicce)
                spike_vec = torch.zeros((length, N_psn), device = devicce)
                # for it in range(1, length):
                #     V_vec[it, :] = (1 - dta/tau_m) * V_vec[it-1, :] + I_ext[it, :] * dta/tau_m
                #     if (V_vec[it-1, :] >= V_thr).any():
                #         for jj in (torch.where(V_vec[it-1, :] >= V_thr)[0]):
                #             spike_vec[it-1, int(jj)] = 1
                #             V_vec[it, int(jj)] = 0
                V_vec[:, :] = I_ext
                while len(torch.where(V_vec > V_thr)[0]) != 0:
                    tn, tspike = torch.where(V_vec.T > V_thr)
                    V_vec[tspike[0], tn[0]] = 1
                    spike_vec[tspike[0], tn[0]] = 1
                    V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]] += revpot[:len(V_vec[tspike[0] + 1 : tspike[0] + 1 + int(200/dta), tn[0]])]
            
                om = torch.empty((patterns+1, N_psn), device = devicce)
                for pat in range(patterns+1):
                    om[pat] = (torch.where(torch.sum(spike_vec[pattime[pat] - int(sigma / dta): pattime[pat] + patlength + int(tau_m/dta) +int(sigma/dta)], axis = 0) != 0, 1, 0))
                omega[cycle] = om
                for i in range(N_psn):
                    cosine[cycle, i] = torch.dot(weights[NI:, i], w02[NI:, i])/(torch.linalg.norm(weights[NI:, i])*torch.linalg.norm(w02[NI:, i]))

        try:
            os.mkdir(f"{path}/{directory_name}/{patterns}+1/")
        except:
            None
        np.save(f"{path}/{directory_name}/{patterns}+1/omega_{task_id}.npy", omega.cpu().numpy())
        np.save(f"{path}/{directory_name}/{patterns}+1/spikes_{task_id}.npy", spikes)
        np.save(f"{path}/{directory_name}/{patterns}+1/b_{task_id}.npy", b.cpu().numpy())
        np.save(f"{path}/{directory_name}/{patterns}+1/a_{task_id}.npy", a.cpu().numpy())
        np.save(f"{path}/{directory_name}/{patterns}+1/wvec_{task_id}.npy", wvec.cpu().numpy())
        np.save(f"{path}/{directory_name}/{patterns}+1/cosine_{task_id}.npy", cosine.cpu().numpy())
        a02 = torch.zeros((NN, N_psn), device = devicce)
        a02[:, :] = torch.from_numpy(np.load(f"{path}/{directory_name}/{patterns}+1/a_{task_id}.npy"))
        b02 = torch.zeros((NN, N_psn), device = devicce)
        b02[:, :] = torch.from_numpy(np.load(f"{path}/{directory_name}/{patterns}+1/b_{task_id}.npy"))
        w02 = a02*b02


