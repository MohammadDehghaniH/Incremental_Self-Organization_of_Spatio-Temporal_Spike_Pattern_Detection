function Model_NET_Incremental(p1,p2)
%p1: seed for random number and p1=1:1:\mu. There are \mu independent simulations
%p2: N_Post: number of post-synaptic neurons. (p2=1: N_Post=7);(p2=1: N_Post=8)
%w =ab, w: W_vec, a: W_veca, b: W_vecb

SDF1=10+(p1-1)*1e6;
rng(SDF1);

N_Post_vec=[4,14,21];
N_Post=N_Post_vec(p2); %number of post-synaptic neurons

Ntrials0 =2000;
%%Step 1: There is no embedded pattern in afferents for 1e4 learning cycles.

Ntrials2 =30000; 
%%Step 2: every embedded pattern is in afferents for 3e4 learning
%%cycles. There is pre-synaptic competition.


desired_S = 2; %desired number of spikes. i.e. r = 2/1 Hz epoch leangth = 1 sec
em_vec=[4,11,17];
em=em_vec(p2); % maximum number of embedded patterns

TTE = 1000; %epoch leanth [all in ms]
delt = 0.1; % integration step.
EpL = TTE/delt; 
times = 1:EpL; % discretization
times = times*delt;

%Parameters

N_Pres = 500;  %number of input neurons
percI = 0.2; %percentage of inhibition
NI = fix(percI*N_Pres); %number of inhibitory neurons

% Generating kernels for excitatory and inhibitory inputs

taumem = 15;   %membrane time constant
tautraceE =3;  %Rise time of excitatory currents
tautraceI = 5; %Decay time of inhibitory currents
tauriseE =0.5; %Decay time of excitatory currents
tauriseI = 1;  %Rise time of inhibitory currents

traceE = exp(-times/tautraceE) - exp(-times/tauriseE);
ettaE=tautraceE/tauriseE;
VnormE = (ettaE.^(ettaE/(ettaE - 1)))/(ettaE-1);
traceE = traceE*VnormE;
traceI = exp(-times/tautraceI) - exp(-times/tauriseI);
ettaI=tautraceI/tauriseI;
VnormI = (ettaI.^(ettaI/(ettaI - 1)))/(ettaI-1);
traceI = traceI*VnormI;

%rates
rateE = 5; %in [Hz] rate of excitatory neuron
rateI = 20 ;%in [Hz] rate of inhibitory neuron

% initial weight vectors
w_max = 1; % maximum amount of excitatory synapsis.

W_veca = 0.1+1e-2*randn(N_Pres,N_Post);
W_veca(W_veca<0)=0;

W_vecb = 0.1+1e-2*randn(N_Pres,N_Post);
W_vecb(W_vecb<0)=0;

%the embedded patterns
Pat_L=50/delt; %pattern length (ms/delt= steps!)
PAT1 = rand(N_Pres, Pat_L,em);
PAT1(1:NI,:,:) = PAT1(1:NI,:,:) < (rateI*delt/1000);
PAT1(NI+1:N_Pres,:,:) = PAT1(NI+1:N_Pres,:,:) < (rateE*delt/1000);

eval(['save PATTERNS/Poisson_1_PRE_HSP_Sig_2/EP_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat PAT1']);


intimeP_vec1=(1./delt)*(200:120:920);
intimeP_vec=intimeP_vec1(randperm(length(intimeP_vec1)));




%Learning rates
de_f=-1e-4;
alpha_1=1e-2; %scaling factor 
C_E=0.9e-4; %excitatory learning rate
C_I = 1e-4;%inhibitory learning rate
tr=200;%transient time
dta=delt/taumem; %
dtaM=(1- dta);
resy = 0.0; %resting potential 
threshy = 1; % threshold potential


%initializations: 
del_b_E  =  zeros(N_Pres-NI,N_Post); %\del b_E
del_a_E  =  zeros(N_Pres-NI,N_Post); %\del a_E
del_b_I  = zeros(NI,N_Post);     %\del b_I and \del a_I
Filterd_Spikes=zeros(1,N_Post);%long-time firing rate

% low-pass filter of spikes
Gamma_star= 0.9; 
Gamma_starM1=1-Gamma_star;

% low-pass filter of eligibility
Gamma_E= 0.99; 
Gamma_EM1=1-Gamma_E;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigha=2; %standard deviation
traceG=normpdf(times-500,0,sigha); %Gaussian kernel
sig_sig=sigha*10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Number of embedded patterns displayed after the neurons learn to respond
%to input with no embedded patterns.
N_AN=2;

for itrial=1:Ntrials0
    
        [I_ext,INP]=Model_Input_NET_WITHOUT_EM(rateI,traceI,rateE,traceE,N_Pres,NI,N_Post,EpL,delt,W_veca.*W_vecb);
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        %Neuron Model
        num_spikes=zeros(N_Post,EpL);
        V_vec=zeros(EpL,N_Post);
        for it = 2:EpL
            V_vec(it,:) = (dtaM*V_vec(it-1,:)) + (I_ext(it-1,:)*dta);
            f_p = find(V_vec(it-1,:) >= threshy);
            if ~isempty(f_p)
                V_vec(it,f_p) =resy; 
                num_spikes(f_p,it) = num_spikes(f_p) + 1; 
            end
        end
        Filterd_Spikes = Gamma_star*Filterd_Spikes + Gamma_starM1*sum(num_spikes(:,tr:EpL),2)';   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        del_b_I = Gamma_E*del_b_I + Gamma_EM1*(INP(1:NI,tr:EpL)*(V_vec(tr:EpL,:)));  

        Filterd_Dw_E= (INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)))...
            -mean((INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0))));
        del_b_E = Gamma_E*del_b_E+ Gamma_EM1*Filterd_Dw_E;

        W_vecb(1:NI,:)=W_vecb(1:NI,:) + C_I*del_b_I;
        W_vecb(NI+1:N_Pres,:)=(1+de_f)*W_vecb(NI+1:N_Pres,:)+...
            alpha_1*(W_vecb(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
            C_E.*del_b_E;
        
        SDF=sum((Filterd_Dw_E>0),2);
        del_a_E = Gamma_E*del_a_E+ Gamma_EM1*((Filterd_Dw_E.*(Filterd_Dw_E>0))-((SDF>1).*sum((Filterd_Dw_E.*(Filterd_Dw_E>0)),2)./(SDF+1e-16)));

        W_veca(1:NI,:)=W_veca(1:NI,:) + C_I*del_b_I;
        W_veca(NI+1:N_Pres,:)=(1+de_f)*W_veca(NI+1:N_Pres,:)+...
            alpha_1*(W_veca(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
            C_E.*del_a_E;

        W_veca = W_veca.*(W_veca > 0.);
        W_vecb = W_vecb.*(W_vecb > 0.);

        W_veca(NI+1:N_Pres,:)=W_veca(NI+1:N_Pres,:).*(W_veca(NI+1:N_Pres,:) <= w_max) + w_max*(W_veca(NI+1:N_Pres,:) > w_max);
        W_vecb(NI+1:N_Pres,:)=W_vecb(NI+1:N_Pres,:).*(W_vecb(NI+1:N_Pres,:) <= w_max) + w_max*(W_vecb(NI+1:N_Pres,:) > w_max);

end

eval(['save EP_0/Poisson_1_PRE_HSP_Sig_2/W_A/test_Wa_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat W_veca']);
eval(['save EP_0/Poisson_1_PRE_HSP_Sig_2/W_B/test_Wb_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat W_vecb']);
eval(['save EP_0/Poisson_1_PRE_HSP_Sig_2/Filterd_Spikes/Filterd_Spikes_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat Filterd_Spikes']);
eval(['save EP_0/Poisson_1_PRE_HSP_Sig_2/del_aE/del_a_E_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat del_a_E']);
eval(['save EP_0/Poisson_1_PRE_HSP_Sig_2/del_abI/del_b_I_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat del_b_I']);
eval(['save EP_0/Poisson_1_PRE_HSP_Sig_2/del_bE/del_b_E_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat del_b_E']);



for em_id=N_AN:em
    
    if em_id ==N_AN
        PAT2=PAT1(:,:,1:N_AN);
        em_ll=N_AN;
    else
        PAT2=PAT1(:,:,em_id);
        em_ll=1;
    end
    PAT3=PAT1(:,:,1:em_id);
    Omega_R_test=zeros(1,Ntrials2);

for itrial=1:Ntrials2
        [I_ext,INP]=Model_Input_NET_WITH_EM(rateI,traceI,rateE,traceE,N_Pres,NI,N_Post,EpL,delt,W_veca.*W_vecb,intimeP_vec,em_ll,PAT2,Pat_L,traceG);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
        %Neuron Model
        num_spikes=zeros(N_Post,EpL);
        V_vec=zeros(EpL,N_Post);

        for it = 2:EpL
            V_vec(it,:) = (dtaM*V_vec(it-1,:)) + (I_ext(it-1,:)*dta);
            f_p = find(V_vec(it-1,:) >= threshy);
            if ~isempty(f_p)
                V_vec(it,f_p) =resy; 
                num_spikes(f_p,it) = num_spikes(f_p) + 1; 
            end
        end

        Filterd_Spikes = Gamma_star*Filterd_Spikes + Gamma_starM1*sum(num_spikes(:,tr:EpL),2)';   
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        del_b_I = Gamma_E*del_b_I + Gamma_EM1*(INP(1:NI,tr:EpL)*(V_vec(tr:EpL,:)));  

        Filterd_Dw_E= (INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)))...
        -mean((INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0))));
        del_b_E = Gamma_E*del_b_E+ Gamma_EM1*Filterd_Dw_E;

        W_vecb(1:NI,:)=W_vecb(1:NI,:) + C_I*del_b_I;
        W_vecb(NI+1:N_Pres,:)=(1+de_f)*W_vecb(NI+1:N_Pres,:)+...
        alpha_1*(W_vecb(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
        C_E.*del_b_E;

        SDF=sum((Filterd_Dw_E>0),2);
        del_a_E = Gamma_E*del_a_E+ Gamma_EM1*((Filterd_Dw_E.*(Filterd_Dw_E>0))-((SDF>1).*sum((Filterd_Dw_E.*(Filterd_Dw_E>0)),2)./(SDF+1e-16)));

        W_veca(1:NI,:)=W_veca(1:NI,:) + C_I*del_b_I;
        W_veca(NI+1:N_Pres,:)=(1+de_f)*W_veca(NI+1:N_Pres,:)+...
        alpha_1*(W_veca(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
        C_E.*del_a_E;

        W_veca = W_veca.*(W_veca > 0.);
        W_vecb = W_vecb.*(W_vecb > 0.);

        W_veca(NI+1:N_Pres,:)=W_veca(NI+1:N_Pres,:).*(W_veca(NI+1:N_Pres,:) <= w_max) + w_max*(W_veca(NI+1:N_Pres,:) > w_max);
        W_vecb(NI+1:N_Pres,:)=W_vecb(NI+1:N_Pres,:).*(W_vecb(NI+1:N_Pres,:) <= w_max) + w_max*(W_vecb(NI+1:N_Pres,:) > w_max);

        if rem(itrial,50)==0
            
            [I_ext_test]=Model_Input_NET_WITH_EM_ORIGINAL(rateI,traceI,rateE,traceE,N_Pres,NI,N_Post,EpL,delt,W_veca.*W_vecb,intimeP_vec,em_id,PAT3,Pat_L,traceG);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
            %Neuron Model
            V_vec_test=zeros(EpL,N_Post);

            for it = 2:EpL
                V_vec_test(it,:) = (dtaM*V_vec_test(it-1,:)) + (I_ext_test(it-1,:)*dta);
                f_p = find(V_vec_test(it-1,:) >= threshy);
                if ~isempty(f_p)
                    V_vec_test(it,f_p) =resy; 
                end
            end

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Neuron model
            V_vec_test2=V_vec_test>=1;
            V_vec_test1=zeros(em_id,N_Post);
            for io=1:em_id
                V_vec_test1(io,:)=sum(V_vec_test2(intimeP_vec(io)+1-sig_sig:intimeP_vec(io)+Pat_L+150+sig_sig,:),1);
            end
            Omega_R_test(1,itrial)=rank(double(V_vec_test1>0))/(em_id);  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        
        end

end

eval(['save EP_',num2str(em_id,'%.i') '/Poisson_1_PRE_HSP_Sig_2/W_A/test_Wa_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat W_veca']);
eval(['save EP_',num2str(em_id,'%.i') '/Poisson_1_PRE_HSP_Sig_2/W_B/test_Wb_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat W_vecb']);
eval(['save EP_',num2str(em_id,'%.i') '/Poisson_1_PRE_HSP_Sig_2/Omega_Vec/test_Omega_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat Omega_R_test']);
eval(['save EP_',num2str(em_id,'%.i') '/Poisson_1_PRE_HSP_Sig_2/Filterd_Spikes/Filterd_Spikes_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat Filterd_Spikes']);
eval(['save EP_',num2str(em_id,'%.i') '/Poisson_1_PRE_HSP_Sig_2/del_aE/del_a_E_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat del_a_E']);
eval(['save EP_',num2str(em_id,'%.i') '/Poisson_1_PRE_HSP_Sig_2/del_abI/del_b_I_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat del_b_I']);
eval(['save EP_',num2str(em_id,'%.i') '/Poisson_1_PRE_HSP_Sig_2/del_bE/del_b_E_',num2str(p1),'_',num2str(N_Post),'_',num2str(p2),  '.mat del_b_E']);

    
end
end




