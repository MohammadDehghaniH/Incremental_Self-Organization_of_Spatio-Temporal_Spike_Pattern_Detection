function Model_One_With_PreHSP(p1,p2)


%p1: seed for random number and p1=1:1:\mu. 
%There are \mu independent simulations (500)
%p2: paEpLern length. p2=1 is for 50 ms;  p2=2 is for 100 ms; p2=3 is for 300 ms

rng(10+(p1-1)*1e6);

%There are 5 steps in this program.  
%%Step 1: There is no embedded paEpLern in afferents for 2000 learning cycles.

%%Step 2: There is one embedded paEpLern in afferents for 10000 learning cycles.

%%Step 3: There is no embedded paEpLern in afferents for 20000 learning cycles.

%%Step 4: There is the embedded paEpLern (from step 2) in afferents for 20000 learning cycles.

%%step 5: There is a new embedded paEpLern in afferents for 10000 learning cycles.
%%(initial conditions are from step 2)

%Data from this program is used in other programs to plot figures. 

Ntrials_1=2000; 
%Ntrials_1: number of learning cycles to learn the background.
%There is no embedded paEpLern in afferents for 2000 learning cycles. (step 1)

Ntrials_2=10000; 
%Ntrials_2: number of learning cycles to learn the background containing the embedded paEpLern (step 2).
%(Note initial conditions in this step are from step1)

Ntrials_3=20000; 
%Ntrials_3: number of learning cycles to learn the background. 
%There is no embedded paEpLern in afferents.
%(Note initial conditions in this step are from step 2)

Ntrials_4=20000; 
%Ntrials_4: number of learning cycles to learn the background containing the embedded paEpLern (step 4).
%(Note initial conditions in this step are from step 3)


Ntrials_5=10000; 
%Ntrials_2: number of learning cycles to learn the background containing a new embedded paEpLern (step 5).
%(Note initial conditions in this step are from step2)


%Spikess: vector of ouPat_Lut spikes.
%ns_0: vector of ouPat_Lut spikes in embedded paEpLern duration
%ns_15:  vector of ouPat_Lut spikes in embedded pater duration + 15ms
%Norm_W: the norm of weight vector in each learning cycle
%Cos_W: cosine between weight vector in each learning cycle with initial weight vector
%em:  em=0: there is no embedded paEpLern in afferents. em=1: there is an embedded paEpLern in afferents.
%w =ab, w: W_vecb, a=1, b: W_vecbb, W_vecb=W_vecbb




N_Post=1;      %number of post-synaptic neuron
desired_S = 2; % desired number of spikes r0 = 2 Hz


resy = 0.0; %resting potential
threshy = 1; % threshold potential

N_Pres = 500; %number of afferents
percI = 20; % inhibitory percentages
NI = fix((percI/100)*N_Pres); % number of inhibitory neurons
TTE = 2000; %total time [all in ms] 2 seconds
delt = 0.1; % integration step.
EpL = TTE/delt; 
times = 1:EpL; % discretization
times = times*delt;


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

W_vecb = 0.01+1e-4*randn(N_Pres,N_Post); % initial weight vector
W_vecb(W_vecb<0)=0;
% note for the case of one neuron all w_a is 1
w_max = 1; % maximum amount of synapsis.



tr=200;%transient time
dta=delt/taumem;
dtaM=(1-dta);

intimeP_vec = 5000; %ms Inserting time of the embedded paEpLern
Pat_L_vec = [50];  %ms paEpLern leangth
Pat_L=Pat_L_vec(p2)/delt; 

% generating randomly embedded paEpLern
PAT1 = rand(N_Pres, Pat_L,2);
PAT1(1:NI,:,:) = 1.*(PAT1(1:NI,:,:) < rateI*delt/1000);
PAT1(NI+1:N_Pres,:,:) = 1.*(PAT1(NI+1:N_Pres,:,:) < rateE*delt/1000);
PAT=PAT1(:,:,1);


%V0I=0; %modification threshold for inhibitory neurons
%V0E=0; %modification threshold for excitatory neurons
%instead of (V_vec(tr:EpL,:)- V0E) I wrote V_vec(tr:EpL,:)
%step 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inall=zeros(N_Pres,2*EpL-1);
Spikess=zeros(Ntrials_1,N_Post);
em=0;


de_f=-1e-4;
alpha_1=1e-2; %scaling factor 
C_E=0.9e-4; %excitatory learning rate
C_I = 1e-4;%inhibitory learning rate

% low-pass filter of spikes
Gamma_star= 0.9; 
Gamma_starM1=1-Gamma_star;

% low-pass filter of eligibility
Gamma_E= 0.99; 
Gamma_EM1=1-Gamma_E;


%initializations: 
del_b_E  =  zeros(N_Pres-NI,N_Post); %\del b_E
del_b_I  =  zeros(NI,N_Post);     %\del b_I and \del a_I
Filterd_Spikes=zeros(1,N_Post);%long-time firing rate


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sigha=2; %standard deviation
traceG=normpdf(times-round(TTE/2),0,sigha); %Gaussian kernel
sig_sig=sigha*10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for itrial=1:Ntrials_1

    [I_ext,INP] = Model_Input(rateI,rateE,delt,intimeP_vec,Pat_L,PAT,W_vecb,inall,NI,N_Pres,EpL,traceI,traceE,em,traceG);
                
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    %Neuron Model
    num_spikes_vec=zeros(N_Post,EpL);
    V_vec=zeros(EpL,N_Post);
    for it = 2:EpL
        V_vec(it,:) = (dtaM*V_vec(it-1,:)) + (I_ext(it-1,:)*dta); 
        if V_vec(it-1,:) >= threshy
            V_vec(it,1)=resy;
            num_spikes_vec(1,it)=1;
        end
    end   
    num_spikes=sum(num_spikes_vec(:,tr:EpL),2);
    Filterd_Spikes = Gamma_star*Filterd_Spikes + Gamma_starM1*num_spikes;    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Spikess(itrial,:)=num_spikes;

    
    del_b_I = Gamma_E*del_b_I + Gamma_EM1*(INP(1:NI,tr:EpL)*(V_vec(tr:EpL,:)));
   
    Filterd_Dw_E= (INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)))...
            -mean(INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)));
    del_b_E = Gamma_E*del_b_E+ Gamma_EM1*Filterd_Dw_E;
    
   
    W_vecb(1:NI,:)=W_vecb(1:NI,:) + C_I*del_b_I;   
    W_vecb(NI+1:N_Pres,:)=(1+de_f)*W_vecb(NI+1:N_Pres,:)+...
        alpha_1*(W_vecb(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
        C_E*del_b_E;
    
    W_vecb=W_vecb.*(W_vecb > 0.);
    W_vecb(NI+1:N_Pres)=(W_vecb(NI+1:N_Pres).*(W_vecb(NI+1:N_Pres) <= w_max) + w_max*(W_vecb(NI+1:N_Pres) > w_max));

end
%

 eval(['save EP_0/Weights/W_',num2str(p1),'_',num2str(p2),  '.mat W_vecb']);
 eval(['save EP_0/Spikes/spike_',num2str(p1),'_',num2str(p2),  '.mat Spikess']);
 eval(['save EP_0/Filterd_Spikes/Filterd_Spikes_',num2str(p1),'_',num2str(p2),  '.mat Filterd_Spikes']);
 eval(['save EP_0/del_b_I_F/del_b_I_',num2str(p1),'_',num2str(p2),  '.mat del_b_I']);
 eval(['save EP_0/del_b_E_F/del_b_E_',num2str(p1),'_',num2str(p2),  '.mat del_b_E']);

%step 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ns_0=zeros(Ntrials_2,N_Post);
ns_15=zeros(Ntrials_2,N_Post);
Spikess=zeros(Ntrials_2,N_Post);
Norm_W=zeros(Ntrials_2,N_Post);
Cos_W=zeros(Ntrials_2,N_Post);
W_vecb_0=W_vecb;
em=1;
for itrial=1:Ntrials_2

    [I_ext,INP] = Model_Input(rateI,rateE,delt,intimeP_vec,Pat_L,PAT,W_vecb,inall,NI,N_Pres,EpL,traceI,traceE,em,traceG);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    %Neuron Model
    num_spikes_vec=zeros(N_Post,EpL);
    V_vec=zeros(EpL,N_Post);
    for it = 2:EpL
         V_vec(it,:) = (dtaM*V_vec(it-1,:)) + (I_ext(it-1,:)*dta); 
        if V_vec(it-1,:) >= threshy
            V_vec(it,1)=resy;
            num_spikes_vec(1,it)=1;
        end
    end   
    num_spikes=sum(num_spikes_vec(:,tr:EpL),2);
    Filterd_Spikes = Gamma_star*Filterd_Spikes + Gamma_starM1*num_spikes;    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ns_0(itrial,:) = sum(num_spikes_vec(:,intimeP_vec(1,1)+1-sig_sig: intimeP_vec(1,1) + Pat_L+sig_sig),2);
    ns_15(itrial,:) = sum(num_spikes_vec(:,intimeP_vec(1,1)+1-sig_sig:intimeP_vec(1,1) + Pat_L+150+sig_sig),2);
    Spikess(itrial,:)=num_spikes;

    del_b_I = Gamma_E*del_b_I + Gamma_EM1*(INP(1:NI,tr:EpL)*(V_vec(tr:EpL,:)));
   
    Filterd_Dw_E= (INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)))...
            -mean(INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)));
    del_b_E = Gamma_E*del_b_E+ Gamma_EM1*Filterd_Dw_E;
    
   
    W_vecb(1:NI,:)=W_vecb(1:NI,:) + C_I*del_b_I;   
    W_vecb(NI+1:N_Pres,:)=(1+de_f)*W_vecb(NI+1:N_Pres,:)+...
        alpha_1*(W_vecb(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
        C_E*del_b_E;
    
    W_vecb=W_vecb.*(W_vecb > 0.);
    W_vecb(NI+1:N_Pres)=(W_vecb(NI+1:N_Pres).*(W_vecb(NI+1:N_Pres) <= w_max) + w_max*(W_vecb(NI+1:N_Pres) > w_max));

    
    Norm_W(itrial,1)=norm(W_vecb);
    Cos_W(itrial,1)=sum(W_vecb.*W_vecb_0)/(norm(W_vecb)*norm(W_vecb_0));


end
eval(['save EP_01/Weights/W_',num2str(p1),'_',num2str(p2),  '.mat W_vecb']);
eval(['save EP_01/Spikes/spike_',num2str(p1),'_',num2str(p2),  '.mat Spikess']);
eval(['save EP_01/Filterd_Spikes/Filterd_Spikes_',num2str(p1),'_',num2str(p2),  '.mat Filterd_Spikes']);
eval(['save EP_01/del_b_I_F/del_b_I_',num2str(p1),'_',num2str(p2),  '.mat del_b_I']);
eval(['save EP_01/del_b_E_F/del_b_E_',num2str(p1),'_',num2str(p2),  '.mat del_b_E']);

eval(['save EP_01/Data_Pattern/EP_',num2str(p1),'_',num2str(p2),  '.mat PAT']);
eval(['save EP_01/Data_ns_0/ER_',num2str(p1),'_',num2str(p2),  '.mat ns_0']);
eval(['save EP_01/Data_ns_15/ER_',num2str(p1),'_',num2str(p2),  '.mat ns_15']);
eval(['save EP_01/Data_Cos/Cos_',num2str(p1),'_',num2str(p2),  '.mat Cos_W']);
eval(['save EP_01/Data_Norm/Norm_',num2str(p1),'_',num2str(p2),  '.mat Norm_W']);

%step 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if p2==1
    
        %ns_0=zeros(Ntrials_3,N_Post);
        ns_15=zeros(Ntrials_3,N_Post);
        Spikess=zeros(Ntrials_3,N_Post);
        Spikess_p=zeros(Ntrials_3,N_Post);
        %Norm_W=zeros(Ntrials_3,N_Post);
        %Cos_W=zeros(Ntrials_3,N_Post);
        %W_vecb_0=W_vecb;
        em21=0;
        em211=1;
        for itrial=1:Ntrials_3
            
            [I_ext,INP] = Model_Input(rateI,rateE,delt,intimeP_vec,Pat_L,PAT,W_vecb,inall,NI,N_Pres,EpL,traceI,traceE,em21,traceG);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
            %Neuron Model
            num_spikes_vec=zeros(N_Post,EpL);
            V_vec=zeros(EpL,N_Post);
            for it = 2:EpL
                 V_vec(it,:) = (dtaM*V_vec(it-1,:)) + (I_ext(it-1,:)*dta); 
                if V_vec(it-1,:) >= threshy
                    V_vec(it,1)=resy;
                    num_spikes_vec(1,it)=1;
                end
            end   
            num_spikes=sum(num_spikes_vec(:,tr:EpL),2);
            Filterd_Spikes = Gamma_star*Filterd_Spikes + Gamma_starM1*num_spikes;    
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            Spikess(itrial,:)=num_spikes;

            del_b_I = Gamma_E*del_b_I + Gamma_EM1*(INP(1:NI,tr:EpL)*(V_vec(tr:EpL,:)));

            Filterd_Dw_E= (INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)))...
                    -mean(INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)));
            del_b_E = Gamma_E*del_b_E+ Gamma_EM1*Filterd_Dw_E;


            W_vecb(1:NI,:)=W_vecb(1:NI,:) + C_I*del_b_I;   
            W_vecb(NI+1:N_Pres,:)=(1+de_f)*W_vecb(NI+1:N_Pres,:)+...
                alpha_1*(W_vecb(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
                C_E*del_b_E;

            W_vecb=W_vecb.*(W_vecb > 0.);
            W_vecb(NI+1:N_Pres)=(W_vecb(NI+1:N_Pres).*(W_vecb(NI+1:N_Pres) <= w_max) + w_max*(W_vecb(NI+1:N_Pres) > w_max));

            %Norm_W(itrial,1)=norm(W_vecb);
            %Cos_W(itrial,1)=sum(W_vecb.*W_vecb_0)/(norm(W_vecb)*norm(W_vecb_0));

            if rem(itrial,50)==0
               
               [I_ext_o] = Model_Input_test(rateI,rateE,delt,intimeP_vec,Pat_L,PAT,W_vecb,inall,NI,N_Pres,EpL,traceI,traceE,em211,traceG);

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
                %Neuron Model
                num_spikes_vec0=zeros(N_Post,EpL);
                V_vec0=zeros(EpL,N_Post);
                for it = 2:EpL
                    V_vec0(it,:) = ((1- dta)*V_vec0(it-1,:)) + I_ext_o(it-1,:)*dta;
                    if V_vec0(it-1,:) >= threshy
                        V_vec0(it,1)=resy;
                        num_spikes_vec0(1,it)=1;
                    end
                end   
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %ns_0(itrial,:) = sum(num_spikes_vec0(:,intimeP_vec(1,1)+1: intimeP_vec(1,1) + Pat_L),2);
                ns_15(itrial,:) = sum(num_spikes_vec0(:,intimeP_vec(1,1)+1-sig_sig:intimeP_vec(1,1) + Pat_L+150+sig_sig),2);
                Spikess_p(itrial,:)=sum(num_spikes_vec0(:,tr:EpL),2);

            end

        end


        eval(['save EP_010/Weights/W_',num2str(p1),'_',num2str(p2),  '.mat W_vecb']);
        eval(['save EP_010/Spikes/spike_',num2str(p1),'_',num2str(p2),  '.mat Spikess']);
        eval(['save EP_010/Spikes/spike_p_',num2str(p1),'_',num2str(p2),  '.mat Spikess_p']);

        eval(['save EP_010/Filterd_Spikes/Filterd_Spikes_',num2str(p1),'_',num2str(p2),  '.mat Filterd_Spikes']);
        eval(['save EP_010/del_b_I_F/del_b_I_',num2str(p1),'_',num2str(p2),  '.mat del_b_I']);
        eval(['save EP_010/del_b_E_F/del_b_E_',num2str(p1),'_',num2str(p2),  '.mat del_b_E']);

        %eval(['save EP_010/Data_Pattern/EP_',num2str(p1),'_',num2str(p2),  '.mat PAT']);
        %eval(['save EP_010/Data_ns_0/ER_',num2str(p1),'_',num2str(p2),  '.mat ns_0']);
        eval(['save EP_010/Data_ns_15/ER_',num2str(p1),'_',num2str(p2),  '.mat ns_15']);
        %eval(['save EP_010/Data_Cos/Cos_',num2str(p1),'_',num2str(p2),  '.mat Cos_W']);
        %eval(['save EP_010/Data_Norm/Norm_',num2str(p1),'_',num2str(p2),  '.mat Norm_W']);

        %step 4
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %ns_0=zeros(Ntrials_4,N_Post);
        ns_15=zeros(Ntrials_4,N_Post);
        Spikess=zeros(Ntrials_4,N_Post);
%         Norm_W=zeros(Ntrials_4,N_Post);
%         Cos_W=zeros(Ntrials_4,N_Post);
%         W_vecb_0=W_vecb;
        em=1;
        for itrial=1:Ntrials_4

            [I_ext,INP] = Model_Input(rateI,rateE,delt,intimeP_vec,Pat_L,PAT,W_vecb,inall,NI,N_Pres,EpL,traceI,traceE,em,traceG);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
            %Neuron Model
            num_spikes_vec=zeros(N_Post,EpL);
            V_vec=zeros(EpL,N_Post);
            for it = 2:EpL
                 V_vec(it,:) = (dtaM*V_vec(it-1,:)) + (I_ext(it-1,:)*dta); 
                if V_vec(it-1,:) >= threshy
                    V_vec(it,1)=resy;
                    num_spikes_vec(1,it)=1;
                end
            end   
            num_spikes=sum(num_spikes_vec(:,tr:EpL),2);
            Filterd_Spikes = Gamma_star*Filterd_Spikes + Gamma_starM1*num_spikes;    
           %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


            %ns_0(itrial,:) = sum(num_spikes_vec(:,intimeP_vec(1,1)+1: intimeP_vec(1,1) + Pat_L),2);
            ns_15(itrial,:) = sum(num_spikes_vec(:,intimeP_vec(1,1)+1-sig_sig:intimeP_vec(1,1) + Pat_L+150+sig_sig),2);
            Spikess(itrial,:)=num_spikes;

            del_b_I = Gamma_E*del_b_I + Gamma_EM1*(INP(1:NI,tr:EpL)*(V_vec(tr:EpL,:)));

            Filterd_Dw_E= (INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)))...
                    -mean(INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)));
            del_b_E = Gamma_E*del_b_E+ Gamma_EM1*Filterd_Dw_E;


            W_vecb(1:NI,:)=W_vecb(1:NI,:) + C_I*del_b_I;   
            W_vecb(NI+1:N_Pres,:)=(1+de_f)*W_vecb(NI+1:N_Pres,:)+...
                alpha_1*(W_vecb(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
                C_E*del_b_E;

            W_vecb=W_vecb.*(W_vecb > 0.);
            W_vecb(NI+1:N_Pres)=(W_vecb(NI+1:N_Pres).*(W_vecb(NI+1:N_Pres) <= w_max) + w_max*(W_vecb(NI+1:N_Pres) > w_max));


           %Norm_W(itrial,1)=norm(W_vecb);
           %Cos_W(itrial,1)=sum(W_vecb.*W_vecb_0)/(norm(W_vecb)*norm(W_vecb_0));

        end
        eval(['save EP_0101/Weights/W_',num2str(p1),'_',num2str(p2),  '.mat W_vecb']);
        eval(['save EP_0101/Spikes/spike_',num2str(p1),'_',num2str(p2),  '.mat Spikess']);
        eval(['save EP_0101/Filterd_Spikes/Filterd_Spikes_',num2str(p1),'_',num2str(p2),  '.mat Filterd_Spikes']);
        eval(['save EP_0101/del_b_I_F/del_b_I_',num2str(p1),'_',num2str(p2),  '.mat del_b_I']);
        eval(['save EP_0101/del_b_E_F/del_b_E_',num2str(p1),'_',num2str(p2),  '.mat del_b_E']);

        %eval(['save EP_0101/Data_Pattern/EP_',num2str(p1),'_',num2str(p2),  '.mat PAT']);
        %eval(['save EP_0101/Data_ns_0/ER_',num2str(p1),'_',num2str(p2),  '.mat ns_0']);
        eval(['save EP_0101/Data_ns_15/ER_',num2str(p1),'_',num2str(p2),  '.mat ns_15']);
        %eval(['save EP_0101/Data_Cos/Cos_',num2str(p1),'_',num2str(p2),  '.mat Cos_W']);
        %eval(['save EP_0101/Data_Norm/Norm_',num2str(p1),'_',num2str(p2),  '.mat Norm_W']);

        %step 5
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Learning the second paEpLern

        eval(['load EP_01/Weights/W_',num2str(p1),'_',num2str(p2),  '.mat W_vecb']);
        eval(['load EP_01/Spikes/spike_',num2str(p1),'_',num2str(p2),  '.mat Spikess']);
        eval(['load EP_01/Filterd_Spikes/Filterd_Spikes_',num2str(p1),'_',num2str(p2),  '.mat Filterd_Spikes']);
        eval(['load EP_01/del_b_I_F/del_b_I_',num2str(p1),'_',num2str(p2),  '.mat del_b_I']);
        eval(['load EP_01/del_b_E_F/del_b_E_',num2str(p1),'_',num2str(p2),  '.mat del_b_E']);


        PAT2=PAT1(:,:,2);
        PAT=PAT1(:,:,1);

        ns_0=zeros(Ntrials_5,N_Post);
        ns_15=zeros(Ntrials_5,N_Post);
        Spikess=zeros(Ntrials_5,N_Post);

        %ns_0_0=zeros(Ntrials_5,N_Post);
        ns_15_0=zeros(Ntrials_5,N_Post);
        Spikess_0=zeros(Ntrials_5,N_Post);

        Norm_W=zeros(Ntrials_5,N_Post);
        Cos_W=zeros(Ntrials_5,N_Post);
        W_vecb_0=W_vecb;
        em=1;
        for itrial=1:Ntrials_5

            [I_ext,INP] = Model_Input(rateI,rateE,delt,intimeP_vec,Pat_L,PAT2,W_vecb,inall,NI,N_Pres,EpL,traceI,traceE,em,traceG);


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
            %Neuron Model
            num_spikes_vec=zeros(N_Post,EpL);
            V_vec=zeros(EpL,N_Post);
            for it = 2:EpL
                 V_vec(it,:) = (dtaM*V_vec(it-1,:)) + (I_ext(it-1,:)*dta); 
                if V_vec(it-1,:) >= threshy
                    V_vec(it,1)=resy;
                    num_spikes_vec(1,it)=1;
                end
            end   
            num_spikes=sum(num_spikes_vec(:,tr:EpL),2);
            Filterd_Spikes = Gamma_star*Filterd_Spikes + Gamma_starM1*num_spikes;    
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            ns_0(itrial,:) = sum(num_spikes_vec(:,intimeP_vec(1,1)+1-sig_sig: intimeP_vec(1,1) + Pat_L+sig_sig),2);
            ns_15(itrial,:) = sum(num_spikes_vec(:,intimeP_vec(1,1)+1-sig_sig:intimeP_vec(1,1) + Pat_L+150+sig_sig),2);
            Spikess(itrial,:)=num_spikes;

            del_b_I = Gamma_E*del_b_I + Gamma_EM1*(INP(1:NI,tr:EpL)*(V_vec(tr:EpL,:)));

            Filterd_Dw_E= (INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)))...
                    -mean(INP(NI+1:N_Pres,tr:EpL)*((V_vec(tr:EpL,:) ).*((V_vec(tr:EpL,:) ) > 0)));
            del_b_E = Gamma_E*del_b_E+ Gamma_EM1*Filterd_Dw_E;


            W_vecb(1:NI,:)=W_vecb(1:NI,:) + C_I*del_b_I;   
            W_vecb(NI+1:N_Pres,:)=(1+de_f)*W_vecb(NI+1:N_Pres,:)+...
                alpha_1*(W_vecb(NI+1:N_Pres,:).*tanh(desired_S-Filterd_Spikes))+...
                C_E*del_b_E;

            W_vecb=W_vecb.*(W_vecb > 0.);
            W_vecb(NI+1:N_Pres)=(W_vecb(NI+1:N_Pres).*(W_vecb(NI+1:N_Pres) <= w_max) + w_max*(W_vecb(NI+1:N_Pres) > w_max));


            Norm_W(itrial,1)=norm(W_vecb);
            Cos_W(itrial,1)=sum(W_vecb.*W_vecb_0)/(norm(W_vecb)*norm(W_vecb_0));

            [I_ext_0] = Model_Input_test(rateI,rateE,delt,intimeP_vec,Pat_L,PAT,W_vecb,inall,NI,N_Pres,EpL,traceI,traceE,em,traceG);   
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
            %Neuron Model
            num_spikes_vec_0=zeros(N_Post,EpL);
            V_vec_0=zeros(EpL,N_Post);
            for it = 2:EpL
                V_vec_0(it,:) = ((1- dta)*V_vec_0(it-1,:)) + I_ext_0(it-1,:)*dta;
                if V_vec_0(it-1,:) >= threshy
                    V_vec_0(it,1)=resy;
                    num_spikes_vec_0(1,it)=1;
                end
            end   

          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %ns_0_0(itrial,:) = sum(num_spikes_vec_0(:,intimeP_vec(1,1)+1: intimeP_vec(1,1) + Pat_L),2);
            ns_15_0(itrial,:) = sum(num_spikes_vec_0(:,intimeP_vec(1,1)+1-sig_sig:intimeP_vec(1,1) + Pat_L+150+sig_sig),2);
            Spikess_0(itrial,:)=sum(num_spikes_vec_0(:,tr:EpL),2);


        end
        eval(['save EP_011/Weights/W_',num2str(p1),'_',num2str(p2),  '.mat W_vecb']);
        eval(['save EP_011/Spikes/spike_',num2str(p1),'_',num2str(p2),  '.mat Spikess']);
        eval(['save EP_011/Filterd_Spikes/Filterd_Spikes_',num2str(p1),'_',num2str(p2),  '.mat Filterd_Spikes']);
        eval(['save EP_011/del_b_I_F/del_b_I_',num2str(p1),'_',num2str(p2),  '.mat del_b_I']);
        eval(['save EP_011/del_b_E_F/del_b_E_',num2str(p1),'_',num2str(p2),  '.mat del_b_E']);

        eval(['save EP_011/Data_Pattern/EP_',num2str(p1),'_',num2str(p2),  '.mat PAT2']);
        %eval(['save EP_011/Data_ns_0/ER_',num2str(p1),'_',num2str(p2),  '.mat ns_0']);
        eval(['save EP_011/Data_ns_15/ER_',num2str(p1),'_',num2str(p2),  '.mat ns_15']);
        eval(['save EP_011/Data_Cos/Cos_',num2str(p1),'_',num2str(p2),  '.mat Cos_W']);
        eval(['save EP_011/Data_Norm/Norm_',num2str(p1),'_',num2str(p2),  '.mat Norm_W']);
        
        eval(['save EP_011/Spikes/spike_2_',num2str(p1),'_',num2str(p2),  '.mat Spikess_0']);
        eval(['save EP_011/Data_ns_15/ER_2_',num2str(p1),'_',num2str(p2),  '.mat ns_15_0']);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

end
