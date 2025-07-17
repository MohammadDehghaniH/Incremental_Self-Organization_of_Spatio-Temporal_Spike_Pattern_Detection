function [I_ext11]=Model_Input_test(rateI,rateE,delt,intimeP_vec,Pat_L,PAT,W_vecb,inall,NI,N_Pres,EpL,traceI,traceE,em,traceG)
    %I_ext: external input
    %INP: corresponding kernel to external input
    BCV1=rand(N_Pres,EpL);
    PATP=zeros(N_Pres,EpL);
    PATP(NI+1:N_Pres,:) = BCV1(NI+1:N_Pres,:) < (rateE*delt/1000); 
    PATP(1:NI,:) = BCV1(1:NI,:) < (rateI*delt/1000);  
    if em>0
        PATP(:, intimeP_vec+1: intimeP_vec + Pat_L) = PAT; 
    end
    
    %%%
    inall0=zeros(N_Pres,EpL);
    BCV2=rand(N_Pres,EpL);
    for i=1:N_Pres
       inall0(i,:) =conv(PATP(i,:), traceG,'same');
    end    
    PATP2=BCV2<inall0*delt;
    
    for i = 1: NI
        inall(i,:) = conv(PATP2(i,:), traceI);        
    end 
    for i = NI+1: N_Pres
        inall(i,:) = conv(PATP2(i,:), traceE);
    end    
    INP = inall(:,1:EpL);   
    
    WW_vecb = W_vecb;
    WW_vecb(1:NI,:) = -W_vecb(1:NI,:);    
    I_ext11(:,1)=WW_vecb(:,1)'*INP; 
end
