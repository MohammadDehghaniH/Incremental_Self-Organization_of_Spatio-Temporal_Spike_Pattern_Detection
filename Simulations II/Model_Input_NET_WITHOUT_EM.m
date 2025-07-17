function [I_ext,INP]=Model_Input_NET_WITHOUT_EM(rateI,traceI,rateE,traceE,N_Pres,NI,N_Post,EpL,delt,W_vecAB)
%I_ext: external input
    %INP: corresponding kernel to external input
    inall=zeros(N_Pres,2*EpL-1);
    I_ext=zeros(EpL,N_Post);
    PATP=rand(N_Pres,EpL);
    PATP(NI+1:N_Pres,:) = PATP(NI+1:N_Pres,:) < (rateE*delt/1000); 
    PATP(1:NI,:) = PATP(1:NI,:) < (rateI*delt/1000);  
 
    
    for i = 1: NI
        inall(i,:) = -1*conv(PATP(i,:), traceI);        
    end 
    
    for i = NI+1: N_Pres
        inall(i,:) = conv(PATP(i,:), traceE);
    end    
    INP = abs(inall(:,1:EpL));   
    
    for iw = 1:N_Post
        I_ext(:,iw)=W_vecAB(:,iw)'*inall(:,1:EpL);
    end
    
end
