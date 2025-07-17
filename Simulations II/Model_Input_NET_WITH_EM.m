function [I_ext,INP]=Model_Input_NET_WITH_EM(rateI,traceI,rateE,traceE,N_Pres,NI,N_Post,EpL,delt,W_vecAB,intimeP_vec,em_l,PAT2,Pat_L,traceG)


    %I_ext: external input
    %INP: corresponding kernel to external input
    inall0=zeros(N_Pres,EpL);
    inall=zeros(N_Pres,2*EpL-1);
    I_ext=zeros(EpL,N_Post);
    PATP=rand(N_Pres,EpL);
    PATP(NI+1:N_Pres,:) = PATP(NI+1:N_Pres,:) < (rateE*delt/1000); 
    PATP(1:NI,:) = PATP(1:NI,:) < (rateI*delt/1000);  
    BCV2=rand(N_Pres,EpL);


    for i=1:em_l
         PATP(:, intimeP_vec(1,i)+1: intimeP_vec(1,i) + Pat_L) = PAT2(:,:,i);  
    end

     
    for i=1:N_Pres
       inall0(i,:) =conv(PATP(i,:), traceG,'same');
    end
    
    PATP2=BCV2<inall0*delt;
    
    for i = 1: NI
        inall(i,:) = -1*conv(PATP2(i,:), traceI);        
    end 
    
    for i = NI+1: N_Pres
        inall(i,:) = conv(PATP2(i,:), traceE);
    end    
    INP = abs(inall(:,1:EpL));   
    
    for iw = 1:N_Post
        I_ext(:,iw)=W_vecAB(:,iw)'*inall(:,1:EpL);
    end
    
    
end
