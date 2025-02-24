%% PREDICT PERFORMANCE CURVE FOR CORRELATION-BASED AAD ALGORITHMS

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

clear; close all; clc;

%% Setup: parameters

params = struct;
params.setup = 'linearDecoder-das2016'; % 'linearDecoder-das2016' or 'VLAAI-fuglsang2018' 
params.baselineWl = [60,30,20,10,5,1]; % baseline decision window length(s) to start from 
params.amountEstimationData = [60,30,10,5,2]; % minutes of estimation data to use
params.nbRep = 10; % number of random permutations per condition/setting to use
params.CI = false; % confidence interval estimation or not
params.B = 1000; % number of bootstrap permutations
params.save = false; params.saveName = 'pick-your-own!'; % save parameters

%% Initialization
switch params.setup 
    case 'linearDecoder-das2016'
        load corrs-linearDecoder-das2016.mat; % attended correlations = 1st column, unattended = 2nd column
    case 'VLAAI-fuglsang2018'
        load corrs-vlaai-fugslang2018.mat; % attended correlations = 1st column, unattended = 2nd column
end
params.windowLengthsTo = paramsAAD.windowLengths;
params.fs = paramsAAD.fs; fs = params.fs;

accGT = acc*100; nbSubjects = size(accGT,1); nbBaseWl = length(params.baselineWl); nbWlTo = length(params.windowLengthsTo);

if params.CI
    accPred = zeros(params.nbRep,nbSubjects,length(params.amountEstimationData),nbBaseWl,nbWlTo,3);
else
    accPred = zeros(params.nbRep,nbSubjects,length(params.amountEstimationData),nbBaseWl,nbWlTo);
end

%% Loop over baseline window lengths, subjects, estimation data amount, repetitions to perform accuracy predictions and performance curve modeling
wb = waitbar(0,'Starting'); ii = 1;
for wlBaseI = 1:nbBaseWl % loop over baseline window lengths
    wlBase = params.baselineWl(wlBaseI);

    for s = 1:nbSubjects % loop over subjects
        Cbaseline = corrs{s,paramsAAD.windowLengths==params.baselineWl(wlBaseI)}; % extract labeled baseline correlations

        for edI = 1:length(params.amountEstimationData) % loop over estimation data amounts
            waitbar(ii/(nbBaseWl*nbSubjects*length(params.amountEstimationData)), wb, sprintf('Progress: %.2f %%', ii/(nbBaseWl*nbSubjects*length(params.amountEstimationData))*100)); ii = ii+1;
            
            for rep = 1:params.nbRep % loop over random repetitions

                %% pick random subset of correlations to limit estimation data
                rp = randperm(size(Cbaseline,1),round(params.amountEstimationData(edI)*60/params.baselineWl(wlBaseI)));
                Ctemp = Cbaseline(rp,:); 
                
                %% Fisher transformation and estimation of parameters on baseline window length
                rhoA = mean(Ctemp(:,1)); rhoU = mean(Ctemp(:,2));
                Z = atanh(Ctemp); muZbase = mean(Z(:,1)-Z(:,2)); stdZbase = std(Z(:,1)-Z(:,2));

                for wlToI = 1:nbWlTo % loop over target window lengths
                    wlTo = params.windowLengthsTo(wlToI);

                    %% Extrapolation of parameters
                    muZto = muZbase-(rhoA-rhoU)/(2*(wlBase*fs-1))+(rhoA-rhoU)/(2*wlTo*fs-1);
                    stdZto = sqrt(2/(wlTo*fs-1))/sqrt(2/(wlBase*fs-1))*stdZbase;

                    %% AAD accuracy prediction
                    accPred(rep,s,edI,wlBaseI,wlToI,1) = (1-cdf('Normal',0,muZto,stdZto))*100;
                
                    %% CI estimation
                    if params.CI
                        accPred(rep,s,edI,wlBaseI,wlToI,2:3) = bootci(params.B,{@(C)extrapolateAcc(C,wlBase,wlTo,fs),Ctemp},'Type','bca');
                    end
                end
            end
        end
    end
end

% save
if params.save
    save(params.saveName,'accGT','accPred','corrs','params','paramsAAD');
end
close(wb);

%% Plotting

% across window lengths for the whole dataset
figure;
A = mean(accPred(:,:,1,:,:,1),2);
for wlBaseI = 1:nbBaseWl
    subplot(floor(sqrt(nbBaseWl)),ceil(nbBaseWl/floor(sqrt(nbBaseWl))),wlBaseI);
    plot(params.windowLengthsTo,mean(accGT,1),'-*','linewidth',2);
    hold on
    plot(params.windowLengthsTo,squeeze(mean(A(:,1,1,wlBaseI,:),1)),'-*','linewidth',2)
    leg = {'ground truth'};
    xlabel('window length to [s]'); ylabel('AAD accuracy [%]'); legend('ground truth','estimated')
    title(['window length from = ',num2str(params.baselineWl(wlBaseI)),', ',num2str(params.amountEstimationData(end)),' min est data']);
end

switch params.setup
    case 'linearDecoder-das2016'
        bestWlBaseInd = find(params.baselineWl==20);
    case 'VLAAI-fuglsang2018'
        bestWlBaseInd = find(params.baselineWl==25);
end

% check best amount of estimation data
accGTf = repmat(accGT,[1,1,params.nbRep,length(params.amountEstimationData),nbBaseWl]); accGTf = permute(accGTf,[3,1,4,5,2]);
absErr = abs(mean(accGTf(:,:,:,bestWlBaseInd,:),2)-mean(accPred(:,:,:,bestWlBaseInd,:,1),2));
figure;
errorbar(params.amountEstimationData,squeeze(mean(mean(absErr,5),1)),squeeze(std(mean(absErr,5),[],1)),'-*','linewidth',2);
xlabel('amount of estimation data [min]'); ylabel('abs err AAD accuracy [pp]'); title('dataset-wide estimation')

% table params
estDatInd = find(params.amountEstimationData==30);
absErr = squeeze(abs(accGTf(:,:,estDatInd,bestWlBaseInd,:)-accPred(:,:,estDatInd,bestWlBaseInd,:,1)));
if params.CI
    withinCI = accPred(:,:,estDatInd,bestWlBaseInd,:,2) <= accGTf(:,:,estDatInd,bestWlBaseInd,:) & accPred(:,:,estDatInd,bestWlBaseInd,:,3) >= accGTf(:,:,estDatInd,bestWlBaseInd,:);
end
