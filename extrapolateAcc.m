function accPred = extrapolateAcc(C,wlBase,wlTo,fs)
% EXTRAPOLATEACC Extrapolate the AAD accuracy from one window length to 
% another.
%
%   Input parameters:
%       C [DOUBLE]: attended and unattended stimulus-response correlations 
%           (nb of windows x 2). First column: attended correlations,
%           second column: unattended correlations
%       wlBase [DOUBLE]: baseline window length at which correlations are
%           measured
%       wlTo [DOUBLE]: target decision window length
%       fs [INTEGER]: sampling frequency
%
%   Output:
%       accPred [DOUBLE]: the predicted accuracy

% Author: Simon Geirnaert, KU Leuven, ESAT & Dept. of Neurosciences
% Correspondence: simon.geirnaert@esat.kuleuven.be

%% Fisher transformation and estimation of parameters on baseline window length
rhoA = mean(C(:,1)); rhoU = mean(C(:,2));
Z = atanh(C);
muZbase = mean(Z(:,1)-Z(:,2)); stdZbase = std(Z(:,1)-Z(:,2));

%% Extrapolation of parameters
muZto = muZbase-(rhoA-rhoU)/(2*(wlBase*fs-1))+(rhoA-rhoU)/(2*wlTo*fs-1);
stdZto = sqrt(2/(wlTo*fs-1))/sqrt(2/(wlBase*fs-1))*stdZbase;

%% AAD accuracy prediction
accPred = (1-cdf('Normal',0,muZto,stdZto))*100;

end