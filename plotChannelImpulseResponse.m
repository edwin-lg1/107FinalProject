close all
clear
clc

%%
% Plot channel impulse response

ch1 = [1, 0.5, 0.75, -2/7];
ch_outdoor = [0.5 1 0 0.63 0 0 0 0 0.25 0 0 0 0.16 0 0 0 0 0 0 0 0 0 0 0 0 0.1];
ch_indoor = [1 0.4365 0.1905 0.0832 0 0.0158 0 0.003];

C = cell(3,1);
C(:,:) = {upsample(ch1,32);upsample(ch_outdoor,32);upsample(ch_indoor,32)};

rx =1:512;
Eb = 1;
n_pow = 0.1;

for i=1:numel(C)
    H = freqz(C{i,:});
    Q_zf = 1./H;
    Q_mmse = conj(H) ./ ((abs(H).^2) + n_pow/Eb);

    q_zf = ifft(Q_zf,numel(C{i,:}));
    q_mmse = ifft(Q_mmse,numel(C{i,:}));

    t = linspace(0,length(q_zf)/32,length(q_zf));

    figure,

    subplot(2,1,1)
    plot(t,q_zf)
    grid minor
    title("Channel: "+i+newline + ...
        "ZF Equalizer - Impulse Response")
%     title("")
    subplot(2,1,2)
    plot(t,q_mmse)
    grid minor
    title("MMSE Equalizer - Impulse Response")
    subtitle("noise power: "+n_pow)
end



