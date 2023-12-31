%% Reset Workspace
close all
clear
clc
%% 
% EE 107 - Communication Systems - Final Project
% Authors: Nicholas Boudreau, Andrew Chen, Edwin Liang-Gilman
%
%% Q1
% Variable parameters:
%   roll-off factor (alpha),
%   truncation length (K)
alpha = 0.5; % [0 0.5 1]
K = 6; % [2 4 6]
samps = 32;

T = 1;
t = linspace(0,1,samps+1);

g1 = sin((pi / T) * t);
g1 = g1 / norm(g1);             % normalize; energy = 1
g2 = rcosdesign(alpha,2*K,samps);
g2 = g2 / norm(g2);
%% Plots Q1
% Half-Sine
figure(1)
plot(t,g1,'LineWidth',1.5);
xlabel('t (s)')
ylabel('g_1(t)')
title('Plot of Half-Sine Pulse')
grid on
figure(2)
freqz(g1)
title('Frequency Response of Half-Sine Pulse')
% SRRC
figure(3)
plot(linspace(-K*T,K*T,2*K*samps + 1),g2,'LineWidth',1.5);
xlabel('t (s)')
ylabel('g_2(t)')
title('Plot of SRRC Pulse')
grid on
figure(4)
freqz(g2)
title('Frequency Response of SRRC Pulse')

%% Change roll-off factor (alpha)
K = 6;
alpha = 0;
g2_a0 = rcosdesign(alpha,2*K,samps);
g2_a0 = g2_a0 / norm(g2_a0);
alpha = 1;
g2_a1 = rcosdesign(alpha,2*K,samps);
g2_a1 = g2_a1 / norm(g2_a1);
figure(5)
hold on
plot(linspace(-K*T,K*T,2*K*samps + 1),g2_a0,'LineWidth',1.5);
plot(linspace(-K*T,K*T,2*K*samps + 1),g2_a1,'LineWidth',1.5);
xlabel('t (s)')
ylabel('g_2(t)')
title('Plot of SRRC Pulse Shaping Function for varying \alpha')
legend('\alpha = 0','\alpha = 1')
grid on
[h0,w0] = freqz(g2_a0);
[h1,w1] = freqz(g2_a1);
figure(6)
hold on
plot(w0/pi,20*log10(abs(h0)),'LineWidth',1.5)
plot(w1/pi,20*log10(abs(h1)),'LineWidth',1.5)
xlabel('Normalized Frequency (x\pi rad/sample)')
ylabel('Magnitude (dB)')
title('Frequency Response of SRRC for varying \alpha')
legend('\alpha = 0','\alpha = 1')
grid on
% Change length of truncated pulse (K)
alpha = 0.5;
K = 2;
g2_a0 = rcosdesign(alpha,2*K,samps);
g2_a0 = g2_a0 / norm(g2_a0);
K = 4;
g2_a1 = rcosdesign(alpha,2*K,samps);
g2_a1 = g2_a1 / norm(g2_a1);
figure(7)
hold on
plot(linspace(-2*T,2*T,2*2*samps + 1),g2_a0,'LineWidth',3);
plot(linspace(-4*T,4*T,2*4*samps + 1),g2_a1,'LineWidth',1.5);
xlabel('t (s)')
ylabel('g_2(t)')
title('Plot of SRRC Pulse Shaping Function for varying K')
legend('K = 2','K = 4')
grid on
[h0,w0] = freqz(g2_a0);
[h1,w1] = freqz(g2_a1);
figure(8)
hold on
plot(w0/pi,20*log10(abs(h0)),'LineWidth',1.5)
plot(w1/pi,20*log10(abs(h1)),'LineWidth',1.5)
xlabel('Normalized Frequency (x\pi rad/sample)')
ylabel('Magnitude (dB)')
title('Frequency Response of SRRC for varying K')
legend('K = 2','K = 4')
grid on
%% Q2 Generate random bit sequence ak, map to antipodal bk 
% 10 bit signal
% ak = [1 0 1 1 0 1 0 0 1 0]; %randi([0, 1],1,10);
rng(0)          % fix for repeatability
n_bits = 50;    % scaling test case
ak = randi([0, 1],1,n_bits);
bk = 2 * ak - 1;
g1_mod = [];
g2_mod = [];

%%
% Half-Sine Modulation
for i = 1:length(ak)-1
   g1_mod = [g1_mod bk(i)*g1(1:samps)];
end
g1_mod = [g1_mod bk(10)*g1];

% SRRC Modulation
srrc_ind = [];
idx0 = 1;
idx1 = (2*K*32)+1;
for i = 1:length(ak)
   srrc_ind(i,idx0:idx1) = bk(i) * g2;
   idx0 = idx0 + 32;
   idx1 = idx1 + 32;
end
for i = 1:length(srrc_ind)
   g2_mod = [g2_mod sum(srrc_ind(:,i))];
end
t_srrc = linspace(0,2*length(ak)+1,length(g2_mod));



%% Plots for modulated signals
figure(9)
t_halfsine = linspace(0,length(ak),samps*length(ak)+1);
plot(t_halfsine,g1_mod,'LineWidth',1.5);
title('Modulated Half-Sine Pulse')
xlabel('T (s)')
ylabel('g_1(t)')
grid on

figure(10)
hold on
for i = 1:length(ak)
   k = find(srrc_ind(i,:));
   plot(t_srrc(k),srrc_ind(i,k),'LineWidth',1.2)
end
title('Plot of Individual SRRC Pulses')
xlabel('t (s)')
ylabel('g_2(t)')
grid on

% Testing out modulation using convolution
figure(11)
plot(t_srrc,g2_mod,'LineWidth',1.5)
title('Modulated SRRC Pulse')
grid minor
%% Q3
% [h_half,w_half] = freqz(g1_mod);
% [h_srrc,w_srrc] = freqz(g2_mod);
%
% figure(12)
% hold on
% plot(w_half,20*log10(abs(h_half)),'LineWidth',1.5)
% plot(w_srrc,20*log10(abs(h_srrc)),'LineWidth',1.5)
% xlabel('Normalized Frequency (x\pi rad/sample)')
% ylabel('Magnitude (dB)')
% title('Spectrum of Modulated Half-Sine and SRRC')
% legend('Half-Sine','SRRC')
% grid on
figure(12)
freqz(g1_mod);
title('Frequency Response of Half-Sine Modulation')
figure(13)
freqz(g2_mod);
title('Frequency Response of SRRC Modulation')
%% Q4 Eye diagrams
eyediagram(g1_mod,32,1,16); % (signal,n_samps,period,offset)
% eyediagram(g2_mod(193:481),32,1);
trim = K*samps;
eyediagram(g2_mod(1+trim:end-trim),32,1); % (193:481)get rid of trailing start/end (K*samps)

%% Channel Effects using local functions
% TODO: Move to separate files

[channel_mod, taps] = channel_effect(g1_mod);
[channel_srrc_mod] = channel_effect(g2_mod);

%% Plot channel impulse response
t = 0:4/length(taps):4-4/length(taps);
figure,
stem(t,taps)
grid minor
title("Channel Impulse Response h[n]")
xlabel("samples (n)")
ylabel("h[n]")
xlim([-0.5, 3.5])

%% Eye Diagrams after channel effects
eyediagram(channel_mod,32,1,16)
eyediagram(channel_srrc_mod(1+trim:end-trim),32,1)
eyediagram(channel_srrc_mod(1+trim:end-trim),16,1)



%% Eye Digrams w/ Noise
% eyediagram(channel_mod,32,1,16)
noisy_halfsine = addnoise(channel_mod);
noisy_srrc = addnoise(channel_srrc_mod);

for i = 1:size(noisy_halfsine,1)
    eyediagram(noisy_halfsine(i,:),32,1,16)
end

for i = 1:size(noisy_srrc,1)
    eyediagram(noisy_srrc(i,1+trim:end-trim),16,1)
end
%% Supporting Local Functions
% Simulate channel effects on modulated signals
function [modulated_channel, filter_coeffs] = channel_effect(modulated_signal)
filter_coeffs = [1, 0.5, 0.75, -2/7]; % TODO: take as input
upsampled_filter_coeffs = upsample(filter_coeffs,32);

modulated_channel = conv(upsampled_filter_coeffs,modulated_signal);
end

% Add White Noise based on signal power or hard coded power level
function [noisy_signal] = addnoise(input_sig)

% noise_power = pwr(input_sig);
noise_power = [1e-3, 1e-2, 1e-1]; %,max(input_sig)]; TODO: automate
sigma = noise_power';
n = sigma.*randn(size(input_sig));
noisy_signal = input_sig + n;
end

% Calculate power of given signal
function [signal_pwr] = pwr(signal)
    signal_pwr = 10*log10(norm(signal));
end