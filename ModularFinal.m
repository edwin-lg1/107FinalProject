%% Reset Workspace
close all
clear
clc

%%
% EE 107 - Communication Systems - Final Project
% Authors: Nicholas Boudreau, Andrew Chen, Edwin Liang-Gilman

%% Specify Parameters
filename = 'octopus.png';

qbits = 8; % [8 16 color depth]
color = 1; % [0 greyscale, 1 color]
pulseshape = 1; % [1 HS, 2 SRRC]
channel = 2; % [1 default, 2 outdoor, 3 indoor]
noisepwr = 1; % [1 LOW 1e-2, 2 HIGH 1e-1]
eq = 2; % [1 zf 2 mmse]

%%
% run
main(filename,qbits,color,pulseshape,channel,eq, noisepwr)







%% main driver 
function main(img_filename, qbits, color, pulseshape, channel, eq, noisepwr)
    % image preproc
    switch color
        case 1
            [Ztres,r,c,m,n,minval,maxval] = ImagePreProcess_color(img_filename, qbits);
        otherwise
            img_filename = 'cameraman.tif';
            [Ztres,r,c,m,n,minval,maxval] = ImagePreProcess_gray(img_filename, qbits);
    end

    img_RGB = imread(img_filename);
    numcol = floor(numel(img_RGB(1,:,1))/2);

    %%%% Bit stream %%%%

    DCTblocks = int2bit(Ztres,8);           % 64x8xNumberBlocks uint8
    stack = zeros(8,8,size(DCTblocks,3));   % 8x8xNumberBlocks double

    %%%% Constants for processing %%%%
    samps = 32;
    Tb = 1;
    K = 6;
    alpha = 0.5;

    Eb = 1;
    
    for z = 1%:size(DCTblocks,3) % r/8*c/8*3 (108)
        disp(z)
        ak = reshape(DCTblocks(:,:,z),[],1);
        ak_double = cast(ak, 'double');
        bk = 2*ak_double-1;
        bk = upsample(bk,32);
        img_bitstream = bk;
        n_bits = size(DCTblocks,1)*size(DCTblocks,2);
        hs_vectorlength = n_bits*32;
        
        switch pulseshape
            case 1
%                 disp("HS Modulation")
                [modulated_sig, pulse, pulse_t, ...
                    Tb, K, alpha, samps] = pulseshape_modulation(img_bitstream, ...
                                                                pulseshape, Tb, K, alpha, samps);
                
                hs_mod = modulated_sig(1:hs_vectorlength);
                channel_input_sig = hs_mod;

            case 2
%                 disp("SRRC Modulation")

                channel_input_sig = [];

        end
            %%%%% Channel Modulation %%%%
        switch channel
            case 1
%                 disp("Channel Definition 1")
                [noisy_received_sig, clean_channel_sig, taps, npwr] = channel_modulation(channel_input_sig, channel, noisepwr);
            case 2
%                 disp("Outdoor Channel")
                [noisy_received_sig, clean_channel_sig, taps, npwr] = channel_modulation(channel_input_sig, channel, noisepwr);
            case 3
%                 disp("Indoor Channel")
                [noisy_received_sig, clean_channel_sig, taps, npwr] = channel_modulation(channel_input_sig, channel, noisepwr);
        end

        if pulseshape
            noisy_received_sig = noisy_received_sig(1:hs_vectorlength);
        else
            noisy_received_sig = [];

        end
            rx_zeronoise = clean_channel_sig;
            rx_noisy = noisy_received_sig;

            %%%% MATCHED FILTERING %%%%

            [rx_zero_noise_mf, matchedfilter] = matched_filtering(rx_zeronoise,pulse);
            [rx_noisy_mf, matchedfilter] = matched_filtering(rx_noisy,pulse);

        if pulseshape
            rx_noisy_mf = rx_noisy_mf(1:hs_vectorlength);
            rx_zero_noise_mf = rx_zero_noise_mf(1:hs_vectorlength);

        else
            rx_noisy_mf = [];
            rx_zero_noise_mf = [];
        end


            %%%% Equalizer %%%%
            [H,w1] = freqz(taps,1,10000,'whole');

        switch eq
            case 1
%                 disp("Zero Forcing Eq")
                [rx_zeronoise_zf, Q_zf] = zf_equalizer(rx_zero_noise_mf,H);
                [rx_noisy_zf, Q_zf] = zf_equalizer(rx_noisy_mf,H);


            case 2
%                 disp("MMSE Eq")
                [rx_zeronoise_mmse,Q_mmse] = mmse(rx_zero_noise_mf,H,0,Eb);
                [rx_noisy_mmse,Q_mmse] = mmse(rx_noisy_mf,H,npwr,Eb);

        end


        if pulseshape
%             rx_zeronoise_zf = rx_zeronoise_zf(1:hs_vectorlength);
%             rx_noisy_zf = rx_noisy_zf(1:hs_vectorlength);
            rx_zeronoise_mmse = rx_zeronoise_mmse(1:hs_vectorlength);
            rx_noisy_mmse = rx_noisy_mmse(1:hs_vectorlength);

        else
%             rx_zeronoise_zf = [];
%             rx_noisy_zf = [];
            rx_zeronoise_mmse =[];
            rx_noisy_mmse =[];
        end

        
               
            bitstream_sampled = zeros(n_bits,1);
        %%%% Sampling and Detection %%%%

        % HS
        if pulseshape
%             disp("HS Sampling and Detection")
            bitstream_sampled(rx_zeronoise_mmse(32:32:end)>0) = 1;

        else
%             bitstream_sampled = [];
        end


%             bitstream_sampled(zero_forced_sig((161:samps:end-1>0))) = 1;
    
            ak_reconstructed = cast(bitstream_sampled,'uint8');
            ak_bin = reshape(ak_reconstructed,64,8);
            ak_recon = bit2int(ak_bin,8);
            
            
            stack(:,:,z) = ak_recon;
            
%             Ztres(:,:,z) == stack(:,:,z)

        
        figure,
%         plot(clean_channel_sig(1:2000))
        hold on
%         plot(noisy_received_sig(1:2000))
%         plot(channel_input_sig(1:2000))
        plot(rx_noisy_mmse(1:2000),'LineWidth',1.5)
        legend()
        grid minor
        
%         figure,
%         plot(rx_noisy_zf)
%         hold on
%         plot(rx_zeronoise_zf)
%         grid minor
    end
    %%%% Plot mmse/zf output %%%%

%     figure
%     subplot(2,1,1)
%     plot(rx_zeronoise_zf)
%     title("Rx No Noise ZF")
%     
%     subplot(2,1,2)
%     plot(rx_noisy_zf)
%     title("Rx Noise:"+npwr+" ZF")

%     subplot(2,2,3)
%     plot(rx_zeronoise_mmse)
%     title("Rx No Noise MMSE")
% 
% 
%     subplot(2,2,4)
%     plot(rx_noisy_mmse)
%     title("Rx Noise:"+npwr+" MMSE")

    [newZ]=ImagePostProcess_color(stack,r,c,m,n,minval,maxval);
    ImagePostProcess_color(Ztres,r,c,m,n,minval,maxval)
end
%% Supporting Local Functions
%

%
% Zero Forcing Equalizer
function [zf_sig, Q_zf] = zf_equalizer(rx,H)
    Q_zf = 1./H;
    Q_zf = Q_zf';

    zf_sig = eq_sig(rx,Q_zf);
end

% MMSE Equalizer
function [mmse_sig,Q_mmse] = mmse(rx,H,noise_pow,Eb)
    H_conj = conj(H);
    H_sq = abs(H).^2;
    Q_mmse = H_conj ./ (H_sq + (noise_pow/Eb));
    Q_mmse = Q_mmse';

    mmse_sig = eq_sig(rx, Q_mmse);
end

%Equalizer Convolution
function [eq_sig] = eq_sig(rx,Q_eq)
    g = fft(rx, length(Q_eq));
    g_eq = g .* Q_eq;
    eq_sig = ifft(g_eq);
end
% 
% % MMSE Eq
% function [mmse_sig,Q_mmse] = mmse(g_rec,H,noise_pow,Eb)
%     H_conj = conj(H);
%     H_sq = abs(H).^2;
%     Q_mmse = H_conj ./ (H_sq + (noise_pow/Eb));
%     Q_mmse = Q_mmse';
% 
%     % Apply MMSE Eq.
%     G = fft(g_rec,length(Q_mmse));
%     G_mmse = G .* Q_mmse;
%     mmse_sig = ifft(G_mmse);
%     % check length
% end
% %
% % zero-forcing equalizer
% function [ZF_ht, ZF_t, ZF_f, ZF_w] = ZFeq(channel_coeffs)
%     [ZF_f, ZF_w] = freqz(1,channel_coeffs,10000);
%     [ZF_ht, ZF_t] = impz(1,channel_coeffs,2048);
% end

%
% matched filtering, provide pulse shape
function [matchedfiltered_sig, matchedfilter] = matched_filtering(sig,pulse)
    matchedfilter = flip(pulse(1:end-1));%fliplr(pulse);
    matchedfiltered_sig = conv(sig,matchedfilter);
end

% channel modulation with choice of 3 defined channels
% returns noisy and clean channel responses, channel coefficients, and
% noise powers
function [noisy_received_sig, clean_channel_sig, channel_coeffs, noisepwr] = channel_modulation(sig, channelselect,noisesel)
    ch1 = [1, 0.5, 0.75, -2/7];
    ch_outdoor = [0.5 1 0 0.63 0 0 0 0 0.25 0 0 0 0.16 0 0 0 0 0 0 0 0 0 0 0 0 0.1];
    ch_indoor = [1 0.4365 0.1905 0.0832 0 0.0158 0 0.003];

    noisepwr_vector = [1e-2, 1e-1]';
   
    switch channelselect
        case 1
            channel_coeffs = ch1;
        case 2
            channel_coeffs = ch_outdoor;
        case 3
            channel_coeffs = ch_indoor;
        otherwise
            channel_coeffs = [];
    end
        upsampled_filter_coeffs = upsample(channel_coeffs,32);
        clean_channel_sig = conv(upsampled_filter_coeffs,sig);

    switch noisesel
        case 1
%             disp("Low Noise Power = 1e-2")
            noisepwr = noisepwr_vector(noisesel);
            n = noisepwr.*randn(size(clean_channel_sig));
            noisy_received_sig = clean_channel_sig + n;
            
        case 2
%             disp("High Noise Power = 1e-1")
            noisepwr = noisepwr_vector(noisesel);
            n = noisepwr.*randn(size(clean_channel_sig));
            noisy_received_sig = clean_channel_sig + n;
    end
end

% modulation with either half-sine or srrc pulses
% returns modulated signal, pulse signal w.r.t. time, and parameters used
function [modulated_sig, pulse, t_pulse, Tb, K, alpha, samps] = pulseshape_modulation(sig, pulseshape, Tb, K, alpha, samps)
    modulated_sig = [];
    
    switch pulseshape
        case 1 % half-sine
            t_pulse = linspace(0,1,samps+1);
            pulse = sin((pi/Tb)*t_pulse);
            pulse = pulse/norm(pulse);
            pulse = pulse(1:32);

            modulated_sig = conv(sig,pulse);

        case 2 % srrc
            t_pulse = linspace(-K*Tb,K*Tb,2*K*samps+1);
            pulse = rcosdesign(alpha, 2*K, samps);
            pulse = pulse/norm(pulse);
            pulse = pulse(1:end-1);     

            modulated_sig = conv(sig,pulse);

    end
end

% Generate side-by-side image of preprocessed/postprocessed input image
% Note the artifacts and crush in reduced color depth when transforming
% 24-bit color to 8-bit/16-bit
function montage_gen(img_in, img_out, col)
    figure,
    montage({img_in(:,1:col,:), img_out(:,col+1:end,:)});
    title("MATLAB's 'peppers.png' before and after processing",FontSize=26);
    subtitle("8-bit DCT coefficients")
end