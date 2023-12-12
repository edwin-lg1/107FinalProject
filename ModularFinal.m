%% Reset Workspace
close all
clear
clc

%%
% EE 107 - Communication Systems - Final Project
% Authors: Nicholas Boudreau, Andrew Chen, Edwin Liang-Gilman

%%
% Modularize process
% 
% image preprocess
filename = 'octopus.jpg.jpeg';
imshow(filename)
% imfinfo(filename)
qbits = 8;
%%
% run
main(filename,qbits)







%% main driver 
function main(img_filename, qbits)
    % image preproc
    [Ztres,r,c,m,n,minval,maxval] = ImagePreProcess_color(img_filename, qbits);
%     [Ztres_gray,r,c,m,n,minval,maxval]=ImagePreProcess_gray(img_filename,qbits)
    img_RGB = imread(img_filename);
%     img_double = im2double(img_RGB);
    numcol = floor(numel(img_RGB(1,:,1))/2);

    % ->
    % bit stream
    % convert size(img_bitstream) to 1-dimensional bit-stream using reshape

    DCTblocks = int2bit(Ztres,8);
    stack = zeros(8,8,size(DCTblocks,3));
    
    for z = 1:size(DCTblocks,3) % 4 blocks
        ak = reshape(DCTblocks(:,:,z),[],1);
        ak_double = cast(ak, 'double');
        bk = 2*ak_double-1;
%         bk = upsample(bk,32);
        img_bitstream = bk;
        n_bits = size(DCTblocks,1)*size(DCTblocks,2);
        % ->
        % modulate with 2 pulse shapes
%         modulated_sig_cellarray = cell(2,1);
%         t_cellarray = cell(2,1);
        
        Tb = 1;
        K = 6;
        alpha = 0.5;
        samps = 32;

        for i = 1%:2 % 1 selects Half-Sine
            [modulated_sig, t, ...
                pulse, pulse_t, Tb, K, alpha, samps] = pulseshape_modulation(img_bitstream, ...
                                                             i, Tb, K, alpha, samps);
        for j = 1%:3 % 1 selects channel 1
            [noisy_received_sig, clean_channel_sig, taps, npwr] = channel_modulation(modulated_sig, j);
        end
        noisy_low = noisy_received_sig(1,:);
        noisy_high = noisy_received_sig(2,:);
        noisy_0 = clean_channel_sig(:);

        [matchedfiltered_sig, matchedfilter] = matched_filtering(noisy_0,pulse);
        
        %EQ
        zero_forced_sig = filter(1,taps,matchedfiltered_sig);
        H = freqz(taps,1);
        noise_pow = 0;
        Eb = 1;
        [mmse_sig,Q_mmse] = mmse(matchedfiltered_sig',H,noise_pow,Eb);

        bitstream_sampled = zeros(n_bits,1);
 
        bitstream_sampled(zero_forced_sig(32:samps:end-(4*32)-1)>0) = 1;

        ak_reconstructed = cast(bitstream_sampled,'uint8');
        ak_bin = reshape(ak_reconstructed,64,8);
        ak_recon = bit2int(ak_bin,8);
        
        
        stack(:,:,z) = ak_recon;
        orig(:,:,z) = Ztres(:,:,1);
        
        

%         [newZ]=ImagePostProcess_color(orig,r,c,m,n,minval,maxval);


%         figure,
%         stem(ak_reconstructed)
%         hold on
%         stem(2*ak_double-1)



%         [ZF_ht, ZF_t, ZF_f, ZF_w] = ZFeq(taps);
        
%         figure,
%         plot(modulated_sig)
%         hold on
%         plot(abs(mmse_sig))
%         grid minor
%         legend()
%         plot(matchedfiltered_sig)
%         plot(zero_forced_sig)



% 
%         figure(99)
%         plot(noisy_received_sig(1,1:10))
%         hold on
%         plot(noisy_received_sig(2,1:10))
%         plot(clean_channel_sig(1:10), 'LineWidth',2)


%     % plot modulated signal
%             figure(i),
%             plot(t,modulated_sig)
      % plot pulse shapes
%             figure(i+2)
%             plot(pulse_t,pulse)
        
        
        end
        



    % ->
    % convolve with 3 channels

    % ->
    % add 3 possible noise levels [0, 0.1, 0.001]
    
    % ->
    % equalize with 2 equalizers
    
    % -> 
%     % sample and detect
%     bk_reconstructed = img_bitstream; %%%temporary testing



    end

    [newZ]=ImagePostProcess_color(stack,r,c,m,n,minval,maxval);
    
   
%     % ->
%     % reshape bitstream into 8x8 for postprocessing
%     img_reshaped_stream = reshape(img_bitstream_received, 8, 8, []);
%     % ->
%     % image postprocess
%     [img_out] = ImagePostProcess_color(img_reshaped_stream,r,c,m,n,minval,maxval);
% 
%     % postproc figures
%     montage_gen(img_RGB, img_out, numcol)
end
%% Supporting Local Functions
%

%
% %
% function []
% 
% MMSE Eq
function [mmse_sig,Q_mmse] = mmse(g_rec,H,noise_pow,Eb)
    H_conj = conj(H);
    H_sq = abs(H).^2;
    Q_mmse = H_conj ./ (H_sq + (noise_pow/Eb));
    Q_mmse = Q_mmse';

    % Apply MMSE Eq.
    G = fft(g_rec,length(Q_mmse));
    G_mmse = G .* Q_mmse;
    mmse_sig = ifft(G_mmse);
end
%
% zero-forcing equalizer
function [ZF_ht, ZF_t, ZF_f, ZF_w] = ZFeq(channel_coeffs)
    [ZF_f, ZF_w] = freqz(1,channel_coeffs,10000);
    [ZF_ht, ZF_t] = impz(1,channel_coeffs,2048);
end

%
% matched filtering, provide pulse shape
function [matchedfiltered_sig, matchedfilter] = matched_filtering(sig,pulse)
    matchedfilter = flip(pulse(1:end-1));%fliplr(pulse);
    matchedfiltered_sig = conv(sig,matchedfilter);
end
%
% channel modulation with choice of 3 defined channels
% returns noisy and clean channel responses, channel coefficients, and
% noise powers
function [noisy_received_sig, clean_channel_sig, filter_coeffs, noisepwr] = channel_modulation(sig, channelselect)
    % channel definition
    ch1 = [1, 0.5, 0.75, -2/7];
    ch_outdoor = [0.5 1 0 0.63 0 0 0 0 0.25 0 0 0 0.16 0 0 0 0 0 0 0 0 0 0 0 0 0.1];
    ch_indoor = [1 0.4365 0.1905 0.0832 0 0.0158 0 0.003];

    noisepwr = [1e-2, 1e-1]';
    


    switch channelselect
        case 1
            filter_coeffs = ch1;
        case 2
            filter_coeffs = ch_outdoor;
        case 3
            filter_coeffs = ch_indoor;
        otherwise
            filter_coeffs = [];
    end
        upsampled_filter_coeffs = upsample(filter_coeffs,32);
        clean_channel_sig = conv(upsampled_filter_coeffs,sig);
        n = noisepwr.*randn(size(clean_channel_sig));
        noisy_received_sig = clean_channel_sig + n;
end



% modulation with either half-sine or srrc pulses
% returns modulated signal, pulse signal w.r.t. time, and parameters used
function [modulated_sig, t, pulse, t_pulse, Tb, K, alpha, samps] = pulseshape_modulation(sig, pulseshape, Tb, K, alpha, samps)
    modulated_sig = [];
    
    switch pulseshape
        case 1 % half-sine
%             modulated_sig = zeros() % how to determine sizing
            % definition
            t_pulse = linspace(0,1,samps+1);
            pulse = sin((pi/Tb)*t_pulse);
            pulse = pulse/norm(pulse);
            % modulation
            for j = 1:length(sig)-1
                modulated_sig = [modulated_sig sig(j)*pulse(1:samps)];
            end
            modulated_sig = [modulated_sig sig(length(sig))*pulse];
            t = linspace(0,length(sig),samps*length(sig)+1);

        case 2 % srrc
            t_pulse = linspace(-K*Tb,K*Tb,2*K*samps+1);
            pulse = rcosdesign(alpha, 2*K, samps);
            pulse = pulse/norm(pulse);

            srrc_ind = [];
            idx0 = 1;
            idx1 = 2*K*samps+1;
            for j = 1:length(sig)
                srrc_ind(j,idx0:idx1) = sig(j)*pulse;
                idx0 = idx0 + samps;
                idx1 = idx1 + samps;
            end

            for k = 1:length(srrc_ind)
                modulated_sig = [modulated_sig sum(srrc_ind(:,k))];
            end
            t = linspace(0,2*length(sig)+1,length(modulated_sig));

    end
end
% % 
% function [srrc_modulated, Tb, samps,]
% 
% end

% Generate side-by-side image of preprocessed/postprocessed input image
% Note the artifacts and crush in reduced color depth when transforming
% 24-bit color to 8-bit/16-bit
function montage_gen(img_in, img_out, col)
    figure,
    montage({img_in(:,1:col,:), img_out(:,col+1:end,:)});
    title("MATLAB's 'peppers.png' before and after processing",FontSize=26);
    subtitle("8-bit DCT coefficients")
end
