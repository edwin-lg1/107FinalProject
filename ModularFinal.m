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
filename = 'peppers.png';
% imfinfo(filename)
imgRGB = imread(filename);

main(filename,8)







%% main driver 
function main(img_filename, qbits)
    % image preproc
    [Ztres,r,c,m,n,minval,maxval] = ImagePreProcess_color(img_filename, qbits);
    img_RGB = imread(img_filename);
    numcol = floor(numel(img_RGB(1,:,1))/2);
    % ->
    % bit stream
    % convert size(img_bitstream) to 1-dimensional bit-stream using reshape
    img_bitstream = reshape(Ztres,[1 numel(Ztres)]);
                

    % ->
    % modulate with 2 pulse shapes
%     temp = img_bitstream
%                 size(temp)
%                 numel(temp)
    % ->
    % convolve with 3 channels

    % ->
    % add 3 possible noise levels [0, 0.1, 0.001]
    
    % ->
    % equalize with 2 equalizers
    
    % -> 
    % sample and detect
    img_bitstream_received = img_bitstream; %%%temporary testing
    % ->
    % reshape bitstream into 8x8 for postprocessing
    img_reshaped_stream = reshape(img_bitstream_received,[8 8 numel(img_bitstream_received)/64]);
    % ->
    % image postprocess
    [img_out] = ImagePostProcess_color(img_reshaped_stream,r,c,m,n,minval,maxval);

    % postproc figures
    montage_gen(img_RGB, img_out, numcol)
end
%% Supporting Local Functions
%

% Generate side-by-side image of preprocessed/postprocessed input image
% Note the artifacts and crush in reduced color depth when transforming
% 24-bit color to 8-bit/16-bit
function montage_gen(img_in, img_out, col)
    figure,
    montage({img_in(:,1:col,:), img_out(:,col+1:end,:)});
    title("MATLAB's 'peppers.png' before and after processing",FontSize=26);
    subtitle("8-bit DCT coefficients")
end
