% image_test: test pre/post-processing on a default MATLAB image
%% 
clear, close all
clc
%%

qbits = 8;	 %% valid options are 8 and 16
filename = 'onion.png';

%% Show a colored image in color
[Ztres,r,c,m,n,minval,maxval] = ImagePreProcess_color(filename,qbits);
% disp(Ztres)
preimRGB = imread(filename);
[newZ] = ImagePostProcess_color(Ztres,r,c,m,n,minval,maxval);
% disp(newZ)

figure,
montage({preimRGB, newZ})
title("MATLAB's 'Onion.png' before and after processing",FontSize=26);
subtitle("8-bit DCT coefficients")

%% Show a colored image as 3 separate (gray) layers
[Ztres,r,c,m,n,minval,maxval] = ImagePreProcess_gray(filename,qbits);
% disp(Ztres)
preimRGB = imread(filename);
[newZ] = ImagePostProcess_gray(Ztres,r,c,m,n,minval,maxval);

figure,
montage({preimRGB, newZ})


%%%
%% Repeat for 16-bit DCT Coefficients
%%%
qbits = 16;
[Ztres,r,c,m,n,minval,maxval] = ImagePreProcess_color(filename,qbits);
preimRGB = imread(filename);
[newZ] = ImagePostProcess_color(Ztres,r,c,m,n,minval,maxval);

figure,
montage({preimRGB, newZ})
title("MATLAB's 'Onion.png' before and after processing",FontSize=26);
subtitle("16-bit DCT coefficients")

imwrite(newZ,"imtestout.png")

%% no uint32/uint64 functions, but exists im2double? Bit-depth of image is 24-bit
qbits = 64;
[Ztres,r,c,m,n,minval,maxval] = ImagePreProcess_color(filename,qbits);
preimRGB = imread(filename);
[newZ] = ImagePostProcess_color(Ztres,r,c,m,n,minval,maxval);

figure,
montage({preimRGB, newZ})
title("MATLAB's 'Onion.png' before and after processing",FontSize=26);
subtitle("16-bit DCT coefficients")

imwrite(newZ,"imtestout.png")