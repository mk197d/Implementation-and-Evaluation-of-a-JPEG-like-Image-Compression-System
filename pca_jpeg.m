clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_folder = 'singleImage'; 
image_files = dir(fullfile(image_folder, '*.png')); 
total_images = length(image_files);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_images = 1; 
random_indices = randperm(total_images, num_images); 
selected_files = image_files(random_indices); 

components_range = 16:2:64; 
num_comp = length(components_range);

quality_factors = 10:10:100;
num_qf = length(quality_factors);
RMSE = zeros(num_images, num_qf);
BPP = zeros(num_images, num_qf);

RMSE_pca = zeros(num_images, num_comp);
BPP_pca = zeros(num_images, num_comp);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
base_Q = [16 11 10 16 24 40 51 61;
          12 12 14 19 26 58 60 55;
          14 13 16 24 40 57 69 56;
          14 17 22 29 51 87 80 62;
          18 22 37 56 68 109 103 77;
          24 35 55 64 81 104 113 92;
          49 64 78 87 103 121 120 101;
          72 92 95 98 112 100 103 99];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for img_idx = 1:num_images

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    img_path = fullfile(image_folder, selected_files(img_idx).name);
    img = double(imread(img_path));
    if size(img, 3) == 3
        % img = rgb2gray(img);
        img = img(:, :, 1);
        % figure;
        % imshow(uint8(img));
    end
    [rows, cols] = size(img);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PADDING TO ENSURE INTEGRAL COUNT OF BLOCKS
    rows_padded = ceil(rows / 8) * 8;
    cols_padded = ceil(cols / 8) * 8;
    padded_img = padarray(img, [rows_padded - rows, cols_padded - cols], 'post');
    num_pixels = rows * cols;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ORIGINAL AND COMPRESSED IMAGE PLOTS
    total_plots = num_comp + 1; 
    plot_cols = ceil(sqrt(total_plots)); 
    plot_rows = ceil(total_plots / plot_cols); 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for qf_idx = 1:num_qf

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % QUANTIZE

        scale =  50 / quality_factors(qf_idx);
        Q = max(round(base_Q * scale), 1);
        
        quantized_coefs = zeros(rows_padded, cols_padded);
        for i = 1:8:rows_padded
            for j = 1:8:cols_padded
                block = padded_img(i:i+7, j:j+7);
                dct_block = dct2(block);
                quantized_block = round(dct_block ./ Q);
                quantized_coefs(i:i+7, j:j+7) = quantized_block;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENCODE AND SAVE
        symbols = unique(quantized_coefs);
        counts = histcounts(quantized_coefs(:), [symbols; max(symbols)+1]);
        huffman_dict = huffmandict(symbols, counts / numel(quantized_coefs));
        compressed_stream = huffmanenco(quantized_coefs(:), huffman_dict);
        
        save('compressed_data.mat', 'compressed_stream', 'huffman_dict', 'rows', 'cols', 'Q');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % LOAD, DECODE AND RECONSTRUCT
        load('compressed_data.mat');
        file_info = dir('compressed_data.mat'); 
        compressed_bytes = file_info.bytes;     
        compressed_bits = compressed_bytes * 8; 

        decoded_stream = huffmandeco(compressed_stream, huffman_dict);
        
        decoded_quantized = reshape(decoded_stream, rows_padded, cols_padded);
        
        reconstructed_img = zeros(rows_padded, cols_padded);
        for i = 1:8:rows_padded
            for j = 1:8:cols_padded
                quantized_block = decoded_quantized(i:i+7, j:j+7);
                dequantized_block = quantized_block .* Q; % Inverse quantization
                idct_block = idct2(dequantized_block);    % Inverse DCT
                reconstructed_img(i:i+7, j:j+7) = idct_block;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CROP
        reconstructed_img = reconstructed_img(1:rows, 1:cols);
        % 
        % subplot(plot_rows, plot_cols, qf_idx + 1);
        % imshow(uint8(reconstructed_img));
        % title(['QF = ', num2str(quality_factors(qf_idx))]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % RMSE
        error = reconstructed_img - img;
        orgNorm = mean(img(:).^2);
        RMSE(img_idx, qf_idx) = sqrt(mean(error(:).^2) / orgNorm);
        BPP(img_idx, qf_idx) = (compressed_bits / num_pixels);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_images = 1; 
random_indices = randperm(total_images, train_images); 
selected_files_ = image_files(random_indices);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
patchSize = 8; 
patches = {}; 
patchCount = 0; 

for i = 1:train_images
    img_path = fullfile(image_folder, selected_files_(i).name);
    im = (imread(img_path));
    if size(im, 3) == 3
        % img = rgb2gray(img);
        im = im(:, :, 1);
    end
    im = double(im);
   
    [H, W] = size(im); 
    
    for j = 1:patchSize:H-patchSize+1
        for k = 1:patchSize:W-patchSize+1
            v = im(j:j+patchSize-1, k:k+patchSize-1);     
            patchCount = patchCount + 1;
            patches{patchCount} = v; 
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CR = zeros(patchSize); CC = CR;
for i=1:patchCount
    v = patches{i};
    CC = CC + v*v'; 
    CR = CR + v'*v; 
end
CC = CC/(patchCount-1);
CR = CR/(patchCount-1);

[VR,DR] = eig(CR);
[VC,DC] = eig(CC);
VR = VR(:,patchSize:-1:1);
VC = VC(:,patchSize:-1:1); 

PCA_basis = kron(VR,VC);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for cnt = 1:num_images
    img_path = fullfile(image_folder, selected_files(cnt).name);
    img = (imread(img_path));
    if size(img, 3) == 3
        img = img(:, :, 1);
    end
    img = double(img);
    
    [H, W] = size(img);
    pad_H = mod(patchSize - mod(H, patchSize), patchSize);
    pad_W = mod(patchSize - mod(W, patchSize), patchSize);
    img_padded = padarray(img, [pad_H, pad_W], 'replicate', 'post');
    
    [H_padded, W_padded] = size(img_padded);
    num_patches = (H_padded / patchSize) * (W_padded / patchSize);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    coefficients = zeros(patchSize^2, num_patches);
    patch_idx = 1;
    for j = 1:patchSize:H_padded
        for k = 1:patchSize:W_padded
            patch = img_padded(j:j+patchSize-1, k:k+patchSize-1);
            patch_vector = patch(:); 
            
            coeff = PCA_basis \ patch_vector; 
            coefficients(:, patch_idx) = coeff;
            
            patch_idx = patch_idx + 1;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    coefficients = round(coefficients);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for mnt = 1:num_comp
        numComponents = components_range(mnt);
        reconstructed_patches = zeros(patchSize, patchSize, num_patches);
        patch_idx = 1;
        
        for j = 1:patchSize:H_padded
            for k = 1:patchSize:W_padded
                coeff_top_k = coefficients(:, patch_idx);
                coeff_top_k(numComponents+1:end) = 0; 
        
                patch_vector_reconstructed = PCA_basis * coeff_top_k;
                patch_reconstructed = reshape(patch_vector_reconstructed, [patchSize, patchSize]);
             
                reconstructed_patches(:, :, patch_idx) = patch_reconstructed;
                patch_idx = patch_idx + 1;
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        reconstructed_image = zeros(H_padded, W_padded);
        patch_idx = 1;
        
        for j = 1:patchSize:H_padded
            for k = 1:patchSize:W_padded
                reconstructed_image(j:j+patchSize-1, k:k+patchSize-1) = reconstructed_patches(:, :, patch_idx);
                patch_idx = patch_idx + 1;
            end
        end
        
        reconstructed_image = reconstructed_image(1:H, 1:W);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        error = reconstructed_image - img;
        orgNorm = mean(img(:).^2);
        RMSE_pca(cnt, mnt) = sqrt(mean(error(:).^2) / orgNorm);
        
        BPP_pca(cnt, mnt) = (numComponents / 64 * 8);
    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% COMPARISON
for img_idx = 1:num_images
    figure;
    xlabel('Bits Per Pixel (BPP)');
    ylabel('RMSE');
    grid on;
    legend show;
    hold on;
    title('RMSE vs BPP');
    plot(BPP(img_idx, :), RMSE(img_idx, :), '-o', 'DisplayName', 'Our JPEG');
    plot(BPP_pca(img_idx, :), RMSE_pca(img_idx, :), '-x', 'DisplayName', 'PCA');
    saveas(gcf, ['cpca_single_doraemon' num2str(img_idx) '.png']);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


