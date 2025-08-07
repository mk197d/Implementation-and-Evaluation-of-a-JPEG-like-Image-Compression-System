clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_folder = 'BWDataset'; 
image_files = dir(fullfile(image_folder, '*.*')); 
total_images = length(image_files);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_images = 4; 
random_indices = randperm(total_images, num_images); 
selected_files = image_files(random_indices); 

quality_factors = 10:10:100; 
num_qf = length(quality_factors);
RMSE = zeros(num_images, num_qf);
BPP = zeros(num_images, num_qf);

RMSE_jpeg = zeros(num_images, num_qf);
BPP_jpeg = zeros(num_images, num_qf);
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
        img = rgb2gray(img);
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
    total_plots = num_qf + 1; 
    plot_cols = ceil(sqrt(total_plots)); 
    plot_rows = ceil(total_plots / plot_cols); 
    
    % figure;
    % subplot(plot_rows, plot_cols, 1);
    % imshow(uint8(img));
    % title('Original Image');
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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FOR COMPARISON WITH JPEG

    % figure;
    % subplot(plot_rows, plot_cols, 1);
    % imshow(uint8(img));
    % title('Original Image JPEG');
    % 
    for qf_idx = 1:num_qf

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % JPEG COMPRESSION USING BUILT-IN FUNCTION
        jpeg_file = 'temp.jpg';
        imwrite(uint8(img), jpeg_file, 'jpg', 'Quality', quality_factors(qf_idx));
        file_info = dir(jpeg_file);
        jpeg_bytes = file_info.bytes;
        BPP_jpeg(img_idx, qf_idx) = (jpeg_bytes * 8) / num_pixels;

        reconstructed_jpeg = double(imread(jpeg_file));
        orgNorm = mean(img(:).^2);
        error = reconstructed_jpeg - img;
        RMSE_jpeg(img_idx, qf_idx) = sqrt(mean(error(:).^2) / orgNorm);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % subplot(plot_rows, plot_cols, qf_idx + 1);
        % imshow(uint8(reconstructed_jpeg));
        % title(['QF = ', num2str(quality_factors(qf_idx))]);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INDIVIDUAL PLOTS
% for img_idx = 1:num_images
%     figure;
%     plot(BPP(img_idx, :), RMSE(img_idx, :), '-o', 'DisplayName', ...
%         ['Image ', num2str(img_idx)]);
%     xlabel('Bits Per Pixel (BPP)');
%     ylabel('RMSE');
%     title('RMSE vs BPP');
%     legend show;
%     grid on;
%     saveas(gcf, ['gray_individual' num2str(img_idx) '.png']);
% end
% 
% CUMULATIVE PLOTS
% figure;
% xlabel('Bits Per Pixel (BPP)');
% ylabel('RMSE');
% title('RMSE vs BPP for 20 images');
% grid on;
% hold on;
% for img_idx = 1:num_images
%     plot(BPP(img_idx, :), RMSE(img_idx, :), '-o', 'DisplayName', ...
%         ['Image ', num2str(img_idx)]);
% end
% hold off;
% saveas(gcf, 'gray_20_plots.png');

% COMPARISON

for img_idx = 1:num_images
    figure;
    xlabel('Bits Per Pixel (BPP)');
    ylabel('RMSE');
    grid on;
    legend show;
    hold on;
    title('RMSE vs BPP');
    plot(BPP(img_idx, :), RMSE(img_idx, :), '-o', 'DisplayName', 'Manual JPEG');
    plot(BPP_jpeg(img_idx, :), RMSE_jpeg(img_idx, :), '-x', 'DisplayName', 'Built-in JPEG');
    saveas(gcf, ['gray_with_jpeg' num2str(img_idx) '.png']);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


