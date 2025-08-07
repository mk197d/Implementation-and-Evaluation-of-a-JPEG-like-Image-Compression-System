clc; clear; close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_folder = 'msrcorid/msrcorid/animals/cows/general'; 
image_files = dir(fullfile(image_folder, '*.jpg')); 
total_images = length(image_files);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_images = 1; 
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
    
    % Convert to YCbCr
    img_ycbcr = rgb2ycbcr(uint8(img)); 
    Y = img_ycbcr(:,:,1); 
    Cb = img_ycbcr(:,:,2);
    Cr = img_ycbcr(:,:,3);
    [rows, cols, ~] = size(img);

   
    Cb_down = imresize(Cb, 0.5, 'bilinear');
    Cr_down = imresize(Cr, 0.5, 'bilinear');

    [down_rows, down_cols] = size(Cb_down);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % PADDING TO ENSURE INTEGRAL COUNT OF BLOCKS
    rows_padded = ceil(rows / 8) * 8;
    cols_padded = ceil(cols / 8) * 8;

    down_rows_padded = ceil(down_rows / 8) * 8;
    down_cols_padded = ceil(down_cols / 8) * 8;

    padded_Y = padarray(Y, [rows_padded - rows, cols_padded - cols], 'post');
    padded_Cb_down = padarray(Cb_down, [down_rows_padded - down_rows, down_cols_padded - down_cols], 'post');
    padded_Cr_down = padarray(Cr_down, [down_rows_padded - down_rows, down_cols_padded - down_cols], 'post');

    num_pixels = rows * cols;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ORIGINAL AND COMPRESSED IMAGE PLOTS
    total_plots = num_qf + 1; 
    plot_cols = ceil(sqrt(total_plots)); 
    plot_rows = ceil(total_plots / plot_cols); 
    
    figure;
    subplot(plot_rows, plot_cols, 1);
    imshow(uint8(img));
    title('Original Image');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for qf_idx = 1:num_qf

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % QUANTIZE
        scale =  50 / quality_factors(qf_idx);
        Q = max(round(base_Q * scale), 1);

        % Quantize each channel independently
        quantized_Y = dct_quantize_block(padded_Y, Q, rows_padded, cols_padded);
        quantized_Cb_down = dct_quantize_block(padded_Cb_down, Q, down_rows_padded, down_cols_padded);
        quantized_Cr_down = dct_quantize_block(padded_Cr_down, Q, down_rows_padded, down_cols_padded);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ENCODE AND SAVE (for each channel)
        [compressed_Y, huffman_dict_Y] = encode_huffman(quantized_Y);
        [compressed_Cb_down, huffman_dict_Cb] = encode_huffman(quantized_Cb_down);
        [compressed_Cr_down, huffman_dict_Cr] = encode_huffman(quantized_Cr_down);
        
        save('compressed_data.mat', 'compressed_Y', 'huffman_dict_Y', ...
            'compressed_Cb_down', 'huffman_dict_Cb', ...
            'compressed_Cr_down', 'huffman_dict_Cr', 'rows', 'cols', 'Q');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % LOAD, DECODE AND RECONSTRUCT (for each channel)
        load('compressed_data.mat');
        file_info = dir('compressed_data.mat'); 
        compressed_bytes = file_info.bytes;     
        compressed_bits = compressed_bytes * 8;

        decoded_Y = decode_huffman(compressed_Y, huffman_dict_Y, rows_padded, cols_padded);
        decoded_Cb_down = decode_huffman(compressed_Cb_down, huffman_dict_Cb, down_rows_padded, down_cols_padded);
        decoded_Cr_down = decode_huffman(compressed_Cr_down, huffman_dict_Cr, down_rows_padded, down_cols_padded);
        
        % Inverse DCT and Dequantization for each channel
        reconstructed_Y = idct_reconstruct(decoded_Y, Q, rows_padded, cols_padded);
        reconstructed_Cb_down = idct_reconstruct(decoded_Cb_down, Q, down_rows_padded, down_cols_padded);
        reconstructed_Cr_down = idct_reconstruct(decoded_Cr_down, Q, down_rows_padded, down_cols_padded);
        
        % Upsample the Cb and Cr channels
        reconstructed_Cb = imresize(reconstructed_Cb_down, 2, 'bilinear'); % Upsample back to original size
        reconstructed_Cr = imresize(reconstructed_Cr_down, 2, 'bilinear'); % Upsample back to original size
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CROP and convert to RGB
        reconstructed_Y = reconstructed_Y(1:rows, 1:cols);
        reconstructed_Cb = reconstructed_Cb(1:rows, 1:cols);
        reconstructed_Cr = reconstructed_Cr(1:rows, 1:cols);
        
        % Combine the YCbCr channels and convert back to RGB
        reconstructed_img_ycbcr = cat(3, reconstructed_Y, reconstructed_Cb, reconstructed_Cr);
        reconstructed_img_rgb = ycbcr2rgb(uint8(reconstructed_img_ycbcr));
        % 
        subplot(plot_rows, plot_cols, qf_idx + 1);
        imshow(uint8(reconstructed_img_rgb));
        title(['QF = ', num2str(quality_factors(qf_idx))]);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Convert reconstructed image to double for RMSE calculation
        reconstructed_img_rgb = double(reconstructed_img_rgb); 
        
        % RMSE
        error = reconstructed_img_rgb - img;
        orgNorm = mean(img(:).^2);
        RMSE(img_idx, qf_idx) = sqrt(mean(error(:).^2) / orgNorm);
        
        BPP(img_idx, qf_idx) = compressed_bits / num_pixels;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % figure;
    % subplot(plot_rows, plot_cols, 1);
    % imshow(uint8(img));
    % title('Original Image JPEG');

    % for qf_idx = 1:num_qf
    % 
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %     % JPEG COMPRESSION USING BUILT-IN FUNCTION
    %     jpeg_file = 'temp.jpg';
    %     imwrite(uint8(img), jpeg_file, 'jpg', 'Quality', quality_factors(qf_idx));
    %     file_info = dir(jpeg_file);
    %     jpeg_bytes = file_info.bytes;
    %     BPP_jpeg(img_idx, qf_idx) = (jpeg_bytes * 8) / num_pixels;
    % 
    %     reconstructed_jpeg = double(imread(jpeg_file));
    %     orgNorm = mean(img(:).^2);
    %     error = reconstructed_jpeg - img;
    %     RMSE_jpeg(img_idx, qf_idx) = sqrt(mean(error(:).^2) / orgNorm);
    %     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 
    %     % subplot(plot_rows, plot_cols, qf_idx + 1);
    %     % imshow(uint8(reconstructed_jpeg));
    %     % title(['QF = ', num2str(quality_factors(qf_idx))]);
    % end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INDIVIDUAL PLOTS
for img_idx = 1:num_images
    figure;
    plot(BPP(img_idx, :), RMSE(img_idx, :), '-o', 'DisplayName', ...
        ['Image ', num2str(img_idx)]);
    xlabel('Bits Per Pixel (BPP)');
    ylabel('RMSE');
    title('RMSE vs BPP');
    legend show;
    grid on;
    saveas(gcf, ['coloured_individual' num2str(img_idx + 4) '.png']);
end

% % CUMULATIVE PLOTS
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
% saveas(gcf, 'coloured_20_plots.png');

% COMPARISON
% for img_idx = 1:num_images
%     figure;
%     xlabel('Bits Per Pixel (BPP)');
%     ylabel('RMSE');
%     grid on;
%     hold on;
%     plot(BPP(img_idx, :), RMSE(img_idx, :), '-o', 'DisplayName', 'Manual JPEG');
%     plot(BPP_jpeg(img_idx, :), RMSE_jpeg(img_idx, :), '-x', 'DisplayName', 'Built-in JPEG');
%     legend show;
%     saveas(gcf, ['coloured_with_jpeg' num2str(img_idx) '.png']);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HELPER FUNCTIONS
%---------------------------------------------------------------------------------------------------------%
function quantized = dct_quantize_block(img, Q, rows_padded, cols_padded)
    quantized = zeros(rows_padded, cols_padded);
    for i = 1:8:rows_padded
        for j = 1:8:cols_padded
            block = img(i:i+7, j:j+7);
            dct_block = dct2(block);
            quantized_block = round(dct_block ./ Q);
            quantized(i:i+7, j:j+7) = quantized_block;
        end
    end
end
%---------------------------------------------------------------------------------------------------------%

%---------------------------------------------------------------------------------------------------------%
function [compressed, huffman_dict] = encode_huffman(quantized_img)
    symbols = unique(quantized_img);
    counts = histcounts(quantized_img(:), [symbols; max(symbols)+1]);
    huffman_dict = huffmandict(symbols, counts / numel(quantized_img));
    compressed = huffmanenco(quantized_img(:), huffman_dict);
end
%---------------------------------------------------------------------------------------------------------%

%---------------------------------------------------------------------------------------------------------%
function decoded = decode_huffman(compressed, huffman_dict, rows_padded, cols_padded)
    decoded_stream = huffmandeco(compressed, huffman_dict);
    decoded = reshape(decoded_stream, rows_padded, cols_padded);
end
%---------------------------------------------------------------------------------------------------------%

%---------------------------------------------------------------------------------------------------------%
function reconstructed = idct_reconstruct(quantized_img, Q, rows_padded, cols_padded)
    reconstructed = zeros(rows_padded, cols_padded);
    for i = 1:8:rows_padded
        for j = 1:8:cols_padded
            quantized_block = quantized_img(i:i+7, j:j+7);
            dequantized_block = quantized_block .* Q; % Inverse quantization
            idct_block = idct2(dequantized_block);    % Inverse DCT
            reconstructed(i:i+7, j:j+7) = idct_block;
        end
    end
end
%---------------------------------------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

