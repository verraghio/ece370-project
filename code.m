%% Setup and Initialization

Tx = 2;    
Rx = 6;    
M = 16;     % Modulation Order 
K_search = 4;   % Paper suggests K=4 for 16-QAM

K = log2(M);    % Bits per symbol

numSc = 12;     % No. of subcarriers
numSym = 7;     % No. of OFDM Symbols

snrRange_dB = 5:2.5:15;     % Defining SNR range in dB
snrRange_linear = 10.^(snrRange_dB / 10);   % dB to Linear 
numTrials = 750;    % no. of independent experiments

totalSymbols = numTrials * numSc * numSym * Tx; 
% This represents the total no. of QAM symbols sent across the entire 
% duration of the simulation for each SNR value.
% SER = Total Errors / Total Symbols

% These arrays will store SER values for different decoding methods. 
SER_ZF = zeros(size(snrRange_linear));
SER_MMSE = zeros(size(snrRange_linear));
SER_ML = zeros(size(snrRange_linear));
SER_HybridZF = zeros(size(snrRange_linear));
SER_HybridMMSE = zeros(size(snrRange_linear));

data = 0:M-1;   % symbol indices
const = qammod(data, M);    % default rectangular 16-QAM

% normalizing to unit average power
const = const / sqrt(mean(abs(const).^2));

%% Main Simulation Loop 

% Running two loops, outer loop iterates over the SNR values and inner loop does the actual estimation/decoding work.
for i = 1:length(snrRange_linear)

    % Initial Symbol Error = 0
    totalSymError_ZF = 0;
    totalSymError_MMSE = 0;
    totalSymError_ML = 0;
    totalSymError_HybridZF = 0;
    totalSymError_HybridMMSE = 0;

    % We repeat the experiment multiple time for the same SNR value so as to get the average 
    for j = 1:numTrials

        totalSymSent = Tx * numSc * numSym;     % 12*7*2 = 168
        totalBits = totalSymSent * K;   % 168*4 = 672
        txBits = randi([0 1], 1, totalBits);  % Generating 672 random integers[0,1]       

        % Modulating 1s and 0s to the constellation points. Normalizing symbol power ensures that our SNR calculation are correct. 
        txSymbols = qammod(txBits.', M, 'InputType', 'bit', 'UnitAveragePower', true).';

        % Spatial Multiplexing. we split the long stream of symbols into Tx no. of columns
        X = reshape(txSymbols, [], Tx);

        current_snr = snrRange_linear(i);
        % SNR = Psignal/Pnoise = 1/Pnoise => Pnoise = 1/SNR
        noise_var = 1 / current_snr;

        for k = 1:size(X, 1)

            % ZF Decoding
            x_k = X(k, :).'; % Grabs the k-th row and performs a transpose. H*x requires x to be a column vector
            H = (randn(Rx, Tx) + 1i*randn(Rx, Tx)) / sqrt(2);  % MIMO channel matrix (rayleigh fading)
            n = sqrt(noise_var/2) * (randn(Rx, 1) + 1i*randn(Rx, 1));

            %System Eqn
            y = H * x_k + n;

            W_zf = pinv(H);
            x_est_zf = W_zf * y;

            detected_syms = qamdemod(x_est_zf, M, 'UnitAveragePower', true);
            actual_syms = qamdemod(x_k, M, 'UnitAveragePower', true);
            totalSymError_ZF = totalSymError_ZF + sum(detected_syms ~= actual_syms);

            % MMSE Decoding
            W_mmse = (H' * H + noise_var * eye(Tx)) \ H';
            x_est_mmse = W_mmse * y;
            detected_syms_mmse = qamdemod(x_est_mmse, M, 'UnitAveragePower', true);
            totalSymError_MMSE = totalSymError_MMSE + sum(detected_syms_mmse ~= actual_syms);

            % ML Decoding 
            min_dist = inf;
            detected_ml = zeros(Tx, 1);

            % Nested loops to check all combinations for both antennas (16 * 16 = 256 checks)
            for i1 = 0:M-1
                for i2 = 0:M-1

                    % Create a candidate symbol vector
                    % We map indices i1, i2 to constellation points
                    sym1 = const(i1+1); 
                    sym2 = const(i2+1);
                    x_cand = [sym1; sym2];
                    
                    % Calculate Euclidean Distance: ||y - H*x_cand||^2
                    dist = sum(abs(y - H * x_cand).^2);
                    
                    % If this is the best match so far, save it
                    if dist < min_dist
                        min_dist = dist;
                        detected_ml = [i1; i2]; % Save indices directly
                    end
                end
            end
            totalSymError_ML = totalSymError_ML + sum(detected_ml ~= actual_syms);

            % Hybrid ML
            estimates_to_refine = [x_est_zf, x_est_mmse]; 
            detected_hybrid = zeros(Tx, 2); % Col 1 = Hybrid-ZF, Col 2 = Hybrid-MMSE

            for method = 1:2
                % Get the initial estimate (Col 1 is ZF, Col 2 is MMSE)
                s_initial = estimates_to_refine(:, method);
                
                % Get K Nearest Neighbors for each antenna 
                candidate_indices = zeros(Tx, K_search);
                
                for ant = 1:Tx
                    % Calculate distance from estimate to all 16 constellation points
                    dists = abs(s_initial(ant) - const); 
                    
                    % Sort and pick the indices of the closest K points
                    [~, sorted_idx] = sort(dists); 
                    candidate_indices(ant, :) = sorted_idx(1:K_search);
                end
                
                % Form Subset G (All combinations of neighbors)
                [idx1, idx2] = ndgrid(candidate_indices(1,:), candidate_indices(2,:));
                
                % Convert indices back to actual complex symbols
                % These are our "Reduced Search Candidates"
                cand_sym1 = const(idx1(:));
                cand_sym2 = const(idx2(:));
                candidate_vectors = [cand_sym1; cand_sym2]; % Size: 2x16
                
                % ML Search within Subset G
                % Calculate distances: || y - H * s_candidate ||^2
                % H * candidates
                rx_preds = H * candidate_vectors; 
                
                % Calculate errors
                errors = sum(abs(y - rx_preds).^2, 1);
                
                % Find the minimum error in this small batch
                [~, best_idx] = min(errors);
                detected_hybrid(:, method) = qamdemod(candidate_vectors(:, best_idx), M, 'UnitAveragePower', true);
            end
        
            % Extract results
            detected_hybrid_zf = detected_hybrid(:, 1);
            detected_hybrid_mmse = detected_hybrid(:, 2);
            
            % --- Count Errors ---
            totalSymError_HybridZF = totalSymError_HybridZF + sum(detected_hybrid_zf ~= actual_syms);
            totalSymError_HybridMMSE = totalSymError_HybridMMSE + sum(detected_hybrid_mmse ~= actual_syms);
        end

    end

    denominator = numTrials * numSc * numSym * Tx;

    SER_ZF(i) = totalSymError_ZF / denominator;
    SER_MMSE(i) = totalSymError_MMSE / denominator;
    SER_ML(i) = totalSymError_ML / denominator;
    SER_HybridZF(i)   = totalSymError_HybridZF / denominator;
    SER_HybridMMSE(i) = totalSymError_HybridMMSE / denominator;

    fprintf('SNR:%4.1f dB | ZF:%.1e | MMSE:%.1e | ML:%.1e | Hyb-ZF:%.1e | Hyb-MMSE:%.1e\n', ...
            snrRange_dB(i), ...
            SER_ZF(i), ...
            SER_MMSE(i), ...
            SER_ML(i), ...
            SER_HybridZF(i), ...
            SER_HybridMMSE(i));
end

%% Plotting the Results
figure;

% 1. Standard Linear Detectors
semilogy(snrRange_dB, SER_ZF,   'b-o', 'LineWidth', 1.5, 'MarkerSize', 7);
hold on;
semilogy(snrRange_dB, SER_MMSE, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 7);

% 2. True Maximum Likelihood (The Benchmark)
semilogy(snrRange_dB, SER_ML,   'k-d', 'LineWidth', 2,   'MarkerSize', 8);

% 3. Proposed Hybrid Detectors
semilogy(snrRange_dB, SER_HybridZF,   'g-^', 'LineWidth', 1.5, 'MarkerSize', 8);
semilogy(snrRange_dB, SER_HybridMMSE, 'm-*', 'LineWidth', 1.5, 'MarkerSize', 8);

% --- Graph Formatting ---
grid on;
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
title(['MIMO Detection Comparison (' num2str(Tx) 'x' num2str(Rx) ' 16-QAM)']);

% Legend
legend('Zero Forcing', 'MMSE', 'Maximum Likelihood (ML)', ...
       'Hybrid ZF-ML', 'Hybrid MMSE-ML', ...
       'Location', 'southwest');

% Clean up axes
xlim([min(snrRange_dB) max(snrRange_dB)]);
ylim([1e-5 1]); % Optional: prevents graph from scaling to -Inf if errors are 0;