close all;
refresh = true;

if refresh || ~exist('X_total', 'var')
    clc;
    clear;
    fprintf('Loading ducks_snapshot_matrix.mat...\n');
    load('ducks_snapshot_matrix.mat'); % This loads the 'ducks' matrix into workspace
    X_total = double(ducks); 
end

X = X_total;
vid_width  = 320;   % e.g., 320 pixels
vid_height = 180;   % e.g., 240 pixels
num_channels = 1;   % 1 for Grayscale, 3 for RGB Color
%% Setting the parameters
X_total_size = size(X_total,2);
frame_start = 1;
frame_end   = X_total_size;
X = X_total(:, frame_start:frame_end);
[n, m] = size(X);
numsnapshots = size(X_total, 2);   % keep this as full video length

dt = 5e-4;                               % True time step (0.0005 seconds) - Given my CONFIG
total_time = dt * (m - 1);               % True total duration of analyzed snapshots
t = (0:dt:total_time)';
%% STARTING MRDMD
L = 4; 
maxJ = 2^(L-1); 
matrices = cell(1, maxJ);       % List of matrices that we update to run DMD through
matrices{1} = X;

% Tracking variables
list_ml = zeros(L, maxJ);           % List of truncation number for each level, bin
list_b = cell(L, maxJ);             % List of starting vector coefficients for each level, bin mode
list_w = cell(L, maxJ);             % List of eigenvalues for each level, bin mode
list_modes = cell(L, maxJ);         % List of modes for each level, bin
list_bin_widths = zeros(L, maxJ);   % List of bin widths for each level, bin
list_t_start = zeros(L, maxJ);      % List of bin start points for each level, bin
list_t_start(1,1) = 1;
list_bin_widths(1,1) = m;
freq_threshold_hz = 1000;

for i = 1:L
    J = 2^(i-1);
    next_matrices = cell(1, 2*J);   % Setting an empty list of matrices for the NEXT level
    count = 1;                      % Used to go to the next spot in the next_matrices list
    
    for j = 1:J
        A = matrices{j};            % Get the next matrix from the previous matrices list
        
        if isempty(A) || size(A, 2) < 10 || any(isnan(A(:)))    % If A is empty or too small, 
            next_matrices{count} = []; count = count + 1;       % then add an empty matrix for the 
            next_matrices{count} = []; count = count + 1;       % next list and skip the process
            continue;
        end
        
        current_start = list_t_start(i, j);                     % For (1,1), starts at 0
        current_width = size(A, 2);
        list_bin_widths(i, j) = current_width;                  % For (1,1), starts at 0
        
        level_dt = dt;                                          % So we set dt = 0.0005 seconds earlier
        [modes, D, b] = dmd(A);                                 % Run DMD
        timeperiod = size(D, 1);                                % Size of D (depends on our r value)
        
        % Calculate continuous frequencies
        freqs = zeros(timeperiod, 1);                   
        for modenum = 1:timeperiod
            lambda = D(modenum, modenum);                       % For each number in our number of modes
            omega = log(lambda) / level_dt;                     % Making this a continuous eigenvalue
            freqs(modenum) = abs(imag(omega)) / (2 * pi);       % Finding the corresponding frequency
        end
        

        [sorted_freqs, sort_idx] = sort(freqs);                 % Sort frequencies ascending
        D = D(sort_idx, sort_idx);                              % Sort D, modes, and b in the same way
        modes = modes(:, sort_idx);
        b = b(sort_idx);
        
        ml = find(sorted_freqs >= freq_threshold_hz, 1, 'first'); % Find the first frequency to be higher than the threshold
        
        if isempty(ml)                                          % If none are higher, it flags ALL MODES
            ml = timeperiod + 1;                                % as being SLOW MODES
            slow_inds = 1:timeperiod;
        else
            slow_inds = 1:(ml-1);                               % Or else it flags all up to that index
        end
        
        if ~isempty(slow_inds)                                  % As long as there exist slow modes
            eigs_slow = diag(D(slow_inds, slow_inds));
            mag = abs(eigs_slow);                               
            over = mag > 1.0;                                   % If the magnitude of the discrete eigenvalues
            eigs_slow(over) = eigs_slow(over) ./ mag(over);     % is greater than 1, project it onto 1 unit circle
                                                                
            time_powers = eigs_slow .^ (0:current_width-1);     % basically Omega Matrix
            slowmatrix = modes(:, slow_inds) * (b(slow_inds) .* time_powers);
        else
            slowmatrix = zeros(n, current_width);               % If there are no slow modes, just make zeros
        end
        
        fastmatrix = A - slowmatrix; 
        % Storing all the variables
        list_ml(i, j) = ml;                                     
        list_w{i, j} = diag(D(slow_inds, slow_inds));           % Store eigenvalues as a vector
        list_b{i, j} = b(slow_inds);
        list_modes{i, j} = modes(:, slow_inds);
        
        % Splitting Step: Split the new fastmatrix into half
        midpoint = floor(current_width / 2);
        A1 = fastmatrix(:, 1:midpoint);
        A2 = fastmatrix(:, (midpoint + 1):end);
        
        left_child_idx = 2*j - 1;
        next_matrices{left_child_idx} = A1;
        right_child_idx = 2*j;
        next_matrices{right_child_idx} = A2;
        if i < L
            list_t_start(i+1, left_child_idx) = current_start;             % Setting starts for the NEXT LEVEL
            list_bin_widths(i+1, left_child_idx) = midpoint;

            list_t_start(i+1, right_child_idx) = current_start + midpoint;
            list_bin_widths(i+1, right_child_idx) = current_width - midpoint;
        end
    end
    matrices = next_matrices; % Update the new matrix list
end


%% Reconstruction Loop

X_rec = zeros(n, m);
for i = 1:L
    J = 2^(i-1);
    for j = 1:J
        if isempty(list_modes{i, j})
            continue;
        end
        modes = list_modes{i, j};
        eigs_slow = list_w{i, j};
        b = list_b{i, j};
        
        t_start = list_t_start(i, j);                                      % Setting start/end - modes only exist locally
        bin_width = list_bin_widths(i, j);
        
        if bin_width == 0
            continue;
        end
        
        t_end = t_start + bin_width - 1;
        mag = abs(eigs_slow);
        over = mag > 1.0;                                                  % Project eigs>1 to 1
        eigs_slow(over) = eigs_slow(over) ./ mag(over);
        
        time_powers = eigs_slow .^ (0:bin_width-1);
        local_rec = modes * (b .* time_powers);
        
        % Protect against matrix index rounding overflows at tree boundaries
        if t_end > m                                                       % Make overflow over m into m
            t_end = m;
            local_rec = local_rec(:, 1:(t_end - t_start + 1));
        end

        X_rec(:, t_start:t_end) = X_rec(:, t_start:t_end) + local_rec;     % Add on the local mode data
    end
end
X_rec = real(X_rec); 

%{
%% Quick display of what we got: 
total_modes = 0;
for i = 1:L
    J = 2^(i-1);
    for j = 1:J
        if ~isempty(list_modes{i,j})
            % The number of columns equals the number of modes in this bin
            num_modes_in_bin = size(list_modes{i,j}, 2); 
            total_modes = total_modes + num_modes_in_bin;
            fprintf('Level %d, Bin %d has %d slow modes\n', i, j, num_modes_in_bin);
        end
    end
end
fprintf('--- Total modes across all time windows: %d ---\n', total_modes);
%}

%% Error Graph

mrdmd_error = NaN(1, numsnapshots); 
for k = 1:m
    true_snapshot = X(:, k);
    rec_snapshot  = X_rec(:, k);
    
    % Shift the tracking index to map precisely onto the absolute X_total timeline
    absolute_idx = frame_start + k - 1;
    mrdmd_error(absolute_idx) = norm(true_snapshot - rec_snapshot) / (norm(true_snapshot) + eps);
end

figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.6, 0.5]);

% Plotting against the absolute indices preserves the alignment layout
plot(mrdmd_error * 100, 'r-', 'LineWidth', 1.5, 'DisplayName', 'MRDMD');
hold on;
xlim([1, numsnapshots]); % Lock the visual frame boundary to the total absolute snapshots
ylim([0 200]);
grid on;
legend('Location', 'best');
xlabel('Snapshot (Absolute Timeline)'); 
ylabel('Error %');
title('Snapshot-wise Reconstruction Performance');




%% MULTI-TARGET EXTRACTION WITH ACTIVE LOOKUP MENU
plot_start_idx = 126; 
num_plot_snaps = X_total_size;
plot_end_idx   = plot_start_idx + num_plot_snaps - 1;
plot_end_idx   = 187;


% Format: [Level, Bin (j), Mode_Index] (Use 0 as a wildcard for ALL)
% Use 1,0,0 - 2,0,0 - 3,0,0 - 4,0,0 for full period-specific reconstruction
target_coordinates = [
    3, 2, 13;   
    %2, 0, 0; 
    %3, 0, 0;
    %4, 0, 0;
];

fprintf('\n===========================================================');
fprintf('\n   AVAILABLE COORDINATES FOR FRAMES %d TO %d', plot_start_idx, plot_end_idx);
fprintf('\n===========================================================\n');


available_modes = cell(L,maxJ);

for i = 1:L
    J = 2^(i-1);
    for j = 1:J
        if isempty(list_modes{i, j}), continue; end
        
        t_start   = list_t_start(i, j);
        bin_width = list_bin_widths(i, j);
        t_end     = t_start + bin_width - 1;
        
        % Check if this bin overlaps with our visual playback window
        if t_start <= plot_end_idx && t_end >= plot_start_idx
            num_modes_available = size(list_modes{i, j}, 2);
            available_modes{i,j} = list_modes{i,j};
            fprintf('● Level %d, Bin %d (Frames %d to %d)\n', i, j, t_start, t_end);
            fprintf('  └─ Available Mode Indices: [');
            for m_idx = 1:num_modes_available
                if m_idx == num_modes_available
                    fprintf('%d', m_idx);
                else
                    fprintf('%d, ', m_idx);
                end
            end
            fprintf(']\n\n');
        end
    end
end
fprintf('===========================================================\n\n');
%{
%% --- EXTRACTION ENGINE ---
X_level_extract = zeros(n, m);
fprintf('--- Extracting Multi-Target Components ---\n');

num_targets = size(target_coordinates, 1);

for idx = 1:num_targets
    req_i    = target_coordinates(idx, 1);
    req_j    = target_coordinates(idx, 2);
    req_mode = target_coordinates(idx, 3);
    
    if req_i > L || req_i < 1
        warning('Target row %d: Level %d out of bounds. Skipping.', idx, req_i);
        continue;
    end
    
    max_J = 2^(req_i-1);
    if req_j == 0
        bins_to_process = 1:max_J;
    else
        if req_j > max_J || req_j < 0
            warning('Target row %d: Bin %d out of bounds for Level %d. Skipping.', idx, req_j, req_i);
            continue;
        end
        bins_to_process = req_j;
    end
    
    for j = bins_to_process
        if isempty(list_modes{req_i, j}), continue; end
        
        modes     = list_modes{req_i, j};
        eigs_slow = list_w{req_i, j};
        b         = list_b{req_i, j};
        t_start   = list_t_start(req_i, j);
        bin_width = list_bin_widths(req_i, j);
        if bin_width == 0, continue; end
        
        t_end = t_start + bin_width - 1;
        num_modes_available = size(modes, 2);
        
        if req_mode == 0
            modes_to_process = 1:num_modes_available;
        else
            if req_mode > num_modes_available || req_mode < 0
                continue; 
            end
            modes_to_process = req_mode;
        end
        
        selected_modes = modes(:, modes_to_process);
        selected_eigs  = eigs_slow(modes_to_process);
        selected_b     = b(modes_to_process);
        
        mag = abs(selected_eigs);
        over = mag > 1.0;
        selected_eigs(over) = selected_eigs(over) ./ mag(over);
        
        time_powers = selected_eigs .^ (0:bin_width-1);
        local_rec   = selected_modes * (selected_b .* time_powers);
        
        if t_end > m
            t_end = m;
            local_rec = local_rec(:, 1:(t_end - t_start + 1));
        end
        
        X_level_extract(:, t_start:t_end) = X_level_extract(:, t_start:t_end) + local_rec;
    end
    
    str_j = iif(req_j==0, 'ALL', num2str(req_j));
    str_m = iif(req_mode==0, 'ALL', num2str(req_mode));
    fprintf('Loaded Entry %d -> Level: %d | Bin: %s | Mode: %s\n', idx, req_i, str_j, str_m);
end

X_level_extract = real(X_level_extract);

% Plotting Loop
h_extract_fig = figure('Position', [150, 150, 1200, 500], 'Name', 'Wildcard Target Mode Viewer');
for k = plot_start_idx:plot_end_idx
    if ~ishandle(h_extract_fig), break; end
    set(0, 'CurrentFigure', h_extract_fig);
    absolute_idx = frame_start + k - 1;
    
    if num_channels == 1
        frame_orig = reshape(X(:,k), [vid_width, vid_height])';
        frame_ext  = reshape(X_level_extract(:,k), [vid_width, vid_height])';
    else
        frame_orig = reshape(X(:,k), [vid_width, vid_height, num_channels]);
        frame_orig = permute(frame_orig, [2, 1, 3]);
        frame_ext = reshape(X_level_extract(:,k), [vid_width, vid_height, num_channels]);
        frame_ext = permute(frame_ext, [2, 1, 3]);
    end
    
    subplot(1, 2, 1); imshow(frame_orig, []);
    title(sprintf('Original (Frame %d)', absolute_idx));
    
    subplot(1, 2, 2); imshow(frame_ext, []);
    title(sprintf('Filtered Composite Reconstruction (Frame %d)', absolute_idx));
    
    drawnow;
    pause(0.1);
end
%}


%% 1. Calculate Dynamically the L4 Boundaries

% HERE WE ARE JUST FINDING WHERE L4 STARTS
col_idx = find(~cellfun(@isempty, available_modes(4, :))); 
num_L4_modes = size(available_modes{4, col_idx}, 2);      
modes_before_L4 = 0;
for r = 1:3
    active_col = find(~cellfun(@isempty, available_modes(r, :)), 1);
    if ~isempty(active_col)
        modes_before_L4 = modes_before_L4 + size(available_modes{r, active_col}, 2);
    end
end
L4_start_col = modes_before_L4 + 1 + 1; 

% HERE BASICALLY ALL WE ARE DOING IS GETTING MY AVAILABLE LEVEL 1 - 3 MODES 
% FROM MY TIME PERIOD, AND ADDING THEM TOGETHER INTO ONE MATRIX

all_modes_matrix = [];
library_modes_matrix = []; 
[num_rows, num_cols] = size(available_modes);
for r = 1:num_rows
    for c = 1:num_cols
        if ~isempty(available_modes{r, c})
            current_modes = available_modes{r, c};
            all_modes_matrix = [all_modes_matrix, current_modes]; %#ok<AGROW>
            if r < 4
                library_modes_matrix = [library_modes_matrix, current_modes]; %#ok<AGROW>
            end
        end
    end
end


refresh_Xi = true;
if exist('Xi', 'var') && ~refresh_Xi
    fprintf('Matrix "Xi" already exists. Skipping regression loop.\n');
else
    % BIG STEP FOR COMPUTATION - USING SVD ON FULL MATRIX TO SAVE MEMORY
    [U_squeeze, S_squeeze, ~] = svd(library_modes_matrix, 'econ');
    library_squeezed = S_squeeze; 
    
    % PROJECTING AVAILABLE LEVEL 4 MODES TO THE SVD SPACE OF LEVELS 1-3
    col_idx = find(~cellfun(@isempty, available_modes(4, :)));
    Y_targets = available_modes{4, col_idx}; % Y_TARGET IS LEVEL 4 MODES
    Y_targets_squeezed = U_squeeze' * Y_targets; 
    
    % M IS STILL THE NUMBER OF 1-3 MODES!!!
    [M_dim, M] = size(library_squeezed);
    num_L4_modes = size(Y_targets, 2); 
    num_features = 1 + M + (M * (M + 1) / 2); 
    toc;

    % STEP 2 - GENERATING THE COMPRESSED DICTIONARY
    fprintf('Building compressed dictionary (Size: %d x %d)...\n', M_dim, num_features);
    tic;
    Theta_tiny = createdict(library_squeezed); 
    toc;
    
    % STEP 3 - STLSQ ALGORITHM - NORMAL EQUATIONS
    lambda = 0.15;       
    max_iter = 25;      
    alpha = 1e-7;       
    
    fprintf('Computing Normal Equations on tiny matrices...\n');
    tic;
    % HERE WE NORMALIZE ALL THE COLUMNS OF OUR THETA DICTIONARY
    column_scales = sqrt(sum(Theta_tiny.^2, 1)); 
    column_scales(column_scales == 0) = 1; 
    Theta_scaled = Theta_tiny ./ column_scales;
    
    % SETTING UP NORMAL EQUATIONS FOR LEAST SQUARES: 
    % FORMAT IS: ATAX = ATB, WHERE Y_PROJECTED IS ATB
    A_full = Theta_scaled' * Theta_scaled; 
    A_full = A_full + alpha * eye(size(A_full, 1)); 
    Y_projected = Theta_scaled' * Y_targets_squeezed; 
    toc;
    
    % Pre-allocate the coefficient matrix
    Xi_scaled = zeros(num_features, num_L4_modes); 
    
    fprintf('Starting STLSQ loop with lambda = %.2f...\n', lambda);
    tic;
    % FOR EACH L4 MODE
    % TRYING TO FIND X IN ATAX = ATB
    for idx = 1:num_L4_modes 
        b = Y_projected(:, idx); % b here is actually AtB.  Inversing AtA.
        xi_active = A_full \ b; 
        active_inds = true(size(xi_active));
        
        for iter = 1:max_iter
            small_inds = abs(xi_active) < lambda;
            if ~any(small_inds & active_inds), break; end
            % FINDING SMALLEST COEFFICIENTS (XI_ACTIVE), SETTING TO ZERO
            xi_active(small_inds) = 0; 
            active_inds(small_inds) = false; 
            if ~any(active_inds), break; end 
            % DOES THIS AGAIN
            xi_active(active_inds) = A_full(active_inds, active_inds) \ b(active_inds);
        end
        Xi_scaled(:, idx) = xi_active;
    end
    
    % Un-scale the coefficients back to physical units
    Xi = Xi_scaled ./ column_scales'; 
    toc;
    
    fprintf('Done! Xi equations computed using 99%% less memory.\n');
end

%% Quick Lambda Diagnostic Sweep (used for next trial)
lambda_test_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4];

fprintf('\n=== LAMBDA DIAGNOSTIC SWEEP ===\n');
for L = lambda_test_values
    % Test a single target mode (e.g., L4 Mode 1) to see how many terms survive
    b = Y_projected(:, 1); 
    xi_test = A_full \ b;

    % Run a fast 5-iter threshold check
    for iter = 1:5
        small_inds = abs(xi_test) < L;
        xi_test(small_inds) = 0;
        % SUPER IMPORTANT. ACTIVE INDEXES ARE THE ONES THAT HAVE NOT BEEN
        % LABELED AS GARBAGE YET. IT'S FULL OF TRUE'S AND FALSE'S.  ONCE A
        % TERM FALLS BELOW LAMBDA, IT'S PUT IN THE GARBAGE. SO WHEN THE NEW
        % ITERATION RUNS, THAT SAME TERM MIGHT FALL BELOW LAMBDA AGAIN.
        % WHY? BECAUSE THE SMALL, USELESS RELATIONSHIPS ARE REMOVED. 
        active_inds = xi_test ~= 0;
        if ~any(active_inds), break; end
        xi_test(active_inds) = A_full(active_inds, active_inds) \ b(active_inds);
    end

    num_active_terms = sum(xi_test ~= 0);
    fprintf('Lambda = %.2f -> Active terms found for Mode 1: %d\n', L, num_active_terms);
end
fprintf('================================\n');

%% 5. Analyze and Print Discovered Level 4 Equations

% BASICALLY JUST GIVING NAMES/LABELS FOR ALL THE TYPES OF TERMS IN THE
% DICTIONARY.  THIS JUST INCLUDES 1, MODES, CROSS-MODES. 

M = size(all_modes_matrix, 2); % Number of linear modes (100)
feature_labels = cell(1, num_features);
% Label 1: The constant term
feature_labels{1} = '1';
% Labels 2 to M+1: The linear modes
for i = 1:M
    feature_labels{i+1} = sprintf('phi_%d', i);
end
% Labels M+2 onward: The cross-product combinations (Mode_i * Mode_j)
col_idx = M + 2;
for i = 1:M
    for j = i:M
        feature_labels{col_idx} = sprintf('(phi_%d * phi_%d)', i, j);
        col_idx = col_idx + 1;
    end
end

fprintf('\n================ DISCOVERED LEVEL 4 EQUATIONS ================\n');

% Loop through each of the Level 4 target modes
for idx = 1:num_L4_modes
    % For each column (corresponding to each level 4 target mode)
    coef_vector = Xi(:, idx);
    % Find the rows where SINDy left a non-zero weight
    active_idx = find(coef_vector ~= 0);

    if isempty(active_idx)
        fprintf('d(L4_mode_%d)/dt = 0  (No driving dynamics found)\n\n', idx);
        continue;
    end

    % Build the equation string piece by piece
    equation_str = sprintf('d(L4_mode_%d)/dt = ', idx);

    for k = 1:length(active_idx)
        row = active_idx(k);
        weight = coef_vector(row);
        label = feature_labels{row};

        % Formatting signs for clean viewing
        if k == 1
            equation_str = sprintf('%s %.4f * %s', equation_str, weight, label);
        else
            if weight > 0
                equation_str = sprintf('%s + %.4f * %s', equation_str, weight, label);
            else
                equation_str = sprintf('%s - %.4f * %s', equation_str, abs(weight), label);
            end
        end
    end

    % Print the final discovered equation
    fprintf('%s\n\n', equation_str);
end
fprintf('==============================================================\n');


%% FUNCTIONS
function [modes, D, b] = dmd(X)
    X1 = X(:, 1:end-1);
    Y = X(:, 2:end);
    [U, S, V] = svd(X1, 'econ');
    sing_vals = diag(S);
    
    thresh = 1e-8 * sing_vals(1);
    r = sum(sing_vals > thresh);
    
    r = min([25, r, size(U, 2)]); 
    if r == 0, r = 1; end
    U = U(:, 1:r);
    S = S(1:r, 1:r);
    V = V(:, 1:r);
    
    A = (U' * Y * V) / S;
    [W, D] = eig(A);
    
    modes = Y * V * (S \ W);
    b = pinv(modes) * X1(:, 1); 
end


function out = iif(cond, trueVal, falseVal)
if cond, out = trueVal; else, out = falseVal; end
end

function Theta = createdict(Modes)
    % Modes size: 57600 x 100
    [N, M] = size(Modes);
    
    % Total columns: 1 (constant) + M (linear) + (M * (M + 1) / 2) (cross terms)
    num_features = 1 + M + (M * (M + 1) / 2); % For 100 modes, this is exactly 5151
    
    % PRE-ALLOCATE THE FULL MATRIX UP FRONT
    Theta = zeros(N, num_features);
    
    % 1. Populate the constant column
    Theta(:, 1) = ones(N, 1);
    
    % 2. Populate the linear terms (columns 2 to M+1)
    Theta(:, 2:M+1) = Modes;
    
    % 3. Populate the cross-product combinations (Mode_i * Mode_j)
    col_idx = M + 2; % Start filling right after the linear terms
    for i = 1:M
        for j = i:M
            Theta(:, col_idx) = Modes(:, i) .* Modes(:, j);
            col_idx = col_idx + 1;
        end
    end
end
