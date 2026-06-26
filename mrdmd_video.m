close all;
refresh = false;

if refresh || ~exist('X_total', 'var')
    clc;
    clear;
    fprintf('Loading ducks_snapshot_matrix.mat...\n');
    load('ducks_snapshot_matrix.mat'); % This loads the 'ducks' matrix into workspace
    X_total = double(ducks); 
end

X = X_total;

%% Setting the parameters
frame_start = 50;
frame_end   = 250;
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
            list_t_start(i+1, left_child_idx) = current_start;             
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
        
        t_start = list_t_start(i, j);
        bin_width = list_bin_widths(i, j);
        
        if bin_width == 0
            continue;
        end
        
        t_end = t_start + bin_width - 1;
        mag = abs(eigs_slow);
        over = mag > 1.0;
        eigs_slow(over) = eigs_slow(over) ./ mag(over);
        
        time_powers = eigs_slow .^ (0:bin_width-1);
        local_rec = modes * (b .* time_powers);
        
        % Protect against matrix index rounding overflows at tree boundaries
        if t_end > m
            t_end = m;
            local_rec = local_rec(:, 1:(t_end - t_start + 1));
        end

        X_rec(:, t_start:t_end) = X_rec(:, t_start:t_end) + local_rec;
    end
end
X_rec = real(X_rec); 


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

%% Plotting Old vs New 
total_spatial_elements = size(ducks, 1);
num_frames = size(ducks, 2);
if total_spatial_elements == 14400        % New highly reduced grayscale matrix
    vid_height = 90;
    vid_width = 160;
    num_channels = 1;
elseif total_spatial_elements == 43200   % Your current dataset
    vid_height = 180;                     
    vid_width = 240;                      
    num_channels = 1;                     
elseif total_spatial_elements == 691200   % Old 360p color matrix
    vid_height = 360;
    vid_width = 640;
    num_channels = 3;
elseif total_spatial_elements == 2764800  % Raw 720p color matrix
    vid_height = 720;
    vid_width = 1280;
    num_channels = 3;
else
    error('Unknown matrix dimension layout.');
end


%% Plotting Old vs New (Quiver Field View)
% Assign the figure to a specific handle variable 'h_fig'
h_fig = figure('Position', [100, 100, 1200, 500], 'Name', 'Ducks Playback');

for k = 1:m
    % Force MATLAB to focus on the playback figure window so it doesn't render on the error graph
    if ishandle(h_fig)
        set(0, 'CurrentFigure', h_fig);
    else
        break; % Exit gracefully if user closes the window early
    end
    
    absolute_idx = frame_start + k - 1;
    
    % Handle single channel dimension mapping with the Transpose Fix
    % Handle single channel dimension mapping with Auto-Scaling
    if num_channels == 1
        frame_orig = reshape(X(:,k), [vid_width, vid_height])';
        frame_rec  = reshape(X_rec(:,k), [vid_width, vid_height])';
    else
        frame_orig = reshape(X(:,k), [vid_width, vid_height, num_channels]);
        frame_orig = permute(frame_orig, [2, 1, 3]);
    
        frame_rec = reshape(X_rec(:,k), [vid_width, vid_height, num_channels]);
        frame_rec = permute(frame_rec, [2, 1, 3]);
    end
    
    % Left side: Original
    subplot(1, 2, 1);
    imshow(frame_orig, []); % <--- Added [] to auto-scale intensity
    title(sprintf('Original (Frame %d)', absolute_idx));
    
    % Right side: Reconstructed
    subplot(1, 2, 2);
    imshow(frame_rec, []);  % <--- Added [] to auto-scale intensity
    title(sprintf('Reconstruction (Frame %d)', absolute_idx));
    
    % Force MATLAB to flush the graphics queue immediately
    drawnow;
    pause(0.01);
end



%% FUNCTIONS
function [modes, D, b] = dmd(X)
    X1 = X(:, 1:end-1);
    Y = X(:, 2:end);
    [U, S, V] = svd(X1, 'econ');
    sing_vals = diag(S);
    
    thresh = 1e-8 * sing_vals(1);
    r = sum(sing_vals > thresh);
    
    r = min([100, r, size(U, 2)]); 
    if r == 0, r = 1; end
    U = U(:, 1:r);
    S = S(1:r, 1:r);
    V = V(:, 1:r);
    
    A = (U' * Y * V) / S;
    [W, D] = eig(A);
    
    modes = Y * V * (S \ W);
    b = pinv(modes) * X1(:, 1); 
end