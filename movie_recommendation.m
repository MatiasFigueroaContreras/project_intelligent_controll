%% Experiment test
folder_path = "./data/ml-100k/";
filename = "u";
num_clusters = 2;
num_neighbors = 200;
num_folds = 5;
k_general = 9;
k_cluster = 2;
[mae, rmse, exec_time] = run_folds_experiment(folder_path, filename, num_clusters, num_neighbors, num_folds, [k_general, k_cluster], true);

fprintf('\n====================================================================\n');
fprintf("\t\t\t\t\t\t FINAL RESULTS \n");
fprintf('MAE: %.14f | RMSE: %.14f | TIME: %.4f sec\n', mae, rmse, exec_time);
fprintf('====================================================================\n');
%% Main Functions

% Description: Runs a cross-validation experiment
% Parameters: folder_path: folder path containing the files
%             filename: base name of the files
%             num_clusters: number of clusters
%             num_neighbors: number of neighbors
%             num_folds: number of folds
%             k_svd: vector with the number of latent factors for SVD
%             verbose: flag for printing results
% Returns: mae: mean absolute error
%          rmse: root mean square error
%          exec_time: execution time
function [mae, rmse, exec_time] = run_folds_experiment(folder_path, filename, num_clusters, num_neighbors, num_folds,  k_svd, verbose)
    mae_acum = 0;
    rmse_acum = 0;
    exec_time = 0;
    for i = 1:num_folds
        tic;
        fold_filename = filename + int2str(i);
        [fold_mae, fold_rmse] = run_experiment(folder_path, fold_filename, num_clusters, num_neighbors, k_svd);
        mae_acum = mae_acum + fold_mae;
        rmse_acum = rmse_acum + fold_rmse;
        fold_toc = toc;
        exec_time = exec_time + fold_toc;
        if(verbose)
            fprintf('---------------------------------------------------------\n');
            fprintf('\t\t\t\t\t ITERATION %d:\n', i);
            fprintf('MAE: %.8f | RMSE: %.8f | TIME: %.4f sec\n', fold_mae, fold_rmse, fold_toc);
            fprintf('---------------------------------------------------------\n');
        end
    end
    
    mae = mae_acum / num_folds;
    rmse = rmse_acum / num_folds;
end

% Description: Runs an experiment
% Parameters: folder_path: folder path containing the files
%             filename: file name
%             num_clusters: number of clusters
%             num_neighbors: number of neighbors
%             k_svd: vector with the number of latent factors for SVD
% Returns: mae: mean absolute error
%          rmse: root mean square error
function [mae, rmse] = run_experiment(folder_path, filename, num_clusters, num_neighbors, k_svd)
    train_extension = ".base";
    test_extension = ".test";

    k_general = k_svd(1);
    k_cluster = k_svd(2);

    % Data reading and preparation
    [~, user_ids, movie_ids, ratings] = load_data(folder_path, filename + train_extension);
    [user_map, user_count, user_indexes] = map_ids_to_indexes(user_ids);
    [movie_map, movie_count, movie_indexes] = map_ids_to_indexes(movie_ids);

    % Feature matrix generation
    features_matrix_nan = generate_features_matrix(user_indexes, movie_indexes, ratings, user_count, movie_count, NaN);
    
    lambda = 1.5; % Regularization parameter
    num_features = k_general; % Number of latent factors
    max_iter = 30; % Maximum iterations
    features_matrix_filled = fill_matrix_svd_als(features_matrix_nan, num_features, lambda, max_iter);

    % Fuzzy C-Means application
    fcm_options = fcmOptions();
    fcm_options.Verbose = false;
    fcm_options.NumClusters = num_clusters;
    [~, U] = fcm(features_matrix_filled, fcm_options);
    U = U';

    % Obtain clusters to which users belong (Defuzzification)
    [~, max_cluster_assignment] = max(U, [], 2);
    clustered_users = cell(num_clusters, 1);
    for i = 1:num_clusters
        clustered_users{i} = find(max_cluster_assignment == i);
    end
    
    cluster_distances = calculate_dist_cluster_matrices(clustered_users, features_matrix_filled, num_clusters);

    lambda_cluster = 1.5;
    num_features_cluster = k_cluster;
    max_iter_cluster = 50;
    cluster_svds = calculate_cluster_svd_als(clustered_users, features_matrix_nan, num_features_cluster, lambda_cluster, max_iter_cluster);

    % Evaluate movie ratings on the test set
    [~, test_user_ids, test_movie_ids, test_ratings] = load_data(folder_path, filename + test_extension);
    predicted_ratings = zeros(size(test_ratings));
    for i = 1:length(test_user_ids)
        user_id = test_user_ids(i);
        movie_id = test_movie_ids(i);
        user_index = user_map(user_id);
        
        if ~isKey(movie_map, movie_id)
            predicted_ratings(i) = 3;
            continue;
        end
        movie_index = movie_map(movie_id);
        user_cluster_number = max_cluster_assignment(user_index);
        user_cluster = clustered_users{user_cluster_number};
        user_cluster_index = user_cluster == user_index;
        user_cluster_dist = cluster_distances{user_cluster_number};

        % Distances within the cluster
        distances = user_cluster_dist(user_cluster_index, :);
        max_num_neighbor = min(num_neighbors, length(user_cluster));
        [~, neighbor_cluster_indexes] = mink(distances, max_num_neighbor);

        % Neighbors and their ratings
        neighbors_indexes = user_cluster(neighbor_cluster_indexes);        
        neighbor_ratings = features_matrix_nan(neighbors_indexes, movie_index);

        mean_rating = mean(neighbor_ratings, 'omitnan');
        
        % Cluster SVDs
        cluster_svd = cluster_svds{user_cluster_number};
        svd_cluster_rating = cluster_svd(user_cluster_index, movie_index);

        if isnan(mean_rating)
            mean_rating = 3;
        end

        predicted_ratings(i) = (mean_rating + svd_cluster_rating) / 2;
    end

    % PD: To make recommendations, TOP-N movies are given by taking 
    %     unwatched movies based on the rating predictions made.
    error = test_ratings - predicted_ratings;
    mae = mean(abs(error));
    rmse = sqrt(mean(error.^2));
end

%% Auxiliary Functions

% Description: Loads data from a text file
% Parameters: folder_path: folder path containing the file
%             filename: name of the file
% Returns: data: matrix with the data
%          user_ids: vector with user IDs
%          movie_ids: vector with movie IDs
%          ratings: vector with ratings
function [data, user_ids, movie_ids, ratings] = load_data(folder_path, filename)
    data = load(folder_path + filename);
    user_ids = data(:, 1);
    movie_ids = data(:, 2);
    ratings = data(:, 3);
end

% Description: Maps IDs to indices
% Parameters: ids: vector with the IDs
% Returns: check_map: map with IDs and their indices
%          count: number of unique IDs
%          indexes: vector with the indices of the IDs
function [check_map, count, indexes] = map_ids_to_indexes(ids)
    check_map = containers.Map("KeyType", "double", "ValueType", "double");
    map = containers.Map("KeyType", "double", "ValueType", "double");
    count = 0;
    indexes = zeros(size(ids));
    for i = 1:length(ids)
        if ~isKey(check_map, ids(i))
            count = count + 1;
            check_map(ids(i)) = count;
            map(count) = ids(i);
        end
        indexes(i) = check_map(ids(i));
    end
end

% Description: Generates a feature matrix
% Parameters: user_indexes: vector with user indices
%             movie_indexes: vector with movie indices
%             ratings: vector with ratings
%             user_count: number of users
%             movie_count: number of movies
%             not_rated_value: value for unrated entries
% Returns: features_matrix: feature matrix
function features_matrix = generate_features_matrix(user_indexes, movie_indexes, ratings, user_count, movie_count, not_rated_value)
    if isnan(not_rated_value)
        features_matrix = NaN(user_count, movie_count);
    else
        features_matrix = not_rated_value * ones(user_count, movie_count);
    end
    for i = 1:length(ratings)
        features_matrix(user_indexes(i), movie_indexes(i)) = ratings(i);
    end
end

% Description: Calculates distances between cluster matrices using Pearson correlation
% Parameters: clustered_users: cells with user indices for each cluster
%             features_matrix: feature matrix
%             num_clusters: number of clusters
% Returns: cluster_distances: cells with distances between cluster matrices
function cluster_distances = calculate_dist_cluster_matrices(clustered_users, features_matrix, num_clusters)
    cluster_distances = cell(num_clusters, 1);
    
    for cluster_idx = 1:num_clusters
        user_indexes = clustered_users{cluster_idx};
        
        cluster_features = features_matrix(user_indexes, :);
        
        cluster_corr_matrix = corr(cluster_features', 'type', 'Pearson');
        
        cluster_dist_matrix = 1 - cluster_corr_matrix;
        cluster_dist_matrix = cluster_dist_matrix ./ max(cluster_dist_matrix(:));
        
        cluster_distances{cluster_idx} = cluster_dist_matrix;
    end
end

% Description: Fills missing entries in a matrix using SVD with ALS
% Parameters: features_matrix_nan: feature matrix with NaN values
%             num_features: number of latent factors
%             lambda: regularization parameter
%             max_iter: maximum number of iterations
% Returns: matrix: feature matrix filled
function matrix = fill_matrix_svd_als(features_matrix_nan, num_features, lambda, max_iter)
    [num_users, num_movies] = size(features_matrix_nan);
    features_matrix = features_matrix_nan;
    features_matrix(isnan(features_matrix)) = 0;

    % Initialize latent factors
    rng(1234);
    U = rand(num_users, num_features);
    V = rand(num_movies, num_features);

    % Mask for existing ratings
    mask = ~isnan(features_matrix_nan);

    for iter = 1:max_iter
        for i = 1:num_users
            V_masked = V(mask(i, :), :);
            R_masked = features_matrix(i, mask(i, :))';
            U(i, :) = (V_masked' * V_masked + lambda * eye(num_features)) \ (V_masked' * R_masked);
        end

        for j = 1:num_movies
            U_masked = U(mask(:, j), :);
            R_masked = features_matrix(mask(:, j), j);
            V(j, :) = (U_masked' * U_masked + lambda * eye(num_features)) \ (U_masked' * R_masked);
        end
    end

    % Reconstruct the matrix
    reconstructed_matrix = U * V';

    % Clamp to valid ranges (1 to 5)
    reconstructed_matrix = max(min(reconstructed_matrix, 5), 1);

    % Fill missing entries (NaN) in the original matrix
    matrix = features_matrix_nan;
    matrix(isnan(features_matrix_nan)) = reconstructed_matrix(isnan(features_matrix_nan));
end

% Description: Calculates SVD matrices for clusters using ALS
% Parameters: clustered_users: cells with user indices for each cluster
%             features_matrix_nan: feature matrix with NaN values
%             num_features: number of latent factors
%             lambda: regularization parameter
%             max_iter: maximum number of iterations
% Returns: cluster_svds: cells with SVD matrices for clusters
function cluster_svds = calculate_cluster_svd_als(clustered_users, features_matrix_nan, num_features, lambda, max_iter)
    cluster_svds = cell(length(clustered_users), 1);
    for i = 1:length(clustered_users)
        user_indexes = clustered_users{i};
        cluster_features = features_matrix_nan(user_indexes, :);
        cluster_svd = fill_matrix_svd_als(cluster_features, num_features, lambda, max_iter);
        cluster_svds{i} = cluster_svd;
    end
end