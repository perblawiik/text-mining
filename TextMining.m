clc, clear, close all

relevant = load("medline/MED.REL");
data = load("text-mining-medline_stemmed.mat");

A = data.A;
queries = data.q;
dictionary = data.dict;

query_number = 15;
fprintf("Query number: %d\n", query_number);

%% 
step_size = 0.001;
query_start = 14;
query_end = query_start;

% Latent Semantic Indexing
% Rank
k = 100;
% Average precision and recall for LSI
[precision_lsi, recall_lsi] = query_match_lsi(A, queries, k, step_size, query_start, query_end);

% Nonnegative Matrix Factorization
% Rank
k = 50;
% Precision and Recall for NMF
[precision_nmf, recall_nmf] = query_match_nmf(A, queries, k, step_size, query_start, query_end);

% Draw graph
figure
plot(recall_lsi, precision_lsi, 'Color', [0 0.4470 0.7410], 'LineStyle', '-', 'Marker', 'o')
hold on;
plot(recall_nmf, precision_nmf, 'Color', [0.8500 0.3250 0.0980], 'LineStyle', '-', 'Marker', 'x')
legend('LSI', 'NMF');
xlabel('Recall');
ylabel('Precision');

%% Latent Semantic Indexing for one query

% Rank
k = 100;
query = queries(:,query_number);
cos_theta = lsi_algorithm(A, query, k);

tolerance = 0.1;
[precision, recall, num_matches] = cosine_matching(relevant,cos_theta,tolerance,query_number);

fprintf("\nLatent Semantic Indexing:\n");
fprintf("Rank: %d\n", k);
fprintf("Tolerance: %.2f\n", tolerance);
fprintf("Number of matches: %d\n", num_matches);
fprintf("Precision: %.2f\n", precision);
fprintf("Recall: %.2f\n", recall);

%% Nonnegative Matrix Factorization for one query
% Rank
k = 50;
query = queries(:,query_number);
cos_theta = nmf_algorithm(A, query, k, 100);

tolerance = 0.1;
[precision, recall, num_matches] = cosine_matching(relevant,cos_theta,tolerance,query_number);

fprintf("\nNonnegative Matrix Factorization:\n");
fprintf("Rank: %d\n", k);
fprintf("Tolerance: %.2f\n", tolerance);
fprintf("Number of matches: %d\n", num_matches);
fprintf("Precision: %.2f\n", precision);
fprintf("Recall: %.2f\n", recall);
