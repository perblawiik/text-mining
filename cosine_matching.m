function [precision, recall, D_t] = cosine_matching(relevant, cos_theta, tolerance, query_number)
    I = find(cos_theta > tolerance);

    D_t = length(I);

    indices = find(ismember(relevant(:,3), I));
    D_r = nnz(relevant(indices, 1) == query_number);
    precision = D_r / D_t;

    N_r = nnz(relevant(:, 1) == query_number);
    recall = D_r / N_r;
end

