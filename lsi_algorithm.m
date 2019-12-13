function [cos_theta] = lsi_algorithm(A, query, k)

    [U_k, S_k, V_k] = svds(A, k);
    
    H_k = S_k * V_k';

    query_k = U_k'*query;

    % All documents
    cos_theta = (query_k'*H_k) / (norm(query_k, 2) * norm(H_k, 2));
end

