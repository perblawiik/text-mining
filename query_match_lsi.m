function [P, R] = query_match_lsi(A, queries, rank, step_size, query_start, query_end)
    relevant = load("medline/MED.REL");
    tolerance_count = (1 / step_size) + 1;

    % Precision and Recall
    P = zeros(tolerance_count,1);
    R = zeros(tolerance_count,1);

    % Rank
    k = rank;
    for query_count = query_start:query_end
        q = queries(:,query_count);
        cos_theta = lsi_algorithm(A, q, k);

        n = 1;
        for tolerance = 0 : step_size : 1.0
            [precision, recall, ~] = cosine_matching(relevant, cos_theta, tolerance, query_count);
            P(n) = P(n) + precision;
            R(n) = R(n) + recall;
            n = n + 1;
        end
    end

    num_queries = query_end - query_start + 1;
    P = P / num_queries;
    R = R / num_queries;
end

