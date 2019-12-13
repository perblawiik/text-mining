function [cos_theta] = nmf_algorithm(A_original, query, k, iterations)
    A = A_original;
    [m, n] = size(A);
    W = zeros(m, k);
    H = zeros(k, n);

    [U_k, ~, V_k] = svds(A, k);

    % Initialization
    W(:,1) = U_k(:,1);
    H(1, :) = V_k(:, 1);
    for j = 2:k
        C = U_k(:,j)*V_k(:,j)';
        C = C.*(C >= 0);
        [u, ~, v] = svds(C, 1);
        W(:,j) = abs(u);
        H(j,:) = abs(v);
    end

    % Normalize A
    for i = 1 : k
        max_value = max(A_original(:,i));
        A(:,i) = A_original(:,i)./max_value;
    end

    epsilon = 0.00001;
    for j = 1:iterations
        W = W.*(W >= 0);
        H = H.*(W'*A)./((W'*W)*H + epsilon);
        H = H.*(H >= 0);
        W = W.*(A*H')./(W*(H*H') + epsilon);

        % Normalize
        for i = 1 : k
            max_value = max(W(:,i));
            W(:,i) = W(:,i)/max_value;
            H(i, :) = H(i, :)*max_value;
        end
    end
    
    % MATLAB built in function
    %[W,H] = nnmf(A, k,'alg','mult');

    % Thin QR decomposition
    [Q,R] = qr(W, 0);

    % Compute the query in the same basis as WH
    query_reduced = inv(R)*Q'*query;
    % Cosinus distance for all documents
    cos_theta = (query_reduced'*H) / (norm(query_reduced, 2) * norm(H, 2));
end

