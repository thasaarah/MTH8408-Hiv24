R = matrix_random = rand(100, 2)


p = 0.9
x=ones(size(R,2))/size(R,2)
l = Int(size(R,2))
function riskreturn(x, R, μ)
    l = Int(size(R,2))
    r = copy(x)
    for j in 1:l
        a = 0
        b = 0
        for i in 1:l
            a += p^(length(r)-i) * log(R[i])
            b += p^(length(r)-i)
        r[j] = exp(a/b)
        end
    end

    # r = mean(R, dims=1) # Calculer la moyenne 
    C = cov(R) # Calculer la covariance 
    risk = x' * C * x # Calcul du risque
    reward = -dot(r, x) # Calcul du rendement
    output = reward + μ*risk # μ doit être défini
    
    return output
end

riskreturn(x, R, 1)

