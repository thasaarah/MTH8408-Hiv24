using YahooFinance, ADNLPModels, LinearAlgebra, Statistics


# Dates de début et de fin pour la récupération des données
start_date = "2018-12-13"
end_date = "2019-06-28"
N = 100 # Number of datapoints

# Récupération des données pour GOOG et BTC-CAD
data_goog = get_symbols("GOOG", start_date, end_date)
data_btc_cad = get_symbols("BTC-CAD", start_date, end_date)

# Truncate outputs
data_goog = values(data_goog["Close"][1:N])
data_btc_cad = values(data_btc_cad["Close"][1:N])

# println(data_goog)
# println(data_btc_cad)

# Calcul des rendements à chaque groupe de donéés
# returns_goog = diff(data_goog) ./ data_goog[1:end-1]
# returns_btc_cad = diff(data_btc_cad) ./ data_btc_cad[1:end-1]

#println(returns_good)
#println(returns_btc_cad)

returns_matrix = hcat(data_goog, data_btc_cad)
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

R = returns_matrix
y = ones(size(R,2))/size(R,2)

f(x) = riskreturn(x,R,1)

f(y)