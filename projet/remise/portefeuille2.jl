using YahooFinance, ADNLPModels, LinearAlgebra, Statistics, Ipopt


# Dates de début et de fin pour la récupération des données
start_date = "2023-09-09"
end_date = "2024-03-24"
N = 100 # Number of datapoints

# Récupération des données pour GOOG et BTC-CAD
data1 = get_symbols("NKE", start_date, end_date)
data2 = get_symbols("BAC", start_date, end_date)
data3 = get_symbols("PFE", start_date, end_date)
data4 = get_symbols("MO", start_date, end_date)

# Sélection des données
data1 = values(data1["Close"][1:N])
data2 = values(data2["Close"][1:N])
data3 = values(data3["Close"][1:N])
data4 = values(data4["Close"][1:N])

returns_matrix = hcat(data1,data2,data3,data4)
function riskreturn(x, R, μ)
    p = 0.9
    l = Int(size(R,2))
    N = Int(size(R,1))
    r = zeros(l)
    for j in 1:l
        a = 0
        b = 0
        for i in 1:N
            a += p^(N-i) * log(R[i,j])
            b += p^(N-i)
        end
        r[j] = exp(a/b)
    end
    C = cov(R) # Calcul de la covariance
    risk = x' * C * x # Calcul du risque

    reward = dot(r, x) # Calcul du rendement
    output = -reward + μ*risk # μ doit être défini

    return output
end

l = Int(size(returns_matrix,2))

μ = 0.5
R = returns_matrix
f(x) = riskreturn(x, R, μ)
c(x) = [sum(x)]
x0 = ones(l)/l
lvar = zeros(l)
uvar = ones(l)
lcon = [1.]
ucon = [1.]

using Ipopt, NLPModelsIpopt


stats_container = Dict()
stats_obj = ones(100)

for mu in 1:100
    # Fonction objectif
    f_mu(x) = riskreturn(x, R, mu)

    # Modèle NLP
    nlp_mu = ADNLPModel(f_mu, x0, lvar, uvar, c, lcon, ucon)

    # Résoudre le modèle avec Ipopt
    stats_mu = ipopt(nlp_mu)
    stats_container[mu] = stats_mu

    # Objectif
    stats_obj[mu] = stats_container[mu].objective
end

for i in stats_obj
    print(i)
    print(",")
end

