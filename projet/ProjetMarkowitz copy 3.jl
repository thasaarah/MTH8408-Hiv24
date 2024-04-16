using YahooFinance, ADNLPModels, LinearAlgebra, Statistics, Ipopt


# Dates de début et de fin pour la récupération des données
start_date = "2023-09-09"
end_date = "2024-03-24"
N = 100 # Number of datapoints

# Récupération des données pour GOOG et BTC-CAD
data_goog = get_symbols("NKE", start_date, end_date)
data_btc_cad = get_symbols("BAC", start_date, end_date)
data_cad_usd = get_symbols("PFE", start_date, end_date)
data_cad_eur = get_symbols("MO", start_date, end_date)

# Truncate outputs
data_goog = values(data_goog["Close"][1:N])
data_btc_cad = values(data_btc_cad["Close"][1:N])
data_cad_usd = values(data_cad_usd["Close"][1:N])
data_cad_eur = values(data_cad_eur["Close"][1:N])

# println(data_goog)
# println(data_btc_cad)

# Calcul des rendements à chaque groupe de donéés
# returns_goog = diff(data_goog) ./ data_goog[1:end-1]
# returns_btc_cad = diff(data_btc_cad) ./ data_btc_cad[1:end-1]

#println(returns_good)
#println(returns_btc_cad)

returns_matrix = hcat(data_goog, data_btc_cad,data_cad_usd,data_cad_eur)
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
    C = cov(R) # Calculer la covariance
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
#nlp = ADNLPModel(f,x0,lvar,uvar,c,lcon,ucon)
using Ipopt, NLPModelsIpopt
#stats = ipopt(nlp)

f01(x) = riskreturn(x, R, 0.1)
nlp01 = ADNLPModel(f01,x0,lvar,uvar,c,lcon,ucon)
stats01 = ipopt(nlp01)

f05(x) = riskreturn(x, R, 0.5)
nlp05 = ADNLPModel(f05,x0,lvar,uvar,c,lcon,ucon)
stats05 = ipopt(nlp05)

f1(x) = riskreturn(x, R, 1)
nlp1 = ADNLPModel(f1,x0,lvar,uvar,c,lcon,ucon)
stats1 = ipopt(nlp1)

f10(x) = riskreturn(x, R, 10)
nlp10 = ADNLPModel(f10,x0,lvar,uvar,c,lcon,ucon)
stats10 = ipopt(nlp10)


f100(x) = riskreturn(x, R, 100)
nlp100 = ADNLPModel(f100,x0,lvar,uvar,c,lcon,ucon)
stats100 = ipopt(nlp100)

println(stats01.objective)
println(stats05.objective)
println(stats1.objective)
println(stats10.objective)
println(stats100.objective)


# returns_matrix = hcat(data_btc_cad,data_cad_usd,data_cad_eur,data_goog)