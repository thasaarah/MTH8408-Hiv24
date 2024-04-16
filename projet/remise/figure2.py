import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

mu = range(1,101,1)
rendements = np.array([-57.28693974562689,-48.89132772503278,-45.29950737313849,-42.908634938298704,-40.99835841161645,-39.34787281635526,-37.85834180713023,-36.55357288351937,-35.41898156118911,-34.4035145597553,-33.47468342806435,-32.61082919868055,-31.796957201840698,-31.022356959142524,-30.279174119757712,-29.56151792056499,-28.864883660353982,-28.185767682627755,-27.521403942783444,-26.869579605138743,-26.2285033265223,-25.59670946252116,-24.97298726340291,-24.356327771057156,-23.745883460671223,-23.140937195865135,-22.540878082685555,-21.945182497743943,-21.353399041567403,-20.76513650127943,-20.18005414308436,-19.59785382491096,-19.018273543120927,-18.441082118033357,-17.866074790502324,-17.293069552398144,-16.721904072145623,-16.152433105703864,-15.58452630585667,-15.018066360109422,-14.452947401088975,-13.889073644029093,-13.326358214374423,-12.76472213525642,-12.204093449972639,-11.64440645892519,-11.085601053971335,-10.52762213598021,-9.970419103709212,-9.413945404015141,-8.858158134981704,-8.303017694839983,-7.7484874706339,-7.194533561478096,-6.641124532006213,-6.088231192235202,-5.535826400601799,-4.983884887374824,-4.43238309602507,-3.881299040458721,-3.3306121762929664,-2.780303284589081,-2.230354366658382,-1.6807485487302856,-1.1314699954200194,-0.5825038310631427,-0.03383606809461526,0.5145464582527381,1.0626561495711115,1.6105046987903364,2.1581031400836537,2.705461894614565,3.252590812524012,3.7994992115129236,4.346195912338068,4.892689271505887,5.438987211418692,5.985097248201612,6.5310265174165565,7.076781797847338,7.622369533522986,8.167795854129658,8.713066593946998,9.25818730943169,9.803163295560267,10.347999601031134,10.892701042418722,11.43727221736225,11.981717516865402,12.526041136776229,13.070247088509866,13.614339209072085,14.158321170436132,14.702196488321084,15.245968530415851,15.789640524088766,16.333215563620968,16.876696616996377,17.420086532280337,17.96338804361551])

plt.figure(figsize=(10, 6))
plt.scatter(mu, rendements, color='red')
plt.xlabel('Facteur de tolérance au risque μ')
plt.ylabel('Retour')
plt.title('Retour en fonction du risque pour portefeuille contenant NKE, BAC, PFE et MO')
plt.legend()
plt.grid(True)
plt.show()


