import numpy as np
import thermo
from thermo import R, N_A, k_B, M, m, Tc

def get_eta0_Pas(T):
    """ From Vogel and Herrmann; output in Pa*s """
    n = [9.9301297115406, 7.2658798096248e-1, 
        -7.4692506744427e-1, 1.0156334572774e-1]
    e = [-1, -2 ,-3, -4]
    o = 0.0
    for n_i, t_i in zip(n, e):
        o += n_i*(Tc/T)**t_i
    return o/1e6

def get_etapluszero(T):
    fB2 = thermo.get_frakB2(T)
    return get_eta0_Pas(T)/np.sqrt(m*k_B*T)*fB2**(2/3)

def get_Beta1_m3(T):
    """ From Vogel, output in m^3/molecule """
    sigma = 0.49154e-9 # [m]
    Tstar = T/260.0 # Fitted in this work
    b = [-19.572881,219.73999,-1015.3226,
         2471.01251, -3375.1717,2491.6597,
         -787.26086,14.085455,-0.34664158]
    p = [-0.25*i for i in range(7)] + [-2.5, -5.5]
    assert(len(b) == len(p))
    Bstar = sum(b_i*Tstar**p_i for b_i,p_i in zip(b,p))
    return Bstar*sigma**3

def get_Beta1plus(T):
    frakB2 = thermo.get_frakB2(T)
    frakB3 = thermo.get_frakB3(T)
    Beta1 = get_Beta1_m3(T)
    return 1/frakB2*(1/3*frakB3/frakB2 + Beta1)

def get_lnUpsilon(T, splus):
    splus1 = 2.0; splus2 = 5.4 
    mAr = 0.63392108; bAr = -0.5339991
    # N.B.: The coefficients in polyval are required to
    # be in *decreasing* order

    if splus > splus2:
        coeffSA = [0.316991, -0.302498, 0.440977]
        x = np.log(splus)
        return np.exp(np.polyval(coeffSA[::-1], x))
    elif splus < splus1:
        a_1 = get_Beta1plus(T)*get_etapluszero(T)
        a_2 = (splus1*(2*mAr-2*a_1)+3*bAr)/splus1**2
        a_3 = (splus1*(a_1-mAr)-2*bAr)/splus1**3
        return np.polyval([a_3, a_2, a_1, 0], splus)
    else:
        return np.polyval([mAr, bAr], splus)

def get_eta(T_K, rho_kgm3):
    splus = thermo.get_splus(T_K, rho_kgm3)
    etaplus0 = get_etapluszero(T_K)
    Upsilon = np.exp(get_lnUpsilon(T_K, splus))
    etaplus = Upsilon - 1.0 + etaplus0
    rhoN = rho_kgm3/M*N_A
    sqmkT = np.sqrt(m*k_B*T_K)
    if splus == 0:
        fB2 = thermo.get_frakB2(T_K)
        fac = etaplus*sqmkT/fB2
    else:
        fac = (rhoN**(2/3)*sqmkT)/splus**(2/3)
    return etaplus*fac

if __name__ == '__main__':
    for T_K, p_kPa, eta_Pas, rho_kgm3 in [
        # Check: experimental measuments from Seibt
        (373.146, 917.29, 1.03E-05, 14.099),
        (373.067, 13797, 6.34E-05, 421.333),
        (373.115, 28928, 8.54E-05, 470.686)
        ]:
        splus = thermo.get_splus(T_K, rho_kgm3)
        rhoN = rho_kgm3/M*N_A
        etacorr_Pas = get_eta(T_K, rho_kgm3)
        dev = 100*(etacorr_Pas/eta_Pas-1)
        print(splus, eta_Pas, dev, '% diff')

""" Output when running script:
0.09103193464785768 1.03e-05 -0.5829275526744726 % diff
2.229202444328692 6.34e-05 -0.6336127278606551 % diff
2.6159146769686377 8.54e-05 -0.8955076274567486 % diff
"""