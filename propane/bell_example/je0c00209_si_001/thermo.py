import numpy as np

R = 8.314472
N_A = 6.02214076e23 # [1/mol]
k_B = R/N_A
M = 0.04409562 # [kg/mol]
m = M/N_A # [kg/molecule]
Tc = 369.89 # [K]
rhoc_kgm3 = 220.4781 # [kg/m^3]
rhoNc = rhoc_kgm3/M*N_A # [molecules/m^3]

# Coefficients from the EOS of Lemmon et al.
n = [0.042910051,1.7313671,-2.4516524,0.34157466,-0.46047898,-0.66847295,0.20889705,
     0.19421381,-0.22917851,-0.60405866,0.066680654,0.017534618,0.33874242,0.22228777,
     -0.23219062,-0.09220694,-0.47575718,-0.017486824]
t = [1.0,0.33,0.8,0.43,0.90,2.46,2.09,0.88,1.09,3.25,4.62,0.76,2.50,2.75,3.05,2.55,8.40,6.75]
d = [4.0,1.0,1.0,2.0,2.0,1.0,3.0,6.0,6.0,2.0,3.0,1.0,1.0,1.0,2.0,2.0,4.0,1.0]
c = [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] # 1 if l>0; 0 otherwise
l = [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,2.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
eta = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.963,1.977,1.917,2.307,2.546,3.28,14.6]
beta = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.33,3.47,3.15,3.19,0.92,18.8,547.8]
gamma = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.684,0.829,1.419,0.817,1.5,1.426,1.093]
epsilon = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.283,0.6936,0.788,0.473,0.8577,0.271,0.948]

# Make sure all arrays are the same length
assert(len(set([len(v) for v in [n,t,d,c,l,eta,beta,gamma,epsilon]])) == 1)

def get_Axy(T, rho_kgm3, itau, idelta):
    tau = Tc/T
    delta = rho_kgm3/rhoc_kgm3
    N = len(gamma)
    summer = 0.0
    if itau == 0 and idelta == 0:
        for i in range(N):
            u_i = -c[i]*delta**l[i]-eta[i]*(delta-epsilon[i])**2-beta[i]*(tau-gamma[i])**2
            summer += n[i]*tau**t[i]*delta**d[i]*np.exp(u_i)
    elif itau == 1 and idelta == 0:
        for i in range(N):
            u_i = -c[i]*delta**l[i]-eta[i]*(delta-epsilon[i])**2-beta[i]*(tau-gamma[i])**2
            fact = -2*beta[i]*tau*(tau-gamma[i]) + t[i]
            summer += n[i]*tau**t[i]*delta**d[i]*np.exp(u_i)*fact
    elif itau == 0 and idelta == 1:
        for i in range(N):
            u_i = -c[i]*delta**l[i]-eta[i]*(delta-epsilon[i])**2-beta[i]*(tau-gamma[i])**2
            fact = -c[i]*l[i]*delta**l[i]-2*eta[i]*delta*(delta-epsilon[i]) + d[i]
            summer += n[i]*tau**t[i]*delta**d[i]*np.exp(u_i)*fact
    else:
        raise ValueError('bad pair of itau,idelta:'+str(tuple(itau,idelta)))
    return summer

def get_splus(T, rho_kgm3):
    alphar = get_Axy(T, rho_kgm3, 0, 0)
    taudalphardtau = get_Axy(T, rho_kgm3, 1, 0)
    return -taudalphardtau+alphar

def get_frakB2(T):
    """ From Lemmon et al.; output in m^3/molecule """
    tau = Tc/T
    N = len(gamma)
    summer = 0
    for k in range(N):
        if d[k] == 1:
            c_k = n[k]*np.exp(-eta[k]*epsilon[k]**2)
            derivB2 = (-2*beta[k]*tau**(t[k]+1)*(gamma[k]-tau)+tau**t[k]*(1-t[k]))
            summer += c_k*np.exp(-beta[k]*(gamma[k]-tau)**2)*derivB2
    return summer/rhoNc

def get_frakB3(T):
    """ From Lemmon et al.; output in m^6/molecule^2 """
    tau = Tc/T
    N = len(gamma)
    summer = 0
    for k in range(N):
        if d[k] == 2:
            paren = (2*beta[k]*tau*(gamma[k]-tau) + t[k] - 1)
            expui = np.exp(-beta[k]*(tau-gamma[k])**2 -epsilon[k]**2*eta[k])
            summer += -2*n[k]*tau**t[k]*paren*expui
        if d[k] == 1:
            # N.B.: Small integers can be exactly represented as floats
            if l[k] == 1.0 and eta[k] == 0 and beta[k] == 0: 
                # An exponential term
                summer += 2*n[k]*tau**t[k]*(t[k]-1)
            elif l[k] == 0.0 and eta[k] != 0 and beta[k] != 0:
                # A Gaussian term
                expui = np.exp(-beta[k]*(gamma[k] - tau)**2 - epsilon[k]**2*eta[k])
                paren = -2*beta[k]*tau*(gamma[k] - tau) - t[k] + 1
                summer += 4*epsilon[k]*eta[k]*n[k]*tau**t[k]*paren*expui
    return summer/rhoNc**2

if __name__ == '__main__':
    # Definition of state point
    T_K, p_kPa, eta_Pas, rho_kgm3 = 373.067, 13797, 6.34E-05, 421.333

    # CoolProp v6.3.0 implements the EOS of Lemmon et al. for propane
    # so we check that the implementation you are reading is consistent with that implementation
    import CoolProp.CoolProp as CP

    splusCP = -CP.PropsSI('Smolar_residual','T',T_K,'Dmass',rho_kgm3,'n-Propane')/(N_A*k_B)
    assert(abs(splusCP-get_splus(T_K, rho_kgm3)) < 1e-10)

    B2 = CP.PropsSI('Bvirial','T',T_K,'Dmass',1e-12,'n-Propane')/N_A
    dB2dT = CP.PropsSI('dBvirial_dT','T',T_K,'Dmass',1e-12,'n-Propane')/N_A
    frakB2 = B2 + T_K*dB2dT
    print('B_2+T*dB_2/dT (CoolProp, this file, difference):')
    print(get_frakB2(T_K), frakB2, get_frakB2(T_K)-frakB2)

    B3 = CP.PropsSI('Cvirial','T',T_K,'Dmass',1e-12,'n-Propane')/N_A**2
    dB3dT = CP.PropsSI('dCvirial_dT','T',T_K,'Dmass',1e-12,'n-Propane')/N_A**2
    frakB3 = B3 + T_K*dB3dT
    print('B_3+T*dB_3/dT (CoolProp, this file, difference):')
    print(get_frakB3(T_K), frakB3, get_frakB3(T_K)-frakB3)
    print('****')
    print('Note: they will note be precisely the same because the code uses exact virial coefficients'
          ', but CoolProp uses numerical approximations that are good but not exact')