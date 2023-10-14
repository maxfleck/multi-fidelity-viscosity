import pandas
import numpy as np 
import os
import scipy.optimize
 
def ITS48to68(T48_K):
    """ From https://www.bipm.org/utils/common/pdf/its-90/SInf_Chapter_1_Introduction_2013.pdf"""
    def T68_minus_T48(t68_Celsius):
        if t68_Celsius < 0:
            b = [8.188411e-03,9.722129e-04,1.009974e-04,2.952294e-06,4.520372e-08,3.863623e-10,1.684889e-12,2.879618e-15]
        elif t68_Celsius > 0 and t68_Celsius < 470:
            b = [2.83469e-04,-4.85523e-04,6.05956e-06,-8.17404e-09,-6.63454e-11,3.11292e-13,-5.65993e-16,3.98137e-19 ]
        else:
            b = [6.0317242e00,-3.2703041e-02, 6.5078688e-05,-6.0234949e-08, 3.0420643e-11,-8.5348347e-15,1.2509557e-18,-7.4707543e-23]
        return sum([b[i]*t68_Celsius**i for i in range(8)])
    def resid(T68):
        t68 = T68-273.15
        return T68_minus_T48(t68) - (T68-T48_K)
    r = scipy.optimize.newton(resid, T48_K)
    return r

def ITS68to90(T68_K):
    """ From https://www.bipm.org/utils/common/pdf/its-90/SInf_Chapter_1_Introduction_2013.pdf"""
    def T90_minus_T68(t90_Celsius):
        b = [0,-0.148759,-0.267408,1.080760,1.269056,-4.089591,-1.871251,7.438081,-3.536296]
        return sum([b[i]*(t90_Celsius/630)**i for i in range(1,9)])
    def resid(T90):
        t90 = T90-273.15
        return T90_minus_T68(t90) - (T90-T68_K)
    r = scipy.optimize.newton(resid, T68_K)
    return r

def convert_to_ITS90(value, old_scale):
    if old_scale == 68:
        return ITS68to90(value)    
    elif old_scale == 48:
        return ITS68to90(ITS48to68(value))
    else:
        return value

def add_T_K90(row):
    # T must always be specified
    assert('T' in row)
    assert('T_units' in row)
    T_units = row['T_units']
    if row['year'] > 1990:
        ITSscale = 90
    elif row['year'] > 1968:
        ITSscale = 68
    else:
        ITSscale = 48

    if T_units == 'K':
        return convert_to_ITS90(row['T'], ITSscale)
    elif T_units == 'C':
        return convert_to_ITS90(row['T']+273.15, ITSscale)
    elif T_units == 'F':
        return convert_to_ITS90((row['T']-32) * 5/9 + 273.15, ITSscale)
    else:
        raise ValueError(T_units)

def add_VDN_kgm3(row, M):
    rho = row['rho']
    rho_units = row['rho_units']
    if pandas.isnull(row['rho']):
        return np.nan
    else:
        if rho_units == 'g/cm3':
            return rho*1000
        elif rho_units == 'mol/L':
            return rho*1000*M
        elif rho_units == 'kg/m^3':
            return rho
        else:
            raise ValueError(rho_units)

def add_p_kPa(row):
    p = row['p']
    p_units = row['p_units']
    if pandas.isnull(row['p']):
        return np.nan
    else:
        if p_units == 'MPa':
            return p*1e3
        elif p_units == 'kPa':
            return p 
        elif p_units == 'atm':
            return p*101.325 
        elif p_units == 'psia':
            return p*6.894757 # https://www.nist.gov/physical-measurement-laboratory/nist-guide-si-appendix-b8
        else:
            raise ValueError(p_units)

def add_eta_Pas(row):
    # eta must always be specified
    assert('eta' in row)
    assert('eta_units' in row)
    eta = row['eta']
    eta_units = row['eta_units']
    if eta_units == 'uP':
        return eta*0.1/1e6
    elif eta_units == 'cP':
        return eta*0.1/1e2
    elif eta_units == 'uPa*s':
        return eta/1e6
    elif eta_units == 'ug/cms':
        return eta*0.1/1e6
    else:
        raise ValueError(eta_units)

def convert_raw_propane():
    here = os.path.abspath(os.path.dirname(__file__))
    data = pandas.read_csv(here+'/propane_dense_raw.csv')

    data['T [K]'] = data.apply(add_T_K90, axis=1)
    del data['T']
    del data['T_units']
    data['P [kPa]'] = data.apply(add_p_kPa, axis=1)
    del data['p']
    del data['p_units']
    data['NVC [PaÂ·s]'] = data.apply(add_eta_Pas, axis=1)
    del data['eta']
    del data['eta_units']
    #import CoolProp.CoolProp as CP
    data['VDN [kg/m^3]'] = data.apply(add_VDN_kgm3, axis=1, M = 44.097 ) #CP.PropsSI('molemass','Propane'))
    del data['rho']
    del data['rho_units']

    data.to_csv(here+'/propane_converted.csv',index=False)

    return data

if __name__ == '__main__':
    import sys
    sys.path.append('../..')
    data = convert_raw_propane()
    
    sec = data[~pandas.isnull(data.secondary)]
    pri = data[pandas.isnull(data.secondary)]

    for df in pri, sec:
        print('***************')
        for doi, gp in df.groupby('doi'):
            assert(len(set(gp['doi']))==1)
            assert(len(set(gp['year']))==1)
            assert(len(set(gp['author']))==1)
            meta = dict(gp.iloc[0])
            year, auth = [meta[k] for k in ['year','author']]
            print(year, auth, len(gp), np.min(gp['P [kPa]'])/1e3, np.max(gp['P [kPa]'])/1e3)
