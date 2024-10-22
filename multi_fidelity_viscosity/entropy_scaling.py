
import numpy as np
from scipy.optimize import curve_fit
import feos
from feos.si import * # SI numbers and constants
from feos.pcsaft import *
from feos.eos import *
from scipy.interpolate import LinearNDInterpolator

def collision_integral( T, p):
    """
    computes analytical solution of the collision integral

    T: reduced temperature
    p: parameters

    returns analytical solution of the collision integral
    """
    A,B,C,D,E,F,G,H,R,S,W,P = p
    return A/T**B + C/np.exp(D*T) + E/np.exp(F*T) + G/np.exp(H*T) + R*T**B*np.sin(S*T**W - P)

def get_omega11(red_temperature):
    """
    computes analytical solution of the omega11 collision integral
    
    red_temperature: reduced temperature
    
    returns omega11
    """
    p11 = [ 
        1.06036,0.15610,0.19300,
        0.47635,1.03587,1.52996,
        1.76474,3.89411,0.0,
        0.0,0.0,0.0
    ]
    return collision_integral(red_temperature,p11)

def get_omega22(red_temperature):
    """
    computes analytical solution of the omega22 collision integral

    red_temperature: reduced temperature

    returns omega22
    """
    p22 = [ 
         1.16145,0.14874,0.52487,
         0.77320,2.16178,2.43787,
         0.0,0.0,-6.435/10**4,
         18.0323,-0.76830,7.27371
        ]
    return collision_integral(red_temperature,p22)

def get_CE_viscosity_reference(temperature, density, saft_parameters):
    """
    computes Chapman-Enskog viscosity reference for an array of temperatures
    uses pc-saft parameters

    temperature: array of temperatures
    saft_parameters: pc saft parameter object build with feos

    returns reference
    """
    epsilon = saft_parameters.pure_records[0].model_record.epsilon_k*KELVIN
    sigma   = saft_parameters.pure_records[0].model_record.sigma*ANGSTROM
    m       = saft_parameters.pure_records[0].model_record.m
    M       = saft_parameters.pure_records[0].molarweight*GRAM/MOL
    red_temperature = temperature/epsilon

    omega22 = get_omega22(red_temperature)

    sigma2 = sigma**2
    M_SI = M

    sq1  = np.sqrt( M_SI * KB * temperature / NAV /np.pi /METER**2 / KILOGRAM**2 *SECOND**2 ) *METER*KILOGRAM/SECOND
    div1 = omega22 * sigma2
    viscosity_reference = 5/16* sq1 / div1 #*PASCAL*SECOND
    
    #return np.ones(len(temperature)) *PASCAL*SECOND *np.min(viscosity_reference/PASCAL/SECOND)
    return viscosity_reference

def get_modCE_viscosity_reference(temperature, density, saft_parameters):
    m = saft_parameters.pure_records[0].model_record.m
    return get_CE_viscosity_reference(temperature, density, saft_parameters)/m
    
    

class LJ_mayer_kabelac():
    """
    wrapper for LJ-type (tabled results in reduced properties) viscosities
    
    falls back to CE if data outside range (bad idea???)
    
    """
    def __init__(self, red_temperature, red_density, red_viscosity):
        
        self.red_temperature = red_temperature
        self.red_density     = red_density
        self.red_viscosity   = red_viscosity
      
        self.red_viscosity_estimator = LinearNDInterpolator( np.array([ red_temperature, red_density ]).T, red_viscosity )
        
        return
        
    def get_LJ_viscosity_reference(self, temperature, density, saft_parameters):
        epsilon = saft_parameters.pure_records[0].model_record.epsilon_k*KELVIN
        sigma   = saft_parameters.pure_records[0].model_record.sigma*ANGSTROM
        M       = saft_parameters.pure_records[0].molarweight*GRAM/MOL    
        
        tt  = self.reduce_temperature( temperature, epsilon )
        rr  = self.reduce_mass_density( density, sigma, M )
        vis = self.unreduce_viscosity(self.red_viscosity_estimator( np.array([tt,rr]).T ), epsilon, sigma, M ) /PASCAL/SECOND     
        
        p   = np.argwhere(np.isnan(vis))
        ttt = temperature/KELVIN
        dummy = get_CE_viscosity_reference( ttt[p]*KELVIN,None,saft_parameters )/PASCAL/SECOND
        vis[p] = dummy
        print(len(p),"datapoints substituded by Chapman-Enskog. THIS IS A VERY BAD IDEA :P")
        return vis*PASCAL*SECOND

    def unreduce_temperature(self, red_temperature, epsilon_kb ):
        return red_temperature*epsilon_kb
    
    def reduce_temperature(self, temperature, epsilon_kb ):
        return temperature/ epsilon_kb    
    
    def unreduce_mass_density(self, red_density, sigma, M):
        return red_density/ sigma**3/NAV *self.M  
    
    def reduce_mass_density(self, density, sigma, M ):
        return density* sigma**3*NAV / M      
    
    def unreduce_viscosity(self, red_viscosity, epsilon_kb, sigma, M):
        return red_viscosity * np.sqrt( epsilon_kb*KB*M/NAV ) / sigma**2
    

def get_diffusion_reference(temperature, density, saft_parameters):
    """
    computes viscosity reference for arrays of temperatures and densities
    based on pc-saft parameters

    temperature: array of temperatures
    density: array of densities
    saft_parameters: pc-saft parameter object build with feos

    returns reference
    """
    epsilon = saft_parameters.pure_records[0].model_record.epsilon_k*KELVIN
    sigma   = saft_parameters.pure_records[0].model_record.sigma*ANGSTROM
    m       = saft_parameters.pure_records[0].model_record.m
    M       = saft_parameters.pure_records[0].molarweight*GRAM/MOL
    red_temperature = temperature/epsilon

    omega11 = get_omega11(red_temperature)

    sigma2 = sigma**2
    M_SI = M
    num_density       = density/M_SI*NAV

    sq2  = np.sqrt( RGAS*temperature/( np.pi*M_SI*m ) /METER**2 *SECOND**2 ) *METER/SECOND
    div2 = sigma2*num_density*omega11 
    diffusion_reference = 3/8/div2*sq2 

    return diffusion_reference


def get_thermal_conductivity_reference(temperature, residual_entropies, saft_parameters):
    """
    computes thermal conductivity reference for arrays of temperatures and residual entropies
    based on pc-saft parameters

    temperature: array of temperatures
    residual_entropies: array of residual entropies
    saft_parameters: pc-saft parameter object build with feos
    
    returns reference
    """
    epsilon = saft_parameters.pure_records[0].model_record.epsilon_k*KELVIN
    sigma   = saft_parameters.pure_records[0].model_record.sigma*ANGSTROM
    m       = saft_parameters.pure_records[0].model_record.m
    M       = saft_parameters.pure_records[0].molarweight*GRAM/MOL

    red_temperature = temperature/epsilon
    omega22 = get_omega22(red_temperature)

    eos = EquationOfState.pcsaft(saft_parameters)
    residual_entropie_crit = State.critical_point(eos).specific_entropy(Contributions.Residual)/ KB /NAV *M/m

    alpha_visc = np.exp(-1.0 * (residual_entropies / residual_entropie_crit))
    ref_CE = (83.235 / 10.0**3) * (temperature/KELVIN / M * (GRAM / MOL) / m)**0.5 / sigma**2*ANGSTROM**2 / omega22 * m
    temperature_star = (temperature / epsilon / m)
    ref_temperature_star = (-0.0167141 * temperature_star + 0.0470581 * temperature_star**2) *(m**2 * sigma**3/ANGSTROM**3 * epsilon/KELVIN) / 10.0**5
    
    thermal_reference = (ref_CE + ref_temperature_star * alpha_visc) * WATT / METER / KELVIN    

    return thermal_reference

def viscosity_correlation(s, a, b, c, d):
    """
    polynomial correlation function for reduced viscosities in reduced residual entropy space

    s: array of reduces resitual entropies
    a,b,c,d: parameters of the polynomial

    returns reduced viscosities
    """
    return a + b*s + c*s**2 + d*s**3

def old_diffusion_correlation(s, a, b, c):
    """
    old polynomial correlation function for reduced diffusion coefficients in reduced residual entropy space

    s: array of reduces resitual entropies
    a,b,c: parameters of the polynomial

    returns reduced diffusion coefficients
    """
    return a - b*( 1. - np.exp(s) )*s**2 + c*s**3

def diffusion_correlation(s, a, b, c, d):
    """
    new polynomial correlation function for reduced diffusion coefficients in reduced residual entropy space
    new term d*s**10 for supercooled liquids only

    s: array of reduces resitual entropies
    a,b,c,d: parameters of the polynomial
    
    returns reduced diffusion coefficients
    """
    return a + b*s - c* ( 1. - np.exp(s) )*s**2 + d*s**10

def thermal_cond_correlation(s, a, b, c, d):
    """
    new polynomial correlation function for reduced thermal conductivity in reduced residual entropy space

    s: array of reduces resitual entropies
    a,b,c,d: parameters of the polynomial
    
    returns reduced thermal conductivity
    """    
    return a + b*s + c*( 1 - np.exp(s) ) + d*s**2
