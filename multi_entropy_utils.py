import numpy as np
import pandas as pd
import glob, os
from feos.si import * # SI numbers and constants
from feos.pcsaft import *
from feos.eos import *
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize,curve_fit

from entropy_scaling import *

import GPy
from emukit.model_wrappers import GPyModelWrapper
from emukit.experimental_design.experimental_design_loop import ExperimentalDesignLoop
from emukit.core import ParameterSpace, ContinuousParameter
from emukit.core.loop import UserFunctionWrapper

import emukit.multi_fidelity
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays

def split_dataset( data, p, units ):
    dummy = {}
    for key in data.keys():
        if key in units.keys() and units[key]:
            helper = data[key]/units[key]
            dummy[key] = helper[p]*units[key]
        else:
            dummy[key] = data[key][p]
    return dummy


def apply_units( data, units ):
    dummy = {}
    for key in data.keys():
        if key in units.keys() and units[key]:
            dummy[key] = np.array(list(data[key])).astype(float)*units[key]
        else:
            dummy[key] = np.array(data[key])
    return dummy


def viscosity_correlation(s, a, b, c, d):
    return a + b*s + c*s**2 + d*s**3

def viscosity_prho_entropy_space( dummy, parameters, viscosity_reference,
                                 rho_SI=GRAM/METER**3, p_SI=BAR, reference="CE" ):
    M = parameters.pure_records[0].molarweight *(GRAM/MOL)
    m = parameters.pure_records[0].model_record.m

    # calculate entropies with PC-SAFT
    eos = EquationOfState.pcsaft(parameters)
    residual_entropies = []
    d_helper = dummy["densities"] / rho_SI   
    p_helper = dummy["pressures"] / p_SI     
    for i,( t,(d,p) ) in enumerate( zip(dummy["temperatures"], zip( dummy["densities"],dummy["pressures"] ) ) ):
        #print( d, d_helper[i],"xxxxx", p, p_helper[i] )
        if not np.isnan(d_helper[i]):
            state = State(eos, temperature=t, density=d/M)
        elif not np.isnan(p_helper[i]):
            state = State(eos, temperature=t, pressure=p )
            d_helper[i] = state.mass_density() / rho_SI 
        else:
            residual_entropies.append( 100000000000 )
            print("miss")
            continue
        s0 = state.specific_entropy(Contributions.ResidualNvt)/ KB /NAV *M/m
        residual_entropies.append(s0)
    
    dummy["densities"] = d_helper * rho_SI
    dummy["residual_entropies"] = np.array(residual_entropies)
    # get references and norm
    dummy["viscosity_reference"] = viscosity_reference( dummy["temperatures"], dummy["densities"], parameters)
    dummy["ln_viscosity_star"] = np.log( dummy["viscosities"]/dummy["viscosity_reference"])
    
    return dummy


def viscosity_entropy_space( dummy, parameters, x_measure="" ):
    M = parameters.pure_records[0].molarweight *(GRAM/MOL)
    m = parameters.pure_records[0].model_record.m

    # calculate entropies with PC-SAFT
    eos = EquationOfState.pcsaft(parameters)
    residual_entropies = []
    if x_measure == "pressure":
        for t,d in zip(dummy["temperatures"], dummy["pressures"]):
            state = State(eos, temperature=t, pressure=d )
            s0 = state.specific_entropy(Contributions.ResidualNvt)/ KB /NAV *M/m
            residual_entropies.append(s0)
    else:            
        for t,d in zip(dummy["temperatures"], dummy["densities"]):
            state = State(eos, temperature=t, density=d/M)
            s0 = state.specific_entropy(Contributions.ResidualNvt)/ KB /NAV *M/m
            residual_entropies.append(s0)

    dummy["residual_entropies"] = np.array(residual_entropies)
    # get references and norm
    dummy["viscosity_reference"] = get_viscosity_reference(dummy["temperatures"], parameters)
    dummy["ln_viscosity_star"] = np.log( dummy["viscosities"]/dummy["viscosity_reference"])

    return dummy

def entropy_plot( xdata, ydata, yfill, markers, colors, labels, save="",
                 msize=12 , fsize=15, alpha=0.3, lsize = 4,
                 markeredgewidth=3, markeredgewidth_no=10 ):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches( 7, 5)   
    plt.gcf().set_size_inches(7, 5)
    
    for i,_ in enumerate(labels):
        if len(xdata[i]) < markeredgewidth_no:
            plt.plot(xdata[i], ydata[i], markers[i],
               color=colors[i], label=labels[i], markersize=msize, 
                     linewidth=lsize,markeredgewidth=markeredgewidth)
        else:
            plt.plot(xdata[i], ydata[i], markers[i],
           color=colors[i], label=labels[i], markersize=msize, linewidth=lsize)
            
        if len(yfill[i]) == len(xdata[i]):
            ax.fill_between(xdata[i], ydata[i] - 1.96*yfill[i], ydata[i] + 1.96*yfill[i], 
                            facecolor=colors[i], alpha=alpha)
    
    plt.xlabel("s*", fontsize=fsize)
    plt.ylabel("ln($\eta$*) ", fontsize=fsize)
    plt.legend(fontsize=fsize,frameon=False)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)    
    if save:
        plt.savefig( save+".png" , bbox_inches='tight')
        plt.savefig( save+".pdf" , bbox_inches='tight')
    plt.show()
    plt.close()
    return

class ansatz_test():
    
    def __init__(self, xdata, ydata, f ):
        self.xdata = xdata
        self.ydata = ydata
        self.f = f
        return
    
    def train(self):
        self.popt, self.pcov = curve_fit(self.f, 
                                         self.xdata, 
                                         self.ydata)        
        return
    
    def predict(self,x):
        return self.f(x, *self.popt), None

class gp_test():
    
    def __init__(self, xdata, ydata, n_restarts=5 ):
        self.xdata = xdata
        self.ydata = ydata
        self.n_restarts = n_restarts

        x_train_gp = np.atleast_2d( self.xdata ).T
        y_train_gp = np.atleast_2d( self.ydata ).T
        model_gpy = GPy.models.GPRegression(x_train_gp, y_train_gp)
        model_gpy.n_restarts = self.n_restarts
        self.model = GPyModelWrapper(model_gpy)
        return
    
    def train(self):

        self.model.model.optimize()
        #self.model = GPyModelWrapper(model_gpy)
        return
    
    def predict(self,x):
        gp_pred, gp_std =  self.model.predict(np.array( np.atleast_2d( x ).T ))
        return np.squeeze(gp_pred), np.squeeze(gp_std)


class multi_fidelity_test():
    
    def __init__(self, x_hi, x_lo, y_hi, y_lo, kernels):
        self.x_hi = x_hi
        self.y_hi = y_hi        
        self.x_lo = x_lo
        self.y_lo = y_lo

        x_train_l = np.atleast_2d( self.x_lo ).T
        y_train_l = np.atleast_2d( self.y_lo ).T
        x_train_h = np.atleast_2d( self.x_hi ).T
        y_train_h = np.atleast_2d( self.y_hi ).T           

        X_train, Y_train = convert_xy_lists_to_arrays([x_train_l, x_train_h], 
                                                    [y_train_l, y_train_h])        
        
        lin_mf_kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        self.gpy_model = GPyLinearMultiFidelityModel(X_train, Y_train, lin_mf_kernel, n_fidelities=2)
        
        return
    
    def train(self):
        self.model = GPyMultiOutputWrapper(self.gpy_model, 2, n_optimization_restarts=5)
        self.model.optimize()
        return
    
    def predict(self,x):
        xx = np.atleast_2d( x ).T
        X = convert_x_list_to_array([xx, xx])
        X_l = X[:len(x)]
        X_h = X[len(x):]
        hf_mean, hf_var = self.model.predict(X_h)
        hf_std = np.sqrt(hf_var)
        return np.squeeze(hf_mean), np.squeeze(hf_std)
        
    def predict_low_fidelity(self,x):
        xx = np.atleast_2d( x ).T
        X = convert_x_list_to_array([xx, xx])
        X_l = X[:len(x)]
        X_h = X[len(x):]
        lf_mean, lf_var = self.model.predict(X_l)
        lf_std = np.sqrt(lf_var)
        return np.squeeze(lf_mean), np.squeeze(lf_std)    
    
def get_entropy_error(xref, yref, f):
    dummy, _ = f.predict(xref)
    return np.mean( np.abs( (dummy - yref)/yref ) )*100

def get_superspace_error(xref, yref, normer, f):
    dummy = backtranslate(xref, normer, f)
    return np.mean( np.abs( (dummy - yref)/yref ) )*100    

def backtranslate(xref, normer, f):
    dummy, _ = f.predict(xref)
    return  np.squeeze( np.exp(dummy)*normer )

def superspace_plot( xdata, ydata, yfill, markers, colors, labels, save="",
                 msize=12 , fsize=15, alpha=0.3, lsize = 4,
                 markeredgewidth=3, markeredgewidth_no=10 ):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches( 7, 5)   
    plt.gcf().set_size_inches(7, 5)
    
    for i,_ in enumerate(labels):
        if len(xdata[i]) < markeredgewidth_no:
            plt.plot(xdata[i], ydata[i], markers[i],
               color=colors[i], label=labels[i], markersize=msize, 
                     linewidth=lsize,markeredgewidth=markeredgewidth)
        else:
            plt.plot(xdata[i], ydata[i], markers[i],
           color=colors[i], label=labels[i], markersize=msize, linewidth=lsize)
            
        if len(yfill[i]) == len(xdata[i]):
            ax.fill_between(xdata[i], ydata[i] - 1.96*yfill[i], ydata[i] + 1.96*yfill[i], 
                            facecolor=colors[i], alpha=alpha)
    
    plt.xlabel("temperature / K", fontsize=fsize)
    plt.ylabel("$\eta$ / Pas", fontsize=fsize)
    plt.legend(fontsize=fsize,frameon=False)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)    
    if save:
        plt.savefig( save+".png" , bbox_inches='tight')
        plt.savefig( save+".pdf" , bbox_inches='tight')
    plt.show()
    plt.close()
    return

def linear_plot( x_hi, x_lo, markers, colors, labels, save="",
                 msize=12 , fsize=15, alpha=0.3, lsize = 4,
                 markeredgewidth=3, markeredgewidth_no=10 ):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_size_inches( 5, 5)   
    plt.gcf().set_size_inches(5, 5)
    
    for i,_ in enumerate(labels):
        plt.plot(x_hi[i], x_lo[i], markers[i],
           color=colors[i], label=labels[i], markersize=msize, linewidth=lsize,markeredgewidth=markeredgewidth)

    plt.xlabel("ln($\eta$*) high", fontsize=fsize)
    plt.ylabel("ln($\eta$*) low", fontsize=fsize)
    if len(labels)>1:
        plt.legend(fontsize=fsize,frameon=False)
    plt.xticks(fontsize=fsize)
    plt.yticks(fontsize=fsize)    
    if save:
        plt.savefig( save+".png" , bbox_inches='tight')
        plt.savefig( save+".pdf" , bbox_inches='tight')
    plt.show()
    plt.close()
    return

