# -*- coding: utf-8 -*-
"""
Function and routines file used in biofire_main.py for MT "How fires shape biodiversity in plant communities: a study using a stochastic dynamical model" (Torrassa, 2023)
"""

import os
import json
import logging
import sys

import numba
from random import random, uniform
import numpy as np
from numpy.random import default_rng
import scipy
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------

# (0) DYNAMIC INTEGRATION PARAMS from json file

# -------------------------------------------------------------------------------------

def settings(file_settings):
    global NP, maxT, initT, NN, N0, N1, rep, H, HY, eps, minfirerettime, firevf

    fjson = open(file_settings)
    ds = json.load(fjson)

    # community initial size
    NP = ds["algorithm"]["generation"]["n_species"]

    # SIMULATION LENGHT
    maxT = ds["algorithm"]["simulation"]["runtime"]     # years
    initT = ds["algorithm"]["simulation"]["initT"]      # years

    rep = ds["algorithm"]["simulation"]["repetition"]   # number of repetition of the simulation

    H   = ds["algorithm"]["simulation"]["delta_day"]    # d, delta t of integration LEAVE IT TO 1 DAY,DO
    eps = ds["algorithm"]["simulation"]["epsilon"]      # y^-1, minimum fire frequency when feedback
    minfirerettime = ds["algorithm"]["simulation"]["minfirerettime"]          # mean fire return time
    firevf = ds["algorithm"]["simulation"]["firevf"]

    # NUMBER OF INTEGRATION TIMESTEP (SAME AS INTEGRATION TIME IN DAYS BECAUSE h=1 d)
    NN = int(maxT * 365) #days
    N0 = int(NN/2) #days
    N1 = int(initT * 365) #days
    HY = H/365                  # y, delta t of integration

# -------------------------------------------------------------------------------------

# (2) ALGORITHM FUNCTIONS

# -------------------------------------------------------------------------------------

# (2.1) DYNAMIC FUNCTION

# -------------------------------------------------------------------------------------

@numba.jit()
def derivs(tt, v, dv):
    """ Tilman equations system extended to NP species """
    A = np.ones(NP)         # parameters for Tilman asymetric hierarchy
    A[0] = 0.0
    ip = 0
    while ip < NP:
        dv[ip] = C[ip]*v[ip] * (1-np.sum(v[0:ip+1])) - M[ip]*v[ip] - v[ip]*np.sum((C*v)[0:ip])*A[ip]
        ip += 1

@numba.jit()
def rk4(tt, y, dydx):
    """ Runge Kutta 4th Order """
    k2 = np.zeros(NP)
    k3 = np.zeros(NP)
    k4 = np.zeros(NP)
    h6 = HY/6.
    k1 = dydx
    y1 = y+0.5*k1*HY
    derivs(tt, y1, k2)
    y2 = y + 0.5*k2*HY
    derivs(tt, y2, k3)
    y3 = y+k3*HY
    derivs(tt, y3, k4)
    yout = y+h6*(k1+2*k2+2*k3+k4)
    return yout

@numba.jit()
def fireocc(v, iifire):
    """ Fire Occurrence function """
    v_out = v*(1-iifire) + v*R*iifire
    return v_out


@numba.jit()
def dyn(b, bout, steps, firevf):
    """ Complete Dynamic of the community """
    i = 0
    iifire = 0
    fv = []
    df = np.zeros(NP)
    f = np.zeros_like(b)
    
    while i < steps:
        f = b
        derivs(i, f, df)
        b = rk4(i, f, df)
        
        # stochastic fire dynamics with veg-feedback and minimum return fire of 2 yrs
        dummy = random()
        numok = round(1./(np.sum(L*b)+eps))*365
        if dummy <= 1./(numok)*H and firevf > (minfirerettime*365.):
            iifire = 1.     # FIRE
            fv.append(firevf/365.)
            firevf = 0.
        else:
            iifire = 0.     # NO FIRE
            firevf = firevf+H
        
        b = fireocc(b, iifire)      # fire occurrence function
        # set to 0.0 b < 1e-10
        # b = (1 - (b < 1e-10))*b
        bout[:,i] = b        
        # if i % 100 == 0 or iifire > 0:
        #     print(i, b, iifire)
        i += 1
    # print(bout[:,0], bout[:,-1])
    return bout, fv

#--------------- (2.2) COMMUNTY FUNCTIONS ---------------

def new_community():
    """
    Function to restart to the basic plant community - in this case, no pfts
    """
    global NP, C, M, R, L, B0ic
    NP = 0
    C = np.array([])
    M = np.array([])
    R = np.array([])
    L = np.array([])
    B0ic = np.array([[]])
    
def initial_conditions(NP, binv=0.0):
    global B0ic, spp
    # bisogna aggiungere un controllo sul numero di PFT, se fossero 100+ allora la condizione iniziale bassa andrebbe cambiata
    spp = np.zeros(NP)
    nINV = np.count_nonzero(spp)
    nNAT = len(spp) - nINV
    B0ic = np.array([])
    bmin = 0.01
    while round(1/(nNAT+1),3) <= bmin:
        bmin = bmin / 2
    AA = np.ones(nNAT)*bmin # all low
    BB = np.ones((nNAT,nNAT))*bmin + np.identity(nNAT)*(0.9-bmin*nNAT) # all low, one high
    CC = np.ones(nNAT)*round(1/(nNAT+1),3) #all highest
    B0ic = np.insert(BB, 0, AA, axis=0)
    B0ic = np.insert(B0ic, nNAT+1, CC, axis=0)
    # initial condition for invasive species
    iINV = np.nonzero(spp)          
    for i in iINV[0]:
        B0ic = np.insert(B0ic, i, np.ones(nNAT+2)*binv, axis=1)
    
def med_community():
    global NP, C, M, R, L, B0ic, spp
    NP = 6
    spp = np.zeros(NP)
    C0 = 0.047
    C1 = 0.053
    C2 = 0.045
    C3 = 0.067
    C4 = 0.11
    C5 = 0.22
    C = np.array([C0, C1, C2, C3, C4, C5])
    M0 = 1/400
    M1 = 1/125
    M2 = 1/50
    M3 = 1/25
    M4 = 1/15
    M5 = 1/40
    M = np.array([M0, M1, M2, M3, M4, M5])
    R0 = 0.85
    R1 = 0.40
    R2 = 0.50
    R3 = 0.50
    R4 = 0.50
    R5 = 0.40
    R = np.array([R0, R1, R2, R3, R4, R5])
    L0 = 1/500
    L1 = 1/20
    L2 = 1/15
    L3 = 1/10
    L4 = 1/10
    L5 = 1/10
    L = np.array([L0, L1, L2, L3, L4, L5])

#-------------------------------------------------------------------

def flammability(rng, frt_min=1, frt_max=1000, size=1):
    """ Generation function for flammability L trait value """
    if frt_min<=1:
        frt_min = 1 + sys.float_info.epsilon #add an epsilon to exclude the lower bound, while the upper bound is automatically excluded in the numpy.Generator.uniform function
    frt = np.power(10, rng.uniform(np.log10(frt_min), np.log10(frt_max),size)) # specific fire return time
    return 1/frt

#-------------------------------------------------------------------
# GENERATION OF LINEAR C-M BASED ON THE MEDITERRANEAN TRAITS
#-------------------------------------------------------------------

def trait_linear_med(rng, t_max=1, t_min=0.01, size=1, sigma=0.007):
    """ 
    
    returns an array of colonization rate traits
        
    Parameters
    ----------
    t_max : float, optional
        The default value is 1
    t_min : float, optional
        The default value is 0.01
    size : int or tuple of ints, optional
        Size of the desired array. Usually correspond to the size of the pft community. The default value is 1.
        
    Returns
    -------
    T : ndarray rounded to the 3rd decimal value
    """
    N = size
    # linear correlation
    ii = np.arange(0,N)+1
    alfa = (t_max-t_min)/(N-1)
    beta = t_max - alfa*N
    # print(f"N={N}, C(i) = {beta} + {alfa} * i")
    # noise
    T = alfa*ii + beta
    
    if sigma!=0:
        for i in range(0,N):
            noise = rng.normal(0, sigma)
            while (T[i]+noise)<0.001: #imposto un limite inferiore sotto il quale C non può scendere
                noise = rng.normal(0, sigma)
            T[i] += noise
    return T

def rand_linear_community_med(rng, nnew):
    global NP, C, M, R, L#, B0ic, spp

    # Parametri per la generazione di C ed M presi dalla comunità del mediterraneo

    I_med = np.arange(1,7)

    C_med = np.array([0.047, 0.053, 0.045, 0.067, 0.11, 0.22])
    reg_ic = scipy.stats.linregress(x=I_med, y=C_med)
    C_min = reg_ic.slope*1 + reg_ic.intercept
    C_max = reg_ic.slope*6 + reg_ic.intercept
    C_reg = reg_ic.intercept + reg_ic.slope*I_med
    C_std = np.std(C_reg - C_med)

    M_med = np.array([1/400, 1/125, 1/50, 1/25, 1/15, 1/40])
    reg_im = scipy.stats.linregress(x=I_med, y=M_med)
    M_min = reg_im.slope*1 + reg_im.intercept
    M_max = reg_im.slope*6 + reg_im.intercept
    M_reg = reg_im.intercept + reg_im.slope*I_med
    M_std = np.std(M_reg - M_med)
    
    m, c = 0, 0
    while np.any(c<=m): #faccio il while solo su M in modo da non dover generare un'altra volta anche C
        c = np.round(trait_linear_med(rng, t_max=C_max, t_min=C_min, size=nnew, sigma=C_std), 5)
        m = np.round(trait_linear_med(rng, t_max=M_max, t_min=M_min, size=nnew, sigma=M_std), 5)
    
    
    r = np.round(rng.uniform(.001,1,size=nnew), 5)
    l = np.round(flammability(rng, frt_min=2, frt_max=500, size=nnew), 5)
    
    # NP = nnew
    # spp = np.zeros(NP)
    C = c
    M = m
    R = r
    L = l
    # initial_conditions(nnew)

    return C,M,R,L

#-------------------------------------------------------------------

def rand_uniform_community(rng, nnew):
    """
    Create a plant community with random uniform traits
    
    Parameters
    ----------
    rng: random generator
    nnew : int or tuple of ints, optional
        Number of the new PFTs. The default value is 1. 
    
    Returns
    -------
    None
    """
    global NP, C, M, R, L, B0ic, spp
    
    c, m = 0., 0.
    while np.any(c<=m):
        c = np.round(rng.uniform(.001,100,size=nnew), 3)
        m = np.round(rng.uniform(.001,10,size=nnew), 3)
    r = np.round(rng.uniform(.001,1,size=nnew), 3)
    l = np.round(flammability(rng, frt_min=2, frt_max=500, size=nnew), 3)
    
    NP = nnew
    spp = np.zeros(NP)
    C = c
    M = m
    R = r
    L = l
    initial_conditions(nnew)

    return C,M,R,L

def rand_uniform_invasive(rng, nnew=1):
    """
    Insert nnew new PFTs with random traits into an exixting community. Hyerarchical position i is random as well
    
    Parameters
    ----------
    rng: random generator
    nnew : int or tuple of ints, optional
        Number of the new PFTs. The default value is 1. 
    
    Returns
    -------
    None
    """
    global NP, C, M, R, L, B0ic
    
    for inew in np.arange(nnew):
        i = int(uniform(0,NP))
        c, m = 0., 0.
        while c<=m:
            c = round(uniform(.001,100),3)
            m = round(uniform(.001,10),3)
        l = round(uniform(.001,1),3)
        r = round(uniform(.001,1),3)
        NP += 1
        C = np.insert(C,i,c)
        M = np.insert(M,i,m)
        R = np.insert(R,i,r)
        L = np.insert(L,i,l)
        
        B0ic = np.insert(B0ic, i, 0.01, axis=1)
        B0ic = np.insert(B0ic, i+1, 0.01, axis=0)
        B0ic[i+1,i]=0.89


#-------------------------------------------------------------------

def trait_linear(rng, t_max=1, t_min=0.01, size=1, sigma=0.5):
    """ 
    
    returns an array of the selected trait
        
    Parameters
    ----------
    rng: random generator
    t_max : float, optional
        The default value is 1
    t_min : float, optional
        The default value is 0.01
    size : int or tuple of ints, optional
        Size of the desired array. Usually correspond to the size of the pft community. The default value is 1.
        
    Returns
    -------
    T : ndarray rounded to the 3rd decimal value
    """
    N = size
    # linear correlation
    ii = np.arange(0,N)+1
    alfa = (t_max-t_min)/(N-1)
    beta = t_max - alfa*N
    # noise
    T = alfa*ii + beta
    
    if sigma!=0:
        gamma = (t_max-t_min)/(N*sigma)
        for i in range(0,N):
            noise = rng.normal(0, gamma)
            while (T[i]+noise)<0.001 or (T[i]+noise)>t_max:
                noise = rng.normal(0, gamma)
            T[i] += noise

    return T


def rand_linear_invasive(rng, nnew=1, sigma=0.5):
    """
    Insert nnew new PFTs with random traits from Random-C-Exponential function into an exixting community.
    Hyerarchical position i is random as well from a uniform distribution.
    
    Parameters
    ----------
    nnew : int or tuple of ints, optional
        Number of the new PFTs. The default value is 1. 
    
    Returns
    -------
    None
    """
    global NP, C, M, R, L, B0ic, spp
    
    for inew in np.arange(nnew):
        NP +=1
        i = rng.integers(0,high=NP,size=1)
        c = np.round(trait_linear(rng, t_max=3, t_min=0.01, size=nnew, sigma=sigma), 3)
        m = np.round(trait_linear(rng, t_max=0.3, t_min=0.01, size=nnew, sigma=sigma), 3)
        r = np.round(rng.uniform(.001,1,size=nnew), 3)
        l = np.round(flammability(rng, frt_min=1, frt_max=1000, size=nnew), 3)
        logging.info(f'Invasive species traits: i={i+1}, C={c},  M={m}, R={r}, L={l}')
        
        C = np.insert(C,i,c)
        M = np.insert(M,i,m)
        R = np.insert(R,i,r)
        L = np.insert(L,i,l)
        
        spp = np.insert(spp,i,1)
        # B0ic = np.insert(B0ic, i, 0.00, axis=1)


def rand_linear_community(rng, nnew, sigma=0.7):
    """
    Create a plant community with random linear traits
    
    Parameters
    ----------
    rng: random generator
    nnew : int or tuple of ints, optional
        Number of the new PFTs. The default value is 1. 
    
    Returns
    -------
    None
    """
    global NP, C, M, R, L, B0ic, spp
    
    c, m = 0., 0.
    while np.any(c<=m):
        c = np.round(trait_linear(rng, t_max=100, t_min=0.01, size=nnew, sigma=sigma), 3)
        m = np.round(trait_linear(rng, t_max=10, t_min=0.01, size=nnew, sigma=sigma), 3)
    r = np.round(rng.uniform(.001,1,size=nnew), 3)
    l = np.round(flammability(rng, frt_min=2, frt_max=500, size=nnew), 3)
    
    C = c
    M = m
    R = r
    L = l
    NP = nnew
    spp = np.zeros(NP)
    initial_conditions(nnew)

    return C,M,R,L
    
#-------------------------------------------------------------------

def trait_exponential(rng, t_max=1, t_min=0.01, size=1, eta=3):
    """ 
    
    returns an array of colonization rate traits
        
    Parameters
    ----------
    rng: random generator
    t_max : float, optional
        The default value is 1
    t_min : float, optional
        The default value is 0.01
    size : int or tuple of ints, optional
        Size of the desired array. Usually correspond to the size of the pft community. The default value is 1.
        
    Returns
    -------
    T : ndarray rounded to the 3rd decimal value
    """
    N = size
    # exponential correlation
    ii = np.arange(0,N)+1
    alfa = (np.log10(t_min)*N - np.log10(t_max)) / (N-1)
    beta = (np.log10(t_max) - np.log10(t_min)) / (N-1)
    # print(f"N={N}, C(i) = 10 ^ ({alfa} + {beta} * i)")
    # noise
    T = 10 ** (alfa + beta*ii)
    
    if eta!=0:
        # new function for the noise
        k = (N-1) / 2**eta
        gamma = T*(10**(beta*k)-1)
        for i in range(0,N):
            noise = rng.normal(0, gamma[i])
            while (T[i]+noise)<0.001 or (T[i]+noise)>t_max:
                noise = rng.normal(0, gamma[i])
            T[i] += noise
    return T

def rand_exponential_invasive(rng, nnew=1, eta=3):
    """
    Insert nnew new PFTs with random traits from Random-C-Exponential function into an exixting community.
    Hyerarchical position i is random as well from a uniform distribution.
    
    Parameters
    ----------
    nnew : int or tuple of ints, optional
        Number of the new PFTs. The default value is 1. 
    
    Returns
    -------
    None
    """
    global NP, C, M, R, L, B0ic, spp
    
    for inew in np.arange(nnew):
        NP +=1
        i = rng.integers(0,high=NP,size=1)
        c = np.round(trait_exponential(rng, t_max=3, t_min=0.01, size=NP, eta=eta), 3)[i]
        m = np.round(trait_exponential(rng, t_max=0.3, t_min=0.01, size=NP, eta=eta), 3)[i]
        r = np.round(rng.uniform(.001,1,size=1), 3)
        l = np.round(flammability(rng, frt_min=2, frt_max=500, size=nnew), 3)
        logging.info(f'Invasive species traits: i={i+1}, C={c},  M={m}, R={r}, L={l}')
        
        C = np.insert(C,i,c)
        M = np.insert(M,i,m)
        R = np.insert(R,i,r)
        L = np.insert(L,i,l)
        
        spp = np.insert(spp,i,1)
        # B0ic = np.insert(B0ic, i, 0.00, axis=1)

def rand_exponential_community(rng, nnew, noise=3):
    """
    Create a plant community with random exponential traits
    
    Parameters
    ----------
    rng: random generator
    nnew : int or tuple of ints, optional
        Number of the new PFTs. The default value is 1. 
    
    Returns
    -------
    None
    """
    global NP, C, M, R, L#, B0ic, spp
    
    c, m = 0., 0.
    while np.any(c<=m):
        c = np.round(trait_exponential(rng, t_max=100, t_min=0.01, size=nnew, eta=noise), 3)
        m = np.round(trait_exponential(rng, t_max=10, t_min=0.01, size=nnew, eta=noise), 3)
    r = np.round(rng.uniform(.001,1,size=nnew), 3)
    l = np.round(flammability(rng, frt_min=1, frt_max=1000, size=nnew), 3)
    
    # NP = nnew
    # spp = np.zeros(NP)
    C = c
    M = m
    R = r
    L = l
    # initial_conditions(nnew)

    return C,M,R,L

#--------------- (2.3) ANALYSES FUNCTIONS ---------------

def equil_til():
    """
    Computes the equilibrium vegetation cover for a system with no fire (Tilman 1994)
    
    Returns
    -------
    beq : ndarray
    """
    # PARAMS FOR IMPERFECT HIERARCHY
    A = np.ones(NP)
    A[0] = 0.0
    # print('Portion occupied at equilibrium (Tilman):')
    # print('i', 'b_eq')
    beq = np.zeros(NP)
    for ii in range(NP):
        beq[ii] = 1 - M[ii]/C[ii] - A[ii]*np.sum(beq[0:ii]*(1+C[0:ii]/C[ii]))
        if beq[ii] < 0:
            beq[ii] = 0.0
        # cond = C[ii] - M[ii] > 0
        # print(ii+1, beq[ii])
    # print('occupied space at equilibrium:', np.sum(beq),'\n')
    return beq
    
def coex_fire(Tfire):
    Tp = - np.log(R) / (C - M)
    cond = Tfire > Tp
    # print(cond)
    return cond

# def richness(b):
#     """returns the Species Richness Index for the array b"""
#     pos = (b > 0.00001)
# #     pos = (b > 0.01)
#     sr = np.sum(pos)
#     return sr

# def evenness(b):
#     """returns the Pielou's Evenness Index for the array b"""
#     p = b / np.sum(b) #relative abundance
#     pos = (p > 0.)
#     h = -np.sum(np.log(p, where=pos)*p) # shannon entropy
#     sr = richness(b) # species richness
#     h_max = np.log(sr) #maximum shannon entropy
#     if h_max > 0:
#         pe = h / h_max
#     else:
#         # pe = np.nan
#         pe = 0.0
#     return pe

# def simpson(b):
#     """returns the Inverse Simpson Index for the array b"""
#     p = b / np.sum(b)
#     p2 = np.power(p,2)
#     isi = 1 / np.sum(p2)
#     return isi

def richness(b, bmin=1e-05):
    """
    Computes the Species Richness Index for one or more input communities
    
    Parameters
    -------
    b : array_like
        vegetation cover of the plants of the communities
    bmin : float, optional
        minimum vegetation cover value to consider a plant present in the community. The default value is 1e-05
    
    Returns
    -------
    sr : int or 1-D ndarray
        number of species coexisting in the input communities
    """
    
    nd = np.ndim(b)
    if nd==1:
        sr = np.sum(b>bmin)
    elif nd==2:
        sr = np.sum(b>bmin, axis=1)
    else:
        print('TROPPE DIMENSIONI, restituirò un NAN!')
        sr = np.nan
    return sr

def evenness(b, nnp):
    """
    Computes the Pielou's Evenness Index for one or more input communities
    
    Parameters
    -------
    b : 1-D or 2-D array_like
        vegetation cover of the plants of the communities
    nnp : int
        maximum number of species in the community
    
    Returns
    -------
    pe : int or 1-D ndarray
        species evenness of the input communities
    """
    
    nd = np.ndim(b)
    if nd==1:
        p = b / np.sum(b) #relative abundance
        pos = (p > 0.)
        h = -np.sum(np.log(p, where=pos)*p) # shannon entropy
        sr = richness(b) # species richness
        h_max = np.log(sr) # maximum shannon entropy
        if h_max > 0:
            pe = h / h_max
        else:
            # pe = np.nan
            pe = 0.0
    elif nd==2:
        p = b/np.repeat(np.expand_dims(np.sum(b, axis=1),1), nnp, axis=1) # relative abundance
        pos = (p > 0.)
        h = -np.sum(np.log(p, where=pos)*p, axis=1) # shannon entropy
        sr = richness(b) # species richness
        h_max = np.log(sr) # maximum shannon entropy
        pe = np.where(h_max>0, h/h_max, 0.)
    else:
        print('TROPPE DIMENSIONI, restituirò un NAN!')
        pe = np.nan
    return pe


def simpson(b, nnp):
    """
    Computes the Inverse Simpson Index for one or more given communities
    
    Parameters
    -------
    b : 1-D or 2-D array_like
        vegetation cover of the plants of the communities
    nnp : int
        maximum number of species in the community
    
    Returns
    -------
    isi : int or 1-D ndarray
        Inverse Simpson Index of the input communities
    """
    
    nd = np.ndim(b)
    if nd==1:
        p = b / np.sum(b)
        p2 = np.power(p,2)
        isi = 1 / np.sum(p2)
    elif nd==2:
        p = b/np.repeat(np.expand_dims(np.sum(b, axis=1),1), nnp, axis=1) #relative abundance
        p2 = np.power(p,2)
        isi = 1 / np.sum(p2, axis=1)
    else:
        print('TROPPE DIMENSIONI, restituirò un NAN!')
        isi = np.nan
    return isi

# Funzioni da controllare!

def coeff_var(trait, bave):
    nd = np.ndim(bave)
    if nd==1:
        cv = np.std(trait, where=(bave>0)) / np.mean(trait, where=(bave>0))
    elif nd==2:
        cv = np.std(trait, where=(bave>0), axis=1) / np.mean(trait, where=(bave>0), axis=1)
    else:
        print('TROPPE DIMENSIONI, restituirò un NAN!')
        cv = np.nan
    return cv
        

def index_disp(trait, bave):
    nd = np.ndim(bave)
    if nd==1:
        dc = np.power(np.std(trait, where=(bave>0)), 2) / np.mean(trait, where=(bave>0))
    elif nd==2:
        dc = np.power(np.std(trait, where=(bave>0), axis=1), 2) / np.mean(trait, where=(bave>0), axis=1)
    else:
        print('TROPPE DIMENSIONI, restituirò un NAN!')
        dc = np.nan
    return dc
    
def BC_distance(arr1, arr2):
    nd = np.ndim(arr1)
    if nd==1:
        bc = 1 - 2*np.sum(np.minimum(arr1,arr2))/np.sum(arr1+arr2)
    elif nd==2:
        bc = 1 - 2*np.sum(np.minimum(arr1,arr2), axis=1)/np.sum(arr1+arr2, axis=1)
    else:
        print('TROPPE DIMENSIONI, restituirò un NAN!')
        bc = np.nan
    return bc

#--------------- (2.3) I/O FUNCTION ---------------

def eco_info():
    logging.info("Species traits:")
    inv = spp==1
    logging.info("Competition [i] - Colonization [C] - Mortality [M] - Fire Response [R] - Flammability [L] - Alien[y/n]")
    for i in range(NP):
        logging.info(f'{i+1}\t{C[i]}\t{M[i]}\t{R[i]}\t{L[i]}\t{inv[i]}')

def eco_info_file(f_info_str):
    global C, M
    with open(f_info_str,'w') as filei:
        np.savetxt(filei, list(zip(C, M, R, L, spp)), fmt='%1.5f', delimiter="\t")

def set_traits_inv(ifile):
    global C, M, R, L, NP, spp
    
    new_community()

    traits = np.loadtxt(ifile)
    C = traits[:,0]
    M = traits[:,1]
    R = traits[:,2]
    L = traits[:,3]
    spp = traits[:,4]
    NP = len(traits[:,0])
    # print(C,M,R,L,spp,NP)
    
    derivs.recompile()
    rk4.recompile()
    fireocc.recompile()
    dyn.recompile()
    
    return

def set_traits(ifile):
    global C, M, R, L, NP
    
    new_community()

    traits = np.loadtxt(ifile)
    C = traits[:,0]
    M = traits[:,1]
    R = traits[:,2]
    L = traits[:,3]
    NP = len(traits[:,0])
    
    derivs.recompile()
    rk4.recompile()
    fireocc.recompile()
    dyn.recompile()
    
    return
    
def get_traits(ifile):
    set_traits(ifile)
    c, m, r, l = C, M, R, L
    return c, m, r, l

#--------------- (2.4) FIGURE FUNCTIONS ---------------

def set_color_tab(N):

    if N<=10:
        cmap = plt.get_cmap('tab10', 10)
        my_tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    elif N<=20:
        cmap = plt.get_cmap('tab20', 20)
        my_tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]

    else:
        nc=int(N/10)
        mytab00 = []

        i=1
        cmap = plt.get_cmap('Blues_r', i+nc+1)  # Blue
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('Oranges_r', i+nc+1)  # Orange
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('Greens_r', i+nc+1)  # Green
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('Reds_r', i+nc+1)  # Red
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('PRGn', i+nc*2+1)  # Purple
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('pink', i+nc+1)  # Brown
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=0
        cmap = plt.get_cmap('PiYG', i+nc*2+1)  # Pink
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=1
        cmap = plt.get_cmap('Greys_r', i+nc+1)  # Grey
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=10
        cmap = plt.get_cmap('gist_stern', i+nc+1)  # Olive
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        i=12
        cmap = plt.get_cmap('ocean', i+nc+1)  # Cyan
        tab = [colors.rgb2hex(cmap(i)[:3]) for i in range(cmap.N)]
        mytab00 = mytab00+tab[i:i+nc]

        my_tab = mytab00

    return my_tab

#------------------------------ (3) MAIN ------------------------------

if __name__ == "__main__":
    rng = default_rng()
    med_community()
    # rand_exponential_pft(rng,2,3)
    rand_exponential_invasive(rng, 1, 3)
    initial_conditions(NP)