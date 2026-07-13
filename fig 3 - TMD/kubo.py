import numpy as np
from tmd_model import *

#Boltzmann's constant
kB = 8.617333262 * 10**-2 #meV/K

#adjoint function
def adj(v):
    return np.conjugate(np.transpose(v))

#transform from layer resolved conductivities to total, chiral, and counterflow conductivities
def reprocess(s): #H otimes sigma_0 for Hadamard gate H
    T = np.array([
        [1, 0,  1, 0],
        [0, 1,  0, 1],
        [-1, 0, 1, 0],
        [0, -1,  0,1]])
    return np.einsum("ai,ij w,bj->ab w", T, s, T)

#fermi function
def fermi(E,mu,T):
    x = (E-mu)/(kB*T)
    temp = np.zeros(np.shape(x))
    #the ones that would overflow
    temp[x>40] = 0.0
    temp[x<-40] = 1.0
    #the ones we will actually compute
    middle = (x>=-40) & (x<=40) #& gives set intersection
    temp[middle] = 1/(1+np.exp(x[middle]))
    return temp

#derivative of fermi function df(E)/dE
def dfermi(E,mu,T):
    x = (E-mu)/(kB*T)
    temp = np.zeros(np.shape(x))
    #the ones that would overflow
    middle = (x>=-40) & (x<=40) #& gives set intersection
    temp[middle] = -1/(2*kB*T*(1+np.cosh(x[middle])))
    return temp

def kubo_fast(k, θ, mu, T, ws, cutoff, valley=+1, a=params["a"], ms=params["ms"], V=params["V"], psi=params["psi"], t_inter=params["t_inter"], nu=params["nu"], eps=0.0, strain_dir=0, displacement=0):
    #diagonalize eigensystem
    vals, vecs = np.linalg.eigh(H_valley(k, θ, cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, valley=valley, eps=eps, strain_dir=strain_dir, nu=nu,displacement=displacement))
    
    #fermi functions and energy differences
    F = fermi(vals,mu,T)
    fermis = F[:,None] - F[None,:] #f(E_i) - f(E_j)
    dE = vals[None,:] - vals[:,None]   # E_j - E_i
    dfermis = dfermi(vals, mu, T)
    #select interband/intraband contributions
    mask = np.abs(dE) > 1e-10 #select energies that aren't the same, including degeneracies
    fermi_ratio = np.zeros(np.shape(dE))
    fermi_ratio[mask] = fermis[mask] / dE[mask]
    fermi_ratio[~mask] = (0*dfermis[None,:] - dfermis[:,None])[~mask]

    #current operators and matrix elements
    J_OPS = np.stack([
        j_valley(k, θ, "1", "x", cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, valley=valley, eps=eps, strain_dir=strain_dir, nu=nu, displacement=displacement),
        j_valley(k, θ, "1", "y", cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, valley=valley, eps=eps, strain_dir=strain_dir, nu=nu, displacement=displacement),
        j_valley(k, θ, "2", "x", cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, valley=valley, eps=eps, strain_dir=strain_dir, nu=nu, displacement=displacement),
        j_valley(k, θ, "2", "y", cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, valley=valley, eps=eps, strain_dir=strain_dir, nu=nu, displacement=displacement),
    ])
    #evaluate everything quickly using einsum
    ops = np.stack([vecs.conj().T @ J @ vecs for J in J_OPS])
    weights = np.einsum("ij,aij,bji->abij",fermi_ratio, ops, ops) #the numerator
    denom = 1 / (ws[:, None, None] - vals[None, None, :] + vals[None, :, None])
    return np.einsum("abij,wij->abw", weights, denom)