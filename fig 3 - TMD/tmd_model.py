import numpy as np

#material parameters for MoTe2 : Wu, et all, PRL 122, 086402 (2019)
params = {
            "a" : 0.3472, #nm
            "ms" : 0.62, #m_e
            "V" : 8, #meV
            "psi" : -89.6*np.pi/180,
            "t_inter" : -8.5, #meV
            "nu" : 0.25 #poisson ratio, from Nano Lett 19, 761 (2019)
            }
#material parameters for WSe2 : Devakul et al, Nat Commun 12, 6730 (2021)
# params = {
#             "a" : 0.3317, #nm
#             "ms" : 0.43, #m_e
#             "V" : 9, #meV
#             "psi" : 128*np.pi/180,
#             "t_inter" : 18, #meV
#             "nu" : 0.19 #poisson ratio, from Appl. Phys. Lett. 104, 203110 (2014)
#             }

def R(phi):
    return np.array([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi),  np.cos(phi)]])


def strain_matrix(eps=0, strain_dir=0, nu=params["nu"]):
    """
    Real-space deformation matrix for uniaxial tensile strain eps at angle strain_dir, with Poisson ratio nu.
    """
    d  = np.array([np.cos(strain_dir),np.sin(strain_dir)]) #the strain direction
    dp = np.array([np.cos(strain_dir+np.pi/2),np.sin(strain_dir+np.pi/2)]) #the perpendicular direction

    u = eps * np.outer(d,d) - nu * eps * np.outer(dp,dp)
    return np.identity(2) + u


def geometry(theta, a=params["a"], nu=params["nu"], eps=0, strain_dir=0):
    """
    convention:
      layer 1 = bottom, strained +epsilon/2, rotated +theta/2
      layer 2 = top, strained -epsilon/2, rotated -theta/2
    Returns ΔK1, ΔK2, G1, G2.
    """

    # bottom: rotated, then strained
    Rb = R(+theta/2) #R inverse transpose is R
    Sb = strain_matrix(eps=+eps/2, strain_dir=strain_dir, nu=nu)
    Sb_inv = np.linalg.inv(Sb).T

    # top: rotated, then strained
    Rt = R(-theta/2) #R inverse transpose is R
    St = strain_matrix(eps=-eps/2, strain_dir=strain_dir, nu=nu)
    St_inv = np.linalg.inv(St).T

    # layer-resolved reciprocal vectors
    Kmono = 4*np.pi/(3*a)
    b1_0 = Kmono * np.array([3/2, -np.sqrt(3)/2])
    b2_0 = Kmono * np.array([3/2,  np.sqrt(3)/2])

    b1_b = Sb_inv @ Rb @ b1_0
    b2_b = Sb_inv @ Rb @ b2_0

    b1_t = St_inv @ Rt @ b1_0
    b2_t = St_inv @ Rt @ b2_0

    # moire reciprocal primitive vectors
    G1 = b1_b - b1_t
    G2 = b2_b - b2_t

    # layer valley locations.
    K0 = (b1_0 + b2_0) / 3 #valley corner
    K_b = Sb_inv @ Rb @ K0
    K_t = St_inv @ Rt @ K0

    #spacing between K points
    qK = K_b - K_t

    ΔK1 = +0.5*qK
    ΔK2 = -0.5*qK

    return [ΔK1, ΔK2, G1, G2]


def H_TMD(k,θ, cutoff, a=params["a"], ms=params["ms"], V=params["V"], psi=params["psi"], t_inter=params["t_inter"], nu=params["nu"], eps=0.0, strain_dir=0, displacement=0):
    k = np.array(k)
    h2_2m = 38.0998211/ms #ms in units of m_e; hbar^2/2m_e nm^2 = 38.1 meV

    ΔK1, ΔK2, G1, G2 = geometry(θ, a=a, eps=eps, strain_dir=strain_dir, nu=nu)

    T0 = t_inter
    T1 = t_inter
    T2 = t_inter

    #dimensions
    nx = 2*cutoff+1
    dim = 1*nx**2 #1 instead of 2!

    #setup matrices
    H_tot = np.zeros((2*dim,2*dim),dtype=complex)
    H_11 = np.zeros((dim,dim),dtype=complex)
    H_22 = np.zeros((dim,dim),dtype=complex)
    H_12 = np.zeros((dim,dim),dtype=complex)
    V_11 = np.zeros((dim,dim),dtype=complex)
    V_22 = np.zeros((dim,dim),dtype=complex)

    #p^2/2m terms
    for i in range(0,dim,1): 
        n1 = i//nx-cutoff
        n2 = i%nx-cutoff
        H_11[i,i] = -h2_2m*np.linalg.norm(k-ΔK1+n1*G1+n2*G2)**2 #bottom
        H_22[i,i] = -h2_2m*np.linalg.norm(k-ΔK2+n1*G1+n2*G2)**2 #top

    #interlayer potential terms
    def idx(n1,n2): #convert between 2d lattice of possible momentum scatterings to a 1D representation (columns/rows of a matrix)
        return (n1 + cutoff)*nx + (n2 + cutoff)
    
    Ub = V*np.exp(1j*psi) #bottom
    Ut = V*np.exp(-1j*psi)#top

    for n1 in range(-cutoff,cutoff+1,1):
        for n2 in range(-cutoff,cutoff+1,1):
            i = idx(n1, n2)

            if(n2<cutoff):
                j = idx(n1,n2+1) # (0,+1) #convention of these is flipped from BM (0,-1)
                V_11[i,j] += Ub
                V_22[i,j] += Ut

            if(n1>-cutoff):
                j = idx(n1-1,n2) #(-1,0)
                V_11[i,j] += Ub
                V_22[i,j] += Ut

            if(n1<cutoff and n2>-cutoff):
                j = idx(n1+1,n2-1) #(+1,-1)
                V_11[i,j] += Ub
                V_22[i,j] += Ut

    V_11 = V_11 + np.conj(V_11.T) #make hermitian
    V_22 = V_22 + np.conj(V_22.T) #make hermitian

    # interlayer tunneling terms
    for n1 in range(-cutoff,cutoff+1):
        for n2 in range(-cutoff, cutoff+1):
            H_12[idx(n1,n2), idx(n1,n2)] += T0 #(0,0) 
            if(n1>-cutoff):
                H_12[idx(n1,n2), idx(n1-1,n2)] += T1 #(-1,0)
            if(n2>-cutoff):
                H_12[idx(n1,n2), idx(n1,n2-1)] += T2 #(0,-1)

    #put them together
    H_tot[0:dim,0:dim] = H_11 + V_11 + displacement/2 * np.eye(dim)
    H_tot[dim:2*dim,dim:2*dim] = H_22 + V_22 - displacement/2 * np.eye(dim)
    H_tot[0:dim,dim:2*dim] = H_12
    H_tot[dim:2*dim,0:dim] = H_12.conj().T

    return H_tot


def H_valley(k,θ, cutoff, valley=+1, a=params["a"], ms=params["ms"], V=params["V"], psi=params["psi"], t_inter=params["t_inter"], nu=params["nu"], eps=0.0, strain_dir=0, displacement=0):
    """
    valley = +1: K valley
    valley = -1: K' valley with H_K'(k) = H_K^*(-k)
    """

    if(valley == +1):
        return H_TMD(k, θ, cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, nu=nu, eps=eps, strain_dir=strain_dir, displacement=displacement)

    elif(valley == -1):
        return H_TMD(-k, θ, cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, nu=nu, eps=eps, strain_dir=strain_dir, displacement=displacement).conj()

    else:
        raise ValueError("valley must be +1 for K or -1 for K'")
		

def j_TMD(k,θ,l,dir, cutoff, a=params["a"], ms=params["ms"], V=params["V"], psi=params["psi"], t_inter=params["t_inter"], nu=params["nu"], eps=0, strain_dir=0, displacement=0):
    """
    Current/velocity operator for the TMD continuum model. j = dH/dk_dir
    l = "1" or "2"
    dir = "x" or "y"
    """

    h2_2m = 38.0998211 / ms #ms in units of m_e; hbar^2/2m_e nm^2 = 38.1 meV

    ΔK1, ΔK2, G1, G2 = geometry(θ, a=a, nu=nu, eps=eps, strain_dir=strain_dir)

    nx = 2*cutoff + 1
    dim = nx**2

    j_tot = np.zeros((2*dim, 2*dim), dtype=complex)
    j_11 = np.zeros((dim, dim), dtype=complex)
    j_22 = np.zeros((dim, dim), dtype=complex)

    if(dir == "x"):
        d = 0
    elif(dir == "y"):
        d = 1
    else:
        raise ValueError("dir must be 'x' or 'y'")

    for i in range(dim):
        n1 = i // nx - cutoff
        n2 = i % nx - cutoff

        q1 = k - ΔK1 + n1*G1 + n2*G2
        q2 = k - ΔK2 + n1*G1 + n2*G2

        if(l == "1"):
            j_11[i, i] = -2 * h2_2m * q1[d]

        elif(l == "2"):
            j_22[i, i] = -2 * h2_2m * q2[d]

        else:
            raise ValueError("l must be '1' or '2'")

    j_tot[0:dim, 0:dim] = j_11
    j_tot[dim:2*dim, dim:2*dim] = j_22

    return j_tot


def j_valley(k,θ,l,dir, cutoff, valley=+1, a=params["a"], ms=params["ms"], V=params["V"], psi=params["psi"], t_inter=params["t_inter"], nu=params["nu"], eps=0.0, strain_dir=0, displacement=0):
    """
    valley = +1: K
    valley = -1: K', using H_K'(k) = H_K^*(-k)
    """

    if(valley == +1):
        return j_TMD(k, θ, l, dir, cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, nu=nu, eps=eps, strain_dir=strain_dir, displacement=displacement)

    elif(valley == -1):
        return -j_TMD(-k, θ, l, dir, cutoff, a=a, ms=ms, V=V, psi=psi, t_inter=t_inter, nu=nu, eps=eps, strain_dir=strain_dir, displacement=displacement).conj()
    
    else:
        raise ValueError("valley must be +1 for K or -1 for K'")