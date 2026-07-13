import numpy as np

#material parameters for TBG: Bistritzer and Macdonald, PNAS 108, 12233 (2011)
params = {
            "a" : 0.246, #nm
            "t_intra" : 0.52, #eV
            "t_inter" : 0.08, #eV
            "nu" : 0.19 #poisson ratio, from Nano Research 8, 1847 (2015)
            }

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


def H_BM(k, θ, cutoff, a=params["a"], t_intra=params["t_intra"], t_inter=params["t_inter"], eps=0, strain_dir=0, nu=params["nu"], displacement=0):

    ΔK1, ΔK2, G1, G2 = geometry(θ, a=a, eps=eps, strain_dir=strain_dir, nu=nu)

    T0 = np.array([[1,1],[1,1]],dtype=complex)
    T1 = np.array([[np.exp(-1j*2*np.pi/3),1],[np.exp(1j*2*np.pi/3),np.exp(-1j*2*np.pi/3)]],dtype=complex)
    T2 = np.array([[np.exp(1j*2*np.pi/3),1],[np.exp(-1j*2*np.pi/3),np.exp(1j*2*np.pi/3)]],dtype=complex)

    twister = np.array([[np.exp(1j*θ/4),0],[0,np.exp(-1j*θ/4)]]) #e^{i(theta/2)tau_z}
    tau_x = np.array([[0,1],[1,0]])
    tau_y = np.array([[0,-1j],[1j,0]])
    tau_x2 = twister@tau_x@twister.conj() #top
    tau_y2 = twister@tau_y@twister.conj()
    tau_x1 = twister.conj()@tau_x@twister #bottom
    tau_y1 = twister.conj()@tau_y@twister

    #dimensions
    nx = 2*cutoff+1
    dim = 2*nx**2

    #setup matrices
    H_tot = np.zeros((2*dim,2*dim),dtype=complex)
    H_11 = np.zeros((dim,dim),dtype=complex)
    H_22 = np.zeros((dim,dim),dtype=complex)
    H_21 = np.zeros((dim,dim),dtype=complex)

    #construct Dirac terms
    for i in range(0,dim//2,1):
        n1 = i//nx-cutoff
        n2 = i%nx-cutoff
        H_11[2*i:2*(i+1),2*i:2*(i+1)] = tau_x1*(k-ΔK1+n1*G1+n2*G2)[0] + tau_y1*(k-ΔK1+n1*G1+n2*G2)[1]
        H_22[2*i:2*(i+1),2*i:2*(i+1)] = tau_x2*(k-ΔK2+n1*G1+n2*G2)[0] + tau_y2*(k-ΔK2+n1*G1+n2*G2)[1]

    #construct tunneling terms
    def idx(n1,n2): #convert between 2d lattice of possible momentum scatterings to a 1D representation (columns/rows of a matrix)
        return 2*((n1 + cutoff)*nx + (n2 + cutoff))
    
    for n1 in range(-cutoff, cutoff+1):
        for n2 in range(-cutoff, cutoff+1):
            H_21[idx(n1,n2):idx(n1,n2)+2, idx(n1,n2):idx(n1,n2)+2] += T0 #(0,0) 
            if(n2<cutoff):
                H_21[idx(n1,n2):idx(n1,n2)+2, idx(n1,n2+1):idx(n1,n2+1)+2] += T1 #(0,1)
            if(n1<cutoff):
                H_21[idx(n1,n2):idx(n1,n2)+2, idx(n1+1,n2):idx(n1+1,n2)+2] += T2 #(1,0)

    #put them together
    H_tot[0:dim,0:dim] = t_intra*H_11 + displacement/2*np.eye(dim)
    H_tot[dim:2*dim,dim:2*dim] = t_intra*H_22 - displacement/2*np.eye(dim)
    H_tot[0:dim,dim:2*dim] = t_inter*H_21.conj().T
    H_tot[dim:2*dim,0:dim] = t_inter*H_21

    return H_tot


def H_valley(k, θ, cutoff, a=params["a"], t_intra=params["t_intra"], t_inter=params["t_inter"], valley=+1, eps=0, strain_dir=0, nu=params["nu"], displacement=0):
    """
    valley = +1: K valley
    valley = -1: K' valley, using time reversal H_K (k) = H_K^*(-k)
    """

    if valley == +1:
        return H_BM(k, θ, cutoff, a=a, t_intra=t_intra, t_inter=t_inter, eps=eps, strain_dir=strain_dir, nu=nu, displacement=displacement)

    elif valley == -1:
        H = H_BM(-k, θ, cutoff, a=a, t_intra=t_intra, t_inter=t_inter, eps=eps, strain_dir=strain_dir, nu=nu, displacement=displacement)
        return H.conj() #removed Delta_K1, Delta_K2

    else:
        raise ValueError("valley must be +1 for K or -1 for K'")
		
def j_BM(k, θ, l, dir, cutoff, a=params["a"], t_intra=params["t_intra"], t_inter=params["t_inter"], eps=0, strain_dir=0, nu=params["nu"], displacement=0, frame="lab"):

    twister = np.array([[np.exp(1j*θ/4),0],[0,np.exp(-1j*θ/4)]])
    tau_x = np.array([[0,1],[1,0]])
    tau_y = np.array([[0,-1j],[1j,0]])
    tau_x2 = twister@tau_x@twister.conj()
    tau_y2 = twister@tau_y@twister.conj()
    tau_x1 = twister.conj()@tau_x@twister
    tau_y1 = twister.conj()@tau_y@twister

    #dimensions
    nx = 2*cutoff+1
    dim = 2*nx**2

    #setup matrices
    j_tot = np.zeros((2*dim,2*dim),dtype=complex)
    j_11 = np.zeros((dim,dim),dtype=complex)
    j_22 = np.zeros((dim,dim),dtype=complex)

    #construct Dirac terms

    if(frame=="lab"):
        for i in range(0,dim//2,1):

            n1 = i//nx-cutoff
            n2 = i%nx-cutoff
            if(dir=="x"):
                if(l=="1"):
                    j_11[2*i:2*(i+1),2*i:2*(i+1)] = tau_x1
                if(l=="2"):
                    j_22[2*i:2*(i+1),2*i:2*(i+1)] = tau_x2
            if(dir=="y"):
                if(l=="1"):
                    j_11[2*i:2*(i+1),2*i:2*(i+1)] = tau_y1
                if(l=="2"):
                    j_22[2*i:2*(i+1),2*i:2*(i+1)] = tau_y2
    
    elif(frame=="bm" or frame=="BM"):
        for i in range(0,dim//2,1):

            n1 = i//nx-cutoff
            n2 = i%nx-cutoff
            if(dir=="x"):
                if(l=="1"):
                    j_11[2*i:2*(i+1),2*i:2*(i+1)] = tau_x
                if(l=="2"):
                    j_22[2*i:2*(i+1),2*i:2*(i+1)] = tau_x
            if(dir=="y"):
                if(l=="1"):
                    j_11[2*i:2*(i+1),2*i:2*(i+1)] = tau_y
                if(l=="2"):
                    j_22[2*i:2*(i+1),2*i:2*(i+1)] = tau_y
    else:
        raise ValueError("frame ",frame," is not lab or BM")

    #put them together
    j_tot[0:dim,0:dim] = t_intra * j_11
    j_tot[dim:2*dim,dim:2*dim] = t_intra * j_22
    return j_tot


def j_valley(k, θ, l, dir, cutoff, a=params["a"], t_intra=params["t_intra"], t_inter=params["t_inter"], valley=+1, eps=0, strain_dir=0, nu=params["nu"], displacement=0, frame="lab"):
    """
    valley = +1: K
    valley = -1: K'
    """

    if valley == +1:
        return j_BM(k, θ, l, dir, cutoff, a=a, t_intra=t_intra, t_inter=t_inter, eps=eps, strain_dir=strain_dir, nu=nu, displacement=displacement, frame=frame)
    elif valley == -1:
        return -j_BM(-k, θ, l, dir, cutoff, a=a, t_intra=t_intra, t_inter=t_inter, eps=eps, strain_dir=strain_dir, nu=nu, displacement=displacement, frame=frame).conj()
    else:
        raise ValueError("valley must be +1 for K or -1 for K'")
