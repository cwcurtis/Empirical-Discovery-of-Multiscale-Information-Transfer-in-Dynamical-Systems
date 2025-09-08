import numpy as np
from tqdm import tqdm

def rk4(lhs, dt, function):
    k1 = dt * function(lhs)
    k2 = dt * function(lhs + k1 / 2.0)
    k3 = dt * function(lhs + k2 / 2.0)
    k4 = dt * function(lhs + k3)
    rhs = lhs + 1.0 / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)
    return rhs

def trajectory(func, ic, start, stop, dt):
    num_dims = np.size(ic)
    num_steps = int((stop - start)/dt)
    traj = np.zeros((num_steps, num_dims))
    traj[0, :] = ic
    for ii in range(1, num_steps):
        traj[ii, :] = rk4(traj[ii-1, :], dt, func)
    return traj

def lorenz63(lhs, rho=28.0, sigma=10.0, beta=8./3.):
    """ Lorenz63 example:
    ODE =>
    dx1/dt = sigma*(x2 - x1)
    dx2/dt = x1*(rho - x3) - x2
    dx3/dt = x1*x2 - beta*x3
    """
    rhs = np.zeros(3)
    rhs[0] = sigma*(lhs[1] - lhs[0])
    rhs[1] = lhs[0]*(rho - lhs[2]) - lhs[1]
    rhs[2] = lhs[0]*lhs[1] - beta*lhs[2]
    return rhs

def lorenz96(lhs, Fval=8.):    
    rhs = -lhs + Fval + ( np.roll(lhs,-1) - np.roll(lhs,2) ) * np.roll(lhs,1)
    return rhs

def rossler(lhs, alpha=0.2, beta=0.2, gamma=5.7):
    """ Rossler system:
    ODE =>
    dx1/dt = -x2 - x3
    dx2/dt = x1 + alpha*x2
    dx3/dt = beta + x3*(x1 - gamma)
    """
    rhs = np.zeros(3)
    rhs[0] = 6.*(-lhs[1] - lhs[2])
    rhs[1] = 6.*(lhs[0] + alpha*lhs[1])
    rhs[2] = 6.*(beta + lhs[2] * (lhs[0] - gamma))
    return rhs

def coupled_lorenz_rossler(lhs,  dt, jj, coupling=5., forcing=0.):
    """ Lorenz63 example:
    ODE =>
    dx1/dt = sigma*(x2 - x1)
    dx2/dt = x1*(rho - x3) - x2 + C*x4**2
    dx3/dt = x1*x2 - beta*x3
    """
    rho=28.0
    sigma=10.0
    beta=8./3.

    fhlf = .5*(forcing[jj]+forcing[jj+1])

    # Lorenz 63
    k1 = dt * lorenz63(lhs)
    k1[1] += dt * coupling * forcing[jj]**2.

    k2 = dt * lorenz63(lhs + k1/2.)
    k2[1] += dt * coupling * fhlf**2.

    k3 = dt * lorenz63(lhs + k2/2.)
    k3[1] += dt * coupling * fhlf**2.

    k4 = dt * lorenz63(lhs + k3)
    k4[1] += dt * coupling * forcing[jj+1]**2.

    rhs = lhs + 1.0 / 6.0 * (k1 + 2.0 * (k2 + k3) + k4)

    return rhs

def generate_coupled(x1min, x1max, x2min, x2max, 
                        x3min, x3max, num_ic=10, dt=0.01, tf=150.0, coupling=.1, delay=1, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    icond3 = np.random.uniform(x3min, x3max, num_ic)
    icond4 = np.random.uniform(x1min, x1max, num_ic)
    icond5 = np.random.uniform(x2min, x2max, num_ic)
    icond6 = np.random.uniform(x3min, x3max, num_ic)
    
    data_mat = np.zeros((num_ic, 6, num_steps+1), dtype=np.float64)
    rossler_solver = lambda x: rossler(x, alpha=.2, beta=.2, gamma=5.7)
    
    for ii in tqdm(range(num_ic), 
                   desc="Generating Rossler system data...", ncols=100):
        data_mat[ii, 3:, 0] = np.array([icond4[ii], icond5[ii], icond6[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, 3:, jj+1] = rk4(data_mat[ii, 3:, jj], dt, rossler_solver)
    
    for ii in tqdm(range(num_ic), 
                   desc="Generating Coupled Lorenz-Rossler system data...", ncols=100):
        data_mat[ii, :3, 0] = np.array([icond1[ii], icond2[ii], icond3[ii]], dtype=np.float64)
        forcing_vec = np.roll(data_mat[ii, 4, :], delay)    
        for jj in range(num_steps):            
            data_mat[ii, :3, jj+1] = coupled_lorenz_rossler(data_mat[ii, :3, jj], dt, jj, coupling, forcing_vec)
        
    return data_mat

def generate_rossler(x1min, x1max, x2min, x2max, 
                        x3min, x3max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    icond3 = np.random.uniform(x3min, x3max, num_ic)
    data_mat = np.zeros((num_ic, 3, num_steps+1), dtype=np.float64)
    for ii in tqdm(range(num_ic), 
                   desc="Generating Rossler system data...", ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii], icond3[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, lorenz63)
    return data_mat

def generate_lorenz63(x1min, x1max, x2min, x2max, 
                      x3min, x3max, num_ic=10000, dt=0.01, tf=1.0, seed=None):
    np.random.seed(seed=seed)
    num_steps = int(tf / dt)
    num_ic = int(num_ic)
    icond1 = np.random.uniform(x1min, x1max, num_ic)
    icond2 = np.random.uniform(x2min, x2max, num_ic)
    icond3 = np.random.uniform(x3min, x3max, num_ic)
    data_mat = np.zeros((num_ic, 3, num_steps+1), dtype=np.float64)
    for ii in tqdm(range(num_ic), 
                   desc='Generating Lorenz63 system data...', ncols=100):
        data_mat[ii, :, 0] = np.array([icond1[ii], icond2[ii], icond3[ii]], dtype=np.float64)
        for jj in range(num_steps):
            data_mat[ii, :, jj+1] = rk4(data_mat[ii, :, jj], dt, lorenz63)
    return data_mat

def generate_lorenz96(Fval, num_ic=15000, dim=8, dt=0.05, tf=20.0, seed=None):
    np.random.seed(seed=seed)
    nsteps = int(tf / dt)
    num_ic = int(num_ic)
    my_lorenz = lambda lhs: lorenz96(lhs, Fval)
    
    iconds = np.zeros((num_ic, dim), dtype=np.float64)
    uhats = np.ones((num_ic, dim), dtype=np.complex128)
    phases = np.exp(1j * 2*np.pi*np.random.rand(num_ic, int(dim/2)))
        
    for ll in range(num_ic):        
        uhats[ll, 1:] = np.squeeze(np.concatenate( (phases[ll, :], np.flip(np.conj(phases[ll, :-1]))) ))
        
    iconds = np.real(np.fft.ifft(uhats, axis=1)*dim)

    data_mat = np.zeros((num_ic, dim, nsteps + 1), dtype=np.float64)
    
    for ii in tqdm(range(num_ic), desc='Generating Lorenz96 system data...', ncols=100):
        data_mat[ii, :, 0] = iconds[ii, :]
        for jj in range(nsteps):
            data_mat[ii, :, jj + 1] = rk4(data_mat[ii, :, jj], dt, my_lorenz)
    return data_mat