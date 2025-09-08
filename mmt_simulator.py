import numpy as np
import scipy.fft as fft
#from scipy.signal import resample

def nonlinearity(wn):
    Ntot = wn.size
    dxbwn = Ntot * fft.ifft(wn)
    return 1j * fft.fft( dxbwn * np.real(np.abs(dxbwn)**2.) )/Ntot # Computation of nonlinearity.  DE-FOCUSING CASE!!!!!!!!!

def rk4(wn, Dtot_hlf, Dtot_fll, dt, force_diss):
    k1 = dt * nonlinearity(wn)
    k2 = dt * nonlinearity((wn +.5*k1)*Dtot_hlf)
    k3 = dt * nonlinearity((wn*Dtot_hlf +.5*k2))
    k4 = dt * nonlinearity((wn*Dtot_fll + k3*Dtot_hlf))
    
    wnp1 = ((wn + k1/6.)*Dtot_fll + (k2/3. + k3/3.)*Dtot_hlf + k4/6.)*force_diss
    return wnp1

def average(wn, Dxa, fwidth):
    wnavg = np.zeros((wn.shape[0], wn.shape[1]), dtype=np.complex128)
    wave_filter = np.exp(-(fwidth*Dxa/(2.*np.pi))**2.)
    wnavg = np.diag(wave_filter) @ wn
    return wnavg

def transferfun(avg, flucsq):
    Ntot = avg.shape[0]
    avgphys = Ntot * fft.ifft(avg, axis=0)
    flucsqphys = Ntot * fft.ifft(flucsq, axis=0)
    return np.mean((np.abs(avgphys))**2. * flucsqphys, axis=0)/np.mean(np.abs(avgphys)**2., axis=0)

def multiscale_decomp(wn, Dxa, fwidths, tskip):
    Ntot = wn.shape[0]
    tvec = np.ones(tskip, dtype=np.complex128)/tskip    
    tfiltermat = np.zeros((wn.shape[1], wn.shape[1]-tskip+1), dtype=np.complex128)
    multiscale_sep = np.zeros((len(fwidths), wn.shape[1]-tskip+1))
    
    for jj in range(wn.shape[1]-tskip+1):
        tfiltermat[jj:jj+tskip, jj] = tvec
    wnred = wn @ tfiltermat

    for cnt, width in enumerate(fwidths):
        avg = average(wnred, Dxa, width)
        fluc = wnred - avg
        flucphys = Ntot * fft.ifft(fluc, axis=0)
        flucfreq = np.imag(fft.fft(flucphys**2., axis=0)/Ntot)
        flucfreqavg = average(flucfreq, Dxa, width)
        
        multiscale_sep[cnt, :] = transferfun(avg,flucfreqavg)
    return multiscale_sep

def mmt_solver(Nval, Tf, ep_val, dt, samp_indx_rate, inertialright):

    Nsteps = int(Tf/dt) # Number of time steps
    
    alpha = .5
    gamma = 2. 
        
    Dxa = np.abs(np.concatenate((np.arange(Nval+1),np.arange(-Nval+1,0,1)),0)) # note, we never need a naked Dx
    Dxalpha = Dxa**alpha   
    
    Dxinv = np.ones(2*Nval, dtype=np.complex128)
    Dxinv[1:] = 1./Dxa[1:]
            
    kforcelow = 6
    kforcehigh = 9

    kforcevals = ep_val**2. * np.ones(kforcehigh-kforcelow+1)
    force = np.zeros(2*Nval, dtype=np.complex128)
    force[kforcelow:kforcehigh+1] = kforcevals
    force[2*Nval-(kforcehigh+1):2*Nval-kforcelow] = kforcevals[::-1]
    
    dminus = 8
    dplus = 8
    klow = kforcelow-1
    kplus = int(Nval/2)
    numinus = ep_val**2.
    nuplus = ep_val**2.

    highfreqdamp = np.zeros(2*Nval, dtype=np.complex128)
    lowfreqdamp = np.zeros(2*Nval, dtype=np.complex128)
    highfreqdamp[kplus:-kplus] = nuplus * (Dxa[kplus:-kplus]/kplus)**dplus
    lowfreqdamp[:klow] = numinus * (klow*Dxinv[:klow])**dminus
    lowfreqdamp[-klow:] = numinus * (klow*Dxinv[-klow:])**dminus
    
    fdvec = force - highfreqdamp - lowfreqdamp
    force_diss = np.exp(dt * fdvec)
    
    Dtot_hlf = np.exp((-1j * Dxalpha) * dt/2.)
    Dtot_fll = np.exp((-1j * Dxalpha) * dt)    
        
    #samp_indx_rate = 400
    dts = samp_indx_rate * dt
    
    measured_time = inertialright/ep_val**2.
    shift_index = int(measured_time/dt)
    measure_index = int((Tf - measured_time)/dts)+1
    
    init_cond_index = int(100./dt)
    mid_cond_index = int(2000./dt)

    hmltnlinear = np.zeros(measure_index, dtype=np.float64)
    hmltnnonlinear = np.zeros(measure_index, dtype=np.float64)
    
    tseries = np.zeros((2*Nval, measure_index), dtype=np.complex128)
    tseries_short = np.zeros((2*Nval, int(init_cond_index*dt/dts)), dtype=np.complex128)
    tseries_mid = np.zeros((2*Nval, int((mid_cond_index-init_cond_index)*dt/dts)+1), dtype=np.complex128)
    
    tavg = np.zeros(2*Nval, dtype=np.float64)
    shrt_cnt = 0
    mid_cnt = 0
    cnt = 0
    
    wn = np.zeros(2*Nval,dtype=np.complex128) # Initialize the solution vector and associated parts of multi-step solver.
    wn[:] = ep_val * Dxinv**(alpha*gamma/2.) * (np.random.randn(2*Nval) + 1j * np.random.randn(2*Nval))# Fourier transform of initial condition
    
    wn0 = np.zeros(2*Nval, dtype=np.complex128)
    wn0[:] = wn
    
    for jj in range(1, Nsteps): # Run the time stepper i.e. how we get from time t_n = ndt to time t_(n+1)
        wn[:] = rk4(wn, Dtot_hlf, Dtot_fll, dt, force_diss)

        if jj%samp_indx_rate == 0:

            if jj <= init_cond_index:
                tseries_short[:, shrt_cnt] = wn[:]
                shrt_cnt += 1

            if jj >= init_cond_index and jj <= mid_cond_index:
                tseries_mid[:, mid_cnt] = wn[:]
                mid_cnt += 1

            if jj >= shift_index:
                action = np.abs(wn)**2.
                tavg += action
                hmltnlinear[cnt] = np.sum( np.abs(2*Nval*fft.ifft(np.sqrt(Dxalpha)*wn))**2. )
                hmltnnonlinear[cnt] = .5 * np.sum( np.abs(2*Nval*fft.ifft(wn))**4. )               
                tseries[:, cnt] = wn
                cnt += 1
                    
    #fwidths = [.95 * (1./inertialleft), .5 * (1./inertialleft), 2 * 1./inertialright, 1.05 * 1./inertialright]
    #tskip = int(10/dts)
    #multiscale_sep = multiscale_decomp(tseries, Dxa, fwidths, tskip)
    return [wn, wn0, hmltnlinear, hmltnnonlinear, tavg/cnt, tseries_short, tseries_mid, tseries]


