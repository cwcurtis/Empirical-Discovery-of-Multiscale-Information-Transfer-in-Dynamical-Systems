import numpy as np
import scipy.fft as fft
import ewt as ewt
import multi_breaks_otsu as mbo
import knn_mi_comp as mi 

def average(wn, kleft, kcut):
    wnavg = np.zeros((wn.shape[0], wn.shape[1]), dtype=np.complex128)
    wnavg[kleft:kcut, :] = wn[kleft:kcut, :]
    wnavg[-kcut:-kleft, :] = wn[-kcut:-kleft, :]
    return wnavg

def transferfun(avg, flucsq):
    Ntot = avg.shape[0]
    avgphys = Ntot * fft.ifft(avg, axis=0)
    flucsqphys = Ntot * fft.ifft(flucsq, axis=0)    
    return np.mean(( np.conj(avgphys) )**2. * flucsqphys, axis=0)

def window_time_average(wn, tskip):
    tvec = np.ones(tskip, dtype=np.complex128)/tskip        
    
    if wn.ndim > 1:    
        width = wn.shape[1]
    else:
        width = wn.size

    tfiltermat = np.zeros((width, width-tskip+1), dtype=np.complex128)            
    for jj in range(width-tskip+1):
        tfiltermat[jj:jj+tskip, jj] = tvec
    
    if wn.ndim > 1:
        wnred = wn @ tfiltermat
    else:
        wnred = np.squeeze(wn[np.newaxis, :] @ tfiltermat)
    
    return wnred

def multiscale_decomp(wn, inertialleft, inertialright, lends, rends, tskip):
    Ntot = wn.shape[0]    
    multiscale_sep = np.zeros((len(lends), wn.shape[1]-tskip+1), dtype=np.float64)
    
    for cnt in range(lends.size):
        avg = average(wn, lends[cnt], rends[cnt])
        fluc = wn - avg
        fluc_cut = average(fluc, inertialleft, inertialright)
        flucphys = Ntot * fft.ifft(fluc_cut, axis=0)
        flucfreq = fft.fft( flucphys**2., axis=0)/Ntot
        flucfreqavg = average(flucfreq, lends[cnt], rends[cnt])
        multiscale_sep[cnt, :] = np.imag(window_time_average(transferfun(avg, flucfreqavg), tskip))
        
    return multiscale_sep

def scale_sep_and_downsampling(tseries, dts, ep_val, avgscl, inertialleft, inertialright, lends, rends):
    tskip = int( (avgscl/ep_val**2.) / dts)
    tseries_red = window_time_average(tseries, tskip)
    multiscale_sep = multiscale_decomp(tseries_red, inertialleft, inertialright, lends, rends, tskip)
    return multiscale_sep, tskip


def zero_average_scale(tseries):
    tavg = np.mean(tseries, axis=1)
    ndims, ntstps = tseries.shape
    trscl = np.zeros((ndims, ntstps), dtype=np.float64)
    for kk in range(ndims):
        var = np.sqrt(np.mean( (tseries[kk, :] - tavg[kk])**2. ))
        trscl[kk, :] = (tseries[kk, :]-tavg[kk])/var
    return trscl
    

def windowed_deprecation(dep_fac, tseries):
    ndims, nstps = tseries.shape
    avg_window = np.ones((dep_fac,1))/dep_fac
    n_dep_stps = int(np.floor(nstps/dep_fac))
    
    tseries_dep = np.zeros((ndims, 2*n_dep_stps))
    for jj in range(n_dep_stps):
        tseries_dep[:, 2*jj] = np.squeeze(tseries[:, jj*dep_fac:(jj+1)*dep_fac] @ avg_window)            
        if jj < n_dep_stps-1:
            tseries_dep[:, 2*jj+1] = np.squeeze(
                tseries[:, int(dep_fac/2) + jj*dep_fac:int(dep_fac/2) + (jj+1)*dep_fac] @ avg_window)            
    
    return tseries_dep

def ewt_scale_separator(multiscale, kvec0, kvals, molfac):
    NUM_SCALES = len(kvec0) + 1

    separated_scales = np.zeros((multiscale.shape[0], multiscale.shape[1], NUM_SCALES))
    for jj in range(multiscale.shape[0]):
        kbreaks, pdist, kmax = mbo.otsu_breaks_builder(multiscale[jj, :], molfac, kvec0)
        pvals = np.ma.log10(pdist(kvals))
        kbreaks = np.sort(kbreaks)
        if kbreaks[0] == 0:
            kbreaks = kbreaks[1:]
            NUM_SCALES -= 1
            print("Clipping off zero break")
        separated_scales[jj, :, :] = ewt.EWT_Decomp(kbreaks, multiscale[jj, :])
    return separated_scales, kbreaks, pvals

def mseries_average_mi_comp(measuredseries, chunks, kneighbrs):
    chunk_size = int(np.floor(measuredseries.size/chunks))
    measured_series_average = np.zeros(chunk_size,dtype=np.float64)

    for jj in range(chunks):
        measured_series_average += measuredseries[jj*chunk_size:(jj+1)*chunk_size]
    measured_series_average /= chunks
    x = np.squeeze(measured_series_average[1:]).reshape(-1,1)
    y = np.squeeze(measured_series_average[:-1]).reshape(-1,1) 
    curr_mi = mi.miknn(x, y, knghbr=kneighbrs)        
    return measured_series_average, curr_mi


def averaging_threshhold(measuredseries, kneighbrs):
    chunks = 1
    measured_series_average, curr_mi = mseries_average_mi_comp(measuredseries, chunks, kneighbrs)
    thrshhold = 1.5*curr_mi
    while( curr_mi < thrshhold ):
        chunks += 2
        measured_series_average, curr_mi = mseries_average_mi_comp(measuredseries, chunks, kneighbrs)        
    return measured_series_average, chunks 
