import os,sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd 
import celerite2 
from celerite2 import terms
from scipy.optimize import minimize
import jax 
import jax.numpy as jnp 

def outlier_rejection(wavelength, fluxcube, fluxcube_err, sigma):
    meanflux = np.nanmedian(fluxcube,axis=0)
    fluxcube_orig = fluxcube
  #  fluxcube_norm = fluxcube.copy() / meanflux
    fluxcube_cleaned = fluxcube.copy()
    fluxcube_err_cleaned = fluxcube_err.copy()

    output_mask = np.ones_like(fluxcube, dtype=bool)

    for time_slice in range(len(fluxcube)):
        
        y = fluxcube[time_slice,:]
        yerr = fluxcube_err[time_slice,:]
        x = wavelength

        finite_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0) 

        x_finite = x[finite_mask]
        y_finite = y[finite_mask]
        yerr_finite = yerr[finite_mask]
        sort_inds = np.argsort(x_finite)
        x_finite = x_finite[sort_inds]
        y_finite = y_finite[sort_inds]
        yerr_finite = yerr_finite[sort_inds]

        mean_value = np.nanmean(y_finite) 
        term1 = terms.Matern32Term(sigma=np.nanstd(y_finite), rho=0.2)
        kernel = term1

        gp = celerite2.GaussianProcess(kernel, mean=mean_value)
        gp.compute(x_finite, yerr=yerr_finite)
        #print("Initial log likelihood: {0}".format(gp.log_likelihood(y_finite)))
        def set_params(params, gp):
            gp.mean = params[0]
            theta = np.exp(params[1:])
            gp.kernel = terms.Matern32Term(
                sigma=theta[0], rho=theta[1]
            ) 
            gp.compute(x_finite, diag=theta[2], quiet=True)
            return gp
        def neg_log_like(params, gp):
            try:
                gp = set_params(params, gp)
                val = -gp.log_likelihood(y_finite)
                return val if np.isfinite(val) else np.inf
            except (ValueError, np.linalg.LinAlgError):
                return np.inf        # invalid step

        initial_params = [mean_value, np.log(np.std(y_finite)), np.log(0.2), np.log(np.std(y_finite))]
        bounds = [(0, np.max(y_finite)),  # Bounds for mean
          (np.log(np.nanstd(y_finite) / 10), np.log(np.nanstd(y_finite) * 10)), #sigma 
          (np.log(0.1), np.log(10)),
            (np.log(np.nanstd(y_finite) / 10), np.log(np.nanstd(y_finite) * 10))] # rho 
        soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,), bounds=bounds)
        opt_gp = set_params(soln.x, gp)
        mu, variance = opt_gp.predict(y_finite, t=x_finite, return_var=True)
       # sigma = np.sqrt(variance)
        residuals = y_finite - mu

        mad_of_residual = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))

        def plot_prediction_opt(gp, t, y, m, s):
            plt.scatter(t, y, alpha=1, c='k', s=6, label="data")
            plt.plot(t, m, label="prediction")
            plt.fill_between(t, m , m + sigma*mad_of_residual, color="C0", alpha=0.8)
            plt.title("optimal prediction")
        #plot_prediction_opt(opt_gp, x_finite, y_finite, mu, sigma)
        #plt.show()

  
        #print(mad_of_residual)
        if time_slice == 100:
            plt.scatter(x_finite, residuals)
            plt.axhline(np.nanmedian(residuals)-10*mad_of_residual, ls='-', c='r')
            plt.axhline(np.nanmedian(residuals)+10*mad_of_residual, ls='-', c='r')
            plt.axhline(np.nanmedian(residuals)-7*mad_of_residual, ls='-', c='b')
            plt.axhline(np.nanmedian(residuals)+7*mad_of_residual, ls='-', c='b')
            plt.axhline(np.nanmedian(residuals)-5*mad_of_residual, ls='-', c='purple')
            plt.axhline(np.nanmedian(residuals)+5*mad_of_residual, ls='-', c='purple')
            plt.axhline(np.nanmedian(residuals)-4*mad_of_residual, ls='-', c='pink')
            plt.axhline(np.nanmedian(residuals)+4*mad_of_residual, ls='-', c='pink')
            plt.savefig(f'trace_sigma_check.png')
            plt.close()
        
        outlier_mask_finite = np.abs(residuals - np.nanmedian(residuals)) > sigma * mad_of_residual

        outlier_indices_in_sorted_finite = np.where(outlier_mask_finite)[0]
        original_indices_in_finite = sort_inds[outlier_indices_in_sorted_finite]
        global_finite_indices = np.where(finite_mask)[0]
        outlier_indices_in_original = global_finite_indices[original_indices_in_finite]

        output_mask[time_slice, outlier_indices_in_original] = False


        fluxcube_cleaned[time_slice, outlier_indices_in_original] = np.nan

        if time_slice == 100:
            plt.scatter(wavelength, fluxcube_orig[time_slice,:], s=12)
            plt.scatter(wavelength, fluxcube_cleaned[time_slice,:], c='r', s=12, zorder=3)
            plt.savefig(f'trace_gp_check.png')
            plt.close()
            #plt.show()
    return output_mask


def compile_spectra(x1dfits,seg, nrs, planet_str, output_dir, aperturekey='FLUX'):
    print(x1dfits)
    x1dfits = fits.open(x1dfits)
    try:
        bjd = x1dfits[1].data['int_mid_BJD_TDB'][1:]
    except IndexError:
        bjd = np.arange(len(x1dfits)-3)
    wavelength = x1dfits[2].data['WAVELENGTH']
    
    fluxcube = []
    fluxcube_err = []
    for i in np.arange(3,len(x1dfits)):
        fluxcube.append(x1dfits[i].data[aperturekey])
        fluxcube_err.append(x1dfits[i].data["FLUX_ERROR"])
    
    fluxcube,fluxcube_err = np.array(fluxcube),np.array(fluxcube_err)
    #print(np.shape(fluxcube))
    if planet_str.lower() == 'toi1130b':
        if int(nrs) == 2:
            col_sigma = 5
        if int(nrs) == 1:
            col_sigma = 10
            wave_mask = wavelength > 2.86
            wavelength = wavelength[wave_mask]
            fluxcube = fluxcube[:,wave_mask]
            fluxcube_err = fluxcube_err[:,wave_mask]
    elif planet_str.lower() == 'wasp107b':
        if int(nrs) == 2:
            col_sigma = 7
        if int(nrs) == 1:
            col_sigma = 10
            wave_mask = wavelength > 2.86
            wavelength = wavelength[wave_mask]
            fluxcube = fluxcube[:,wave_mask]
            fluxcube_err = fluxcube_err[:,wave_mask]
    elif planet_str.lower() == 'toi1130c':
        if int(nrs) == 2:
            col_sigma = 5
        if int(nrs) == 1:
            col_sigma = 10
            wave_mask = wavelength > 2.86
            wavelength = wavelength[wave_mask]
            fluxcube = fluxcube[:,wave_mask]
            fluxcube_err = fluxcube_err[:,wave_mask]
    #else:
    #    print('Planet not supported. Please test the column pixel rejection manually using templtae above this message in new_unpack.py')
    #    exit()

    if os.path.exists(f'{output_dir}/{planet_str}_mask_check_NRS{nrs}_{seg}_{col_sigma}MAD.npy'):
        print('Column rejection files already exist...')
        outlier_mask = np.load(f'{output_dir}/{planet_str}_mask_check_NRS{nrs}_{seg}_{col_sigma}MAD.npy')
        #for i in range(len(outlier_mask[:,0])):
        #    print(f"Total points masked {i}: {np.sum(~outlier_mask[i,:])} out of {outlier_mask[i,:].size}")
        fraction_outliers_per_column = np.mean(~outlier_mask, axis=0)
        reject_wavelength_mask = fraction_outliers_per_column > 0.5
        fluxcube[:, reject_wavelength_mask] = np.nan
        plt.imshow(outlier_mask, aspect='auto')
        plt.savefig(f'{output_dir}/column_rejection_mask_NRS{nrs}_{seg}_{col_sigma}MAD')
        plt.show()
        print("Columns rejected have been set to NaN.")
    else: 
        print('Running column rejection...')
        outlier_mask = outlier_rejection(wavelength, fluxcube, fluxcube_err, sigma=col_sigma)
        #for i in range(len(outlier_mask[:,0])):
        #    print(f"Total points masked {i}: {np.sum(~outlier_mask[i,:])} out of {outlier_mask[i,:].size}")
        data = np.save(f'{output_dir}/{planet_str}_mask_check_NRS{nrs}_{seg}_{col_sigma}MAD.npy', arr=outlier_mask)
        fraction_outliers_per_column = np.mean(~outlier_mask, axis=0)
        reject_wavelength_mask = fraction_outliers_per_column > 0.5
        fluxcube[:, reject_wavelength_mask] = np.nan

        plt.imshow(outlier_mask, aspect='auto')
        plt.savefig(f'{output_dir}/column_rejection_mask_NRS{nrs}_{seg}_{col_sigma}MAD')
        plt.show()
        print("Columns rejected have been set to NaN.")
    
    return bjd,wavelength,fluxcube,fluxcube_err

def get_whitelightlc(fluxcube,fluxcube_err):
   # print("whitelight")
    whitelightflux = np.nanmean(fluxcube,axis=1) #/ np.nansum(weights,axis=1)
   # print("normalising")

    #whitelightflux = np.nanmean(fluxcube,axis=1)
    #whitelightflux /= np.nanmedian(whitelightflux)
    
    return whitelightflux

def get_rms_colour(fluxcube,fluxcube_err,wavelength):
    #meanflux = np.nanmedian(fluxcube,axis=0)
    #fluxcube /= meanflux

    wavebin_rms = []
    i = 0
    binsize = 10
    print(fluxcube.shape)
    while i < len(fluxcube[0]):
        #flux_i = np.nanmedian(fluxcube[:,i:i+binsize],axis=1)
        flux_i = get_whitelightlc(fluxcube[:,i:i+binsize],fluxcube_err[:,i:i+binsize])
        flux_i/=np.nanmedian(flux_i)

        #plt.plot(flux_i, '.')
        #plt.show()
        mad = np.nanmedian(abs(np.diff(flux_i)))
        wave_i = np.nanmedian(wavelength[i:i+binsize])
        wavebin_rms.append([wave_i,mad])
        i += binsize
        print(i, mad)
    wavebin_rms = np.array(wavebin_rms)

    return wavebin_rms

def join_whitelight(whitelightlist):

    whitelightnew = list(whitelightlist[0])
    for i in np.arange(1,len(whitelightlist)):
        #prev_flux = np.nanmedian(whitelightlist[i-1][-5:])
        #new_flux = np.nanmedian(whitelightlist[i][:5])
        #offset = new_flux-prev_flux
        #whitelightlist[i] -= offset
        whitelightnew += list(whitelightlist[i])

    return np.array(whitelightnew)/np.nanmedian(whitelightnew)
                       

def unpack_lc(file_pattern, nrs, planet_str, output_dir, bins_wanted=50):
    x1dfitslist = np.sort(glob.glob(file_pattern))
    
    fluxcubeall = []
    fluxcubeall_err = []
    wavelength = None
    t = []
    for i in range(len(x1dfitslist)):
        x1dfits = x1dfitslist[i]
        bjd,wl,fluxcube,fluxcube_err = compile_spectra(x1dfits,i, nrs, planet_str, output_dir, aperturekey='FLUX')
        fluxcubeall += list(fluxcube)
        fluxcubeall_err += list(fluxcube_err)
        wavelength = wl
        t.append(bjd)

    t = np.concatenate(t)
    fluxcube = np.array(fluxcubeall)
    fluxcube_err = np.array(fluxcubeall_err)

 
    # Create binned light curves
    binsize = bins_wanted
    wavelengths = []
    binned_lcs = []
    binned_lcs_err = []
    i = 0

    while i < len(wavelength):
        # Get wavelength for this bin
        wave_i = np.nanmedian(wavelength[i:i+binsize])
        wavelengths.append(wave_i)
        
        flux_i = get_whitelightlc(fluxcube[:,i:i+binsize], fluxcube_err[:,i:i+binsize])
        flux_i = flux_i / np.nanmedian(flux_i)  # Normalize
        mad_i = np.repeat(1.4826 * np.nanmedian(np.abs(flux_i - np.nanmedian(flux_i))), len(flux_i))
        binned_lcs_err.append(mad_i)
        binned_lcs.append(flux_i)
        i += binsize
    

    # Convert to arrays
    wavelengths = jnp.array(wavelengths)  
    indiv_y = jnp.array(binned_lcs)  
    yerr = jnp.array(binned_lcs_err)

    # Original masks
    mask_indiv_y = ~jnp.any(jnp.isnan(indiv_y), axis=1) 
    mask_yerr = ~jnp.any(jnp.isnan(yerr), axis=1)
    
    # Add wavelength mask
    valid_rows = mask_indiv_y & mask_yerr  
    
    indiv_y = indiv_y[valid_rows, :]    
    yerr = yerr[valid_rows, :]
    wavelengths = wavelengths[valid_rows]
    
    return wavelengths, t, indiv_y, yerr

def diagplot(x1dfitslist,aperturekey, nrs, planet_str,output_dir, wl_csv_dir):
    bjdall = []
    whitelightlist = []
    fluxcubeall = []
    fluxcubeall_err = []
    for i in range(len(x1dfitslist)):
        x1dfits = x1dfitslist[i]
        bjd,wavelength,fluxcube,fluxcube_err = compile_spectra(x1dfits,i, nrs, planet_str, output_dir, aperturekey=aperturekey)

        whitelightlist.append(get_whitelightlc(fluxcube,fluxcube_err))
        fluxcubeall += list(fluxcube)
        fluxcubeall_err += list(fluxcube_err)
        bjdall += list(bjd)

    bjdall = np.array(bjdall)
    fluxcubeall = np.array(fluxcubeall)
    fluxcubeall_err = np.array(fluxcubeall_err)

   # wave_df = pd.DataFrame(data={'fluxcubeall': fluxcubeall, 'fluxcubeall_err': fluxcubeall_err, 'wavelength':wavelength})
   # wave_df.to_csv('multiwavelength_wasp107.csv')
    lc = join_whitelight(whitelightlist)

    #lc_mad = np.nanmedian(abs(np.diff(lc)))
    #lc = np.where(lc>np.nanmedian(lc)+5*lc_mad, np.nan, lc)
    mad = np.nanmedian(abs(np.diff(lc)))*1e6


    
    #wavelength_mask = (wavelength > 3.)*((wavelength < 4.8))
    #wavebinrms = get_rms_colour(fluxcubeall[:,wavelength_mask],fluxcubeall_err[:,wavelength_mask],wavelength[wavelength_mask])
    wavebinrms = get_rms_colour(fluxcubeall, fluxcubeall_err, wavelength)
    #lcmask = lc>1.05
    #lc = lc[~lcmask]
    #bjdall = bjdall[~lcmask]

    plt.figure(figsize=(5,7))
    plt.subplots_adjust(left=0.15,right=0.95,bottom=0.1,top=0.95)

    plt.subplot(211)
   # print(len(lc))
    plt.scatter(bjdall,lc,s=1,color='k')
    plt.title(aperturekey+" MAD="+str(np.floor(mad))+"ppm")
    plt.xlabel("BJD-"+str(np.floor(bjdall[0])),fontsize=12)
    plt.ylabel(aperturekey,fontsize=12)

    plt.subplot(212)
    plt.scatter(wavebinrms[:,0],wavebinrms[:,1]*1e6,s=1,color='k')
    plt.xlabel("Wavelength [$\mu$m]",fontsize=12)
    plt.ylabel("MAD (ppm)",fontsize=12)
    plt.ylim(0,4000)
   # print(bjdall-np.floor(bjdall[0]),lc)
    
    plt.savefig(f'{output_dir}/{planet_str}_raw_whitelight.png')
    plt.close()
    #print(wavelength)
    data = {'time':bjdall, 'flux':lc}
    df_data = pd.DataFrame(data=data)
    df_data.to_csv(wl_csv_dir)
   # plt.show()

    
'''
if __name__ == "__main__":

    
    #keylist = ['FLUX','FLUX_NODRIFT','FLUX_1.5FWHM','FLUX_2.0FWHM','FLUX_3.0FWHM','FLUX_4.0FWHM','FLUX_5.0FWHM','FLUX_BKORDER1']
    keylist = ['FLUX']
    nrs = 2
    pl_name = 'TOI1130b'
    wl_csv_dir = 'TOI1130b_NIRSPEC/TOI1130b_NIRSPEC_NRS2_WHITELIGHT.csv'
    #x1dfitslist = np.sort(glob.glob("V1298/jw02149002001_04102_00001-seg00*_nrs2_x1dints.fits"))
    x1dfitslist = np.sort(glob.glob(f"TOI1130b_NIRSPEC/*_nrs{nrs}_x1dints.fits"))
    #x1dfitslist = np.sort(glob.glob("HD15337/transit1/*nrs1*.fits"))
    for aperturekey in keylist:
        diagplot(x1dfitslist,aperturekey, nrs, wl_csv_dir)
'''
