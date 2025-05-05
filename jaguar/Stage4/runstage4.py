
import os
import sys
import glob
from functools import partial
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro_ext.distributions as distx
import numpyro_ext.optim as optimx
import matplotlib.pyplot as plt
import pandas as pd
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
from exotic_ld import StellarLimbDarkening
from jaguar.Stage4.plotting     import plot_map_fits, plot_map_residuals, plot_transmission_spectrum
from jaguar.Stage4.unpack_trace import unpack_trace
import numpyro_ext
import argparse
import yaml
# ---------------------
# Model Functions
# ---------------------
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
def jitter_value(value, jitter_fraction, key):
    """Applies random jitter scaled by value and fraction."""
    jitter = jitter_fraction * value * jax.random.normal(key)
    return value + jitter

def compute_lc_from_params(params, t):
    """Computes a single light curve + linear trend from parameters."""
    orbit = TransitOrbit(
        period=params["period"],
        duration=params["duration"],
        impact_param=params["b"],
        time_transit=params["t0"],
        radius_ratio=params["rors"],
    )
    lc_transit = limb_dark_light_curve(orbit, params["u"])(t)
    trend = params["c"] + params["v"] * (t - jnp.min(t))
    return lc_transit + trend

def whitelight_model(t, yerr, y=None, prior_params=None):
  
    logD = numpyro.sample("logD", dist.Normal(jnp.log(prior_params['duration']), 1e-2))
    duration = numpyro.deterministic("duration", jnp.exp(logD))

    t0 = numpyro.sample("t0", dist.Normal(prior_params['t0'], 1e-2))

    _b = numpyro.sample("_b", dist.Uniform(-2.0, 2.0)) 
    b = numpyro.deterministic('b', jnp.abs(_b))  
    #b = numpyro.deterministic('b', prior_params['b'])    
    u = numpyro.sample("u", distx.QuadLDParams())
    
    depths = numpyro.sample('depths', dist.TruncatedNormal(prior_params['rors']**2, prior_params['rors']**2 * 0.1, low=0.0, high=1.0))
    rors = numpyro.deterministic("rors", jnp.sqrt(depths))
    c = numpyro.sample("c", dist.Normal(1.0, 0.01))
    v = numpyro.sample("v", dist.Normal(0.0, 0.01))
    params = {
        "period": prior_params['period'], "duration": duration, "t0": t0, "b": b,
        "rors": rors, "u": u, "c": c, "v": v
    }
    y_model = compute_lc_from_params(params, t)
    
    numpyro.sample('obs', dist.Normal(y_model, yerr), obs=y)

def vectorized_model(t, yerr, y=None, mu_duration=None, mu_t0=None, mu_depths=None, PERIOD=None, trend_fixed=None, ld_fixed=None):
    """Vectorized numpyro model for multiple light curves."""
    num_lcs = jnp.atleast_2d(yerr).shape[0]

    logD = numpyro.sample("logD", dist.Normal(jnp.log(mu_duration), 0.001))
    duration = numpyro.deterministic("duration", jnp.exp(logD))

    t0 = numpyro.sample("t0", dist.Normal(mu_t0, 1e-2))

    _b = numpyro.sample("_b", dist.Uniform(-2.0, 2.0)) 
    b = numpyro.deterministic('b', jnp.abs(_b))    

    if ld_fixed is None:
        u = numpyro.sample("u", distx.QuadLDParams().expand([num_lcs]))
    else:
        u = numpyro.deterministic("u_fixed", ld_fixed)

    depths = numpyro.sample('depths', dist.TruncatedNormal(mu_depths, mu_depths * 0.1, low=0.0, high=1.0).expand([num_lcs]))
    rors = numpyro.deterministic("rors", jnp.sqrt(jnp.atleast_1d(depths)))

    if trend_fixed is None:
        c = numpyro.sample("c", dist.Normal(1.0, 0.01).expand([num_lcs]))
        v = numpyro.sample("v", dist.Normal(0.0, 0.01).expand([num_lcs]))
    else:
        c = jnp.asarray(trend_fixed["c"])
        v = jnp.asarray(trend_fixed["v"])
        numpyro.deterministic("c_fixed", c)
        numpyro.deterministic("v_fixed", v)

    params = {
        "period": PERIOD, "duration": duration, "t0": t0, "b": b,
        "rors": rors, "u": u, "c": c, "v": v
    }

    y_model = jax.vmap(
        compute_lc_from_params,
        in_axes=(
            { "period": None, "duration": None, "t0": None, "b": None,
              "rors": 0, "u": 0, "c": 0, "v": 0 },
            None,
         )
      )(params, t)

    numpyro.sample("obs", dist.Normal(y_model, yerr), obs=y)

def get_samples(model, key, t, yerr, indiv_y, init_params, trend_fixed=None, ld_fixed=None):
    """Runs MCMC using NUTS to get posterior samples."""
 
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(
            model,
            regularize_mass_matrix=False,
            init_strategy=numpyro.infer.init_to_value(values=init_params),
        ),
        num_warmup=1000,
        num_samples=1000,
        progress_bar=True, 
        jit_model_args=True
    )
    mcmc.run(key, t, yerr, y=indiv_y, trend_fixed=trend_fixed, ld_fixed=ld_fixed)
    return mcmc.get_samples()

def compute_aic(n, residuals, k):
    """Computes an approximate AIC value."""
    rss = np.sum(np.square(residuals))
    rss = rss if rss > 1e-10 else 1e-10 
    aic = 2*k + n * np.log(rss/n)
    return aic

def get_limb_darkening(sld, wavelengths, num_lcs):
    """Gets limb darkening coefficients using exotic_ld."""
    us, us_sigmas = sld.compute_quadratic_ld_coeffs(
        wavelength_range=[min(wavelengths)*10000, max(wavelengths)*10000],
        mode="JWST_NIRSpec_G395H",
        return_sigmas=True
    )
    U_mu = jnp.array(us)
    # U_sigma = jnp.array(us_sigmas) 
    U_mu = jnp.broadcast_to(U_mu, (num_lcs, 2))
    # U_sigma = jnp.broadcast_to(U_sigma, (num_lcs, 2)) 
    return U_mu #, U_sigma

def fit_model_map(t, yerr, indiv_y, init_params,
                 mu_duration, mu_t0, mu_depths, PERIOD,
                 key=None, trend_fixed=None, ld_fixed=None):
    """Fits the model using MAP optimization."""
    if key is None:
        key = jax.random.PRNGKey(111) 

    vmodel = partial(vectorized_model,
                     mu_duration=mu_duration, mu_t0=mu_t0, mu_depths=mu_depths,
                     PERIOD=PERIOD, trend_fixed=trend_fixed, ld_fixed=ld_fixed)

    soln = optimx.optimize(vmodel, start=init_params)(key, t, yerr, y=indiv_y)

    if trend_fixed is not None:
        soln["c"] = np.array(trend_fixed["c"])
        soln["v"] = np.array(trend_fixed["v"])
    if ld_fixed is not None:
        soln["u"] = np.array(ld_fixed)

    return soln

def fit_polynomial(x, y, poly_orders):
    """Fits polynomials and selects the best order using AIC."""
    best_order = None
    best_aic = np.inf
    best_coeffs = None

    for deg in poly_orders:
        y_med = np.median(y, axis=0)
        coeffs = np.polyfit(x, y_med, deg)
        pred = np.polyval(coeffs, x)
        aic = compute_aic(len(x), y_med - pred, k=deg+1) 

        if aic < best_aic:
            best_aic = aic
            best_order = deg
            best_coeffs = coeffs

    return best_coeffs, best_order, best_aic

def plot_poly_fit(x, y, coeffs, order, xlabel, ylabel, title, save_path):
    """Plots data and its polynomial fit, then saves it."""
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = np.polyval(coeffs, x_fit)

    y_med = np.median(y, axis=0)
    y_err = np.std(y, axis=0)
    fig, ax = plt.subplots(figsize=(8, 5)) 
    ax.errorbar(x, y_med, yerr=y_err, fmt='o', ls='', ecolor='k', mfc='k', mec='k')
    ax.plot(x_fit, y_fit, '-', label=f'Poly fit (deg {order})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved polynomial fit plot to {save_path}")

def save_results(wavelengths, samples, nrs, csv_filename):
    """Computes and saves transmission spectrum results to CSV."""
    rors_median = np.median(samples["rors"], axis=0)
    rors_16 = np.percentile(samples["rors"], 16, axis=0)
    rors_84 = np.percentile(samples["rors"], 84, axis=0)
    rors_err = (rors_84 - rors_16) * 0.5

    depth_median = rors_median**2
    depth_err = 2 * rors_median * rors_err

    output_data = np.column_stack((wavelengths, depth_median, depth_err))
    np.savetxt(
        csv_filename, output_data, delimiter=",",
        header="wavelength,depth,depth_err", comments=""
    )
    print(f"Transmission spectroscopy data saved to {csv_filename}")


# ---------------------
# Main Analysis
# ---------------------
def validate_config(cfg):
    # top-level keys we expect
    required = {"planet", "stellar", "flags", "resolution_bins", "outlier_clip",
                "path", "input_dir", "output_dir", "host_device"}
    allowed  = required  
    allowed = required.union({"nrs_order"})

    missing = required - cfg.keys()
    extra   = cfg.keys() - allowed
    if missing:
        raise KeyError(f"Config is missing required keys: {missing}")
    if extra:
        raise KeyError(f"Config contains unrecognized keys: {extra}")

    p_req = {"name", "period", "duration", "t0", "b", "rprs"}
    p_extra = set(cfg["planet"]) - p_req
    p_miss  = p_req - set(cfg["planet"])
    if p_miss or p_extra:
        raise KeyError(f"planet section bad. missing={p_miss}, unknown={p_extra}")

    s_req = {"feh", "teff", "logg", "ld_model", "ld_data_path"}
    s_extra = set(cfg["stellar"]) - s_req
    s_miss  = s_req - set(cfg["stellar"])
    if s_miss or s_extra:
        raise KeyError(f"stellar section bad. missing={s_miss}, unknown={s_extra}")

def main(config_path):
    """
    Load the yaml at `config_path` and do your plotting/unpacking, etc.
    """
    cfg = yaml.safe_load(open(config_path))
    validate_config(cfg)


    nrs = cfg['nrs_order']
    planet_cfg = cfg['planet']
    stellar_cfg = cfg['stellar']
    flags = cfg.get('flags', {})
    bins = cfg.get('resolution_bins', {})
    outlier_clip = cfg.get('outlier_clip', {})
    planet_str = planet_cfg['name']

    # Paths
    base_path = cfg.get('path', '.')
    input_dir = os.path.join(base_path, cfg.get('input_dir', planet_str + '_NIRSPEC'))
    output_dir = os.path.join(base_path, cfg.get('output_dir', planet_str + '_RESULTS'))
    if not os.path.exists(input_dir):
        print(f"Error: No Planet Folder detected. Need directory: {input_dir}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    # device (please use gpu!!)
    host_device = cfg.get('host_device', 'gpu').lower()
    numpyro.set_platform('gpu' if host_device == 'gpu' else 'cpu')
    key_master = jax.random.PRNGKey(555)

    # flags
    interpolate_trend = flags.get('interpolate_trend', False)
    interpolate_ld = flags.get('interpolate_ld', False)
    need_lowres = flags.get('need_lowres', True)

    # binning nm seperation
    high_resolution_bins = bins.get('high', 1)
    low_resolution_bins = bins.get('low', 100)

    # outlier clipping
    whitelight_sigma = outlier_clip.get('whitelight_sigma', 4)
    spectroscopic_sigma = outlier_clip.get('spectroscopic_sigma', 4)

    # plajet priors
    PERIOD_FIXED = planet_cfg['period']
    PRIOR_DUR = planet_cfg['duration']
    PRIOR_T0 = planet_cfg['t0']
    PRIOR_B = planet_cfg['b']
    PRIOR_RPRS = planet_cfg['rprs']
    PRIOR_DEPTH = PRIOR_RPRS ** 2

    # stellar parameters
    stellar_feh = stellar_cfg['feh']
    stellar_teff = stellar_cfg['teff']
    stellar_logg = stellar_cfg['logg']
    ld_model = stellar_cfg.get('ld_model', 'mps1')
    ld_data_path = stellar_cfg.get('ld_data_path', '../exotic_ld_data')

    # file pattern and white-light CSV
    file_pattern = os.path.join(input_dir, f"*_nrs{nrs}_x1dints.fits")
    wl_csv = os.path.join(output_dir, f"{planet_str}_NIRSPEC_NRS{nrs}_WHITELIGHT.csv")

    # gen or load white-light curve
    if not os.path.exists(wl_csv):
        keylist = ['FLUX']
        x1dfitslist = np.sort(glob.glob(file_pattern))
        if len(x1dfitslist) == 0:
            print(f"Error: No input Stage 3 files in {input_dir}")
            sys.exit(1)
        for aperturekey in keylist:
            unpack_trace.diagplot(x1dfitslist,aperturekey, nrs, planet_str, output_dir, wl_csv)
    wl = pd.read_csv(wl_csv)

    wl_time = wl.time.values
    wl_flux = wl.flux.values
    wl_flux_err = 1.4826 * np.nanmedian(np.abs(wl_flux - np.nanmedian(wl_flux)))

    if not os.path.exists(f'{output_dir}/{planet_str}_NIRSPEC_whitelight_outlier_mask.npy'):
        print('Fitting whitelight for outliers and bestfit parameters')
        prior_params_wl = {
                "duration": PRIOR_DUR, "t0": PRIOR_T0,
                "rors": jnp.sqrt(PRIOR_DEPTH), 'period': PERIOD_FIXED, '_b': PRIOR_B,
                'u': numpyro_ext.distributions.QuadLDParams().sample(jax.random.PRNGKey(2345)),
                'c': 1.0, 'v': 0.0, 'logD': jnp.log(PRIOR_DUR), 'b': PRIOR_B, 'depths': PRIOR_DEPTH
            }


                        
        soln = optimx.optimize(whitelight_model, start=prior_params_wl)(key_master, wl_time, wl_flux_err, y=wl_flux, prior_params=prior_params_wl)

       # map_params = {'duration': soln['duration'], 't0': soln['t0'], 'rors':soln['rors'], 'period': PERIOD_FIXED, 'b': PRIOR_B,
       #               'u': soln['u'], 'c': soln['c'], 'v': soln['v']}
       # mod = compute_lc_from_params(map_params, wl_time)
       # plt.scatter(wl_time, wl_flux, c='k')
       # plt.plot(wl_time, mod, c='r')
       # plt.show()

        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(
                whitelight_model,
                regularize_mass_matrix=False,
                init_strategy=numpyro.infer.init_to_value(values=soln),
            ),
            num_warmup=1000,
            num_samples=1000,
            progress_bar=True, 
            jit_model_args=True
        )
        mcmc.run(key_master, wl_time, wl_flux_err, y=wl_flux, prior_params=prior_params_wl)
    
        wl_samples = mcmc.get_samples() 

        bestfit_params_wl = {'duration': jnp.nanmedian(wl_samples['duration']), 't0': jnp.nanmedian(wl_samples['t0']),
                            'b': jnp.nanmedian(wl_samples['b']), 'rors': jnp.nanmedian(wl_samples['rors']),
                            'c': jnp.nanmedian(wl_samples['c']), 'v': jnp.nanmedian(wl_samples['v']),
                            'period': PERIOD_FIXED, 'u': jnp.nanmedian(wl_samples['u'], axis=0)
                            }
        print(bestfit_params_wl)
        wl_transit_model = compute_lc_from_params(bestfit_params_wl, wl_time)

        wl_residual = wl_flux - wl_transit_model 

        #plt.scatter(wl_time, wl_flux, c='k')
        #plt.plot(wl_time, wl_transit_model, c='r')
       # plt.show()
        
        wl_sigma = 1.4826 * np.nanmedian(np.abs(wl_residual - np.nanmedian(wl_residual)))
      
        wl_mad_mask = np.abs(wl_residual - np.nanmedian(wl_residual)) > whitelight_sigma * wl_sigma

        wl_sigma_post_clip = 1.4826 * np.nanmedian(np.abs(wl_residual[~wl_mad_mask] - np.nanmedian(wl_residual[~wl_mad_mask])))

        plt.scatter(wl_time[~wl_mad_mask], wl_flux[~wl_mad_mask], c='k')
        plt.plot(wl_time, wl_transit_model, c='r')
        plt.title(f'Sigma {round(wl_sigma_post_clip*1e6)} PPM')
        plt.savefig(f'{output_dir}/{planet_str}_NIRSPEC_nrs{nrs}_whitelight_check.png')
        plt.close()

        np.save(f'{output_dir}/{planet_str}_NIRSPEC_whitelight_outlier_mask.npy', arr=wl_mad_mask)
        df = pd.DataFrame(bestfit_params_wl)
        df.to_csv(f'{output_dir}/{planet_str}_NIRSPEC_whitelight_bestfit_params.csv')
        print(f'Saved whitelight parameters to {output_dir}/{planet_str}_NIRSPEC_whitelight_bestfit_params.csv')
        DURATION_BASE = jnp.array(bestfit_params_wl['duration'])
        T0_BASE = jnp.array(bestfit_params_wl['t0'])
        B_BASE = jnp.array(bestfit_params_wl['b'])
        RORS_BASE = jnp.array(bestfit_params_wl['rors'])
        DEPTH_BASE = RORS_BASE**2
    else: 
        print(f'Whitelight outliers and bestfit parameters already exist, skipping whitelight fit. If you want to fit whitelight please delete {output_dir}/{planet_str}_NIRSPEC_whitelight_outlier_mask.npy')
        wl_mad_mask = np.load(f'{output_dir}/{planet_str}_NIRSPEC_whitelight_outlier_mask.npy')
        bestfit_params_wl = pd.read_csv(f'{output_dir}/{planet_str}_NIRSPEC_whitelight_bestfit_params.csv')
        DURATION_BASE = jnp.array(bestfit_params_wl['duration'][0])
        T0_BASE = jnp.array(bestfit_params_wl['t0'][0])
        B_BASE = jnp.array(bestfit_params_wl['b'][0])
        RORS_BASE = jnp.array(bestfit_params_wl['rors'][0])
        DEPTH_BASE = RORS_BASE**2
    #cut_residual = wl_residual[~wl_mad_mask] 
    #cut_mad = np.nanmedian(np.abs(cut_residual - np.nanmedian(cut_residual))) * 1e6 
  



    x1dfitslist = glob.glob(file_pattern)
    if not x1dfitslist:
        print(f"Error: No files found matching pattern: {file_pattern}")
        sys.exit(1)
    print(f"Found {len(x1dfitslist)} input file(s).")

    key_lr, key_hr, key_map_lr, key_mcmc_lr, key_map_hr, key_mcmc_hr, key_prior_pred = jax.random.split(key_master, 7)

    sld = StellarLimbDarkening(
        M_H=stellar_feh, Teff=stellar_teff, logg=stellar_logg, ld_model=ld_model,
        ld_data_path=ld_data_path
    )

    high_bins = high_resolution_bins
    low_bins = low_resolution_bins

    # is low-resolution analysis needed
    need_lowres_analysis = interpolate_trend or interpolate_ld or need_lowres

    # init vars that might be set in low-res analysis
    trend_fixed_hr = None
    ld_fixed_hr = None
    best_poly_coeffs_c = None
    best_poly_coeffs_v = None
    best_poly_coeffs_u1 = None
    best_poly_coeffs_u2 = None

    # --- Low-resolution Analysis ---
    if need_lowres_analysis:
        print(f"\n--- Running Low-Resolution Analysis (Binned every {low_bins} nm) ---")
        wavelengths_lr, t_lr, indiv_y_lr, yerr_lr = unpack_trace.unpack_lc(file_pattern, nrs, planet_str, output_dir, low_bins) 

        ##### APPLY OUTLIER MASK HERE ####
        #wavelengths_lr = wavelengths_lr[~wl_mad_mask]
 
        t_lr = t_lr[~wl_mad_mask]
        indiv_y_lr = indiv_y_lr[:, ~wl_mad_mask]
        yerr_lr = yerr_lr[:, ~wl_mad_mask]

        num_lcs_lr = yerr_lr.shape[0]
        print(f"Low-res: {num_lcs_lr} wavelength bins.")

        DEPTHS_BASE_LR = jnp.full(num_lcs_lr, DEPTH_BASE)

    
        U_mu_lr = get_limb_darkening(sld, wavelengths_lr, num_lcs_lr)


        init_params_lr = {
            "logD": jnp.log(DURATION_BASE), "t0": T0_BASE, "_b": B_BASE,
            "u": U_mu_lr, "depths": DEPTHS_BASE_LR, 
            "c": jnp.ones(num_lcs_lr), "v": jnp.zeros(num_lcs_lr),
        }

     

        print("Sampling low-res model using MCMC to find median coefficients...")
    
        samples_lr = get_samples(
            partial(vectorized_model,
                    mu_duration=DURATION_BASE, mu_t0=T0_BASE,
                    mu_depths=DEPTHS_BASE_LR, PERIOD=PERIOD_FIXED),
            key_mcmc_lr, t_lr, yerr_lr, indiv_y_lr, init_params_lr
        )

        trend_c_lr = np.array(samples_lr["c"])
        trend_v_lr = np.array(samples_lr["v"])
        ld_u_lr = np.array(samples_lr["u"])

        map_params_lr = {
            "duration": np.median(samples_lr["duration"]),
            "t0": np.median(samples_lr["t0"]),
            "b": np.median(samples_lr["b"]),
            "rors": np.median(samples_lr["rors"], axis=0),
            "u": np.median(ld_u_lr, axis=0), "c": np.median(trend_c_lr, axis=0),
            "v": np.median(trend_v_lr, axis=0), "period": PERIOD_FIXED
        }
        print(map_params_lr)    

        lc_transit_all = jax.vmap(
            lambda r, u: limb_dark_light_curve(
                            TransitOrbit(
                              period=  PERIOD_FIXED,
                              duration=map_params_lr["duration"],
                              impact_param=map_params_lr["b"],
                              time_transit=map_params_lr["t0"],
                              radius_ratio=r
                            ),
                            u
                         )(t_lr),
            in_axes=(0, 0)
        )(map_params_lr["rors"], map_params_lr["u"])
        t0min = jnp.min(t_lr)
        trend_all = (
          map_params_lr["c"][:, None]
          + map_params_lr["v"][:, None] * (t_lr[None, :] - t0min)
        )  

        model_all = lc_transit_all + trend_all
        residuals = indiv_y_lr - model_all   

        medians = np.nanmedian(residuals, axis=1, keepdims=True)
        sigmas    = 1.4826 * np.nanmedian(np.abs(residuals - medians), axis=1, keepdims=True)

        point_mask = np.abs(residuals - medians) > spectroscopic_sigma * sigmas   

        time_mask = np.any(point_mask, axis=0)  

        n_channels, n_times = indiv_y_lr.shape

        if n_channels > 0:
            nrows = int(np.ceil(np.sqrt(n_channels)))
            ncols = int(np.ceil(n_channels / nrows))
        else:
            nrows, ncols = 1, 1 

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3), squeeze=False)

        axes = axes.flatten()

        plot_count = 0 

        for ch in range(n_channels):
            outlier_mask = point_mask[ch]

            if outlier_mask.any():
                ax = axes[plot_count] 
                ax.scatter(t_lr, indiv_y_lr[ch, :], c="k", label="data", s=10)

                ax.scatter(t_lr[outlier_mask], indiv_y_lr[ch, outlier_mask],
                            c="r", label="outliers", s=10) 

                plot_count += 1 #
        for i in range(plot_count, nrows * ncols):
            axes[i].axis('off') 

        if plot_count > 0:
            plt.tight_layout() 
            plt.savefig(f"{output_dir}/{planet_str}_NIRSPEC_spectro_outliers.png") 
        else:
            print("No channels with outliers found to plot.")

        plt.close(fig) 
        valid = ~time_mask     
        t_lr       = t_lr[valid]
        indiv_y_lr = indiv_y_lr[:, valid]
        yerr_lr    = yerr_lr[:, valid]
    

        print("Plotting low-resolution fits and residuals...")
        plot_map_fits(t_lr, indiv_y_lr, yerr_lr, wavelengths_lr, map_params_lr,
                      {"period": PERIOD_FIXED},
                      f"{output_dir}/lowres_fits_{planet_str}_nrs{nrs}_{low_bins}bins.png", ncols=5)
        plot_map_residuals(t_lr, indiv_y_lr, yerr_lr, wavelengths_lr, map_params_lr,
                           {"period": PERIOD_FIXED},
                           f"{output_dir}/lowres_residuals_{planet_str}_nrs{nrs}_{low_bins}bins.png", ncols=5)

        # Polynomial Fitting for Interpolation
        poly_orders = [1, 2, 3, 4]
        wl_lr = np.array(wavelengths_lr)

        if interpolate_trend:
            print("Fitting polynomials to trend coefficients...")
            best_poly_coeffs_c, best_order_c, _ = fit_polynomial(wl_lr, trend_c_lr, poly_orders)
            best_poly_coeffs_v, best_order_v, _ = fit_polynomial(wl_lr, trend_v_lr, poly_orders)
            print(f"Selected polynomial degrees: c={best_order_c}, v={best_order_v}")
            plot_poly_fit(wl_lr, trend_c_lr, best_poly_coeffs_c, best_order_c,
                         "Wavelength (μm)", "Trend coefficient c", "Trend Offset (c) Polynomial Fit",
                         f"{output_dir}/{planet_str}_trend_c_polyfit_nrs{nrs}_{low_bins}bins.png")
            plot_poly_fit(wl_lr, trend_v_lr, best_poly_coeffs_v, best_order_v,
                         "Wavelength (μm)", "Trend coefficient v", "Trend Slope (v) Polynomial Fit",
                         f"{output_dir}/{planet_str}_trend_v_polyfit_nrs{nrs}_{low_bins}bins.png")

        if interpolate_ld:
            print("Fitting polynomials to limb darkening coefficients...")
            best_poly_coeffs_u1, best_order_u1, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 0], poly_orders)
            best_poly_coeffs_u2, best_order_u2, _ = fit_polynomial(wl_lr, ld_u_lr[:, :, 1], poly_orders)
            print(f"Selected polynomial degrees: u1={best_order_u1}, u2={best_order_u2}")
            plot_poly_fit(wl_lr, ld_u_lr[:,:, 0], best_poly_coeffs_u1, best_order_u1,
                         "Wavelength (μm)", "LD coefficient u1", "Limb Darkening u1 Polynomial Fit",
                         f"{output_dir}/{planet_str}_u1_polyfit_nrs{nrs}_{low_bins}bins.png")
            plot_poly_fit(wl_lr, ld_u_lr[:,:, 1], best_poly_coeffs_u2, best_order_u2,
                         "Wavelength (μm)", "LD coefficient u2", "Limb Darkening u2 Polynomial Fit",
                         f"{output_dir}/{planet_str}_u2_polyfit_nrs{nrs}_{low_bins}bins.png")

        ##### MAKE SPECTROSCOPIC MASK HERE ####
    
        '''
        for i in range(num_lcs_lr):
            rors_i = map_params_lr['rors'][i]
            u_i = map_params_lr['u'][i]
            c_i = map_params_lr['c'][i]
            v_i = map_params_lr['v'][i]
            orbit = TransitOrbit(
                period=PERIOD_FIXED,
                duration=map_params_lr['duration'],
                impact_param=map_params_lr["b"],
                time_transit=map_params_lr["t0"],
                radius_ratio=rors_i,
            )
            lc_transit = limb_dark_light_curve(orbit, u_i)(t_lr)
            trend = c_i + v_i * (t_lr - jnp.min(t_lr))
            model = lc_transit + trend

            spectro_residual = indiv_y_lr[i] - model 
            spectro_residual_mad = 1.4826 * np.nanmedian(np.abs(spectro_residual - np.nanmedian(spectro_residual)))
            spectro_residual_mask = np.abs(spectro_residual - np.nanmedian(spectro_residual)) > 5 * spectro_residual_mad
            plt.scatter(t_lr[i], indiv_y_lr[i], c='k')
            plt.scatter(t_lr[~spectro_residual_mask][i], indiv_y_lr[~spectro_residual_mask][i], c='r')
            plt.savefig(f'spectro_curve_{i}.png')
            plt.show()
        '''

    # --- High-resolution Analysis ---
    print(f"\n--- Running High-Resolution Analysis (Binned at {high_bins} nm) ---")


    wavelengths_hr, t_hr, indiv_y_hr, yerr_hr = unpack_trace.unpack_lc(file_pattern, nrs, planet_str, output_dir, high_bins) 

    ##### APPLY OUTLIER MASK HERE ####
    #wavelengths_hr = wavelengths_hr[~wl_mad_mask]
    t_hr = t_hr[~wl_mad_mask]
    indiv_y_hr = indiv_y_hr[:, ~wl_mad_mask]
    yerr_hr = yerr_hr[:, ~wl_mad_mask]
    if need_lowres_analysis:
        valid = ~time_mask     
        t_hr       = t_hr[valid]
        indiv_y_hr = indiv_y_hr[:, valid]
        yerr_hr    = yerr_hr[:, valid]

    num_lcs_hr = yerr_hr.shape[0]
    print(f"High-res: {num_lcs_hr} wavelength bins.")

    DEPTHS_BASE_HR = jnp.full(num_lcs_hr, DEPTH_BASE)
    wl_hr = np.array(wavelengths_hr) 


    if interpolate_trend:
        c_interp_hr = np.polyval(best_poly_coeffs_c, wl_hr)
        v_interp_hr = np.polyval(best_poly_coeffs_v, wl_hr)
        trend_fixed_hr = {"c": jnp.array(c_interp_hr), "v": jnp.array(v_interp_hr)}
        print("Using interpolated trend parameters for high-res analysis.")
    else:
        trend_fixed_hr = None 
        print("Fitting trend parameters in high-res analysis.")


    if interpolate_ld:
        u1_interp_hr = np.polyval(best_poly_coeffs_u1, wl_hr)
        u2_interp_hr = np.polyval(best_poly_coeffs_u2, wl_hr)
        ld_fixed_hr = jnp.array(np.column_stack((u1_interp_hr, u2_interp_hr)))
        print("Using interpolated limb darkening parameters for high-res analysis.")
        U_mu_hr_init = ld_fixed_hr 
    else:
        ld_fixed_hr = None
        U_mu_hr_init = get_limb_darkening(sld, wavelengths_hr, num_lcs_hr)
        print("Fitting limb darkening parameters in high-res analysis (initialized from exotic_ld).")

    # Initial Parameters (High Res)
    init_params_hr = {
        "logD": jnp.log(DURATION_BASE), "t0": T0_BASE, "_b": B_BASE,
        "depths": DEPTHS_BASE_HR,
        "u": U_mu_hr_init,
        "c": jnp.ones(num_lcs_hr) if trend_fixed_hr is None else trend_fixed_hr["c"],
        "v": jnp.zeros(num_lcs_hr) if trend_fixed_hr is None else trend_fixed_hr["v"],
    }

    # --- Prior Predictive Chcecks  ---
    '''
    print("Generating prior predictive samples for corner plot...")
    n_prior_samples = 2000 # Number of samples to draw from the prior
    prior_predictive = numpyro.infer.Predictive(vectorized_model, num_samples=n_prior_samples)
    # Use the high-res setup, providing base values as means for the prior distributions
    # Pass None for y, as we only want samples from the priors, not conditioned on data
    prior_samples = prior_predictive(
        key_prior_pred, # Use a dedicated key
        t=t_hr,         # Need shape info from t
        yerr=yerr_hr,   # Need shape info from yerr
        y=None,         # Set y=None to sample from priors
        mu_duration=DURATION_BASE,
        mu_t0=T0_BASE,
        mu_depths=DEPTHS_BASE_HR, # Prior mean for depths
        PERIOD=PERIOD_FIXED,
        trend_fixed=trend_fixed_hr, # Reflect if trend is fixed in the actual run
        ld_fixed=ld_fixed_hr        # Reflect if LD is fixed in the actual run
    )

    print("Formatting samples for arviz...")
    # Reshape samples for arviz: (chains, draws, *shape) -> here 1 chain
    converted_prior_samples = {
        p: np.expand_dims(prior_samples[p], axis=0) for p in prior_samples if p != 'obs' # Exclude 'obs' samples
    }
    prior_samples_inf_data = az.from_dict(converted_prior_samples)

    print("Creating corner plot of priors...")
    # Select key parameters to plot
    # Note: 'rors' is deterministic, so it's derived. We sample 'depths'.
    # We sample '_b', derive 'b'.
    # We sample 'logD', derive 'duration'.
    plot_vars = ["t0", "duration", "b", "depths"] # Adjust as needed
    fig_corner = corner.corner(
        prior_samples_inf_data,
        var_names=plot_vars,
        # truths=[T0_BASE, DURATION_BASE, B_BASE, DEPTH_BASE], # Example truths (careful with dimensions if plotting vector params like depths)
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 10},
    )
    corner_plot_filename = f"corner_priors_toi1130b_nrs{nrs}.png"
    fig_corner.savefig(corner_plot_filename)
    print(f"Saved corner plot to {corner_plot_filename}")
    plt.close(fig_corner) # Close figure
    # plt.show() # Uncomment if you want interactive display
    '''


    # --- High-Res MAP ---
    if num_lcs_hr >= 25:
      map_soln_hr = jax.tree_map(lambda x: np.array(x), init_params_hr)
    else: 
      print("Computing MAP solution for high-resolution data...")
      map_soln_hr = fit_model_map(
          t_hr, yerr_hr, indiv_y_hr, init_params_hr,
          mu_duration=DURATION_BASE,
          mu_t0=T0_BASE,
          mu_depths=DEPTHS_BASE_HR,
          PERIOD=PERIOD_FIXED,
          key=key_map_hr,
          trend_fixed=trend_fixed_hr, # Pass fixed trend if calculated
          ld_fixed=ld_fixed_hr       # Pass fixed LD if calculated
      )
      print(jnp.shape(indiv_y_hr)[0] )
      if jnp.shape(indiv_y_hr)[0] <= 25:
        print("Plotting high-resolution MAP fits and residuals...")
        plot_map_fits(t_hr, indiv_y_hr, yerr_hr, wavelengths_hr, map_soln_hr,
                      {"period": PERIOD_FIXED},
                      f"{output_dir}/highres_MAP_fits_{planet_str}_nrs{nrs}_{high_bins}bins.png", ncols=5)
        plot_map_residuals(t_hr, indiv_y_hr, yerr_hr, wavelengths_hr, map_soln_hr,
                          {"period": PERIOD_FIXED},
                          f"{output_dir}/highres_MAP_residuals_{planet_str}_nrs{nrs}_{high_bins}bins.png", ncols=5)
    
    # --- High-Res MCMC  ---
    print("Running MCMC for high-resolution transmission spectrum...")

    init_params_mcmc_hr = map_soln_hr.copy()
    if '_b' not in init_params_mcmc_hr and 'b' in init_params_mcmc_hr:
         init_params_mcmc_hr['_b'] = init_params_mcmc_hr['b']
    if 'logD' not in init_params_mcmc_hr and 'duration' in init_params_mcmc_hr:
        init_params_mcmc_hr['logD'] = jnp.log(init_params_mcmc_hr['duration'])
    if 'depths' not in init_params_mcmc_hr and 'rors' in init_params_mcmc_hr:
        init_params_mcmc_hr['depths'] = np.array(init_params_mcmc_hr['rors'])**2


    # remove parameters that are fixed and shouldn't be initialized if they ended up in map_soln_hr
    if trend_fixed_hr is not None:
        init_params_mcmc_hr.pop('c', None)
        init_params_mcmc_hr.pop('v', None)
        init_params_mcmc_hr.pop('c_fixed', None) 
        init_params_mcmc_hr.pop('v_fixed', None)
    if ld_fixed_hr is not None:
        init_params_mcmc_hr.pop('u', None)
        init_params_mcmc_hr.pop('u_fixed', None)

    init_params_mcmc_hr.pop('duration', None)
    init_params_mcmc_hr.pop('b', None)
    init_params_mcmc_hr.pop('rors', None)

    samples_hr = get_samples(
        partial(vectorized_model,
                mu_duration=DURATION_BASE,
                mu_t0=T0_BASE,
                mu_depths=DEPTHS_BASE_HR,
                PERIOD=PERIOD_FIXED),
        key_mcmc_hr, t_hr, yerr_hr, indiv_y_hr, init_params_mcmc_hr, 
        trend_fixed=trend_fixed_hr, # Pass fixed trend if calculated
        ld_fixed=ld_fixed_hr       # Pass fixed LD if calculated
    )

    # Plot and Save Transmission Spectrum
    print("Plotting and saving final transmission spectrum...")
    plot_transmission_spectrum(wavelengths_hr, samples_hr["rors"],
                               f"{output_dir}/{planet_str}_NIRSPEC_SPECTRUM_nrs{nrs}_{high_bins}bins.png")
    save_results(wavelengths_hr, samples_hr, nrs, f"{output_dir}/{planet_str}_NIRSPEC_SPECTRUM_nrs{nrs}_{high_bins}bins.csv")

    if "u" in samples_hr:
        u_samples = np.array(samples_hr["u"]) # Shape: (n_samples, n_lcs, 2)

        # Calculate median and uncertainties for u1
        u1_median = np.nanmedian(u_samples[:, :, 0], axis=0)
        u1_16 = np.nanpercentile(u_samples[:, :, 0], 16, axis=0)
        u1_84 = np.nanpercentile(u_samples[:, :, 0], 84, axis=0)
        u1_err_low = u1_median - u1_16
        u1_err_high = u1_84 - u1_median
        u1_yerr = np.array([u1_err_low, u1_err_high]) # Format for errorbar

        # Calculate median and uncertainties for u2
        u2_median = np.nanmedian(u_samples[:, :, 1], axis=0)
        u2_16 = np.nanpercentile(u_samples[:, :, 1], 16, axis=0)
        u2_84 = np.nanpercentile(u_samples[:, :, 1], 84, axis=0)
        u2_err_low = u2_median - u2_16
        u2_err_high = u2_84 - u2_median
        u2_yerr = np.array([u2_err_low, u2_err_high]) # Format for errorbar

        # Plot u1 with uncertainties
        plt.figure(figsize=(10, 6)) # Optional: Adjust figure size
        plt.errorbar(wavelengths_hr, u1_median, yerr=u1_yerr, fmt='o', markersize=4,
                    capsize=3, elinewidth=1, markeredgewidth=1, mfc='w',mec='k', label='u1 Median ± 1σ')
        plt.xlabel('Wavelength (micron)')
        plt.ylabel('u1')
        plt.title(f'Limb Darkening Coefficient u1 vs Wavelength (NRS{nrs})')
        plt.tight_layout() # Optional: Adjust layout
        u1_save_path = f'{output_dir}/{planet_str}_NIRSPEC_u1_nrs{nrs}_{high_bins}bins.png'
        plt.savefig(u1_save_path)
        print(f"Saved u1 plot with uncertainties to {u1_save_path}")
        plt.close() # Close the figure to free memory

        # Plot u2 with uncertainties
        plt.figure(figsize=(10, 6)) # Optional: Adjust figure size
        plt.errorbar(wavelengths_hr, u2_median, yerr=u2_yerr, fmt='o', markersize=4,
                    capsize=3, elinewidth=1, markeredgewidth=1, mfc='w',mec='k', label='u2 Median ± 1σ')
        plt.xlabel('Wavelength (micron)')
        plt.ylabel('u2')
        plt.title(f'Limb Darkening Coefficient u2 vs Wavelength (NRS{nrs})')
        plt.tight_layout() # Optional: Adjust layout
        u2_save_path = f'{output_dir}/{planet_str}_NIRSPEC_u2_nrs{nrs}_{high_bins}bins.png'
        plt.savefig(u2_save_path)
        print(f"Saved u2 plot with uncertainties to {u2_save_path}")
        plt.close() # Close the figure

    else:
        print("LD coefficients were fixed—skipping u₁–u₂ plots.")

    oot_mask = (t_hr < T0_BASE - 1 * DURATION_BASE) | (t_hr > T0_BASE + 1 * DURATION_BASE)

    def calc_rms(y_bin):
        baseline = y_bin[oot_mask]
        return jnp.sqrt(jnp.mean(jnp.square(jnp.diff(baseline)))) / jnp.sqrt(2.0)

    rms_vals = jax.vmap(calc_rms)(indiv_y_hr)

    # Plot it
    plt.figure(figsize=(8,5))
    plt.scatter(wavelengths_hr, rms_vals, c='k')
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Baseline RMS")
    plt.title("Out‑of‑Transit RMS vs Wavelength")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{planet_str}_rms_vs_wavelength_nrs{nrs}_{high_bins}bins.png')
    plt.close()
    print(f"Saved RMS vs λ plot to {output_dir}/{planet_str}_rms_vs_wavelength_nrs{nrs}_{high_bins}bins.png")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    args = p.parse_args()
    main(args.config)