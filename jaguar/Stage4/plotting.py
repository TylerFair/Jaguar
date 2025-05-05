import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.transit import TransitOrbit
import jax.numpy as jnp

def plot_map_fits(t, indiv_y, yerr, wavelengths, map_params, transit_params, filename, ncols=3):
    """
    Plot the MAP fits for each wavelength. The transit parameters (period, duration, 
    impact parameter, and transit time) are provided via transit_params, and detrend_offset 
    is the value used to detrend the light curve.
    """
    nrows = int(np.ceil(len(wavelengths) / ncols))
    fig = plt.figure(figsize=(15, 3 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0, wspace=0)
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    colors = plt.cm.winter(norm(wavelengths))
    
    for i in range(len(wavelengths)):
        ax = plt.subplot(gs[i // ncols, i % ncols])
        # Extract MAP parameters for this wavelength
        rors_i = map_params['rors'][i]
        u_i = map_params['u'][i]
        c_i = map_params['c'][i]
        v_i = map_params['v'][i]
        
        orbit = TransitOrbit(
            period=transit_params["period"],
            duration=map_params["duration"],
            impact_param=map_params["b"],
            time_transit=map_params["t0"],
            radius_ratio=rors_i,
        )
        model = limb_dark_light_curve(orbit, u_i)(t)
        model = model + (c_i + v_i * (t - jnp.min(t)))
        
        ax.errorbar(t, indiv_y[i], yerr=yerr[i], fmt='.', alpha=0.3,
                    color=colors[i], label='Data', ms=1, zorder=2)
        ax.plot(t, model, c='k', alpha=1, lw=2.8,
                label='MAP Model', zorder=3)
        ax.text(0.05, 0.95, f'λ = {wavelengths[i]:.3f} μm',
                transform=ax.transAxes, fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename, dpi=200)
    return fig


def plot_map_residuals(t, indiv_y, yerr, wavelengths, map_params, transit_params, filename, ncols=3):
    """
    Plot the residuals for each wavelength using the transit parameters provided via transit_params.
    """
    nrows = int(np.ceil(len(wavelengths) / ncols))
    fig = plt.figure(figsize=(15, 3 * nrows))
    gs = gridspec.GridSpec(nrows, ncols, hspace=0, wspace=0)
    norm = plt.Normalize(wavelengths.min(), wavelengths.max())
    colors = plt.cm.winter(norm(wavelengths))
    
    for i in range(len(wavelengths)):
        ax = plt.subplot(gs[i // ncols, i % ncols])
        rors_i = map_params['rors'][i]
        u_i = map_params['u'][i]
        c_i = map_params['c'][i]
        v_i = map_params['v'][i]
        
        orbit = TransitOrbit(
            period=transit_params["period"],
            duration=map_params["duration"],
            impact_param=map_params["b"],
            time_transit=map_params["t0"],
            radius_ratio=rors_i,
        )
        model = limb_dark_light_curve(orbit, u_i)(t)
        model = model + (c_i + v_i * (t - jnp.min(t)))
        residuals = indiv_y[i] - model
        
        ax.errorbar(t, residuals, yerr=yerr[i], fmt='.', alpha=0.3,
                    ms=1, color=colors[i])
        ax.axhline(y=0, color='k', alpha=1, lw=2.8, zorder=3)
        ax.text(0.05, 0.95, f'λ = {wavelengths[i]:.3f} μm',
                transform=ax.transAxes, fontsize=10)
        rms = np.std(residuals)
        ax.text(0.05, 0.85, f'RMS = {rms:.6f}',
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(filename, dpi=200)
    return fig


def plot_transmission_spectrum(wavelengths, rors_posterior, filename):
    """
    Plot the transmission spectrum from MCMC results.
    
    Parameters:
    -----------
    wavelengths : array-like
        The wavelengths at which the spectrum is measured
    rors_posterior : array-like
        The posterior samples for rors (radius ratio)
    """
    # Compute median and error bounds for rors
    rors_med = jnp.median(rors_posterior, axis=0)
    rors_low = jnp.percentile(rors_posterior, 16, axis=0)
    rors_high = jnp.percentile(rors_posterior, 84, axis=0)

    rors_err = (rors_high - rors_low) / 2.0

    depth_median = rors_med**2
    depth_err = 2 * rors_med * rors_err
    fig = plt.figure(figsize=(10, 8))
    

   
    # Just plot the transmission spectrum
    plt.errorbar(wavelengths, depth_median, 
                    yerr=depth_err,
                    fmt='o', color='orange', ecolor='gray', label='Vectorized MCMC')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("$(R_p / R_s)^2$")
    plt.legend()
    plt.savefig(filename, dpi=200)

    return fig
