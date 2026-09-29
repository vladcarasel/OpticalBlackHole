from __future__ import annotations

import os
import sys

# Make shared modules importable no matter where the script is run from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _d in ("General",):
    _p = os.path.join(_REPO_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap

from geodesics import schwarzschild_geodesic
from ray_tracing_snell import ray_trace, ray_trace_with_outgoing
from annuli import annulus_edges_with_half_ends, sample_piecewise_constant
from Schwarzchild import refractive_index_schwarzschild
from constants import P0


def _build_uniform_annuli(b_inf, P_min, P_max, n_annuli, n_at_P0=1.0):
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    _, n_values = sample_piecewise_constant(
        refractive_index_schwarzschild, edges, b_inf, n_at_P0, P_max
    )
    return edges, n_values#builds the annulus edges and the corresponding refractive index values at the centers of the annuli, sampling n at the centers of the annuli, with endpoint overrides for the outermost and innermost annuli.


def _build_uniform_annuli_inneredge(b_inf, P_min, P_max, n_annuli, n_at_P0=1.0):
    """
    Same annulus geometry as _build_uniform_annuli, but each annulus samples
    n at its INNER edge instead of at its centre (with endpoint overrides).
    Uses the same n(P0)=1 normalisation as _build_uniform_annuli
    """
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    n_values = refractive_index_schwarzschild(
        edges[1:], b_inf, n_at_P0=n_at_P0, P0=P_max#we sample n at the inner edge of each annulus, which corresponds to edges[1:] since edges are ordered from outer to inner.
    )
    return edges, n_values


def _b_hat_at_P0(b_inf, P0_val):
    if np.isclose(b_inf, 0.0):
        return 0.0
    return 1.0 / np.sqrt(b_inf ** (-2) + 2.0 * P0_val ** (-3))


def _phi_offset_for_entry(B0, P0_val):
    ratio = np.clip(B0 / P0_val, -1.0, 1.0)
    return np.arcsin(ratio)#we calculate the initial phi offset for the ray to enter the system with the specified impact parameter B0 at radius P0_val.
#due to rotational symmetry, adding an offset at entry amounts to rotating my whole trajectory by this angle.
#in ray tracing, we put phi=0 in the first annulus. but for plot we want the ray to enter the system at y=B0, for this we need to add angle offset of arcsin(B0/P0).
def _make_symmetric_plasma():#this is just the colormap used for ploting
    n_half = 128
    c1 = plt.cm.plasma(np.linspace(0.0, 1.0, n_half))
    c2 = plt.cm.plasma(np.linspace(1.0, 0.0, n_half))
    return LinearSegmentedColormap.from_list('sym_plasma', np.vstack([c1, c2]))


def _add_annulus_circles(ax, edges, color='0.72', lw=0.5, alpha=0.9):
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    for e in edges:
        ax.plot(e * np.cos(theta), e * np.sin(theta),
                color=color, lw=lw, alpha=alpha, zorder=0)#we plot the annulus edges as circles on the given axes


def _interp_phi_on_radius(rho_query, rho_geo, phi_geo):
    return np.interp(rho_query, rho_geo[::-1], phi_geo[::-1])#we interpolate the geodesic's phi values at the given query radius rho_query.




def _n_paper(P, b_inf):

    P = np.asarray(P, dtype=float)
    return np.sqrt(1.0 + 2.0 * b_inf ** 2 / P ** 3)


def _sample_inner_edge(edges, b_inf):
    
    #Sample n at the inner edge of every annulus: n_i = n(R_i).

    
    edges = np.asarray(edges, dtype=float)
    return _n_paper(edges[1:], b_inf)


def _sample_centre_with_ends(edges, b_inf):

    edges = np.asarray(edges, dtype=float)
    N = len(edges) - 1
    if N == 1:
        return _n_paper(edges[-1:], b_inf)
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_vals = _n_paper(centres, b_inf)
    n_vals[0]  = _n_paper(np.array([edges[0]]),  b_inf)[0]   # P_max
    n_vals[-1] = _n_paper(np.array([edges[-1]]), b_inf)[0]   # P_min
    return n_vals


def make_figure_5(sampler=_sample_inner_edge,
                  out_filename='results/ray_tracing/figure5_annulus_number.png',
                  compare_at_turn_radius=False):

    P_min = 2.0
    b_inf_range = np.linspace(0.0, 5.0, 50)
    n_annuli_range = np.arange(1, 51)

    delta_phi = np.full((len(b_inf_range), len(n_annuli_range)), np.nan)

    for ib, b_inf in enumerate(b_inf_range):
        # b_inf = 0 is a pure radial ray: phi is identically 0 along ray and
        # geodesic, so the deviation is 0 and the recursion is degenerate.
        if np.isclose(b_inf, 0.0):
            delta_phi[ib, :] = 0.0
            continue

        rho_geo, phi_geo = schwarzschild_geodesic(b_inf, P0, P_end=P_min + 1e-3)
        if len(rho_geo) < 2:
            continue
        phi_geo_h = _interp_phi_on_radius(P_min, rho_geo, phi_geo)

        # Paper convention: ray enters from vacuum (n=1) with impact parameter b_inf.
        B_outside = b_inf

        for ia, n_ann in enumerate(n_annuli_range):
            try:
                edges = annulus_edges_with_half_ends(P_min, P0, n_ann)
                n_vals = sampler(edges, b_inf)

                # ray_trace hardcodes const_nb = n_values[0] * B0.  To make the
                # conserved Snell invariant equal to 1 * B_outside = b_inf, we
                # feed it an effective B0 such that n_values[0] * B0_eff = b_inf.
                B0_eff = B_outside / n_vals[0]#we find B0 the impact parameter inside the first annulus.
                #we use this B0_eff inside the ray tracing algorithm

                rho_ray, phi_ray, status = ray_trace(
                    edges, n_vals, B0_eff, return_status=True
                )

                if status["reached_inner"]:
                    delta_phi[ib, ia] = np.abs(
                        np.degrees(phi_ray[-1] - phi_geo_h)#if it reached inner we store the value of the deviation at the horizon, in degrees.
                    )
                elif compare_at_turn_radius and status["turn_radius"] is not None:
                    # Ray turned at rho_turn > P_min.  Compare its azimuth
                    # (phi_ray[-1] is the phi at the turning point, since the
                    # ingoing-only ray_trace stops there) against the geodesic
                    # evaluated at the SAME radius.
                    rho_turn = status["turn_radius"]
                    if rho_turn >= rho_geo.min():
                        phi_geo_turn = _interp_phi_on_radius(
                            rho_turn, rho_geo, phi_geo
                        )
                        delta_phi[ib, ia] = np.abs(
                            np.degrees(phi_ray[-1] - phi_geo_turn)#if the ray turns before reaching the horizon, we compare its phi at the turning radius against the geodesic's phi at that same radius
                        )
                    else:
                        delta_phi[ib, ia] = np.nan#if the turning point is at a radius smaller than the smallest radius reached by the geodesic, we cannot compare and we leave the cell NaN
                else:
                    delta_phi[ib, ia] = np.nan#if the ray didn't reach the inner radius and we are not comparing at the turn radius, we leave the cell NaN
            except Exception:
                delta_phi[ib, ia] = np.nan#if our ray tunrs at some radius larger than P_min, we leave the cell NaN, which will be plotted as white, matching the paper's convention that the deviation at the horizon is undefined if the ray doesn't reach the horizon.

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    cmap = plt.cm.Purples.copy()
    cmap.set_bad(color='0.85')

    B, NA = np.meshgrid(b_inf_range, n_annuli_range, indexing='ij')
    im = ax.contourf(
        B,
        NA,
        np.ma.masked_invalid(np.clip(delta_phi, 0.0, 30.0)),
        levels=np.arange(0.0, 30.01, 3.0),
        cmap=cmap,
        extend='max',
    )
    cb = fig.colorbar(im, ax=ax, ticks=[0, 6, 12, 18, 24, 30])
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}\ (^{\circ})$', fontsize=13)

    ax.set_xlabel(r'$\hat{b}_\infty$', fontsize=15)
    ax.set_ylabel('no. of annuli', fontsize=13)
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(1.0, 50.0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    fig.savefig(out_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_filename}")


def make_figure_6():

    b_inf_0 = 3.0
    P_min = 2.0
    n_annuli = 16

    B0_ref = _b_hat_at_P0(b_inf_0, P0)
    phi0_ref = _phi_offset_for_entry(B0_ref, P0)

    rho_geo, phi_geo = schwarzschild_geodesic(b_inf_0, P0, P_end=P_min + 1e-3)
    X_geo = rho_geo * np.cos(phi0_ref + phi_geo)
    Y_geo = rho_geo * np.sin(phi0_ref + phi_geo)

    edges_ref, n_vals_ref = _build_uniform_annuli(b_inf_0, P_min, P0, n_annuli)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    sym_cmap = _make_symmetric_plasma()
    plasma_cmap = plt.cm.plasma
    purples = plt.cm.Purples.copy()
    purples.set_bad(color='0.85')
    puor= plt.cm.PuOr.copy()
    puor.set_bad(color='0.85')

    ax = axes[0, 0]
    _add_annulus_circles(ax, edges_ref)

    dn_values = np.linspace(0.0, 0.5, 50)
    norm_dn = Normalize(vmin=0.0, vmax=0.5)

    for dn in reversed(dn_values):
        n_shifted = n_vals_ref + dn
        try:
            rho_r, phi_r = ray_trace(edges_ref, n_shifted, B0_ref)
            X_r = rho_r * np.cos(phi0_ref + phi_r)
            Y_r = rho_r * np.sin(phi0_ref + phi_r)
            ax.plot(X_r, Y_r, color=plasma_cmap(norm_dn(dn)),
                    lw=1.5, solid_capstyle='round', zorder=2)
        except Exception:
            pass

    ax.plot(X_geo, Y_geo, '--', color='0.35', lw=2.0, alpha=0.95, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=plasma_cmap, norm=norm_dn)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\Delta n$', fontsize=13)
    ax.set(xlabel=r'$X/M$', ylabel=r'$Y/M$', xlim=(-2, 6), ylim=(0, 6), aspect='equal')
    ax.text(-0.14, 1.02, 'a', transform=ax.transAxes, fontsize=18, fontweight='bold')

    ax = axes[0, 1]
    dn_fine = np.linspace(0.0, 0.5, 80)
    rho_eval = np.linspace(P_min, P0, 200)
    DPhi_b = np.full((len(dn_fine), len(rho_eval)), np.nan)

    for i, dn in enumerate(dn_fine):
        n_shifted = n_vals_ref + dn
        try:
            rho_ray, phi_ray, status = ray_trace(edges_ref, n_shifted, B0_ref, return_status=True)
            phi_geo_i = _interp_phi_on_radius(rho_ray, rho_geo, phi_geo)
            dphi = np.abs(np.degrees(phi_ray - phi_geo_i))
            s = np.argsort(rho_ray)
            DPhi_b[i, :] = np.interp(rho_eval, rho_ray[s], dphi[s], left=np.nan, right=np.nan)
        except Exception:
            pass

    RR, DN = np.meshgrid(rho_eval, dn_fine)
    im = ax.contourf(
        RR,
        DN,
        np.ma.masked_invalid(np.clip(DPhi_b, 0, 10)),
        levels=np.linspace(0, 10, 11),
        cmap=purples,
        extend='max',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 5, 10])
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}\ (^{\circ})$', fontsize=13)
    ax.set(xlabel=r'$R/M$', ylabel=r'$\Delta n$', xlim=(P_min, P0), ylim=(0, 0.5))
    ax.text(-0.14, 1.02, 'b', transform=ax.transAxes, fontsize=18, fontweight='bold')

    ax = axes[1, 0]
    _add_annulus_circles(ax, edges_ref)

    db0_ratios = np.linspace(-0.1, 0.1, 51)
    norm_db = Normalize(vmin=-0.1, vmax=0.1)

    for db_ratio in db0_ratios:
        B0 = B0_ref * (1.0 + db_ratio)
        phi0 = _phi_offset_for_entry(B0, P0)
        try:
            rho_r, phi_r = ray_trace(edges_ref, n_vals_ref, B0)
            X_r = rho_r * np.cos(phi_r + phi0)
            Y_r = rho_r * np.sin(phi_r + phi0)
            ax.plot(X_r, Y_r, color=sym_cmap(norm_db(db_ratio)),
                    lw=1.5, solid_capstyle='round', zorder=2)
        except Exception:
            pass

    ax.plot(X_geo, Y_geo, '--', color='0.35', lw=2.0, alpha=0.95, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=sym_cmap, norm=norm_db)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\Delta B_0 / B_0$', fontsize=13)
    ax.set(xlabel=r'$X/M$', ylabel=r'$Y/M$', xlim=(-2, 6), ylim=(0, 6), aspect='equal')
    ax.text(-0.14, 1.02, 'c', transform=ax.transAxes, fontsize=18, fontweight='bold')

    ax = axes[1, 1]
    db0_fine = np.linspace(-0.1, 0.1, 80)
    DPhi_d = np.full((len(db0_fine), len(rho_eval)), np.nan)

    for i, db_ratio in enumerate(db0_fine):
        B0 = B0_ref * (1.0 + db_ratio)
        try:
            rho_ray, phi_ray, status = ray_trace(edges_ref, n_vals_ref, B0, return_status=True)
            phi_geo_i = _interp_phi_on_radius(rho_ray, rho_geo, phi_geo)
            dphi = np.degrees(phi_ray - phi_geo_i)
            s = np.argsort(rho_ray)
            DPhi_d[i, :] = np.interp(rho_eval, rho_ray[s], dphi[s], left=np.nan, right=np.nan)
        except Exception:
            pass

    RR, DB = np.meshgrid(rho_eval, db0_fine)
    im = ax.contourf(
        RR,
        DB,
        np.ma.masked_invalid(np.clip(DPhi_d, -30, 30)),
        levels=np.linspace(-30, 30, 13),
        cmap=puor,
        extend='both',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[-30, -20, -10, 0, 10, 20, 30])
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}\ (^{\circ})$', fontsize=13)
    ax.set(xlabel=r'$R/M$', ylabel=r'$\Delta B_0 / B_0$', xlim=(P_min, P0), ylim=(-0.1, 0.1))
    ax.text(-0.14, 1.02, 'd', transform=ax.transAxes, fontsize=18, fontweight='bold')

    fig.tight_layout()
    os.makedirs('results/ray_tracing', exist_ok=True)
    fig.savefig('results/ray_tracing/figure6_error_analysis.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved results/ray_tracing/figure6_error_analysis.png")


def make_figure_6_full(builder=_build_uniform_annuli,
                       out_filename='results/ray_tracing/figure6_error_analysis_full.png'):
    #Pass _build_uniform_annuli_inneredge for inner-edge sampling.
    print("=" * 60)
    print("Generating Figure 6 (with outgoing branch)")
    print("=" * 60)

    b_inf_0 = 4.0
    P_min = 2.0
    n_annuli = 16

    B0_ref = _b_hat_at_P0(b_inf_0, P0)
    phi0_ref = _phi_offset_for_entry(B0_ref, P0)

    rho_geo, phi_geo = schwarzschild_geodesic(b_inf_0, P0, P_end=P_min + 1e-3)
    X_geo = rho_geo * np.cos(phi0_ref + phi_geo)
    Y_geo = rho_geo * np.sin(phi0_ref + phi_geo)

    edges_ref, n_vals_ref = builder(b_inf_0, P_min, P0, n_annuli)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 11.0))
    sym_cmap = _make_symmetric_plasma()
    plasma_cmap = plt.cm.plasma
    purples = plt.cm.Purples.copy()
    purples.set_bad(color='0.85')
    puor = plt.cm.PuOr.copy()
    puor.set_bad(color='0.85')

    
    ax = axes[0, 0]
    _add_annulus_circles(ax, edges_ref)

    dn_values = np.linspace(0.0, 0.5, 50)
    norm_dn = Normalize(vmin=0.0, vmax=0.5)

    for dn in reversed(dn_values):
        n_shifted = n_vals_ref + dn
        try:
            rho_r, phi_r = ray_trace_with_outgoing(edges_ref, n_shifted, B0_ref)
            X_r = rho_r * np.cos(phi0_ref + phi_r)
            Y_r = rho_r * np.sin(phi0_ref + phi_r)
            ax.plot(X_r, Y_r, color=plasma_cmap(norm_dn(dn)),
                    lw=1.5, solid_capstyle='round', zorder=2)
        except Exception:
            pass

    ax.plot(X_geo, Y_geo, '--', color='0.35', lw=2.0, alpha=0.95, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=plasma_cmap, norm=norm_dn)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\Delta n$', fontsize=13)
    ax.set(xlabel=r'$X/M$', ylabel=r'$Y/M$',
           xlim=(-P0, P0), ylim=(-P0, P0), aspect='equal')
    ax.text(-0.14, 1.02, 'a', transform=ax.transAxes, fontsize=18, fontweight='bold')

    ax = axes[0, 1]
    dn_fine = np.linspace(0.0, 0.5, 80)
    rho_eval = np.linspace(P_min, P0, 200)
    DPhi_b = np.full((len(dn_fine), len(rho_eval)), np.nan)

    for i, dn in enumerate(dn_fine):
        n_shifted = n_vals_ref + dn
        try:
            rho_ray, phi_ray, status = ray_trace(
                edges_ref, n_shifted, B0_ref, return_status=True
            )
            phi_geo_i = _interp_phi_on_radius(rho_ray, rho_geo, phi_geo)
            dphi = np.abs(np.degrees(phi_ray - phi_geo_i))
            s = np.argsort(rho_ray)
            DPhi_b[i, :] = np.interp(rho_eval, rho_ray[s], dphi[s],
                                     left=np.nan, right=np.nan)
        except Exception:
            pass

    RR, DN = np.meshgrid(rho_eval, dn_fine)
    im = ax.contourf(
        RR, DN,
        np.ma.masked_invalid(np.clip(DPhi_b, 0, 10)),
        levels=np.linspace(0, 10, 11),
        cmap=purples, extend='max',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 5, 10])
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}\ (^{\circ})$', fontsize=13)
    ax.set(xlabel=r'$R/M$', ylabel=r'$\Delta n$',
           xlim=(P_min, P0), ylim=(0, 0.5))
    ax.text(-0.14, 1.02, 'b', transform=ax.transAxes, fontsize=18, fontweight='bold')

    
    ax = axes[1, 0]
    _add_annulus_circles(ax, edges_ref)

    db0_ratios = np.linspace(-0.1, 0.1, 51)
    norm_db = Normalize(vmin=-0.1, vmax=0.1)

    for db_ratio in db0_ratios:
        B0 = B0_ref * (1.0 + db_ratio)
        phi0 = _phi_offset_for_entry(B0, P0)
        try:
            rho_r, phi_r = ray_trace_with_outgoing(edges_ref, n_vals_ref, B0)
            X_r = rho_r * np.cos(phi_r + phi0)
            Y_r = rho_r * np.sin(phi_r + phi0)
            ax.plot(X_r, Y_r, color=sym_cmap(norm_db(db_ratio)),
                    lw=1.5, solid_capstyle='round', zorder=2)
        except Exception:
            pass

    ax.plot(X_geo, Y_geo, '--', color='0.35', lw=2.0, alpha=0.95, zorder=4)

    sm = plt.cm.ScalarMappable(cmap=sym_cmap, norm=norm_db)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$\Delta B_0 / B_0$', fontsize=13)
    ax.set(xlabel=r'$X/M$', ylabel=r'$Y/M$',
           xlim=(-P0, P0), ylim=(-P0, P0), aspect='equal')
    ax.text(-0.14, 1.02, 'c', transform=ax.transAxes, fontsize=18, fontweight='bold')

    
    ax = axes[1, 1]
    db0_fine = np.linspace(-0.1, 0.1, 80)
    DPhi_d = np.full((len(db0_fine), len(rho_eval)), np.nan)

    for i, db_ratio in enumerate(db0_fine):
        B0 = B0_ref * (1.0 + db_ratio)
        try:
            rho_ray, phi_ray, status = ray_trace(
                edges_ref, n_vals_ref, B0, return_status=True
            )
            phi_geo_i = _interp_phi_on_radius(rho_ray, rho_geo, phi_geo)
            dphi = np.degrees(phi_ray - phi_geo_i)
            s = np.argsort(rho_ray)
            DPhi_d[i, :] = np.interp(rho_eval, rho_ray[s], dphi[s],
                                     left=np.nan, right=np.nan)
        except Exception:
            pass

    RR, DB = np.meshgrid(rho_eval, db0_fine)
    im = ax.contourf(
        RR, DB,
        np.ma.masked_invalid(np.clip(DPhi_d, -30, 30)),
        levels=np.linspace(-30, 30, 13),
        cmap=puor, extend='both',
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                      ticks=[-30, -20, -10, 0, 10, 20, 30])
    cb.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}\ (^{\circ})$', fontsize=13)
    ax.set(xlabel=r'$R/M$', ylabel=r'$\Delta B_0 / B_0$',
           xlim=(P_min, P0), ylim=(-0.1, 0.1))
    ax.text(-0.14, 1.02, 'd', transform=ax.transAxes, fontsize=18, fontweight='bold')

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    fig.savefig(out_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_filename}")


if __name__ == '__main__':
    make_figure_5()  # default: inner-edge sampling, NaN at turn
    make_figure_5(
        sampler=_sample_centre_with_ends,
        out_filename='results/ray_tracing/figure5_annulus_number_centre.png',
        compare_at_turn_radius=True,
    )
    make_figure_6()
    make_figure_6_full()  # default: centre+endpoints sampler, full (outgoing) traces
    make_figure_6_full(
        builder=_build_uniform_annuli_inneredge,
        out_filename='results/ray_tracing/figure6_error_analysis_full_inneredge.png',
    )