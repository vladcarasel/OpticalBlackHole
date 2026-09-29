import os
import sys

# Make shared modules importable no matter where the script is run from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _d in ("General", "Ray Tracing/Code_RayTracing"):
    _p = os.path.join(_REPO_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fdfd import simulate_schwarzschild, simulate_kerr_newman
from geodesics import schwarzschild_geodesic_xy, kerr_newman_geodesic_xy
from ray_tracing_snell import build_schwarzschild_annuli, build_kn_annuli
from cases import SCHWARZSCHILD_CASES, KERR_NEWMAN_CASES



WAVELENGTH = 0.5       # um
M = 2.5                # um (so R_S = 5 um)
P0 = 6.0               # dimensionless outer radius
RESOLUTION = 10        # grid points per wavelength (paper uses ~25)
N_ANNULI_SCH = 16      # Schwarzschild: 16 annuli 
N_ANNULI_KN = 21       # Kerr-Newman: 21 annuli 
N_PML = 12             # PML cells per side

# Visual style
CMAP = 'magma'         
ANNULUS_COLOR = '#cccccc'
GEODESIC_COLOR = '#ffffff'
LABEL_COLOR = '#ffffff'
RH_COLOR = '#ff6699'   # horizon: distinct accent
RS_COLOR = '#66ccff'   # Schwarzschild radius: blue accent (always shown)
RMIN_COLOR = '#aaaaaa' # inner edge of simulated annular system
R0_COLOR = '#ffffff'   # outer edge of optical BH



def _circle(ax, R, color, lw=1.2, ls='-', alpha=0.9, label=None):
    theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(R * np.cos(theta), R * np.sin(theta),
            color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def _label_on_circle(ax, R, angle_deg, text, color=LABEL_COLOR,
                     fontsize=10, fontweight='bold', bbox_facecolor='black'):
    #Places a text label tangent to a circle of radius R at angle_deg.
    ang = np.deg2rad(angle_deg)
    text_kwargs = dict(
        color=color, fontsize=fontsize, fontweight=fontweight,
        ha='center', va='center',
    )
    if bbox_facecolor is not None:
        text_kwargs["bbox"] = dict(
            facecolor=bbox_facecolor, edgecolor='none',
            alpha=0.55, pad=1.5,
        )
    ax.text(R * np.cos(ang), R * np.sin(ang), text, **text_kwargs)


def _add_annulus_edges(ax, edges, M_scale, color=ANNULUS_COLOR,
                        lw=0.3, alpha=0.4):
    #Draw annulus edges as faint circles
    theta = np.linspace(0, 2 * np.pi, 300)
    for e in edges:
        r = e * M_scale
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color=color, lw=lw, alpha=alpha)


def _add_poynting_vectors(ax, fdfd, step=8, color=LABEL_COLOR):
    #Overlay Poynting vectors scaled by 1/R as in the paper
    S_x, S_y = fdfd.compute_poynting()

    x_sub = fdfd.x[::step]#sub sampling means we take every `step`-th point in the x and y directions to plot the Poynting vectors. This reduces the number of vectors plotted, making the visualization cleaner and less cluttered, while still capturing the overall flow of energy.
    y_sub = fdfd.y[::step]# we sub-sample the Poynting vector field for cleaner visualization, and we use the same step for both x and y to maintain the grid structure of the vectors.
    Sx_sub = S_x[::step, ::step]
    Sy_sub = S_y[::step, ::step]

    XX, YY = np.meshgrid(x_sub, y_sub, indexing='ij')
    R = np.sqrt(XX**2 + YY**2)

    Smag = np.sqrt(Sx_sub**2 + Sy_sub**2)#we compute the magnitude of the Poynting vector at the sub-sampled points
    scale = 1.0 / np.maximum(R, 0.5)
    Smag_scaled = Smag * scale

    thresh = 0.02 * np.max(Smag_scaled)
    mask = Smag_scaled > thresh

    if np.any(mask):
        ax.quiver(XX[mask], YY[mask],#quiver plots arrows with components Sx_sub and Sy_sub at the locations XX and YY, but only for the points where the scaled magnitude of the Poynting vector exceeds the threshold. The arrows are colored according to the specified color, with an alpha for transparency, and no automatic scaling (scale=None) since we manually scale the vectors by 1/R.
                  Sx_sub[mask] * scale[mask],
                  Sy_sub[mask] * scale[mask],
                  color=color, alpha=0.55, scale=None,
                  headwidth=3, headlength=4, width=0.002)


def _geodesic_rotated_xy(rho, phi, b_inf, P0, M):
    #Rotate the geodesic so it enters the BH at x = b_inf*M from below, matching the beam's impact parameter.
    
    if len(rho) < 2:
        return np.array([]), np.array([])

    # Entry point on outer circle: (b_inf, -sqrt(P0^2 - b_inf^2))
    if b_inf >= P0:
        phi_offset = -np.pi / 2
    else:
        phi_offset = np.arctan2(-np.sqrt(P0**2 - b_inf**2), b_inf)

    X_geo = rho * np.cos(phi + phi_offset) * M
    Y_geo = rho * np.sin(phi + phi_offset) * M

    # Prepend straight vertical line from bottom of plot to entry point
    entry_x = X_geo[0]
    entry_y = Y_geo[0]
    y_bottom = -P0 * M * 1.05
    X_line = np.array([entry_x, entry_x])
    Y_line = np.array([y_bottom, entry_y])

    X = np.concatenate([X_line, X_geo])
    Y = np.concatenate([Y_line, Y_geo])
    return X, Y

#Schwarzschild fdfd simulations

def make_figure_3():

    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                              facecolor='black')
    axes = axes.ravel()

    R_S = 2.0 * M
    R0_phys = P0 * M

    for idx, case in enumerate(SCHWARZSCHILD_CASES[:4]):
        ax = axes[idx]
        b_inf = case['b_inf']
        label = case.get('panel', chr(ord('a') + idx))

        print(f"\n--- Panel {label}: b_inf = {b_inf} ---")

        fdfd = simulate_schwarzschild(
            b_inf=b_inf, P_min=2.0, P0=P0,
            n_annuli=N_ANNULI_SCH, wavelength=WAVELENGTH, M=M,
            resolution=RESOLUTION, N_pml=N_PML
        )

        # Plot |E|/max|E|
        E_norm = np.abs(fdfd.E_z) / np.max(np.abs(fdfd.E_z))
        im = ax.pcolormesh(fdfd.x, fdfd.y, E_norm.T,
                           cmap=CMAP, vmin=0, vmax=0.5,
                           shading='auto', rasterized=True)

        # Annulus edges (faint)
        edges, _ = build_schwarzschild_annuli(b_inf, 2.0, P0, N_ANNULI_SCH)
        _add_annulus_edges(ax, edges, M)

        # Outer edge of optical BH
        _circle(ax, R0_phys, R0_COLOR, lw=1.5)
        # Schwarzschild radius (= horizon for Schwarzschild)
        _circle(ax, R_S, RS_COLOR, lw=1.6)
        _label_on_circle(ax, R_S, 30, r'$R_S$', color=RS_COLOR)

        # True geodesic
        _, _, rho_geo, phi_geo = schwarzschild_geodesic_xy(
            b_inf, P0, P_end=2.01)
        X_geo, Y_geo = _geodesic_rotated_xy(rho_geo, phi_geo, b_inf, P0, M)
        ax.plot(X_geo, Y_geo, color=GEODESIC_COLOR, lw=2.5, alpha=0.95)

        _add_poynting_vectors(ax, fdfd, step=12)

        # Panel labels
        ax.text(0.05, 0.92, f'{label}', transform=ax.transAxes,
                fontsize=18, fontweight='bold', color=LABEL_COLOR)
        ax.text(0.05, 0.85, rf'$\hat{{b}}_\infty = {b_inf:g}$',
                transform=ax.transAxes, fontsize=12, color=LABEL_COLOR)

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('μm', color=LABEL_COLOR)
        ax.set_ylabel('μm', color=LABEL_COLOR)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.tick_params(colors=LABEL_COLOR)
        for spine in ax.spines.values():
            spine.set_color(LABEL_COLOR)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$|E|/\max|E|$', color=LABEL_COLOR)
        cb.ax.tick_params(colors=LABEL_COLOR)
        cb.outline.set_edgecolor(LABEL_COLOR)

    fig.suptitle('Numerical simulations of optical Schwarzschild black holes',
                 fontsize=14, color=LABEL_COLOR, y=0.98)
    fig.tight_layout()

    os.makedirs('results/fdfd', exist_ok=True)
    out = 'results/fdfd/figure3_schwarzschild_fdfd.png'
    fig.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved {out}")

#Kerr-Newman FDFD simulations


def make_figure_4():
    print("=" * 60)
    print("Generating Figure 4: Kerr-Newman FDFD simulations")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12),
                              facecolor='black')
    axes = axes.ravel()

    R_S = 2.0 * M
    R0_phys = P0 * M

    for idx, case in enumerate(KERR_NEWMAN_CASES):
        ax = axes[idx]
        label = case['panel']
        a = case['a']
        rho_Q = case['rho_Q']
        ell_sign = case['ell_sign']
        b_inf = case['b_inf']
        P_min = case['P_min']

        print(f"\n--- Panel {label}: a={a}, rho_Q={rho_Q}, "
              f"b_inf={b_inf}, P_min={P_min:.3f} ---")

        fdfd = simulate_kerr_newman(
            a=a, rho_Q=rho_Q, b_inf=b_inf, ell_sign=ell_sign,
            P_min=P_min, P0=P0, n_annuli=N_ANNULI_KN,
            wavelength=WAVELENGTH, M=M, resolution=RESOLUTION,
            N_pml=N_PML
        )

        E_norm = np.abs(fdfd.E_z) / np.max(np.abs(fdfd.E_z))
        im = ax.pcolormesh(fdfd.x, fdfd.y, E_norm.T,
                           cmap=CMAP, vmin=0, vmax=0.5,
                           shading='auto', rasterized=True)

        edges, _ = build_kn_annuli(a, rho_Q, b_inf, ell_sign,
                                    P_min, P0, N_ANNULI_KN)
        _add_annulus_edges(ax, edges, M)

        # Outer edge of optical BH
        _circle(ax, R0_phys, R0_COLOR, lw=1.5)

        # Three radius circles, all with distinct styles & colors
        # 1) Schwarzschild radius R_S = 2M  (always drawn for comparison)
        _circle(ax, R_S, RS_COLOR, lw=1.4, ls='-')
        # 2) Outer event horizon R_h (if it exists, i.e. no naked singularity)
        disc = 1.0 - a**2 - rho_Q**2
        R_h = (1.0 + np.sqrt(disc)) * M if disc >= 0 else None
        if R_h is not None:
            _circle(ax, R_h, RH_COLOR, lw=1.4, ls='--')
        # 3) Inner edge of the SIMULATED annular system (where the absorbing core begins). Distinct from R_h whenever P_min > P_h.
        R_min = P_min * M
        _circle(ax, R_min, RMIN_COLOR, lw=1.0, ls=':')

        # Labels placed on (or near) the appropriate circles
        # Stagger label angles so they don't overlap.
        _label_on_circle(ax, R_S, 60, r'$R_S$', color=RS_COLOR)
        if R_h is not None and abs(R_h - R_S) > 0.4:
            _label_on_circle(ax, R_h, 120, r'$R_h$', color=RH_COLOR)
        elif R_h is not None:
            # When R_h is close to R_S (e.g. extremal RN), put R_h label below
            _label_on_circle(ax, R_h, -60, r'$R_h$', color=RH_COLOR)
        if abs(R_min - (R_h if R_h is not None else 0)) > 0.4 \
                and abs(R_min - R_S) > 0.4:
            _label_on_circle(ax, R_min, 200, r'$R_\min$',
                             color=RMIN_COLOR, fontsize=9)

        # True geodesic
        _, _, rho_geo, phi_geo = kerr_newman_geodesic_xy(
            a, rho_Q, b_inf, ell_sign, P0, P_min + 0.01
        )
        X_geo, Y_geo = _geodesic_rotated_xy(rho_geo, phi_geo, b_inf, P0, M)
        ax.plot(X_geo, Y_geo, color=GEODESIC_COLOR, lw=2.5, alpha=0.95)

        _add_poynting_vectors(ax, fdfd, step=12)

        # Panel label
        ax.text(0.05, 0.92, label, transform=ax.transAxes,
                fontsize=18, fontweight='bold', color=LABEL_COLOR)

        # Subtitle: BH parameters and impact parameter
        subtitle = (rf'$\hat{{a}}={a:+.2f}$, $\rho_Q={rho_Q:.2f}$, '
                    rf'$\hat{{b}}_\infty={b_inf:g}$')
        ax.text(0.05, 0.02, subtitle, transform=ax.transAxes,
                fontsize=10, color=LABEL_COLOR)

        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_xlabel('μm', color=LABEL_COLOR)
        ax.set_ylabel('μm', color=LABEL_COLOR)
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.tick_params(colors=LABEL_COLOR)
        for spine in ax.spines.values():
            spine.set_color(LABEL_COLOR)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$|E|/\max|E|$', color=LABEL_COLOR)
        cb.ax.tick_params(colors=LABEL_COLOR)
        cb.outline.set_edgecolor(LABEL_COLOR)

    fig.suptitle('Numerical simulations of optical Kerr–Newman black holes',
                 fontsize=14, color=LABEL_COLOR, y=0.98)
    fig.tight_layout()

    os.makedirs('results/fdfd', exist_ok=True)
    out = 'results/fdfd/figure4_kerr_newman_fdfd.png'
    fig.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved {out}")


# Phase/wavefront visualization (Re(E_z) and angle(E_z))


# Diverging colormap for Re(E_z): zero-centered, symmetric.
PHASE_CMAP_RE = 'RdBu_r'
# Cyclic colormap for angle(E_z): respects the 2*pi wraparound.
PHASE_CMAP_ANGLE = 'twilight'


def _plot_phase_panel(ax, fdfd, mode, vmax_re=0.3):
    
    if mode == 'real':
        field = np.real(fdfd.E_z) / np.max(np.abs(fdfd.E_z))
        im = ax.pcolormesh(
            fdfd.x, fdfd.y, field.T,
            cmap=PHASE_CMAP_RE,
            vmin=-vmax_re, vmax=vmax_re,
            shading='auto', rasterized=True,
        )
    elif mode == 'angle':
        # Mask points where |E_z| is essentially zero, since the phase
        # there is just numerical noise and clutters the plot.
        E = fdfd.E_z
        amp = np.abs(E)
        mask = amp < 1e-3 * np.max(amp)
        phase = np.angle(E)
        phase_masked = np.ma.array(phase, mask=mask)
        im = ax.pcolormesh(
            fdfd.x, fdfd.y, phase_masked.T,
            cmap=PHASE_CMAP_ANGLE,
            vmin=-np.pi, vmax=np.pi,
            shading='auto', rasterized=True,
        )
    else:
        raise ValueError(f"mode must be 'real' or 'angle', got {mode!r}")

    return im


def _decorate_phase_panel(ax, fdfd, label, subtitle, edges, M_scale,
                           geodesic_xy, R_circles):
    
    _add_annulus_edges(ax, edges, M_scale)

    for R, color, ls, lbl, ang in R_circles:
        _circle(ax, R, color, lw=1.4, ls=ls)
        if lbl is not None:
            _label_on_circle(ax, R, ang, lbl, color=color,
                             bbox_facecolor=None)

    if geodesic_xy is not None:
        X_geo, Y_geo = geodesic_xy
        # Use a darker line on the diverging/cyclic backgrounds so it stays
        # visible regardless of local field sign.
        ax.plot(X_geo, Y_geo, color='black', lw=2.6, alpha=0.85)
        ax.plot(X_geo, Y_geo, color='white', lw=1.0, alpha=0.9)

    ax.text(0.05, 0.92, label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.7, pad=2))
    if subtitle:
        ax.text(0.05, 0.02, subtitle, transform=ax.transAxes,
                fontsize=10, color='black',
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.7, pad=2))

    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_xlabel('μm')
    ax.set_ylabel('μm')
    ax.set_aspect('equal')


def _make_phase_figure(cases, kind, mode, outname, suptitle):

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), facecolor='white')
    axes = axes.ravel()

    R_S = 2.0 * M
    R0_phys = P0 * M

    for idx, case in enumerate(cases[:4]):
        ax = axes[idx]

        if kind == 'schwarzschild':
            b_inf = case['b_inf']
            label = case.get('panel', chr(ord('a') + idx))

            print(f"\n--- Panel {label}: b_inf = {b_inf} ---")

            fdfd = simulate_schwarzschild(
                b_inf=b_inf, P_min=2.0, P0=P0,
                n_annuli=N_ANNULI_SCH, wavelength=WAVELENGTH, M=M,
                resolution=RESOLUTION, N_pml=N_PML
            )
            edges, _ = build_schwarzschild_annuli(b_inf, 2.0, P0,
                                                   N_ANNULI_SCH)

            _, _, rho_geo, phi_geo = schwarzschild_geodesic_xy(
                b_inf, P0, P_end=2.01)
            geodesic_xy = _geodesic_rotated_xy(rho_geo, phi_geo,
                                                b_inf, P0, M)

            R_circles = [
                (R0_phys, '#444444', '-', None, 0),
                (R_S,     '#0066aa', '-', r'$R_S$', 30),
            ]
            subtitle = rf'$\hat{{b}}_\infty = {b_inf:g}$'

        elif kind == 'kn':
            label = case['panel']
            a = case['a']
            rho_Q = case['rho_Q']
            ell_sign = case['ell_sign']
            b_inf = case['b_inf']
            P_min = case['P_min']

            print(f"\n--- Panel {label}: a={a}, rho_Q={rho_Q}, "
                  f"b_inf={b_inf}, P_min={P_min:.3f} ---")

            fdfd = simulate_kerr_newman(
                a=a, rho_Q=rho_Q, b_inf=b_inf, ell_sign=ell_sign,
                P_min=P_min, P0=P0, n_annuli=N_ANNULI_KN,
                wavelength=WAVELENGTH, M=M, resolution=RESOLUTION,
                N_pml=N_PML
            )
            edges, _ = build_kn_annuli(a, rho_Q, b_inf, ell_sign,
                                        P_min, P0, N_ANNULI_KN)

            _, _, rho_geo, phi_geo = kerr_newman_geodesic_xy(
                a, rho_Q, b_inf, ell_sign, P0, P_min + 0.01)
            geodesic_xy = _geodesic_rotated_xy(rho_geo, phi_geo,
                                                b_inf, P0, M)

            disc = 1.0 - a**2 - rho_Q**2
            R_h = (1.0 + np.sqrt(disc)) * M if disc >= 0 else None
            R_min_phys = P_min * M

            R_circles = [(R0_phys, '#444444', '-', None, 0),
                         (R_S,     '#0066aa', '-', r'$R_S$', 60)]
            if R_h is not None:
                R_circles.append((R_h, '#cc3366', '--', r'$R_h$', 120))
            if abs(R_min_phys - (R_h or 0)) > 0.4:
                R_circles.append((R_min_phys, '#666666', ':',
                                   r'$R_\min$', 200))

            subtitle = (rf'$\hat{{a}}={a:+.2f}$, $\rho_Q={rho_Q:.2f}$, '
                        rf'$\hat{{b}}_\infty={b_inf:g}$')
        else:
            raise ValueError(f"unknown kind: {kind!r}")

        im = _plot_phase_panel(ax, fdfd, mode=mode)
        _decorate_phase_panel(ax, fdfd, label, subtitle, edges, M,
                                geodesic_xy, R_circles)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if mode == 'real':
            cb.set_label(r'$\mathrm{Re}(E_z)/\max|E_z|$')
        else:
            cb.set_label(r'$\arg(E_z)$ [rad]')
            cb.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
            cb.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0',
                                r'$\pi/2$', r'$\pi$'])

    fig.suptitle(suptitle, fontsize=14, y=0.98)
    fig.tight_layout()

    os.makedirs('results/fdfd', exist_ok=True)
    out = f'results/fdfd/{outname}'
    fig.savefig(out, dpi=200, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved {out}")


def make_figure_3_phase():
    #Re(E_z) and arg(E_z) for the Schwarzschild 
    _make_phase_figure(
        SCHWARZSCHILD_CASES, kind='schwarzschild', mode='real',
        outname='figure3_schwarzschild_ReEz.png',
        suptitle=r'Wavefronts $\mathrm{Re}(E_z)$ — Schwarzschild',
    )
    _make_phase_figure(
        SCHWARZSCHILD_CASES, kind='schwarzschild', mode='angle',
        outname='figure3_schwarzschild_argEz.png',
        suptitle=r'Phase $\arg(E_z)$ — Schwarzschild',
    )


def make_figure_4_phase():
    #Re(E_z) and arg(E_z) for Kerr–Newman 
    _make_phase_figure(
        KERR_NEWMAN_CASES, kind='kn', mode='real',
        outname='figure4_kerr_newman_ReEz.png',
        suptitle=r'Wavefronts $\mathrm{Re}(E_z)$ — Kerr–Newman',
    )
    _make_phase_figure(
        KERR_NEWMAN_CASES, kind='kn', mode='angle',
        outname='figure4_kerr_newman_argEz.png',
        suptitle=r'Phase $\arg(E_z)$ — Kerr–Newman',
    )
if __name__ == "__main__":
    make_figure_3()
    make_figure_4()
    make_figure_3_phase()
    make_figure_4_phase()
