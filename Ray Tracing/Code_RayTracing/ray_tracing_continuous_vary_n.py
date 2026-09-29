import os
import sys

# Make shared modules importable no matter where the script is run from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _d in ("General",):
    _p = os.path.join(_REPO_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import warnings

from Kerr_Newman import refractive_index_kn_continuous, kerr_newman_radius, kerr_newman_photon_sphere, kerr_newman_discontinuity_radius
from Schwarzchild import refractive_index_schwarzschild
from geodesics import schwarzschild_geodesic_xy, kerr_newman_geodesic_xy


METRIC_TYPE = "kerr_newman"   # options: "schwarzschild", "kerr_newman"

B_FIXED = 5.409            # Impact parameter for base refractive index
R_HORIZON = 2.0            # Schwarzschild radius (event horizon)
R_START = 500                # Starting radius (outer boundary), must be very far to be in the asymptotic regime
R_PLOT_MAX = 6.0           # Maximum radius for plotting refractive index profile

KERR_A = 0              # Angular momentum parameter (a)
KERR_CHARGE = 1.0         # Charge parameter (Q)


STEP_SIZE = 0.05           # Step size in affine parameter (ds)
MAX_STEPS = 20000          # Maximum integration steps

DELTA_N_MIN = 0.0          # Minimum refractive index perturbation
DELTA_N_MAX = 0.5          # Maximum refractive index perturbation
N_DELTA_VALUES = 200       # Number of delta_n values to sample

IMPACT_PARAMS = [B_FIXED]      # List of impact parameters for ray tracing

B_SCAN_MIN = 1.0           # Minimum impact parameter for diagnostic scans
B_SCAN_MAX = 8.0           # Maximum impact parameter for diagnostic scans
N_B_SCAN = 40              # Number of impact parameters in scan

PLOT_XLIM = (-6, 6)        # X-axis limits for Cartesian plot
PLOT_YLIM = (-6, 6)        # Y-axis limits for Cartesian plot
PLOT_CARTESIAN = False     # Set True to also generate the standalone Cartesian plot


class OpticalBlackHoleDual:

    def __init__(self, b_fixed=B_FIXED, r_start=R_START, r_horizon=R_HORIZON):
        """Initialize the optical black hole."""
        self.b_fixed = b_fixed
        self.r_start = r_start
        self.r_horizon = r_horizon
        
        
    def compute_refractive_index(self, r):
        if METRIC_TYPE == "schwarzschild":
            return refractive_index_schwarzschild(
                r,
                b_inf=self.b_fixed,
                n_at_P0=1.0,
                P0=R_PLOT_MAX
            )

        elif METRIC_TYPE == "kerr_newman":
            kn_ell_sign = int(np.sign(self.b_fixed)) if self.b_fixed != 0 else +1
            return refractive_index_kn_continuous(
                r,
                a=KERR_A,
                rho_Q=KERR_CHARGE,
                b_inf=self.b_fixed,
                ell_sign=kn_ell_sign,
                P0=R_PLOT_MAX,
                n0=1.0
            )

        else:
            raise ValueError("Unknown METRIC_TYPE")
    
    
    
    def trace_ray_continuous(self, impact_param, delta_n=0.0, ds=STEP_SIZE, max_steps=MAX_STEPS):
        def grad_ln_n(x, y):
            r = np.sqrt(x**2 + y**2)
            if r < self.r_horizon:
                return 0.0, 0.0

            eps = 1e-5

            n_r = self.compute_refractive_index(r) + delta_n
            n_r_plus = self.compute_refractive_index(r + eps)
            n_r_minus = self.compute_refractive_index(r - eps)

            dn_dr = (n_r_plus - n_r_minus) / (2 * eps)

            factor = dn_dr / n_r

            return factor * (x / r), factor * (y / r)

        # Initial conditions
        x, y = -self.r_start, impact_param
        vx, vy = 1.0, 0.0

        x_path, y_path = [x], [y]
        hit_horizon = False

        for step in range(max_steps):
            r = np.sqrt(x**2 + y**2)

            if r < self.r_horizon:
                hit_horizon = True
                break

            if x > self.r_start:
                break

            # RK4 integration
            def step_func(x, y, vx, vy):
                ax, ay = grad_ln_n(x, y)
                return vx, vy, ax, ay

            k1_vx, k1_vy, k1_ax, k1_ay = step_func(x, y, vx, vy)
            k2_vx, k2_vy, k2_ax, k2_ay = step_func(
                x + 0.5*ds*k1_vx, y + 0.5*ds*k1_vy,
                vx + 0.5*ds*k1_ax, vy + 0.5*ds*k1_ay
            )
            k3_vx, k3_vy, k3_ax, k3_ay = step_func(
                x + 0.5*ds*k2_vx, y + 0.5*ds*k2_vy,
                vx + 0.5*ds*k2_ax, vy + 0.5*ds*k2_ay
            )
            k4_vx, k4_vy, k4_ax, k4_ay = step_func(
                x + ds*k3_vx, y + ds*k3_vy,
                vx + ds*k3_ax, vy + ds*k3_ay
            )

            vx += (ds/6.0) * (k1_ax + 2*k2_ax + 2*k3_ax + k4_ax)
            vy += (ds/6.0) * (k1_ay + 2*k2_ay + 2*k3_ay + k4_ay)

            # Normalize velocity
            v_norm = np.sqrt(vx**2 + vy**2)
            vx /= v_norm
            vy /= v_norm

            x += (ds/6.0) * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
            y += (ds/6.0) * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)

            x_path.append(x)
            y_path.append(y)
        

        return np.array(x_path), np.array(y_path), hit_horizon
    
    def trace_ray(self, impact_param, delta_n=0.0, ds=STEP_SIZE, max_steps=MAX_STEPS):
        return self.trace_ray_continuous(impact_param, delta_n, ds, max_steps)


def compute_angular_deviation(x_path, y_path):
    if len(x_path) < 2:
        return np.nan
    
    # Initial angle (coming from -x direction, so θ = π)
    theta_initial = np.pi
    
    # Final angle (from last two points)
    theta_final = np.arctan2(y_path[-1] - y_path[-2], x_path[-1] - x_path[-2])
    
    # Angular deviation
    delta_theta = abs(theta_final - theta_initial)
    
    return np.degrees(delta_theta)


def crosses_radius(r_array, r_val):
    return np.any((r_array[:-1] - r_val) * (r_array[1:] - r_val) <= 0)


def _interp_phi_at_r(r_query, r_path, phi_path):
    sort_idx = np.argsort(r_path)
    return np.interp(r_query, r_path[sort_idx], phi_path[sort_idx])


def _display_y_for_orbit(y_values, impact_param):
    return -y_values if impact_param < 0 else y_values


def _wrap_angle(delta_phi):
    return (delta_phi + np.pi) % (2.0 * np.pi) - np.pi


def _crosses_radius_grid(r_array, r_grid):
    if len(r_array) < 2:
        return np.zeros_like(r_grid, dtype=bool)

    seg_min = np.minimum(r_array[:-1], r_array[1:])
    seg_max = np.maximum(r_array[:-1], r_array[1:])
    return np.any((r_grid[:, None] >= seg_min) & (r_grid[:, None] <= seg_max), axis=1)


def compute_geodesic_error_reference(b_fixed, actual_event_horizon):
    """
    Geodesic in the same convention used by other_methods:
    rho starts at R_PLOT_MAX and phi starts from 0 at that radius.
    """
    if METRIC_TYPE == "schwarzschild":
        _, _, r_geo, phi_geo = schwarzschild_geodesic_xy(
            b_inf=b_fixed,
            P0=R_PLOT_MAX,
            P_end=R_HORIZON,
        )

    elif METRIC_TYPE == "kerr_newman":
        ell_sign = int(np.sign(b_fixed)) if b_fixed != 0 else +1
        geo_end = actual_event_horizon if actual_event_horizon is not None else R_HORIZON
        _, _, r_geo, phi_geo = kerr_newman_geodesic_xy(
            a=KERR_A,
            rho_Q=KERR_CHARGE,
            b_inf=b_fixed,
            ell_sign=ell_sign,
            P0=R_PLOT_MAX,
            P_end=geo_end,
        )
    else:
        raise ValueError("Unknown METRIC_TYPE")

    min_r_idx = np.argmin(r_geo)
    return r_geo[:min_r_idx + 1], phi_geo[:min_r_idx + 1]


def _ray_error_coordinates(x_path, y_path, r_outer=R_PLOT_MAX):
    """
    Convert a Cartesian continuous ray to the rho/phi convention used by
    other_methods.ray_trace: ingoing branch only, phi=0 at r_outer.
    """
    r_path = np.sqrt(x_path**2 + y_path**2)
    min_r_idx = np.argmin(r_path)

    r_ing = r_path[:min_r_idx + 1]
    theta_ing = np.unwrap(np.arctan2(y_path[:min_r_idx + 1], x_path[:min_r_idx + 1]))

    if len(r_ing) < 2 or r_ing.min() >= r_outer:
        return np.array([]), np.array([])

    sort_idx = np.argsort(r_ing)
    r_sorted = r_ing[sort_idx]
    theta_sorted = theta_ing[sort_idx]

    theta_outer = np.interp(r_outer, r_sorted, theta_sorted)
    keep = r_ing <= r_outer

    r_ray = np.concatenate(([r_outer], r_ing[keep]))
    phi_ray = np.concatenate(([0.0], theta_outer - theta_ing[keep]))

    order = np.argsort(r_ray)
    _, unique_idx = np.unique(r_ray[order], return_index=True)
    keep_order = order[unique_idx]

    return r_ray[keep_order], phi_ray[keep_order]


def _positive_error_scale(finite_errors, min_span=1.0, max_span=10.0, percentile=98.0):
    """
    Return a dynamic positive color scale for absolute angular errors.
    The upper limit follows the data, but never exceeds the requested cap.
    """
    if finite_errors.size == 0:
        return 0.0, min_span

    upper = np.nanpercentile(finite_errors, percentile)
    if not np.isfinite(upper) or upper == 0:
        upper = np.nanmax(finite_errors)

    if not np.isfinite(upper) or upper == 0:
        upper = min_span

    upper = min(max(upper, min_span), max_span)
    return 0.0, upper


def create_error_analysis_plot(obh, delta_n_range=(DELTA_N_MIN, DELTA_N_MAX),
                                n_delta_values=N_DELTA_VALUES, b_fixed=B_FIXED, actual_event_horizon=None):
    
    delta_n_values = np.linspace(delta_n_range[0], delta_n_range[1], n_delta_values)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colormap
    cmap = plt.cm.plasma_r
    norm = Normalize(vmin=delta_n_range[0], vmax=delta_n_range[1])
    
    if METRIC_TYPE == "schwarzschild":
        x_geo, y_geo, r_geo, phi_geo = schwarzschild_geodesic_xy(
            b_inf=b_fixed,
            P0=R_START,
            P_end=obh.r_horizon
        )

    elif METRIC_TYPE == "kerr_newman":
        # b_fixed > 0  →  ell_sign = +1  (co-rotating)
        # b_fixed < 0  →  ell_sign = -1  (counter-rotating)
        kn_ell_sign = int(np.sign(b_fixed)) if b_fixed != 0 else +1
        geo_end = actual_event_horizon if actual_event_horizon is not None else obh.r_horizon
        x_geo, y_geo, r_geo, phi_geo = kerr_newman_geodesic_xy(
            a=KERR_A,
            rho_Q=KERR_CHARGE,
            b_inf=b_fixed,
            ell_sign=kn_ell_sign,   # must match beam geometry, not np.sign(KERR_A)
            P0=R_START,
            P_end=geo_end
        )

    x_geo = -x_geo
    phi_geo = np.pi - phi_geo
    r_geo_error, phi_geo_error = compute_geodesic_error_reference(
        b_fixed,
        actual_event_horizon,
    )

    print(f"Tracing {n_delta_values} rays for error analysis...")
    traced_rays = []
    for delta_n in delta_n_values:
        x, y, hit_horizon = obh.trace_ray(b_fixed, delta_n=delta_n)
        traced_rays.append((delta_n, x, y, hit_horizon))
    
    ax1.plot(
        x_geo,
        _display_y_for_orbit(y_geo, b_fixed),
        'k--',
        linewidth=2,
        label='Geodesic (Δn=0)',
        alpha=0.7,
    )
    
    for delta_n, x, y, _ in traced_rays:
        color = cmap(norm(delta_n))
        ax1.plot(x, _display_y_for_orbit(y, b_fixed), color=color, alpha=0.6, linewidth=1.5)
    


    center = (0, 0)
    if METRIC_TYPE == 'schwarzschild':
        radius = 1.5 * 2
    elif METRIC_TYPE == 'kerr_newman':
        radius = kerr_newman_photon_sphere(np.abs(KERR_A), KERR_CHARGE, kn_ell_sign)
    circle = Circle(center, radius, fill=False, edgecolor='blue', linewidth=2, label='Photon Sphere')
    ax1.add_patch(circle)

    horizon_circle = Circle((0, 0), obh.r_horizon, color='black', zorder=10, label='Simulated BH')
    ax1.add_patch(horizon_circle)
    
    if actual_event_horizon is not None and actual_event_horizon < obh.r_horizon:
        inner_horizon = Circle(
            (0, 0), actual_event_horizon,
            color='gray', fill=False, linestyle='--', linewidth=1.5, zorder=11
        )
        ax1.add_patch(inner_horizon)

    ax1.set_xlim(PLOT_XLIM)
    ax1.set_ylim(PLOT_YLIM)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X / M', fontsize=12)
    ax1.set_ylabel('Y / M', fontsize=12)
    ax1.set_title('Ray Trajectories with δn Perturbations', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    

    r_grid = np.linspace(obh.r_horizon, R_PLOT_MAX, 300)
    heat_matrix = np.full((len(delta_n_values), len(r_grid)), np.nan)

    for i, (delta_n, x, y, _) in enumerate(traced_rays):
        r_ing, phi_ing = _ray_error_coordinates(x, y)

        if len(r_ing) < 2:
            continue

        r_common_min = max(r_geo_error.min(), r_ing.min())
        r_common_max = min(r_geo_error.max(), r_ing.max())

        if r_common_min >= r_common_max:
            continue

        common_mask = ((r_ing >= r_common_min)
                       & (r_ing <= r_common_max))
        r_sample = r_ing[common_mask]
        phi_ray_at_sample = phi_ing[common_mask]
        phi_geo_at_ray = np.interp(r_sample, r_geo_error[::-1], phi_geo_error[::-1])
        delta_phi_deg = np.abs(np.degrees(phi_ray_at_sample - phi_geo_at_ray))

        sort_idx = np.argsort(r_sample)
        heat_matrix[i, :] = np.interp(
            r_grid,
            r_sample[sort_idx],
            delta_phi_deg[sort_idx],
            left=np.nan,
            right=np.nan,
        )

    error_masked = np.ma.masked_invalid(heat_matrix)
    finite_errors = error_masked.compressed()
    error_lower, error_upper = _positive_error_scale(
        finite_errors,
        min_span=1.0,
        max_span=10.0,
    )
    error_levels = np.linspace(error_lower, error_upper, 16)
    error_ticks = np.linspace(error_lower, error_upper, 6)

    cmap_err = plt.cm.Purples.copy()
    cmap_err.set_bad(color='0.85')
    ax2.set_facecolor('0.85')

    pcm = ax2.contourf(
        r_grid,
        delta_n_values,
        np.ma.masked_invalid(np.clip(error_masked, error_lower, error_upper)),
        levels=error_levels,
        cmap=cmap_err,
        extend='max',
        corner_mask=False,
    )
    cbar2 = plt.colorbar(pcm, ax=ax2, ticks=error_ticks, format='%.3g')
    cbar2.set_label(r'$\Phi_{\rm ray} - \Phi_{\rm geo}$ (°)', fontsize=11)

    ax2.axvline(x=obh.r_horizon, color='white', linestyle='--', linewidth=1.2,
                alpha=0.8, label='Simulated BH')
    if actual_event_horizon is not None and actual_event_horizon < obh.r_horizon:
        ax2.axvline(x=actual_event_horizon, color='gray', linestyle='--', linewidth=1.2,
                    alpha=0.8, label='Event Horizon')

    ax2.set_xlabel('R / M', fontsize=12)
    ax2.set_ylabel('Δn', fontsize=12)
    ax2.set_title('Angular Deviation from Geodesic', fontsize=14)
    ax2.set_xlim([obh.r_horizon, R_PLOT_MAX])
    ax2.set_ylim([delta_n_range[0], delta_n_range[1]])
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    return fig


def create_cartesian_visualization(obh, delta_n_range=(DELTA_N_MIN, DELTA_N_MAX), 
                                   n_delta_values=N_DELTA_VALUES, 
                                   impact_params=None, actual_event_horizon=None):
    if impact_params is None:
        impact_params = IMPACT_PARAMS
    
    delta_n_values = np.linspace(delta_n_range[0], delta_n_range[1], n_delta_values)
    
    fig, ax_cart = plt.subplots(1, 1, figsize=(10, 10))
    
    cmap = plt.cm.plasma
    norm = plt.cm.colors.Normalize(vmin=delta_n_range[0], vmax=delta_n_range[1])
    
    for b in impact_params:
        for delta_n in delta_n_values:
            x, y, hit_horizon = obh.trace_ray(b, delta_n, max_steps=MAX_STEPS)
            if len(x) > 0:  
                color = cmap(norm(delta_n))
                ax_cart.plot(x, _display_y_for_orbit(y, b), color=color, alpha=0.6, linewidth=1.5)
    
    horizon_circle = Circle((0, 0), obh.r_horizon, color='black', zorder=10, label='Simulated BH')
    ax_cart.add_patch(horizon_circle)
    
    # Dotted physical event horizon inside disk (counter-rotating KN only).
    if actual_event_horizon is not None and actual_event_horizon < obh.r_horizon:
        inner_horizon = Circle(
            (0, 0), actual_event_horizon,
            color='gray', fill=False, linestyle='--', linewidth=1.5, zorder=11
        )
        ax_cart.add_patch(inner_horizon)
    
    center = (0, 0)
    if METRIC_TYPE == 'schwarzschild':
        radius = 1.5 * 2
    elif METRIC_TYPE == 'kerr_newman':

        kn_ell_sign = int(np.sign(obh.b_fixed)) if obh.b_fixed != 0 else +1
        radius = kerr_newman_photon_sphere(np.abs(KERR_A), KERR_CHARGE, kn_ell_sign)

    circle = Circle(center, radius, fill=False, edgecolor='blue', linewidth=2, label = 'Photon Sphere')

    ax_cart.add_patch(circle)    
    ax_cart.set_xlim(PLOT_XLIM)
    ax_cart.set_ylim(PLOT_YLIM)
    ax_cart.set_aspect('equal')
    ax_cart.set_xlabel('X / M', fontsize=14)
    ax_cart.set_ylabel('Y / M', fontsize=14)
    ax_cart.legend()
    
    method_name = "Continuous Gradient"
    ax_cart.set_title(f'Optical Black Hole: Ray Trajectories ({method_name})', fontsize=16)
    ax_cart.grid(True, alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_cart)
    cbar.set_label('Δn', fontsize=12)
    
    plt.tight_layout()
    return fig


def main():
    
    method_name = "Continuous Gradient"    
    if METRIC_TYPE == "schwarzschild":
        r_horizon = R_HORIZON
        actual_event_horizon = R_HORIZON
    elif METRIC_TYPE == "kerr_newman":
        actual_event_horizon = kerr_newman_radius(KERR_A, KERR_CHARGE)

        kn_ell_sign = int(np.sign(B_FIXED)) if B_FIXED != 0 else +1

        if kn_ell_sign == -1:
            P_star = kerr_newman_discontinuity_radius(KERR_A, KERR_CHARGE, B_FIXED)
            if P_star is not None and P_star > actual_event_horizon:
                # Discontinuity is outside the event horizon; shift simulation boundary.
                r_horizon = P_star + 0.1
                print(f"\n  Note: Counter-rotating KN – discontinuity detected at P_* = {P_star:.3f}")
                print(f"        Setting simulated black hole radius to {r_horizon:.3f}")
                print(f"        Actual event horizon at {actual_event_horizon:.3f} (gray dotted ring inside disk)")
            else:
                r_horizon = actual_event_horizon
                if P_star is not None:
                    print(f"\n  Note: Counter-rotating KN – P_* = {P_star:.3f} is inside event horizon")
                else:
                    print(f"\n  Note: Counter-rotating KN – no discontinuity (discriminant negative)")
        else:
            r_horizon = actual_event_horizon
            print(f"\n  Note: Co-rotating KN – r_sim = r_horizon = {actual_event_horizon:.3f}")
    
    obh = OpticalBlackHoleDual(
        b_fixed=B_FIXED,
        r_start=R_START,
        r_horizon=r_horizon,)
    
    #parameters recap
    print("\nSimulation parameters:")
    print(f"  Solver method: {method_name}")
    print(f"  b_fixed: {B_FIXED}")
    print(f"  Outer radius: {R_START}")
    if METRIC_TYPE == "kerr_newman" and actual_event_horizon < r_horizon:
        print(f"  Simulated BH radius: {r_horizon}")
        print(f"  Actual event horizon: {actual_event_horizon}")
    else:
        print(f"  Horizon radius: {r_horizon}")
    print(f"  Impact parameters: {IMPACT_PARAMS}")
    print(f"  Delta_n range: [{DELTA_N_MIN}, {DELTA_N_MAX}]")
    print(f"  Number of delta_n values: {N_DELTA_VALUES}")
    
    fig_error = create_error_analysis_plot(
        obh,
        delta_n_range=(DELTA_N_MIN, DELTA_N_MAX),
        n_delta_values=N_DELTA_VALUES,
        b_fixed=B_FIXED,
        actual_event_horizon=actual_event_horizon if METRIC_TYPE == "kerr_newman" else None
    )
    filename ='dual_solver_error_analysis_continuous.png'
    fig_error.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    
    if PLOT_CARTESIAN:
        fig2 = create_cartesian_visualization(
            obh,
            delta_n_range=(DELTA_N_MIN, DELTA_N_MAX),
            n_delta_values=N_DELTA_VALUES,
            impact_params=IMPACT_PARAMS,
            actual_event_horizon=actual_event_horizon if METRIC_TYPE == "kerr_newman" else None
        )

        filename2 ='dual_solver_cartesian_continuous.png'
        fig2.savefig(filename2, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename2}")
    else:
        print("\nSkipping Cartesian visualization (PLOT_CARTESIAN = False)")
    
    
    plt.show()


if __name__ == "__main__":
    main()
