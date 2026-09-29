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
import warnings
from matplotlib.colors import Normalize


from Kerr_Newman import (
    refractive_index_kn_continuous,
    kerr_newman_radius,
    kerr_newman_photon_sphere,
    kerr_newman_discontinuity_radius,
)
from Schwarzchild import refractive_index_schwarzschild
from geodesics import schwarzschild_geodesic_xy, kerr_newman_geodesic_xy


METRIC_TYPE = "kerr_newman"   # "schwarzschild" or "kerr_newman"

B_NOMINAL = 5.409            # Nominal (central) impact parameter
R_HORIZON = 2.0            # Schwarzschild event horizon radius (Schwarzschild mode only)
R_START   = 500.0          # Starting radius — must be far in the asymptotic regime
R_PLOT_MAX = 6.0           # Reference radius for refractive-index normalisation

KERR_A      = 0          # Dimensionless spin  (a)
KERR_CHARGE = 1.0          # Charge parameter    (Q)

STEP_SIZE = 0.05           # Affine-parameter step ds
MAX_STEPS = 20000          # Maximum RK4 steps

DELTA_B_RATIO_MIN = -0.1     # Minimum fractional delta_b / |b_ref|
DELTA_B_RATIO_MAX =  0.1     # Maximum fractional delta_b / |b_ref|
DELTA_B_MIN    = DELTA_B_RATIO_MIN * np.abs(B_NOMINAL)      # Minimum delta_b
DELTA_B_MAX    = DELTA_B_RATIO_MAX * np.abs(B_NOMINAL)      # Maximum delta_b
N_DELTA_VALUES =  200      # Number of delta_b values

PLOT_XLIM = (-6, 6)
PLOT_YLIM = (-6, 6)


def get_r_sim_and_horizon(b_actual):

    if METRIC_TYPE == "schwarzschild":
        return R_HORIZON, R_HORIZON

    elif METRIC_TYPE == "kerr_newman":
        actual_event_horizon = kerr_newman_radius(KERR_A, KERR_CHARGE)
        ell_sign = int(np.sign(b_actual)) if b_actual != 0 else +1

        if ell_sign == -1:          # counter-rotating
            P_star = kerr_newman_discontinuity_radius(KERR_A, KERR_CHARGE, b_actual)
            if P_star is not None and P_star > actual_event_horizon:
                r_sim = P_star + 0.1
                return r_sim, actual_event_horizon
            else:
                return actual_event_horizon, actual_event_horizon
        else:                       # co-rotating
            return actual_event_horizon, actual_event_horizon

    else:
        raise ValueError(f"Unknown METRIC_TYPE: {METRIC_TYPE!r}")


class OpticalBlackHoleVaryB:

    def __init__(self, r_start=R_START):
        self.r_start = r_start

    # ------------------------------------------------------------------
    def compute_refractive_index(self, r, b_actual):
        if METRIC_TYPE == "schwarzschild":
            return refractive_index_schwarzschild(
                r,
                b_inf=b_actual,
                n_at_P0=1.0,
                P0=R_PLOT_MAX,
            )

        elif METRIC_TYPE == "kerr_newman":
        
            ell_sign = int(np.sign(b_actual)) if b_actual != 0 else +1
            return refractive_index_kn_continuous(
                r,
                a=KERR_A,
                rho_Q=KERR_CHARGE,
                b_inf=b_actual,
                ell_sign=ell_sign,
                P0=R_PLOT_MAX,
                n0=1.0,
            )

        else:
            raise ValueError(f"Unknown METRIC_TYPE: {METRIC_TYPE!r}")

    # ------------------------------------------------------------------
    def trace_ray_continuous(self, b_actual, r_horizon, ds=STEP_SIZE, max_steps=MAX_STEPS):
    

        def grad_ln_n(x, y):
            r = np.sqrt(x**2 + y**2)
            if r < r_horizon:
                return 0.0, 0.0
            eps = 1e-5
            n_r      = self.compute_refractive_index(r, b_actual)
            n_r_plus = self.compute_refractive_index(r + eps, b_actual)
            n_r_minus= self.compute_refractive_index(r - eps, b_actual)
            dn_dr    = (n_r_plus - n_r_minus) / (2.0 * eps)
            factor   = dn_dr / n_r
            return factor * (x / r), factor * (y / r)

        x, y   = -self.r_start, b_actual
        vx, vy = 1.0, 0.0

        x_path, y_path = [x], [y]
        hit_horizon = False

        for _ in range(max_steps):
            r = np.sqrt(x**2 + y**2)

            if r < r_horizon:
                hit_horizon = True
                break
            if x > self.r_start:
                break

            def step_func(xi, yi, vxi, vyi):
                axi, ayi = grad_ln_n(xi, yi)
                return vxi, vyi, axi, ayi

            k1_vx, k1_vy, k1_ax, k1_ay = step_func(x, y, vx, vy)
            k2_vx, k2_vy, k2_ax, k2_ay = step_func(
                x + 0.5*ds*k1_vx, y + 0.5*ds*k1_vy,
                vx + 0.5*ds*k1_ax, vy + 0.5*ds*k1_ay,
            )
            k3_vx, k3_vy, k3_ax, k3_ay = step_func(
                x + 0.5*ds*k2_vx, y + 0.5*ds*k2_vy,
                vx + 0.5*ds*k2_ax, vy + 0.5*ds*k2_ay,
            )
            k4_vx, k4_vy, k4_ax, k4_ay = step_func(
                x + ds*k3_vx, y + ds*k3_vy,
                vx + ds*k3_ax, vy + ds*k3_ay,
            )

            vx += (ds / 6.0) * (k1_ax + 2*k2_ax + 2*k3_ax + k4_ax)
            vy += (ds / 6.0) * (k1_ay + 2*k2_ay + 2*k3_ay + k4_ay)

            v_norm = np.sqrt(vx**2 + vy**2)
            vx /= v_norm
            vy /= v_norm

            x += (ds / 6.0) * (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx)
            y += (ds / 6.0) * (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy)

            x_path.append(x)
            y_path.append(y)

        return np.array(x_path), np.array(y_path), hit_horizon


def compute_geodesic(b_nominal, actual_event_horizon):
    
    if METRIC_TYPE == "schwarzschild":
        x_geo, y_geo, r_geo, phi_geo = schwarzschild_geodesic_xy(
            b_inf=b_nominal,
            P0=R_START,
            P_end=R_HORIZON,
        )

    elif METRIC_TYPE == "kerr_newman":
        # ell_sign: same convention as compute_refractive_index
        ell_sign = int(np.sign(b_nominal)) if b_nominal != 0 else +1
        # End at the true event horizon for the geodesic integration
        x_geo, y_geo, r_geo, phi_geo = kerr_newman_geodesic_xy(
            a=KERR_A,
            rho_Q=KERR_CHARGE,
            b_inf=b_nominal,
            ell_sign=ell_sign,
            P0=R_START,
            P_end=actual_event_horizon,
        )
    else:
        raise ValueError(f"Unknown METRIC_TYPE: {METRIC_TYPE!r}")


    x_geo   = -x_geo
    phi_geo = np.pi - phi_geo

    return x_geo, y_geo, r_geo, phi_geo


def compute_geodesic_error_reference(b_nominal, actual_event_horizon):
    """
    Geodesic in the same convention used by other_methods:
    rho starts at R_PLOT_MAX and phi starts from 0 at that radius.
    """
    if METRIC_TYPE == "schwarzschild":
        _, _, r_geo, phi_geo = schwarzschild_geodesic_xy(
            b_inf=b_nominal,
            P0=R_PLOT_MAX,
            P_end=R_HORIZON,
        )

    elif METRIC_TYPE == "kerr_newman":
        ell_sign = int(np.sign(b_nominal)) if b_nominal != 0 else +1
        _, _, r_geo, phi_geo = kerr_newman_geodesic_xy(
            a=KERR_A,
            rho_Q=KERR_CHARGE,
            b_inf=b_nominal,
            ell_sign=ell_sign,
            P0=R_PLOT_MAX,
            P_end=actual_event_horizon,
        )
    else:
        raise ValueError(f"Unknown METRIC_TYPE: {METRIC_TYPE!r}")

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




def crosses_radius(r_array, r_val):
    """Return True if r_array crosses r_val at least once."""
    return np.any((r_array[:-1] - r_val) * (r_array[1:] - r_val) <= 0)


def _crosses_radius_grid(r_array, r_grid):
    """Vectorized version of crosses_radius for many query radii."""
    if len(r_array) < 2:
        return np.zeros_like(r_grid, dtype=bool)

    seg_min = np.minimum(r_array[:-1], r_array[1:])
    seg_max = np.maximum(r_array[:-1], r_array[1:])
    return np.any((r_grid[:, None] >= seg_min) & (r_grid[:, None] <= seg_max), axis=1)


def _interp_phi_at_r(r_query, r_path, phi_path):
    """
    Interpolate phi along a ray path at a given radius.
    Sorts by r so np.interp can be applied (handles the ingoing leg).
    """
    sort_idx = np.argsort(r_path)
    return np.interp(r_query, r_path[sort_idx], phi_path[sort_idx])


def _symmetric_error_scale(finite_errors, min_span=1.0, max_span=30.0, percentile=98.0):
    """
    Return a readable symmetric color scale for signed angular errors.
    Uses a high percentile so one extreme ray does not wash out smaller
    variations, while still showing clipped extremes via colorbar extension.
    """
    if finite_errors.size == 0:
        return -min_span, min_span

    max_abs = np.nanpercentile(np.abs(finite_errors), percentile)
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = np.nanmax(np.abs(finite_errors))

    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = min_span

    max_abs = min(max(max_abs, min_span), max_span)
    return -max_abs, max_abs



def create_error_analysis_figure(
    obh,
    b_nominal,
    actual_event_horizon,
    delta_b_ratio_range=(DELTA_B_RATIO_MIN, DELTA_B_RATIO_MAX),
    n_delta_values=N_DELTA_VALUES,
):
   
    b_reference_abs = np.abs(b_nominal)
    if b_reference_abs == 0:
        raise ValueError("b_nominal must be nonzero for a relative delta_b scan")

    db_ratio_min = max(delta_b_ratio_range[0], DELTA_B_RATIO_MIN)
    db_ratio_max = min(delta_b_ratio_range[1], DELTA_B_RATIO_MAX)
    if db_ratio_min >= db_ratio_max:
        raise ValueError("delta_b_ratio_range must overlap [-0.1, 0.1]")

    delta_b_ratios = np.linspace(db_ratio_min, db_ratio_max, n_delta_values)
    delta_b_values = delta_b_ratios * b_reference_abs

    fig, (ax_traj, ax_error) = plt.subplots(1, 2, figsize=(16, 7))

    max_abs_delta_b_ratio = max(abs(db_ratio_min), abs(db_ratio_max))
    cmap_rays = plt.cm.plasma_r
    norm_rays = Normalize(vmin=0.0, vmax=max_abs_delta_b_ratio)

    x_geo, y_geo, r_geo, phi_geo = compute_geodesic(b_nominal, actual_event_horizon)
    y_plot_sign = -1.0 if b_nominal < 0 else 1.0
    r_geo_error, phi_geo_error = compute_geodesic_error_reference(
        b_nominal,
        actual_event_horizon,
    )
    ax_traj.plot(
        x_geo,
        y_plot_sign * y_geo,
        'k--',
        linewidth=2,
        label='Geodesic (Δb=0)',
        zorder=10,
    )


    r_sim_max           = 0.0   # largest simulated horizon seen — drives the disk size
    any_counter_rotating = False
    error_data           = []   # list of (delta_b / |b_ref|, r_sample, delta_phi_deg)

    for i, (delta_b_ratio, delta_b) in enumerate(zip(delta_b_ratios, delta_b_values)):
        if i % 10 == 0:
            print(f"  Progress: {i}/{n_delta_values}")

        b_actual = b_nominal + delta_b

        r_sim, _ = get_r_sim_and_horizon(b_actual)
        r_sim_max = max(r_sim_max, r_sim)

        if METRIC_TYPE == "kerr_newman" and b_actual < 0:
            any_counter_rotating = True

    
        x, y, hit_horizon = obh.trace_ray_continuous(b_actual, r_horizon=r_sim)

        if len(x) < 2:
            continue

        color = cmap_rays(norm_rays(abs(delta_b_ratio)))
        ax_traj.plot(x, y_plot_sign * y, color=color, alpha=0.5, linewidth=1.5)

        r_sample_all, phi_ray_all = _ray_error_coordinates(x, y)

        if len(r_sample_all) < 2:
            continue

        r_common_min = max(r_geo_error.min(), r_sample_all.min())
        r_common_max = min(r_geo_error.max(), r_sample_all.max())

        if r_common_min >= r_common_max:
            continue

        common_mask = ((r_sample_all >= r_common_min)
                       & (r_sample_all <= r_common_max))
        r_sample = r_sample_all[common_mask]
        phi_ray_at_sample = phi_ray_all[common_mask]
        phi_geo_at_sample = np.interp(r_sample, r_geo_error[::-1], phi_geo_error[::-1])
        delta_phi_deg = np.degrees(phi_ray_at_sample - phi_geo_at_sample)

        error_data.append((delta_b_ratio, r_sample, delta_phi_deg))


    ax_traj.add_patch(Circle(
        (0, 0), r_sim_max, color='black', zorder=10, label='Simulated BH'
    ))

    if (METRIC_TYPE == "kerr_newman"
            and any_counter_rotating
            and actual_event_horizon < r_sim_max):
        ax_traj.add_patch(Circle(
            (0, 0), actual_event_horizon,
            color='gray', fill=False, linestyle='--', linewidth=1.5,
            zorder=11, label='True Event Horizon',
        ))

    if METRIC_TYPE == "schwarzschild":
        ps_radius = 3.0   # 3M (= 1.5 × r_s for r_s = 2M)
    elif METRIC_TYPE == "kerr_newman":
        kn_ell_sign = int(np.sign(b_nominal)) if b_nominal != 0 else +1
        ps_radius = kerr_newman_photon_sphere(np.abs(KERR_A), KERR_CHARGE, kn_ell_sign)

    ax_traj.add_patch(Circle(
        (0, 0), ps_radius, fill=False, edgecolor='blue', linewidth=2, label='Photon Sphere'
    ))

    ax_traj.set_xlim(PLOT_XLIM)
    ax_traj.set_ylim(PLOT_YLIM)
    ax_traj.set_aspect('equal')
    ax_traj.set_xlabel('X / M', fontsize=14)
    ax_traj.set_ylabel('Y / M', fontsize=14)
    ax_traj.set_title(
        f'Ray Trajectories — Impact Parameter Variation ({METRIC_TYPE})', fontsize=14
    )
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(fontsize=10, loc='upper right')

    sm1 = plt.cm.ScalarMappable(cmap=cmap_rays, norm=norm_rays)
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax_traj)
    cbar1.set_label('|Δb| / |b_ref|', fontsize=12)

    print("Building angular-deviation contour plot...")

    if error_data:
        r_min_all = r_sim_max
        r_max_all = R_PLOT_MAX

        if r_min_all < r_max_all:
            r_grid      = np.linspace(r_min_all, r_max_all, 200)
            db_grid     = np.array([d[0] for d in error_data])
            error_grid  = np.full((len(db_grid), len(r_grid)), np.nan)

            for i, (delta_b, r_vals, err_vals) in enumerate(error_data):
                # r_vals is already monotonically increasing, so np.interp is safe.
                error_grid[i, :] = np.interp(
                    r_grid, r_vals, err_vals, left=np.nan, right=np.nan
                )

            error_masked = np.ma.masked_invalid(error_grid)

            cmap_err = plt.cm.PuOr.copy()
            cmap_err.set_bad(color='0.85')
            ax_error.set_facecolor('0.85')
            finite_errors = error_grid[np.isfinite(error_grid)]

            if finite_errors.size > 0:
                err_vmin, err_vmax = _symmetric_error_scale(finite_errors)
                err_levels = np.linspace(err_vmin, err_vmax, 13)
                err_ticks = np.linspace(err_vmin, err_vmax, 7)
                clipped_errors = np.ma.masked_invalid(
                    np.clip(error_masked, err_vmin, err_vmax)
                )

                contour = ax_error.contourf(
                    r_grid,
                    db_grid,
                    clipped_errors,
                    levels=err_levels,
                    cmap=cmap_err,
                    extend='both',
                    corner_mask=False,
                )
                cbar2 = plt.colorbar(
                    contour,
                    ax=ax_error,
                    ticks=err_ticks,
                )
                cbar2.set_label('Φ_ray − Φ_geo  (degrees)', fontsize=12)

            ax_error.axhline(y=0, color='white', linestyle='--', linewidth=2, alpha=0.8,
                             label='Δb = 0')
            ax_error.axvline(x=r_sim_max, color='white', linestyle='--', linewidth=2,
                             alpha=0.8, label='Simulated BH')

            if (METRIC_TYPE == "kerr_newman"
                    and any_counter_rotating
                    and actual_event_horizon < r_sim_max):
                ax_error.axvline(x=actual_event_horizon, color='gray', linestyle='--',
                                 linewidth=1.5, alpha=0.8, label='True Event Horizon')

            ax_error.set_xlim(r_min_all, R_PLOT_MAX)
            ax_error.legend(fontsize=10)

    ax_error.set_xlabel('r / M', fontsize=14)
    ax_error.set_ylabel('Δb / |b_ref|', fontsize=14)
    ax_error.set_title('Angular Deviation from Geodesic', fontsize=14)
    ax_error.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig



def main():
    if METRIC_TYPE == "schwarzschild":
        actual_event_horizon = R_HORIZON
        print(f"\n  Schwarzschild mode  |  r_horizon = {R_HORIZON}")

    elif METRIC_TYPE == "kerr_newman":
        actual_event_horizon = kerr_newman_radius(KERR_A, KERR_CHARGE)
        print(f"\n  Kerr-Newman mode  |  a = {KERR_A},  Q = {KERR_CHARGE}")
        print(f"  Actual event horizon: {actual_event_horizon:.4f} M")

        kn_ell_sign_nom = int(np.sign(B_NOMINAL)) if B_NOMINAL != 0 else +1
        if kn_ell_sign_nom == +1:
            print(f"  Nominal ray: co-rotating (b_nominal = {B_NOMINAL} > 0)")
        else:
            print(f"  Nominal ray: counter-rotating (b_nominal = {B_NOMINAL} < 0)")
            P_star_nom = kerr_newman_discontinuity_radius(KERR_A, KERR_CHARGE, B_NOMINAL)
            if P_star_nom is not None and P_star_nom > actual_event_horizon:
                print(f"  Discontinuity P_* = {P_star_nom:.4f}  →  r_sim = {P_star_nom + 0.1:.4f}")

        if DELTA_B_MIN + B_NOMINAL < 0 < DELTA_B_MAX + B_NOMINAL:
            print(
                f"  NOTE: sweep crosses b = 0  "
                f"({B_NOMINAL + DELTA_B_MIN:.3f} … {B_NOMINAL + DELTA_B_MAX:.3f}).\n"
                f"        Counter-rotating rays will have their own r_sim."
            )
    else:
        raise ValueError(f"Unknown METRIC_TYPE: {METRIC_TYPE!r}")

    # -------------------------------------------------- simulation
    obh = OpticalBlackHoleVaryB(r_start=R_START)

    print(f"\nSimulation parameters:")
    print(f"  Metric:           {METRIC_TYPE}")
    print(f"  b_nominal:        {B_NOMINAL}")
    print(f"  delta_b / |b_ref| range: [{DELTA_B_RATIO_MIN}, {DELTA_B_RATIO_MAX}]")
    print(f"  delta_b range:           [{DELTA_B_MIN}, {DELTA_B_MAX}]")
    print(f"  N_DELTA_VALUES:   {N_DELTA_VALUES}")
    print(f"  R_START:          {R_START}")
    print(f"  R_PLOT_MAX:       {R_PLOT_MAX}")

    fig = create_error_analysis_figure(
        obh,
        b_nominal=B_NOMINAL,
        actual_event_horizon=actual_event_horizon,
        delta_b_ratio_range=(DELTA_B_RATIO_MIN, DELTA_B_RATIO_MAX),
        n_delta_values=N_DELTA_VALUES,
    )

    filename = f"vary_b_{METRIC_TYPE}.png"
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {filename}")


    plt.show()


if __name__ == "__main__":
    main()
