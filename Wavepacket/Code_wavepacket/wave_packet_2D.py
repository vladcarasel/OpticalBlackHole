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
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import warnings

from Schwarzchild import refractive_index_schwarzschild
from Kerr_Newman import (refractive_index_kn_continuous, kerr_newman_radius,
                         kerr_newman_photon_sphere, kerr_newman_discontinuity_radius)


MODE = "kerr_newman"  # Options: "schwarzschild" or "kerr_newman"

USE_BLACK_BACKGROUND = True   #just for style

USE_GAUSSIAN_PACKET = True     # True = Gaussian intensity, False = uniform intensity
RENDER_R_MIN = 10.0            #Animation starts only from when the ray is within this radius

PACKET_DURATION = 100          # Duration in frames. Use np.inf for infinite (no trailing edge)

B_INF = -5.0                    # Impact parameter for refractive index normalization, positive for co-rotating, negative for counter-rotating
R_HORIZON = 2.0                # Physical event horizon radius
R_START = 200.0                # Starting radius (outer boundary)
P0_NORMALIZATION = 6.0         # Reference radius for index normalization
IMPACT_PARAM = B_INF           # Single impact parameter for wave packet


KN_SPIN = 0.5                  # Dimensionless spin parameter â (0 to 1). DO NOT PUT NEGATIVE SIGNS FOR COUNTER-ROTATING
KN_CHARGE = 0.3                # Dimensionless charge parameter ρ_Q

N_ANNULI = 200                 # Number of concentric annuli

STEP_SIZE = 0.05               # Step size in affine parameter (ds)
MAX_STEPS = 20000              # Maximum integration steps


# Wavelength configuration
LAMBDA_MIN = 380.0             # for uniform mode
LAMBDA_MAX = 750.0             # for uniform mode
LAMBDA_CENTER = 550.0          # for Gaussian mode
LAMBDA_SIGMA = 100.0           # for Gaussian mode
LAMBDA_RANGE_SIGMA = 3.0       # Plot wavelengths within \pm sigma

N_WAVELENGTHS = 30             # Number of wavelengths to simulate

DISPERSION_MODEL = "cauchy_strong"  
CAUCHY_B = 10**4               #  (nm²)
CAUCHY_A = 1.5


MAX_FRAMES = 1000              # Number of frames in animation
FRAME_INTERVAL = 30            # Milliseconds between frames

PLOT_XLIM = (-10, 10)          # X-axis limits for Cartesian plot
PLOT_YLIM = (-10, 10)          # Y-axis limits for Cartesian plot

COLOR_TRANSITION_LENGTH = 100  # Number of steps for white→color transition

ALPHA_SCALE = 0.9              # Maximum alpha (transparency) for central wavelength
WIDTH_MIN = 0.5                # Minimum line width for dimmest wavelengths
WIDTH_MAX = 3.0                # Maximum line width for brightest wavelengths


def get_color_scheme():
    if USE_BLACK_BACKGROUND:
        return {
            'bg': 'black',
            'horizon': 'white',
            'text': 'white',
            'grid': 'white',
            'grid_alpha': 0.15
        }
    else:
        return {
            'bg': 'white',
            'horizon': 'black',
            'text': 'black',
            'grid': 'black',
            'grid_alpha': 0.3
        }


def gaussian_intensity(wavelength, center, sigma):
    return np.exp(-0.5 * ((wavelength - center) / sigma)**2)


def wavelength_to_rgb(wavelength):
    wl_clamped = np.clip(wavelength, 380, 750)
    lambda_norm = (wl_clamped - 380) / (750 - 380)
    hue = 0.75 - lambda_norm * 0.75
    saturation = 1.0
    value = 1.0
    rgb = hsv_to_rgb([hue, saturation, value])
    return rgb


def add_colors(colors, weights=None):
    if weights is None:
        weights = np.ones(len(colors))
    colors = np.array(colors)
    weights = np.array(weights).reshape(-1, 1)
    combined = np.sum(colors * weights, axis=0)
    max_val = np.max(combined)
    if max_val > 0:
        combined = combined / max_val
    return np.clip(combined, 0, 1)


class OpticalBlackHoleDispersive:
    def __init__(self, mode="schwarzschild", b_inf=B_INF, r_start=R_START,
                 r_horizon=R_HORIZON, n_annuli=N_ANNULI, cauchy_b=CAUCHY_B,
                 p0_norm=P0_NORMALIZATION, kn_spin=KN_SPIN, kn_charge=KN_CHARGE,
                 kn_ell_sign=None, cauchy_a=CAUCHY_A):

        self.mode = mode.lower()
        self.b_inf = b_inf
        self.r_start = r_start
        self.r_horizon = r_horizon          # physical event horizon (always kept)
        self.n_annuli = n_annuli
        self.cauchy_b = cauchy_b
        self.p0_norm = p0_norm
        self.cauchy_a = cauchy_a

        self.kn_spin = kn_spin
        self.kn_charge = kn_charge

        # Beam: x=-r_start, y=b_inf, vx=+1 to L_z = x*vy - y*vx = -b_inf
        # ell_sign = sign(L_z) = -sign(b_inf)
        if b_inf == 0:
            geometric_ell_sign = +1  #not interesting case, its degenerate
        else:
            geometric_ell_sign = int(np.sign(b_inf))

        if kn_ell_sign is None:
            self.kn_ell_sign = geometric_ell_sign
        else:
            self.kn_ell_sign = kn_ell_sign
            if self.mode == "kerr_newman" and kn_ell_sign != geometric_ell_sign:
                warnings.warn(
                    f"kn_ell_sign={kn_ell_sign} contradicts beam geometry: "
                    f"beam at y={b_inf:+.3f} has L_z={-b_inf:+.3f}, implying "
                    f"ell_sign={geometric_ell_sign}.  The refractive-index "
                    f"calculation will use the overridden value {kn_ell_sign}, "
                    f"which is physically inconsistent with the drawn trajectory."
                )

        self.annuli_radii = np.linspace(r_horizon, r_start, n_annuli + 1)
        self.annuli_centers = (self.annuli_radii[:-1] + self.annuli_radii[1:]) / 2

        if self.mode not in ["schwarzschild", "kerr_newman"]:
            raise ValueError(f"Invalid mode: {self.mode}. Select 'schwarzschild' or 'kerr_newman'")

        self.r_sim = r_horizon
        self.kn_p_star = None   # discontinuity radius P_* (set below for counter-rotating KN)

        #Counter-rotating Kerr-Newman: check whether P_* lies outside the horizon. If yes, shift the simulation boundary to P_* + 0.1 to skip the discontinuity.
        if self.mode == "kerr_newman" and self.kn_ell_sign == -1:
            try:
                p_star = kerr_newman_discontinuity_radius(kn_spin, kn_charge, b_inf)
                r_hor_kn = kerr_newman_radius(kn_spin, kn_charge)
                if p_star is not None and p_star > r_hor_kn:
                    self.r_sim = p_star + 0.1
                    self.kn_p_star = p_star
            except Exception as exc:
                warnings.warn(f"Could not compute P_* for KN counter-rotating mode: {exc}")

    def n_base(self, r):
        r = np.asarray(r, dtype=float)

        if self.mode == "schwarzschild":
            n = refractive_index_schwarzschild(
                P=r,
                b_inf=self.b_inf,
                n_at_P0=1.0,
                P0=self.p0_norm
            )
        elif self.mode == "kerr_newman":
            n = refractive_index_kn_continuous(
                P=r,
                a=self.kn_spin,
                rho_Q=self.kn_charge,
                b_inf=self.b_inf,
                ell_sign=self.kn_ell_sign,
                P0=self.p0_norm,
                n0=1.0
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return n

    def n_wavelength(self, r, wavelength):
        n_geo = self.n_base(r)
        cauchy_term = self.cauchy_b / (wavelength**2)
        n_total = n_geo * (self.cauchy_a + cauchy_term)
        return n_total

    def grad_ln_n(self, x, y, wavelength):
        r = np.sqrt(x**2 + y**2)

        if r < self.r_sim:
            inward_pull = -0.1 / (r + 0.01)
            return inward_pull * (x / r), inward_pull * (y / r)

        # Numerical gradient calculation
        dr = 1e-6
        r_plus = r + dr
        r_minus = max(r - dr, self.r_sim + 1e-8)

        n_plus  = self.n_wavelength(r_plus,  wavelength)
        n_minus = self.n_wavelength(r_minus, wavelength)
        n       = self.n_wavelength(r,       wavelength)

        if abs(n) < 1e-12:
            return 0.0, 0.0

        # Central difference for derivative approx
        dn_dr      = (n_plus - n_minus) / (r_plus - r_minus)
        d_ln_n_dr  = dn_dr / n

        if r < 1e-6:
            return 0.0, 0.0

        d_ln_n_dx = d_ln_n_dr * (x / r)
        d_ln_n_dy = d_ln_n_dr * (y / r)

        return d_ln_n_dx, d_ln_n_dy

    def trace_ray_wavelength(self, impact_parameter, wavelength, max_steps=MAX_STEPS):
        x = -self.r_start
        y = impact_parameter

        # moving to the right at beginning
        vx = 1.0
        vy = 0.0

        x_traj = [x]
        y_traj = [y]

        hit_horizon = False

        for step in range(max_steps):
            r = np.sqrt(x*x + y*y)

            if r < self.r_sim:
                hit_horizon = True
                break

            if x > self.r_start and r > self.r_start:
                break

            ax_val, ay_val = self.grad_ln_n(x, y, wavelength)

            vx += ax_val * STEP_SIZE
            vy += ay_val * STEP_SIZE

            # Normalize!
            v = np.sqrt(vx*vx + vy*vy)
            if v > 0:
                vx /= v
                vy /= v

            x += vx * STEP_SIZE
            y += vy * STEP_SIZE

            x_traj.append(x)
            y_traj.append(y)

        return np.array(x_traj), np.array(y_traj), hit_horizon


def create_animation(obh, filename="wave_packet_generalized.gif",
                     impact_param=IMPACT_PARAM, max_frames=MAX_FRAMES,
                     interval=FRAME_INTERVAL):
    colors = get_color_scheme()

    if USE_GAUSSIAN_PACKET:
        lambda_min = LAMBDA_CENTER - LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
        lambda_max = LAMBDA_CENTER + LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
        wavelengths = np.linspace(lambda_min, lambda_max, N_WAVELENGTHS)
        intensities = gaussian_intensity(wavelengths, LAMBDA_CENTER, LAMBDA_SIGMA)
        mode_text = f'Gaussian Packet (λ₀={LAMBDA_CENTER:.0f}nm, σ={LAMBDA_SIGMA:.0f}nm)'
    else:
        wavelengths = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_WAVELENGTHS)
        intensities = np.ones(N_WAVELENGTHS)
        mode_text = f'Multiple Wavelengths ({LAMBDA_MIN:.0f}-{LAMBDA_MAX:.0f}nm)'

    rays = []
    for wavelength, intensity in zip(wavelengths, intensities):
        base_color = wavelength_to_rgb(wavelength)

        if USE_GAUSSIAN_PACKET:
            alpha     = ALPHA_SCALE * intensity
            linewidth = WIDTH_MIN + (WIDTH_MAX - WIDTH_MIN) * intensity
        else:
            alpha     = 0.7
            linewidth = 2.0

        ray = {
            "x": -obh.r_start,
            "y": impact_param,
            "vx": 1.0,
            "vy": 0.0,
            "wavelength": wavelength,
            "base_color": base_color,
            "current_color": np.array([1.0, 1.0, 1.0]),
            "intensity": intensity,
            "alpha": alpha,
            "linewidth": linewidth,
            "path_x": [-obh.r_start],
            "path_y": [impact_param],
            "alive": True,
            "status": "propagating",
            "birth_frame": 0,
            "full_x": [-obh.r_start],
            "full_y": [impact_param],
            "frames_since_death": 0,
        }
        rays.append(ray)

    print(f"Advancing rays from R={obh.r_start} to R={RENDER_R_MIN}...")
    for ray in rays:
        while True:
            r = np.sqrt(ray["x"]**2 + ray["y"]**2)
            if r <= RENDER_R_MIN:
                break  
            ax_val, ay_val = obh.grad_ln_n(ray["x"], ray["y"], ray["wavelength"])
            ray["vx"] += ax_val * STEP_SIZE
            ray["vy"] += ay_val * STEP_SIZE
            v = np.sqrt(ray["vx"]**2 + ray["vy"]**2)
            if v > 0:
                ray["vx"] /= v
                ray["vy"] /= v
            ray["x"] += ray["vx"] * STEP_SIZE
            ray["y"] += ray["vy"] * STEP_SIZE

        # Reset path history so the visible trail starts at R = RENDER_R_MIN
        ray["full_x"] = [ray["x"]]
        ray["full_y"] = [ray["y"]]


    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(colors['bg'])
    ax.set_facecolor(colors['bg'])
    ax.tick_params(colors=colors['text'])

    lines      = []
    glow_lines = []
    for ray in rays:
        glow, = ax.plot([], [], color=ray["current_color"],
                        alpha=ray["alpha"]*0.3, linewidth=ray["linewidth"]*2,
                        solid_capstyle='round')
        line, = ax.plot([], [], color=ray["current_color"],
                        alpha=ray["alpha"], linewidth=ray["linewidth"],
                        solid_capstyle='round')
        glow_lines.append(glow)
        lines.append(line)

    horizon_circle = Circle((0, 0), obh.r_sim, color=colors['horizon'], zorder=10)
    ax.add_patch(horizon_circle)

    if obh.r_sim > obh.r_horizon:
        inner_horizon = Circle(
            (0, 0), obh.r_horizon,
            color='gray', fill=False, linestyle='--', linewidth=1.5, zorder=11
        )
        ax.add_patch(inner_horizon)

    ax.set_xlim(PLOT_XLIM)
    ax.set_ylim(PLOT_YLIM)
    ax.set_aspect('equal')
    ax.set_xlabel('X / M', fontsize=14, color=colors['text'])
    ax.set_ylabel('Y / M', fontsize=14, color=colors['text'])

    mode_display = obh.mode.replace('_', '-').title()
    if obh.mode == "kerr_newman":
        title = f'{mode_display} BH (â={obh.kn_spin}, ρQ={obh.kn_charge})\n{mode_text}'
    else:
        title = f'{mode_display} BH\n{mode_text}'

    ax.set_title(title, fontsize=16, color=colors['text'])
    ax.grid(True, alpha=colors['grid_alpha'], color=colors['grid'])

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                         fontsize=12, color=colors['text'], va='top')

    # Info text
    info_lines = [f'Mode: {mode_display}', f'b∞ = {obh.b_inf:.1f}']
    if obh.mode == "kerr_newman":
        rot_label = 'Counter-rotating' if obh.kn_ell_sign == -1 else 'Co-rotating'
        info_lines.append(f'â = {obh.kn_spin:.2f}, ρQ = {obh.kn_charge:.2f}')
        info_lines.append(rot_label)
        if obh.kn_p_star is not None:
            info_lines.append(f'P_* = {obh.kn_p_star:.3f}')
            info_lines.append(f'r_sim = {obh.r_sim:.3f}')
    info_text = ax.text(0.98, 0.02, '\n'.join(info_lines),
                        transform=ax.transAxes, fontsize=10,
                        color=colors['text'], ha='right', va='bottom')

    def update(frame):
        for ray, line, glow in zip(rays, lines, glow_lines):
            if ray["alive"]:
                r = np.sqrt(ray["x"]**2 + ray["y"]**2)

                if r <= obh.r_sim:
                    ray["alive"] = False
                    ray["status"] = "absorbed"
                    continue

                if ray["x"] > obh.r_start and r > obh.r_start:
                    ray["alive"] = False
                    ray["status"] = "escaped"
                    continue

                ax_val, ay_val = obh.grad_ln_n(ray["x"], ray["y"], ray["wavelength"])

                #velocities
                ray["vx"] += ax_val * STEP_SIZE
                ray["vy"] += ay_val * STEP_SIZE

                # Normalize
                v = np.sqrt(ray["vx"]**2 + ray["vy"]**2)
                if v > 0:
                    ray["vx"] /= v
                    ray["vy"] /= v

                #pos
                ray["x"] += ray["vx"] * STEP_SIZE
                ray["y"] += ray["vy"] * STEP_SIZE

                ray["full_x"].append(ray["x"])
                ray["full_y"].append(ray["y"])
            else:
                if PACKET_DURATION != np.inf:
                    ray["frames_since_death"] += 1

        for ray, line, glow in zip(rays, lines, glow_lines):
            x_arr = np.array(ray["full_x"])
            y_arr = np.array(ray["full_y"])
            total_steps = len(x_arr)

            if total_steps < 2:
                line.set_data([], [])
                glow.set_data([], [])
                continue

            if PACKET_DURATION == np.inf:
                display_x = x_arr
                display_y = y_arr
            else:
                tail_offset = ray["frames_since_death"]
                head_idx    = total_steps
                tail_idx    = max(0, total_steps - int(PACKET_DURATION) + tail_offset)

                if tail_idx >= head_idx:
                    line.set_data([], [])
                    glow.set_data([], [])
                    continue

                display_x = x_arr[tail_idx:head_idx]
                display_y = y_arr[tail_idx:head_idx]

            if len(display_x) < 2:
                line.set_data([], [])
                glow.set_data([], [])
                continue

            r_display = np.sqrt(display_x**2 + display_y**2)
            outside   = r_display >= obh.r_sim
            display_x = display_x[outside]
            display_y = display_y[outside]

            if len(display_x) < 2:
                line.set_data([], [])
                glow.set_data([], [])
                continue

            line.set_data(display_x, display_y)
            glow.set_data(display_x, display_y)

            dispersion_factor = min(1.0, total_steps / COLOR_TRANSITION_LENGTH)
            white             = np.array([1.0, 1.0, 1.0])
            spectral          = np.array(ray["base_color"])
            current_color     = white * (1 - dispersion_factor) + spectral * dispersion_factor

            ray["current_color"] = current_color
            line.set_color(current_color)
            glow.set_color(current_color)

        frame_text.set_text(f'Frame: {frame}/{max_frames}')

        return lines + glow_lines + [frame_text, info_text]

    anim = animation.FuncAnimation(
        fig, update, frames=max_frames,
        interval=interval, blit=False
    )

    anim.save(filename, writer='pillow', fps=30)
    print(f"Animation saved: {filename}")

    plt.close(fig)


def create_static_plot(obh, impact_param=IMPACT_PARAM):
    colors = get_color_scheme()

    if USE_GAUSSIAN_PACKET:
        lambda_min = LAMBDA_CENTER - LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
        lambda_max = LAMBDA_CENTER + LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
        wavelengths = np.linspace(lambda_min, lambda_max, N_WAVELENGTHS)
        intensities = gaussian_intensity(wavelengths, LAMBDA_CENTER, LAMBDA_SIGMA)
        title = f'Gaussian Wave Packet (λ₀={LAMBDA_CENTER:.0f}nm, σ={LAMBDA_SIGMA:.0f}nm)'
    else:
        wavelengths = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_WAVELENGTHS)
        intensities = np.ones(N_WAVELENGTHS)
        title = f'Multiple Wavelengths Dispersion ({LAMBDA_MIN:.0f}-{LAMBDA_MAX:.0f}nm)'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(colors['bg'])

    for ax in [ax1, ax2]:
        ax.set_facecolor(colors['bg'])
        ax.tick_params(colors=colors['text'])

    for wavelength, intensity in zip(wavelengths, intensities):
        x, y, hit_horizon = obh.trace_ray_wavelength(impact_param, wavelength)
        base_color = wavelength_to_rgb(wavelength)

        if USE_GAUSSIAN_PACKET:
            alpha     = ALPHA_SCALE * intensity
            linewidth = WIDTH_MIN + (WIDTH_MAX - WIDTH_MIN) * intensity
        else:
            alpha     = 0.7
            linewidth = 2.0

        ax1.plot(x, y, color=base_color, alpha=alpha, linewidth=linewidth)

    horizon_circle = Circle((0, 0), obh.r_sim, color=colors['horizon'], zorder=10)
    ax1.add_patch(horizon_circle)

    # Dotted physical event horizon inside disk (counter-rotating KN only)
    if obh.r_sim > obh.r_horizon:
        inner_horizon = Circle(
            (0, 0), obh.r_horizon,
            color='gray', fill=False, linestyle='--', linewidth=1.5, zorder=11
        )
        ax1.add_patch(inner_horizon)

    ax1.set_xlim(PLOT_XLIM)
    ax1.set_ylim(PLOT_YLIM)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X / M', fontsize=14, color=colors['text'])
    ax1.set_ylabel('Y / M', fontsize=14, color=colors['text'])

    mode_display = obh.mode.replace('_', '-').title()
    if obh.mode == "kerr_newman":
        title_str = f'{mode_display} (â={obh.kn_spin}, ρQ={obh.kn_charge})\n{title}'
    else:
        title_str = f'{mode_display}\n{title}'

    ax1.set_title(title_str, fontsize=16, color=colors['text'])
    ax1.grid(True, alpha=colors['grid_alpha'], color=colors['grid'])

    r_values = [2.5, 3.0, 4.0, 6.0]
    wavelength_range = np.linspace(wavelengths[0], wavelengths[-1], 100)

    for r in r_values:
        n_values = [obh.n_wavelength(r, wl) for wl in wavelength_range]
        ax2.plot(wavelength_range, n_values, linewidth=2, label=f'r = {r:.1f}M',
                 color=colors['text'] if colors['bg'] == 'white' else None)

    if USE_GAUSSIAN_PACKET:
        ax2.axvline(LAMBDA_CENTER, color=colors['text'], linestyle='--',
                    alpha=0.5, label='λ₀')

    ax2.set_xlabel('Wavelength (nm)', fontsize=14, color=colors['text'])
    ax2.set_ylabel('n(λ)', fontsize=14, color=colors['text'])
    ax2.set_title(f'Dispersion Profile\n{mode_display}', fontsize=16,
                  color=colors['text'])
    ax2.grid(True, alpha=colors['grid_alpha'], color=colors['grid'])
    ax2.legend(fontsize=10, facecolor=colors['bg'], edgecolor=colors['text'])

    legend = ax2.get_legend()
    for text in legend.get_texts():
        text.set_color(colors['text'])

    plt.tight_layout()
    return fig


def main():

    obh = OpticalBlackHoleDispersive(
        mode=MODE,
        b_inf=B_INF,
        r_start=R_START,
        r_horizon=R_HORIZON,
        n_annuli=N_ANNULI,
        cauchy_b=CAUCHY_B,
        p0_norm=P0_NORMALIZATION,
        kn_spin=KN_SPIN,
        kn_charge=KN_CHARGE,
        # kn_ell_sign is derived automatically from IMPACT_PARAM geometry
    )

    #some infos printed when running
    print("\nConfiguration:")
    print(f"  Mode: {MODE.upper()}")
    if MODE == "kerr_newman":
        print(f"  Spin (â): {KN_SPIN}")
        print(f"  Charge (ρQ): {KN_CHARGE}")
        rot_label = 'Counter-rotating' if obh.kn_ell_sign == -1 else 'Co-rotating'
        print(f"  Angular momentum sign: {obh.kn_ell_sign}  ({rot_label})  [auto-derived from beam geometry]")
        if obh.kn_p_star is not None:
            print(f"  P_* (discontinuity radius): {obh.kn_p_star:.4f}")
            print(f"  Physical event horizon r_horizon: {obh.r_horizon:.4f}")
            print(f"  Simulation boundary r_sim (P_* + 0.1): {obh.r_sim:.4f}")
            print(f"  → Black disk radius = r_sim; dotted ring = r_horizon")
        else:
            print(f"  No P_* discontinuity outside horizon → r_sim = r_horizon = {obh.r_horizon:.4f}")
    print(f"  Background: {'Black' if USE_BLACK_BACKGROUND else 'White'}")
    print(f"  Wave packet: {'Gaussian' if USE_GAUSSIAN_PACKET else 'Uniform wavelengths'}")
    print(f"  Packet duration: {'Infinite' if PACKET_DURATION == np.inf else f'{PACKET_DURATION} frames'}")

    print("\nDispersion parameters:")
    print(f"  Model: {DISPERSION_MODEL}")
    print(f"  Cauchy B: {CAUCHY_B}")

    if USE_GAUSSIAN_PACKET:
        print("\nGaussian packet parameters:")
        print(f"  Center wavelength: {LAMBDA_CENTER} nm")
        print(f"  Spectral width (σ): {LAMBDA_SIGMA} nm")
        lambda_min = LAMBDA_CENTER - LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
        lambda_max = LAMBDA_CENTER + LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
        print(f"  Wavelength range: {lambda_min:.1f} - {lambda_max:.1f} nm")
    else:
        print("\nUniform wavelength parameters:")
        print(f"  Wavelength range: {LAMBDA_MIN} - {LAMBDA_MAX} nm")

    print(f"\nNumber of wavelengths: {N_WAVELENGTHS}")
    print(f"Impact parameter: {IMPACT_PARAM}")

    fig_static = create_static_plot(obh, impact_param=IMPACT_PARAM)
    filename_static = f'wave_packet_{MODE}_static.png'
    fig_static.savefig(filename_static, dpi=150, bbox_inches='tight',
                       facecolor=fig_static.get_facecolor())
    print(f"  Saved: {filename_static}")

    create_animation(
        obh,
        filename=f"wave_packet_{MODE}.gif",
        impact_param=IMPACT_PARAM,
        max_frames=MAX_FRAMES,
        interval=FRAME_INTERVAL
    )

    print("Simulation complete!")
    plt.show()


if __name__ == "__main__":
    main()