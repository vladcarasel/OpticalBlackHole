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
from mpl_toolkits.mplot3d import Axes3D
import warnings
from Schwarzchild import refractive_index_schwarzschild
from Kerr_Newman import (refractive_index_kn_continuous, kerr_newman_radius,
                         kerr_newman_discontinuity_radius)


MODE = "schwarzschild"   # Options: "schwarzschild" or "kerr_newman"

USE_BLACK_BACKGROUND = True
USE_GAUSSIAN_PACKET  = True
PACKET_DURATION      = 100

B_INF            = 5.0
R_HORIZON        = 2.0
R_START          = 200.0   # Physical starting radius; rays are pre-advanced to RENDER_R_MIN
RENDER_R_MIN     = 10.0    # Only paths/animation inside this radius are rendered
P0_NORMALIZATION = 6.0

# Kerr-Newman specific parameters 
KN_SPIN     = 0.5    # Dimensionless spin parameter â (0 to 1)
KN_CHARGE   = 0.3    # Dimensionless charge parameter ρ_Q

N_ANNULI = 200

STEP_SIZE = 0.05
MAX_STEPS = 20000

IMPACT_PARAM = B_INF

LAMBDA_CENTER      = 550.0
LAMBDA_SIGMA       = 100.0
LAMBDA_RANGE_SIGMA = 3.0

N_WAVELENGTHS = 15   #


DISPERSION_MODEL = "cauchy_strong"
CAUCHY_B = 10**4              
CAUCHY_A = 1.5

ENVELOPE_SAMPLES  = 50   # Number of points along trajectory to show envelope
WAVELENGTH_PHASES = 8    # Number of wavelengths to show phase oscillations

MAX_FRAMES     = 600
FRAME_INTERVAL = 30

PLOT_XLIM = (-10, 10)
PLOT_YLIM = (-10, 10)

COLOR_TRANSITION_LENGTH = 100
ALPHA_SCALE = 0.9
WIDTH_MIN   = 0.5
WIDTH_MAX   = 3.0



def get_color_scheme():
    if USE_BLACK_BACKGROUND:
        return {
            'bg': 'black',
            'horizon': 'white',
            'text': 'white',
            'grid': 'white',
            'grid_alpha': 0.15,
            'pane': (0.05, 0.05, 0.05)
        }
    else:
        return {
            'bg': 'white',
            'horizon': 'black',
            'text': 'black',
            'grid': 'black',
            'grid_alpha': 0.3,
            'pane': (0.95, 0.95, 0.95)
        }


def gaussian_intensity(wavelength, center, sigma):
    return np.exp(-0.5 * ((wavelength - center) / sigma)**2)


def wavelength_to_rgb(wavelength):
    wl_clamped  = np.clip(wavelength, 380, 750)
    lambda_norm = (wl_clamped - 380) / (750 - 380)
    hue         = 0.75 - lambda_norm * 0.75
    return hsv_to_rgb([hue, 1.0, 1.0])


class OpticalBlackHoleDispersive:

    def __init__(self, mode="schwarzschild", b_inf=B_INF, r_start=R_START,
                 r_horizon=R_HORIZON, n_annuli=N_ANNULI, cauchy_b=CAUCHY_B,
                 p0_norm=P0_NORMALIZATION, kn_spin=KN_SPIN, kn_charge=KN_CHARGE,
                 kn_ell_sign=None, cauchy_a=CAUCHY_A):
        
        self.mode       = mode.lower()
        self.b_inf      = b_inf
        self.r_start    = r_start
        self.r_horizon  = r_horizon
        self.n_annuli   = n_annuli
        self.cauchy_b   = cauchy_b
        self.p0_norm    = p0_norm
        self.cauchy_a   = cauchy_a

        self.kn_spin    = kn_spin
        self.kn_charge  = kn_charge

        # Beam: x=-r_start, y=b_inf, vx=+1  →  L_z = -b_inf
        if b_inf == 0:
            geometric_ell_sign = +1   # degenerate
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

        self.annuli_radii   = np.linspace(r_horizon, r_start, n_annuli + 1)
        self.annuli_centers = 0.5 * (self.annuli_radii[:-1] + self.annuli_radii[1:])

        if self.mode not in ["schwarzschild", "kerr_newman"]:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'schwarzschild' or 'kerr_newman'")

        self.r_sim      = r_horizon
        self.kn_p_star  = None   # set below for counter-rotating KN

        if self.mode == "kerr_newman" and self.kn_ell_sign == -1:
            try:
                p_star    = kerr_newman_discontinuity_radius(kn_spin, kn_charge, b_inf)
                r_hor_kn  = kerr_newman_radius(kn_spin, kn_charge)
                if p_star is not None and p_star > r_hor_kn:
                    self.r_sim     = p_star + 0.1
                    self.kn_p_star = p_star
            except Exception as exc:
                warnings.warn(f"Could not compute P_* for KN counter-rotating mode: {exc}")

    def n_base(self, r):
        r = np.asarray(r, dtype=float)

        if self.mode == "schwarzschild":
            n = refractive_index_schwarzschild(
                P=r, b_inf=self.b_inf, n_at_P0=1.0, P0=self.p0_norm
            )
        elif self.mode == "kerr_newman":
            n = refractive_index_kn_continuous(
                P=r, a=self.kn_spin, rho_Q=self.kn_charge,
                b_inf=self.b_inf, ell_sign=self.kn_ell_sign,
                P0=self.p0_norm, n0=1.0
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return n

    def n_wavelength(self, r, wavelength):
        n_geo   = self.n_base(r)
        n_total = n_geo * (self.cauchy_a + self.cauchy_b / (wavelength**2))
        return n_total

    def grad_ln_n(self, x, y, wavelength):
        r = np.sqrt(x**2 + y**2)

        if r < self.r_sim:
            inward_pull = -0.1 / (r + 0.01)
            return inward_pull * (x / r), inward_pull * (y / r)

        dr      = 1e-6
        r_plus  = r + dr
        r_minus = max(r - dr, self.r_sim + 1e-8)

        n_plus  = self.n_wavelength(r_plus,  wavelength)
        n_minus = self.n_wavelength(r_minus, wavelength)
        n       = self.n_wavelength(r,       wavelength)

        if abs(n) < 1e-12:
            return 0.0, 0.0

        dn_dr     = (n_plus - n_minus) / (r_plus - r_minus)
        d_ln_n_dr = dn_dr / n

        if r < 1e-6:
            return 0.0, 0.0

        return d_ln_n_dr * (x / r), d_ln_n_dr * (y / r)

    def trace_ray_wavelength(self, impact_parameter, wavelength, max_steps=MAX_STEPS):
        x, y    = -self.r_start, impact_parameter
        vx, vy  = 1.0, 0.0
        x_traj, y_traj = [x], [y]
        hit_horizon = False

        for _ in range(max_steps):
            r = np.sqrt(x*x + y*y)

            if r < self.r_sim:
                hit_horizon = True
            if x > self.r_start and r > self.r_start:
                break

            ax_val, ay_val = self.grad_ln_n(x, y, wavelength)
            vx += ax_val * STEP_SIZE
            vy += ay_val * STEP_SIZE

            v_mag = np.sqrt(vx*vx + vy*vy)
            if v_mag > 0:
                vx, vy = vx / v_mag, vy / v_mag

            x += vx * STEP_SIZE
            y += vy * STEP_SIZE
            x_traj.append(x)
            y_traj.append(y)

        return np.array(x_traj), np.array(y_traj), hit_horizon


def compute_envelope_evolution(obh, impact_param=IMPACT_PARAM):
    lambda_min  = LAMBDA_CENTER - LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
    lambda_max  = LAMBDA_CENTER + LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
    wavelengths = np.linspace(lambda_min, lambda_max, N_WAVELENGTHS)
    intensities = gaussian_intensity(wavelengths, LAMBDA_CENTER, LAMBDA_SIGMA)

    all_paths = []
    for wavelength in wavelengths:
        x_full, y_full, hit_horizon = obh.trace_ray_wavelength(impact_param, wavelength)

        r_arr = np.sqrt(x_full**2 + y_full**2)
        enter = np.where(r_arr < RENDER_R_MIN)[0]
        if len(enter) > 0:
            x_crop = x_full[enter[0]:]
            y_crop = y_full[enter[0]:]
        else:
        
            x_crop = x_full
            y_crop = y_full

        all_paths.append({
            'x':          x_crop,
            'y':          y_crop,
            'wavelength': wavelength,
            'color':      wavelength_to_rgb(wavelength)
        })

    max_length = max(len(p['x']) for p in all_paths)

    envelope_data = []
    for step in range(0, max_length, max(1, max_length // ENVELOPE_SAMPLES)):
        if all(step >= len(p['x']) for p in all_paths):
            break

        center_idx = len(wavelengths) // 2
        x_c = (all_paths[center_idx]['x'][step]
               if step < len(all_paths[center_idx]['x'])
               else all_paths[center_idx]['x'][-1])
        y_c = (all_paths[center_idx]['y'][step]
               if step < len(all_paths[center_idx]['y'])
               else all_paths[center_idx]['y'][-1])

        positions = []
        for path in all_paths:
            if step < len(path['x']):
                positions.append([path['x'][step], path['y'][step]])

        if len(positions) > 1:
            positions = np.array(positions)
            width = np.std(np.linalg.norm(positions - [x_c, y_c], axis=1))
        else:
            width = 0.0

        envelope_data.append({'x': x_c, 'y': y_c, 'width': width, 'step': step})

    return envelope_data, all_paths, wavelengths, intensities


def _draw_horizon_rings(ax, obh, colors, z_level=0.0):
    theta = np.linspace(0, 2 * np.pi, 100)
    sim_x = obh.r_sim * np.cos(theta)
    sim_y = obh.r_sim * np.sin(theta)
    sim_z = np.full_like(theta, z_level)
    ax.plot(sim_x, sim_y, sim_z,
            color=colors['horizon'], linewidth=3, label='Simulation boundary')

    if obh.r_sim > obh.r_horizon:
        hor_x = obh.r_horizon * np.cos(theta)
        hor_y = obh.r_horizon * np.sin(theta)
        hor_z = np.full_like(theta, z_level)
        ax.plot(hor_x, hor_y, hor_z,
                color='gray', linewidth=1.5, linestyle='--',
                label='Event horizon')


def create_3d_visualization(obh, impact_param=IMPACT_PARAM):
    colors = get_color_scheme()

    print("Computing envelope evolution...")
    envelope_data, all_paths, wavelengths, intensities = compute_envelope_evolution(obh, impact_param)

    fig = plt.figure(figsize=(16, 12))
    ax  = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor(colors['bg'])
    ax.set_facecolor(colors['bg'])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(colors['grid'])
    ax.yaxis.pane.set_edgecolor(colors['grid'])
    ax.zaxis.pane.set_edgecolor(colors['grid'])
    ax.grid(True, alpha=colors['grid_alpha'], color=colors['grid'])

    center_idx  = len(wavelengths) // 2
    center_path = all_paths[center_idx]

    x_c = center_path['x']
    y_c = center_path['y']
    z_c = np.zeros_like(x_c)

    r_vals = np.sqrt(x_c**2 + y_c**2)

    inside = np.where(r_vals < obh.r_sim)[0]

    if len(inside) > 0:
        cut_idx = inside[0]
    else:
        cut_idx = len(r_vals)

    x_plot = x_c[:cut_idx]
    y_plot = y_c[:cut_idx]
    z_plot = z_c[:cut_idx]

    if len(x_plot) > 1:
        ax.plot(x_plot, y_plot, z_plot,
                color=colors['text'],
                linewidth=2.5,
                label='Central wavelength path',
                zorder=5)

    for i, env in enumerate(envelope_data):
        if i % 3 != 0:
            continue

        x_c, y_c = env['x'], env['y']
        width     = env['width']

        if width < 1e-6:
            continue

        if i < len(envelope_data) - 1:
            dx = envelope_data[i+1]['x'] - x_c
            dy = envelope_data[i+1]['y'] - y_c
        else:
            dx = x_c - envelope_data[i-1]['x']
            dy = y_c - envelope_data[i-1]['y']

        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-6:
            continue

        perp_x = -dy / norm
        perp_y =  dx / norm

        angles = np.linspace(0, 2 * np.pi, 50)
        x_ring = x_c + width * perp_x * np.cos(angles)
        y_ring = y_c + width * perp_y * np.cos(angles)
        z_ring = width * np.sin(angles)

        alpha = 0.3 if i % 2 == 0 else 0.15
        ax.plot(x_ring, y_ring, z_ring,
                color=colors['text'], alpha=alpha, linewidth=1)

    for i, wl_idx in enumerate(range(0, len(wavelengths),
                                     max(1, len(wavelengths) // WAVELENGTH_PHASES))):
        if wl_idx >= len(all_paths):
            break

        path       = all_paths[wl_idx]
        wavelength = path['wavelength']
        color      = path['color']
        intensity  = intensities[wl_idx]

        k            = 2 * np.pi / (wavelength / LAMBDA_CENTER)
        arc_param    = np.arange(len(path['x'])) * STEP_SIZE
        z_oscillation = 0.5 * np.sin(k * arc_param)

        r_vals = np.sqrt(path['x']**2 + path['y']**2)

        inside = np.where(r_vals < obh.r_sim)[0]

        if len(inside) > 0:
            cut_idx = inside[0]
        else:
            cut_idx = len(r_vals)

        x_plot = path['x'][:cut_idx]
        y_plot = path['y'][:cut_idx]
        z_plot = z_oscillation[:cut_idx]

        if len(x_plot) > 1:
            ax.plot(x_plot, y_plot, z_plot,
                    color=color,
                    alpha=0.6 * intensity,
                    linewidth=1.5,
                    label=f'{wavelength:.0f} nm' if i < 5 else None)

    _draw_horizon_rings(ax, obh, colors, z_level=0.0)

    ax.set_xlabel('X / M', fontsize=14, color=colors['text'])
    ax.set_ylabel('Y / M', fontsize=14, color=colors['text'])
    ax.set_zlabel('Wave Amplitude', fontsize=14, color=colors['text'])

    mode_display = obh.mode.replace('_', '-').title()
    if obh.mode == "kerr_newman":
        title_str = (f'3D Wave Packet: {mode_display} '
                     f'(â={obh.kn_spin}, ρQ={obh.kn_charge})\n'
                     f'(Envelope Spreading & Wavelength Dephasing)')
    else:
        title_str = f'3D Wave Packet: {mode_display}\n(Envelope Spreading & Wavelength Dephasing)'

    ax.set_title(title_str, fontsize=16, color=colors['text'], pad=20)

    ax.tick_params(colors=colors['text'])
    ax.xaxis.label.set_color(colors['text'])
    ax.yaxis.label.set_color(colors['text'])
    ax.zaxis.label.set_color(colors['text'])

    ax.set_xlim(-11, 11)
    ax.set_ylim(-11, 11)
    ax.set_zlim(-2, 2)
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    ax.legend(loc='upper left', fontsize=9,
              facecolor=colors['bg'], edgecolor=colors['text'],
              labelcolor=colors['text'])
    plt.tight_layout()
    return fig, ax


def create_3d_animation(obh, filename="wave_packet_3d.gif", impact_param=IMPACT_PARAM,
                        max_frames=MAX_FRAMES, interval=FRAME_INTERVAL):
    colors = get_color_scheme()

    print("Computing envelope evolution for animation...")
    envelope_data, all_paths, wavelengths, intensities = compute_envelope_evolution(obh, impact_param)

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')

    fig.patch.set_facecolor(colors['bg'])
    ax.set_facecolor(colors['bg'])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(colors['grid'])
    ax.yaxis.pane.set_edgecolor(colors['grid'])
    ax.zaxis.pane.set_edgecolor(colors['grid'])
    ax.grid(True, alpha=colors['grid_alpha'], color=colors['grid'])

    _draw_horizon_rings(ax, obh, colors, z_level=0.0)

    ax.set_xlabel('X / M', fontsize=12, color=colors['text'])
    ax.set_ylabel('Y / M', fontsize=12, color=colors['text'])
    ax.set_zlabel('Wave Amplitude', fontsize=12, color=colors['text'])

    mode_display = obh.mode.replace('_', '-').title()
    if obh.mode == "kerr_newman":
        title_str = (f'3D Wave Packet: {mode_display} '
                     f'(â={obh.kn_spin}, ρQ={obh.kn_charge})')
    else:
        title_str = f'3D Wave Packet: {mode_display}'

    ax.set_title(title_str, fontsize=14, color=colors['text'])

    ax.tick_params(colors=colors['text'])
    ax.set_xlim(-15, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-3, 3)

    lines = []
    for _ in range(len(wavelengths)):
        line, = ax.plot([], [], [], linewidth=1.5)
        lines.append(line)

    frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                           fontsize=12, color=colors['text'])

    info_lines = [f'Mode: {mode_display}', f'b∞ = {obh.b_inf:.1f}']
    if obh.mode == "kerr_newman":
        rot_label = 'Counter-rotating' if obh.kn_ell_sign == -1 else 'Co-rotating'
        info_lines.append(f'â = {obh.kn_spin:.2f}, ρQ = {obh.kn_charge:.2f}')
        info_lines.append(rot_label)
        if obh.kn_p_star is not None:
            info_lines.append(f'P_* = {obh.kn_p_star:.3f}')
            info_lines.append(f'r_sim = {obh.r_sim:.3f}')
    info_text = ax.text2D(0.98, 0.02, '\n'.join(info_lines),
                          transform=ax.transAxes, fontsize=9,
                          color=colors['text'], ha='right', va='bottom')

    
    abs_cross_idx = []
    for path in all_paths:
        r_full  = np.sqrt(path['x']**2 + path['y']**2)
        inside  = np.where(r_full < obh.r_sim)[0]
        abs_cross_idx.append(inside[0] if len(inside) > 0 else len(r_full))

    def update(frame):
        speed_factor  = 2.0
        current_step  = int(frame * speed_factor)

        for i, (line, path, wl, intensity) in enumerate(
                zip(lines, all_paths, wavelengths, intensities)):
            p_len  = len(path['x'])
            c_step = min(current_step, p_len - 1)

            if PACKET_DURATION != np.inf:
                start_idx = max(0, c_step - int(PACKET_DURATION))
                end_idx   = c_step
            else:
                start_idx, end_idx = 0, c_step

            cross = abs_cross_idx[i]

            if start_idx >= cross:
                line.set_data([], [])
                line.set_3d_properties([])
                continue

            
            end_idx_clipped = min(end_idx, cross)

            x_data = path['x'][start_idx:end_idx_clipped]
            y_data = path['y'][start_idx:end_idx_clipped]

            if len(x_data) < 2:
                line.set_data([], [])
                line.set_3d_properties([])
                continue

            k         = 2 * np.pi / (wl / LAMBDA_CENTER)
            arc_param = np.arange(start_idx, end_idx_clipped) * STEP_SIZE
            z_data    = 0.5 * np.sin(k * arc_param)

            x_plot = x_data
            y_plot = y_data
            z_plot = z_data

            if len(x_plot) < 2:
                line.set_data([], [])
                line.set_3d_properties([])
            else:
                line.set_data(x_plot, y_plot)
                line.set_3d_properties(z_plot)
                line.set_color(path['color'])
                line.set_alpha(0.6 * intensity)

        frame_text.set_text(f'Frame: {frame}/{max_frames}')
        ax.view_init(elev=20, azim=-50)

        return lines + [frame_text, info_text]

    anim = animation.FuncAnimation(
        fig, update, frames=max_frames, interval=interval, blit=False
    )

    anim.save(filename, writer='pillow', fps=30)
    print(f"Animation saved: {filename}")

    plt.close(fig)


def main():
    print("="*70)
    print("3D WAVE PACKET CHROMATIC DISPERSION - GENERALIZED")
    print("="*70)

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
        # kn_ell_sign derived automatically from IMPACT_PARAM geometry
    )

    #some useful parameters printed before to recall
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
            print(f"  → Solid ring = r_sim; dotted ring = r_horizon")
        else:
            print(f"  No P_* discontinuity outside horizon → r_sim = r_horizon = {obh.r_horizon:.4f}")
    print(f"  Background: {'Black' if USE_BLACK_BACKGROUND else 'White'}")
    print(f"  R_START = {R_START},  RENDER_R_MIN = {RENDER_R_MIN}")
    print(f"  (Rays pre-advanced from R={R_START} to R={RENDER_R_MIN} before rendering)")
    print(f"  Dispersion model: {DISPERSION_MODEL}")
    wl_lo = LAMBDA_CENTER - LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
    wl_hi = LAMBDA_CENTER + LAMBDA_RANGE_SIGMA * LAMBDA_SIGMA
    print(f"  Wavelength range: {wl_lo:.0f} - {wl_hi:.0f} nm")
    print(f"  Number of wavelengths: {N_WAVELENGTHS}")


    fig_3d, ax_3d = create_3d_visualization(obh, impact_param=IMPACT_PARAM)
    filename_static = f'wave_packet_3d_{MODE}_static.png'
    fig_3d.savefig(filename_static, dpi=150, bbox_inches='tight',
                   facecolor=fig_3d.get_facecolor())
    print(f"  Saved: {filename_static}")


    create_3d_animation(obh, filename=f"wave_packet_3d_{MODE}.gif",
                        impact_param=IMPACT_PARAM,
                        max_frames=MAX_FRAMES,
                        interval=FRAME_INTERVAL)

    print("3D Visualization complete!")

    plt.show()


if __name__ == "__main__":
    main()