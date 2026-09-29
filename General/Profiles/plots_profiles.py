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

from constants import P0, DEFAULT_NUM_ANNULI, DEFAULT_NUM_POINTS
from cases import SCHWARZSCHILD_CASES, KERR_NEWMAN_CASES
from Schwarzchild import refractive_index_schwarzschild
from Kerr_Newman import refractive_index_kn_continuous
from annuli import annulus_edges_with_half_ends, sample_piecewise_constant

def _schwarzschild_plot_colors(count):
    base = ["blue", "orange", "green", "red"]
    if count <= len(base):
        return base[:count]
    repeats = (count + len(base) - 1) // len(base)
    return (base * repeats)[:count]


def make_schwarzschild_step_profiles(P_min=2.0, P_max=P0, n_annuli=DEFAULT_NUM_ANNULI):
    
    b_inf_values = [case["b_inf"] for case in SCHWARZSCHILD_CASES]
    colors = _schwarzschild_plot_colors(len(b_inf_values))

    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)

    plt.figure(figsize=(8, 5))
    for b_inf, color in zip(b_inf_values, colors):
        _, n_values = sample_piecewise_constant(
            refractive_index_schwarzschild, edges, b_inf, 1.0, P_max
        )
        x, y = _step_data(edges, n_values)
        plt.step(x, y, where="post", color=color, label=f"b_hat_inf = {b_inf}")

    plt.xlabel("Radius R/M")
    plt.ylabel("Refractive Index n")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("results/profiles", exist_ok=True)
    plt.savefig("results/profiles/schwarzschild_step_profiles.png", dpi=200)
    plt.close()


def make_schwarzschild_continuous_profiles(P_min=2.0, P_max=P0):
    
    b_inf_values = [case["b_inf"] for case in SCHWARZSCHILD_CASES]
    colors = _schwarzschild_plot_colors(len(b_inf_values))

    radii = np.linspace(P_min, P_max, DEFAULT_NUM_POINTS)

    plt.figure(figsize=(8, 5))
    for b_inf, color in zip(b_inf_values, colors):
        n_values = refractive_index_schwarzschild(radii, b_inf, n_at_P0=1.0, P0=P_max)
        plt.plot(radii, n_values, color=color, label=f"b_hat_inf = {b_inf}")

    plt.xlabel("Radius $R/M$")
    plt.ylabel("Refractive Index $n(R/M)$")
    plt.title("Scalar Refractive Index Profile")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("results/profiles", exist_ok=True)
    plt.savefig("results/profiles/schwarzschild_continuous_profiles.png", dpi=200)
    plt.close()



def _step_data(edges, values):
    x = edges[::-1]
    y = np.r_[values[::-1], values[::-1][-1]]
    return x, y


def make_kerr_newman_profiles():
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.ravel()

    for ax, case in zip(axes, KERR_NEWMAN_CASES):
        edges = annulus_edges_with_half_ends(case["P_min"], case["P_max"], case["n_annuli"])
        _, values = sample_piecewise_constant(
            refractive_index_kn_continuous,
            edges,
            case["a"],
            case["rho_Q"],
            case["b_inf"],
            case["ell_sign"],
            case["P_max"],
            1.0,
        )

        x, y = _step_data(edges, values)
        ax.step(x, y, where="post", linewidth=1.5)

        ax.set_xlim(case["P_min"], case["P_max"])
        ax.set_yscale("log")
        ax.set_xlabel(r"$R/M$")
        ax.set_ylabel(r"$n$")
        ax.text(0.06, 0.88, case["panel"], transform=ax.transAxes, fontsize=14, fontweight="bold")

    fig.suptitle("Kerr–Newman scalar refractive index profiles", fontsize=16)
    fig.tight_layout()

    os.makedirs("results/profiles", exist_ok=True)
    fig.savefig("results/profiles/kerr_newman_profiles.png", dpi=250)
    plt.close(fig)


def make_schwarzschild_profiles():
    make_schwarzschild_step_profiles()
    make_schwarzschild_continuous_profiles()


if __name__ == "__main__":
    make_schwarzschild_profiles()
    make_kerr_newman_profiles()