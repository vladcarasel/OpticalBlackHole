# Optical Black Hole

This repository contains numerical tools and generated results for optical analogues of Schwarzschild and Kerr-Newman black holes.

## Folders

- `General/`: shared physics helpers, refractive-index profiles, annulus construction, constants, and figure-generation utilities.
- `FDFD/`: finite-difference frequency-domain simulations for optical black-hole wave propagation.
- `Ray Tracing/`: geometric ray-tracing simulations, including Snell-law annular propagation and error analysis.
- `Wavepacket/`: time-domain wavepacket simulations and rendered 2D/3D outputs.

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/vladcarasel/OpticalBlackHole.git
cd OpticalBlackHole
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, use `python` instead of `python3`, and run the `cd` and `python` parts of each command below as two separate commands.

## How to run

Run each script from inside its own folder. Figures are saved relative to that folder.

| Figures | Command | Output | Approx. time |
|---|---|---|---|
| Refractive-index profiles | `cd General/Profiles && python make_profiles.py` | `results/profiles/` | seconds |
| Discontinuity radius | `cd General && python discontinuity_radius.py` | shown on screen | seconds |
| Snell ray tracing vs. annulus number | `cd "Ray Tracing/Code_RayTracing" && python error_annuli_snell.py` | `results/ray_tracing/` | seconds |
| Snell ray tracing error analysis | `cd "Ray Tracing/Code_RayTracing" && python error_nb_snell.py` | `results/ray_tracing/` | seconds |
| Continuous ray tracing, varying `b` | `cd "Ray Tracing/Code_RayTracing" && python ray_tracing_continuous_vary_b.py` | `vary_b_<metric>.png` | 1–2 hours |
| Continuous ray tracing, varying `n` | `cd "Ray Tracing/Code_RayTracing" && python ray_tracing_continuous_vary_n.py` | `dual_solver_*.png` | 1–2 hours |
| Schwarzschild FDFD | `cd FDFD/code && python schwarzschild_fdfd.py` | `results/fdfd/` | ~2 min |
| Kerr-Newman FDFD | `cd FDFD/code && python kerr_newman_fdfd.py` | `results/fdfd/` | ~2 min |
| 2D wavepacket | `cd Wavepacket/Code_wavepacket && python wave_packet_2D.py` | `wave_packet_<mode>.gif` + static PNG | ~5 min |
| 3D wavepacket | `cd Wavepacket/Code_wavepacket && python wave_packet_3D.py` | `wave_packet_3d_<mode>.gif` + static PNG | ~3 min |

Times are for a laptop running one script at a time. The FDFD solver uses several GB of RAM, so don't run the two FDFD scripts at the same time.

The continuous ray-tracing and wavepacket scripts generate one case per run. Choose the case with the constants at the top of each file (for example `METRIC_TYPE`, `MODE`, the impact parameter, spin `a` and charge `Q`). The figures already committed under each `results/` or `Simulation*/` folder were produced this way.
