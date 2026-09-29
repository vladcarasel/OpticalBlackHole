import numpy as np

DEFAULT_NUM_ANNULI = 200
DEFAULT_P0 = 6.0


def annulus_edges_with_half_ends(P_min, P_max, n_annuli):
    """Returns edges ordered from outer to inner with half-width ends."""
    if n_annuli < 1:
        raise ValueError("n_annuli must be at least 1")
    if n_annuli == 1:
        return np.array([P_max, P_min], dtype=float)

    total_width = P_max - P_min
    interior_width = total_width / (n_annuli - 1)
    widths = [interior_width / 2.0] + [interior_width] * (n_annuli - 2) + [interior_width / 2.0]
    
    edges = [P_max]
    current = P_max
    for w in widths:
        current -= w
        edges.append(current)
    return np.array(edges, dtype=float)

def annulus_centers_from_edges(edges):
    """Calculates centers as the average of adjacent edges."""
    return 0.5 * (edges[:-1] + edges[1:])

def sample_piecewise_constant(index_function, edges, *args, **kwargs):
    """Samples the index function at centers, preserving endpoint values."""
    centers = annulus_centers_from_edges(edges)
    values = index_function(centers, *args, **kwargs)
    values = np.asarray(values, dtype=float)
    
    # Paper rule: replace outermost/innermost with P_max/P_min values
    values[0] = index_function(np.array([edges[0]]), *args, **kwargs)[0]
    if len(values) > 1:
        values[-1] = index_function(np.array([edges[-1]]), *args, **kwargs)[0]
    return centers, values


def refractive_index_schwarzschild(P, b_inf, n_at_P0=1.0, P0=DEFAULT_P0):
    r"""
    Schwarzschild scalar refractive index:
    $$n(P) \propto \sqrt{b_{\infty}^{-2} + 2 P^{-3}}$$
    """
    P = np.asarray(P, dtype=float)
    raw = np.sqrt(b_inf**-2 + 2.0 * P**-3)
    raw_P0 = np.sqrt(b_inf**-2 + 2.0 * P0**-3)
    scale = n_at_P0 / raw_P0
    return scale * raw

def schwarzschild_annuli_profile(
    b_inf, P_min,
    P_max=DEFAULT_P0,
    n_annuli=DEFAULT_NUM_ANNULI,
    n_at_P0=1.0,
    P0=DEFAULT_P0,
    ):
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    centers, values = sample_piecewise_constant(refractive_index_schwarzschild, edges, b_inf, n_at_P0, P0)
    return centers, values