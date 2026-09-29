import numpy as np


def annulus_edges_with_half_ends(P_min, P_max, n_annuli):
    if n_annuli < 1: raise ValueError("n_annuli must be at least 1")
    if n_annuli == 1: return np.array([P_max, P_min], dtype=float)
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
    return 0.5 * (edges[:-1] + edges[1:])

def sample_piecewise_constant(index_function, edges, *args, **kwargs):
    centers = annulus_centers_from_edges(edges)
    values = index_function(centers, *args, **kwargs)
    values = np.asarray(values, dtype=float)
    values[0] = index_function(np.array([edges[0]]), *args, **kwargs)[0]
    if len(values) > 1:
        values[-1] = index_function(np.array([edges[-1]]), *args, **kwargs)[0]
    return centers, values


def delta_hat(P, a, rho_Q):
    r"""Eq. (17): $$\hat{\Delta} = P^2 - 2P + a^2 + \rho_Q^2$$"""
    P = np.asarray(P, dtype=float)
    return P**2 - 2.0 * P + a**2 + rho_Q**2

def V_pm(P, a, rho_Q, ell_sign=+1):
    P = np.asarray(P, dtype=float)
    D = delta_hat(P, a, rho_Q)
    D = np.maximum(D, 0.0)
    sqrtD = np.sqrt(D)
    denom = (P**2 + a**2) ** 2 - a**2 * D
    V_plus = (a * (2.0 * P - rho_Q**2) + ell_sign * P**2 * sqrtD) / denom
    V_minus = (a * (2.0 * P - rho_Q**2) - ell_sign * P**2 * sqrtD) / denom
    return V_plus, V_minus

def b_hat_kn_exact(P, a, rho_Q, b_inf, ell_sign=+1):
    P = np.asarray(P, dtype=float)
    D = delta_hat(P, a, rho_Q)
    D = np.maximum(D, 0.0)
    V_plus, V_minus = V_pm(P, a, rho_Q, ell_sign)
    b_inf_inv = 1.0 / b_inf
    X = (D - a**2) + (2.0 * P - rho_Q**2) * a * b_inf_inv
    B = (P**2 + a**2) ** 2 - a**2 * D
    C = (b_inf_inv - V_plus) * (b_inf_inv - V_minus)
    radicand = np.maximum(P**2 * (X**2) + (D**2) * B * C, 0.0)
    out = (P**2 * X) / np.sqrt(radicand)
    
    # Clean up non-finite values at the horizon
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = out[~bad][-1] if np.any(~bad) else 1.0 
    return out

def refractive_index_kn_continuous(P, a, rho_Q, b_inf, ell_sign=+1, P0=6.0, n0=1.0):
    r"""Eq. (23): $n(P) \propto 1 / \hat{b}(P)$ normalized at P0."""
    P = np.asarray(P, dtype=float)
    bP = b_hat_kn_exact(P, a, rho_Q, b_inf, ell_sign)
    bP0 = b_hat_kn_exact(np.array([P0]), a, rho_Q, b_inf, ell_sign)[0]
    return (n0 / (1.0 / bP0)) * (1.0 / bP)

def kerr_newman_annuli_profile(a, rho_Q, b_inf, P_min, P_max=6.0, n_annuli=200, ell_sign=+1, n0=1.0, P0=6.0):
    edges = annulus_edges_with_half_ends(P_min, P_max, n_annuli)
    centers, values = sample_piecewise_constant(refractive_index_kn_continuous, edges, a, rho_Q, b_inf, ell_sign, P0, n0)
    return centers, values

def kerr_newman_radius(a, rho_Q, M=1.0, return_dimensionless=True):
    discriminant = 1.0 - a**2 - rho_Q**2
    
    if discriminant < 0:
        raise ValueError("No horizon exists (naked singularity): a^2 + rho_Q^2 > 1")

    P_plus = 1.0 + np.sqrt(discriminant)

    if return_dimensionless:
        return P_plus
    else:
        return M * P_plus
    


def kerr_newman_photon_sphere(a, rho_Q, prograde=True, M=1.0, return_dimensionless=True,
                             P_min=1.0, P_max=10.0, num_points=10000):

    sign = -1.0 if prograde else +1.0

    def f(P):
        P = np.asarray(P)
        inside = P - rho_Q**2
        inside = np.maximum(inside, 0.0)
        return P**2 - 3.0*P + 2.0*rho_Q**2 + sign * 2.0*a*np.sqrt(inside)

    # Scan for sign change
    P_vals = np.linspace(P_min, P_max, num_points)
    f_vals = f(P_vals)

    # Find zero crossing
    sign_changes = np.where(np.diff(np.sign(f_vals)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError("No photon sphere root found in the given range")

    i = sign_changes[0]
    P1, P2 = P_vals[i], P_vals[i+1]

    # Bisection refinement
    for _ in range(60):
        P_mid = 0.5 * (P1 + P2)
        if f(P1) * f(P_mid) <= 0:
            P2 = P_mid
        else:
            P1 = P_mid

    P_root = 0.5 * (P1 + P2)

    if return_dimensionless:
        return P_root
    else:
        return M * P_root
    

def kerr_newman_discontinuity_radius(a, rho_Q, b_inf, M=1.0, return_dimensionless=True):
    term1 = 1.0 - a / b_inf
    discriminant = term1 * (term1 - rho_Q**2)
    
    # Check if discriminant is negative
    if discriminant < 0:
        return None
    
    P_star = term1 + np.sqrt(discriminant)
    
    if return_dimensionless:
        return P_star
    else:
        return M * P_star