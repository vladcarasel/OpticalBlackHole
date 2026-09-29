import os
import sys

# Make shared modules importable no matter where the script is run from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _d in ("General", "Ray Tracing/Code_RayTracing"):
    _p = os.path.join(_REPO_ROOT, *_d.split('/'))
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _pml_sigma(N_pml, d_pml, k0, power=3, R0=1e-8):#The parameter R0 represents the desired reflection coefficient at the outer edge of the PML layer. A value of 1e-8 means that we want the PML to reflect only a very small fraction (10^-8) of the incident wave back into the computational domain, which ensures that the PML effectively absorbs outgoing waves with minimal reflection.

    #Polynomial PML conductivity profile.
    #Returns sigma array of length N_pml.
    
    if N_pml == 0:
        return np.array([])
    sigma_max = -(power + 1) * np.log(R0) / (2.0 * d_pml)# The maximum conductivity sigma_max is calculated based on the desired reflection coefficient R0 and the thickness of the PML layer d_pml.
    d = np.linspace(0, d_pml, N_pml + 1)
    d = 0.5 * (d[:-1] + d[1:])  # cell centers
    #example: if d_pml = 1 and N_pml = 4, then d will be [0.125, 0.375, 0.625, 0.875], which are the centers of the four PML cells that span from 0 to 1.
    return sigma_max * (d / d_pml) ** power# The conductivity profile sigma is calculated as a polynomial function of the distance d from the inner edge of the PML layer. The power parameter controls how quickly the conductivity increases within the PML, with higher powers leading to a more gradual increase. 

def build_pml_stretch(N, N_pml, dx, k0):
    
    #Build 1D array of complex PML stretch factors s(x) for N cells.
    #s = 1 + sigma / (i * k0)  =  1 - i * sigma / k0
    #PML regions are at both ends of the domain.
    
    s = np.ones(N, dtype=complex)
    if N_pml == 0:
        return s
    d_pml = N_pml * dx
    sigma = _pml_sigma(N_pml, d_pml, k0)

    # Left PML: cells 0..N_pml-1, distance increases from center outward
    s[:N_pml] = 1.0 - 1j * sigma[::-1] / k0

    # Right PML: cells N-N_pml..N-1
    s[N - N_pml:] = 1.0 - 1j * sigma / k0

    return s #s is an array of complex stretch factors that modify the spatial derivatives in the FDFD equations to implement the PML. The stretch factors are designed to absorb outgoing waves 




class FDFD2D:
    

    def __init__(self, Lx, Ly, dx, wavelength, N_pml=12):
        
        self.dx = dx
        self.wavelength = wavelength
        self.k0 = 2.0 * np.pi / wavelength

        # Grid dimensions including PML
        self.Nx_phys = int(round(Lx / dx))
        self.Ny_phys = int(round(Ly / dx))
        self.N_pml = N_pml
        self.Nx = self.Nx_phys + 2 * N_pml
        self.Ny = self.Ny_phys + 2 * N_pml
        self.N = self.Nx * self.Ny

        # Physical coordinates (cell centers, including PML)
        self.x = (np.arange(self.Nx) - N_pml + 0.5) * dx - Lx / 2
        self.y = (np.arange(self.Ny) - N_pml + 0.5) * dx - Ly / 2

        # Default: free space, we initialize e_r as an array of 1, and J_z as 0s
        self.eps_r = np.ones((self.Nx, self.Ny), dtype=complex)
        self.J_z = np.zeros((self.Nx, self.Ny), dtype=complex)

        # Build PML, we get the stretch factors on x and y, using the previous helper function
        self.sx = build_pml_stretch(self.Nx, N_pml, dx, self.k0)
        self.sy = build_pml_stretch(self.Ny, N_pml, dx, self.k0)
        self.E_z = None  

    def _index2d(self, ix, iy):
        #Flatten 2D index (ix, iy) -> 1D index.
        return ix * self.Ny + iy
    
    def set_annular_permittivity(self, center_x, center_y, edges, n_values,
                                  inner_eps=None):
        
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')# Create 2D arrays of x and y coordinates for each grid point. 
        R = np.sqrt((XX - center_x)**2 + (YY - center_y)**2) 
        #R is a 2D array where each element R[i, j] gives the distance from the center (center_x, center_y) to the grid point at coordinates (XX[i, j], YY[i, j]). This allows us to classify each grid point into the correct annulus based on its distance from the center and the defined edges of the annuli.
        eps = np.ones_like(R, dtype=complex)  # free space outside

        # Fill annuli from outside in
        for i in range(len(n_values)):
            R_outer = edges[i]
            R_inner = edges[i + 1]
            mask = (R <= R_outer) & (R > R_inner)#1 if at distance R from the center we are in the annuli defined by R_outer and R_inner, 0 otherwise. This mask is used to identify which grid points belong to the current annulus being processed in the loop.
            eps[mask] = n_values[i] ** 2#for each annulus, eps=n^2 in that annulus.

        # Inner core (absorbing)
        R_min = edges[-1]
        if inner_eps is None:
            inner_eps = n_values[-1]**2 - 1j * np.pi  # damping factor
        mask_inner = R <= R_min
        eps[mask_inner] = inner_eps

        self.eps_r = eps#self.eps_r is the 2D array that represents the relative permittivity at each grid point in the computational domain. After running this function, self.eps_r will contain the permittivity values corresponding to the annular structure defined by the edges and n_values, as well as the inner core if specified
    def set_gaussian_beam_source(self, B0, beam_sigma, R0_phys,
                                  center_x=0.0, center_y=0.0,
                                  wg_strip_width=None):

        lam = self.wavelength  
        if wg_strip_width is None:
            wg_strip_width = 2.0 * lam#set the wifth of absorbing stripd to 2lambda

        J = np.zeros((self.Nx, self.Ny), dtype=complex)#2D array representing the source term J_z in the FDFD equations. We will populate this array with the Gaussian beam profile at the appropriate location in the grid. 
        x_beam = center_x + B0

        # Gaussian envelope in x, centered at the beam position
        envelope = np.exp(-((self.x - x_beam)**2) / (2.0 * beam_sigma**2))#it is a 1D array that 
        #example: envelope[i] = exp(-((self.x[i] - x_beam)**2) / (2.0 * beam_sigma**2)). Same length as self.x. Each value represents the value of the envelope at the corresponding x coordinate(from self.x), which is the center of each cell.

        # Truncate to waveguide channel (full width 2*lambda, ± lambda)
        wg_left = x_beam - lam#left end coordinate of absorbing strip
        wg_right = x_beam + lam#right end coordinate of absorbing strip
        wg_mask = (self.x >= wg_left) & (self.x <= wg_right)# mask which is 1 if the x coordinate is within the waveguide, 0 if not
        envelope *= wg_mask#envelope now becomes 0 outside the waveguide channel, and retains its Gaussian shape within the channel.

        # Place source at bottom edge of physical domain (just inside PML)
        y_start = self.y[self.N_pml + 1]#starts at bottom, from the first cell outside the absorber.
        iy_src = np.argmin(np.abs(self.y - y_start))#this find the index of self.y which gives y_start, such that self.y[iy_src]=y_start.
        J[:, iy_src] = envelope#we have a horizontal line og gaussian source at the bottom of the domain, right above the pml at Y_start.
        self.J_z = J
        #note that J is a 2D array, which is now all 0 except for the row corresponding to iy_src, which contains the Gaussian beam profile defined by the envelope. 
        
        # Two narrow vertical absorbers, one just left of x = wg_left and one
        # just right of x = wg_right. They run from the bottom of the physical
        # domain (y = -Ly/2) up to where the beam enters the BH outer circle.
        XX, YY = np.meshgrid(self.x, self.y, indexing='ij')
        #again for computational speed;
        R = np.sqrt((XX - center_x)**2 + (YY - center_y)**2)#R[i][j]=distance from center to grid point (i,j).

        # Vertical extent: from bottom up to the BH entry point.
        # The BH outer edge intersects x = x_beam at
        #     y_entry = center_y - sqrt(R0_phys^2 - B0^2)   (when |B0| < R0)
        # If |B0| >= R0 the beam never enters the BH; cap the strips at y=0.
        if abs(B0) < R0_phys:#if the beam enters the BH;
            y_entry = center_y - np.sqrt(R0_phys**2 - B0**2)#easy pythagoras, this is the y coordinate at which the beam enters the BH starting with an initail impact parameter, B0.
        else:
            y_entry = center_y#if the beam never enters the BH, we still want to place the absorbing strips up to the center line (y=0) to prevent lateral spreading of the beam before it reaches the BH region.

        # Strip 1: left side, x in (wg_left - w, wg_left)
        # Strip 2: right side, x in (wg_right, wg_right + w)
        #wg_left=left x coordinate of the strip.
        #wg_right=right x coordinate of the strip.
        left_strip = (XX > wg_left - wg_strip_width) & (XX < wg_left)#2D array. left_strip[i][j] is 1 if the grid point (i,j) is in the left absorbing strip region, which is defined as the area between x = wg_left - wg_strip_width and x = wg_left.
        #this is 1 for grid points which are in the left strip region, 0 otherwise.
        right_strip = (XX > wg_right) & (XX < wg_right + wg_strip_width)#same procedure
        vertical_extent = (YY < y_entry) & (YY > -self.Ny_phys * self.dx / 2)
        #2D array, vertical_extent[i][j] is 1 if the grid point (i,j) is below the y_entry line and above the bottom edge of the physical domain, which defines the vertical extent of the absorbing strips. 
        # Make absolutely sure we never overwrite the black hole interior
        outside_BH = R > R0_phys

        wg_absorb = vertical_extent & outside_BH & (left_strip | right_strip)#2d array, wg_absorb[i][j] is 1 if the grid point (i,j) is within the vertical extent of the absorbing strips, outside the black hole region, and within either the left or right strip region. This mask identifies the grid points where we want to set the permittivity to create the absorbing waveguide strips.
        self.eps_r[wg_absorb] = 1.0 - 1j * np.pi#we set the permittivity in the absorbing strip regions to a complex value with a negative imaginary part, which creates an absorbing medium  
    def build_system(self):
        
        Nx, Ny, N = self.Nx, self.Ny, self.N
        dx = self.dx
        k0 = self.k0

        # PML stretch at half-grid points (edges between cells)
        sx_fwd = np.ones(Nx, dtype=complex)#1D array currently containg Nx ones. sx_fwd[i] represents the stretch factor at the forward edge of the cell in the x direction;
        sx_fwd[:-1] = 0.5 * (self.sx[:-1] + self.sx[1:])#we average the stretch factors at the cell centers to get the stretch factor at the edge between cells. sx_fwd[i] = 0.5 * (sx[i] + sx[i+1]) for i=0 to Nx-2. This gives us the stretch factor at the forward edge of each cell
        sx_fwd[-1] = self.sx[-1]#the last edge (at the right boundary) uses the stretch factor of the last cell center, since there is no cell beyond it.

        #we now do the same thing backwards to get the stretch factor at the backward edge of each cell. 
        sx_bwd = np.ones(Nx, dtype=complex)
        sx_bwd[1:] = 0.5 * (self.sx[:-1] + self.sx[1:])
        sx_bwd[0] = self.sx[0]

        sy_fwd = np.ones(Ny, dtype=complex)#same for y;
        sy_fwd[:-1] = 0.5 * (self.sy[:-1] + self.sy[1:])
        sy_fwd[-1] = self.sy[-1]

        sy_bwd = np.ones(Ny, dtype=complex)
        sy_bwd[1:] = 0.5 * (self.sy[:-1] + self.sy[1:])
        sy_bwd[0] = self.sy[0]

        dx2 = dx**2#shortcut;

        # Flatten indices: m = ix*Ny + iy;just for indexing
        ix_all = np.repeat(np.arange(Nx), Ny)  
        iy_all = np.tile(np.arange(Ny), Nx)     
        m_all = np.arange(N)                    #m_all is the flattened index for each grid point, where m_all[i] corresponds to the grid point at (ix_all[i], iy_all[i]). 
        #example: if Nx=3 and Ny=4, then ix_all will be [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], iy_all will be [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], and m_all will be [0, 1, 2, ..., N-1]. 
        #so if i want to acces grid point (1,2), m=1*4+2=6. I have to call ix_all[6] and iy_all[6], m_all[6]=6, ix_all[6]=1, iy_all[6]=2.  
        # Stretched inverse factors for all grid points
        ISX_F = 1.0 / sx_fwd[ix_all]#just to store 1/stretch_factors.
        ISX_B = 1.0 / sx_bwd[ix_all]
        ISY_F = 1.0 / sy_fwd[iy_all]
        ISY_B = 1.0 / sy_bwd[iy_all]
        # a sparse matrix is a matrix that is mostly filled with zeros, and only a few non-zero entries. In the context of the FDFD method, the system matrix A is typically very large but also very sparse, because each grid point only interacts with its immediate neighbors 
        # arrays to represent a sparse matrix by storing only the non-zero entries and their corresponding row and column indices. 
        rows_list = []
        cols_list = []
        vals_list = []

        # 1) Diagonal
        diag_val = -(ISX_F + ISX_B + ISY_F + ISY_B) / dx2 + k0**2 * self.eps_r.ravel()#ravel does the flattening self.eps_r[i]=
        rows_list.append(m_all)#m_all=[0, 1, 2, ..., N-1], 
        cols_list.append(m_all)#m_all=[0, 1, 2, ..., N-1], so this is the diagonal of the matrix A, where the row and column indices are the same (m_all), 
        vals_list.append(diag_val)
#diag_val is a 1D array of length N, where diag_val[i]= diagonal entry of matrix A corresponding to grid point (ix_all[i], iy_all[i]).  
        # 2) x+1 neighbor (m + Ny) -- only for ix < Nx-1
        mask_xp = ix_all < Nx - 1# 1D array of length N, mask_xp[1]=1 if the grid point (ix_all[1], iy_all[1]) has a neighbor in the +x direction 
        idx_src = m_all[mask_xp]#gives us the flattened indices of the grid points that have a neighbor in the +x direction, which are the source points for the x+1 neighbor interaction.
        idx_dst = idx_src + Ny#gives us the flattened indices of the neighboring grid points in the +x direction, which are the destination points for the x+1 neighbor interaction. Since each row corresponds to a grid point and each column corresponds to a grid point, adding Ny to the source index gives us the index of the neighbor in the +x direction.
        rows_list.append(idx_src)
        cols_list.append(idx_dst)
        vals_list.append(ISX_F[mask_xp] / dx2)

        # 3) x-1 neighbor (m - Ny) only for ix > 0
        mask_xm = ix_all > 0
        idx_src = m_all[mask_xm]
        idx_dst = idx_src - Ny
        rows_list.append(idx_src)
        cols_list.append(idx_dst)
        vals_list.append(ISX_B[mask_xm] / dx2)

        # 4) y+1 neighbor (m + 1) only for iy < Ny-1
        mask_yp = iy_all < Ny - 1
        idx_src = m_all[mask_yp]
        idx_dst = idx_src + 1
        rows_list.append(idx_src)
        cols_list.append(idx_dst)
        vals_list.append(ISY_F[mask_yp] / dx2)

        # 5) y-1 neighbor (m - 1) only for iy > 0
        mask_ym = iy_all > 0
        idx_src = m_all[mask_ym]
        idx_dst = idx_src - 1
        rows_list.append(idx_src)
        cols_list.append(idx_dst)
        vals_list.append(ISY_B[mask_ym] / dx2)
# so by doing these, we add the contributions from the neighboring grid points in the +x, -x, +y, and -y directions to the sparse matrix A
        # Assemble
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        vals = np.concatenate(vals_list)

        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()#this is a function that build the sparse matrix A  using the rows, cols, and vals arrays, and then converts it for efficient solving. 
        b = -self.J_z.ravel()#this is the flattened version of the source term J_z

        return A, b

    def solve(self):
        #Build and solve the FDFD system
    
        A, b = self.build_system()
        self.E_z = spsolve(A, b).reshape(self.Nx, self.Ny)


        return self.E_z#the result is a 2D array, where Ez[i][j]=the electric field at the grid point (i,j) in the computational domain. 

    def compute_H_fields(self):
        #Compute H_x, H_y from E_z using finite differences
        if self.E_z is None:
            raise ValueError("Solve first.")

        dx = self.dx
        k0 = self.k0

        # H_x = (1/(i*omega*mu)) * dE_z/dy
        # H_y = -(1/(i*omega*mu)) * dE_z/dx
        # With mu=1 and omega = k0: factor = 1/(i*k0)

        H_x = np.zeros_like(self.E_z)#2D array to store the H_x field, initialized to zeros. 
        H_y = np.zeros_like(self.E_z)#2D array to store the H_y field, initialized to zeros.

        # Central differences
        H_x[:, 1:-1] = (self.E_z[:, 2:] - self.E_z[:, :-2]) / (2 * dx * 1j * k0)
        H_y[1:-1, :] = -(self.E_z[2:, :] - self.E_z[:-2, :]) / (2 * dx * 1j * k0)
#aproximation of derivatives, Hx=dEz/dy, Hy=-dEz/dx
#dEz/dy at point (i,j) is approximated by (E_z[i, j+1] - E_z[i, j-1]) / (2*dx), and dE_z/dx at point (i,j) is approximated by (E_z[i+1, j] - E_z[i-1, j]) / (2*dx). The factors of 1/(i*k0) come from the relationship between the electric and magnetic fields in the frequency domain for TM polarization.
        return H_x, H_y

    def compute_poynting(self):
        #Compute time-averaged Poynting vector Re(E x H*/2)
        H_x, H_y = self.compute_H_fields()
        E = self.E_z

        # S = Re(E x H*) / 2
        # For TM mode: S_x = Re(E_z * H_y*)/2, S_y = -Re(E_z * H_x*)/2
        S_x = 0.5 * np.real(E * np.conj(H_y))
        S_y = -0.5 * np.real(E * np.conj(H_x))

        return S_x, S_y #S_x and S_y are 2D arrays representing the x and y components of the time-averaged Poynting vector at each grid point in the computational domain. 




def simulate_schwarzschild(b_inf, P_min=2.0, P0=6.0, n_annuli=16,
                            wavelength=0.5, M=2.5, resolution=15,
                            N_pml=12):
    
    from ray_tracing_snell import build_schwarzschild_annuli#returns the edges and n_values for the annular structure of the optical Schwarzschild black hole.

    # Physical dimensions
    R0 = P0 * M       # outer radius in um
    R_S = 2.0 * M     # Schwarzschild radius in um

    # Domain: 60 lambda x 60 lambda (paper specification)
    L = 60 * wavelength  # 30 um
    dx = wavelength / resolution

    fdfd = FDFD2D(L, L, dx, wavelength, N_pml)

    # Build annular system (dimensionless)
    edges_dim, n_values = build_schwarzschild_annuli(b_inf, P_min, P0, n_annuli)
    # Convert edges to physical units
    edges_phys = edges_dim * M

    # Set permittivity
    inner_eps = n_values[-1]**2 - 1j * np.pi#absorbption in the inner core.
    fdfd.set_annular_permittivity(0.0, 0.0, edges_phys, n_values, inner_eps)#returns eps_r, which is the 2D array of permittivity values for each grid.

    # Set Gaussian beam source (beam enters from below, +y direction)
    B0_phys = b_inf * M  # impact parameter in um (x-offset)
    beam_sigma = wavelength / 2.0  # delta = lambda/2 (paper)
    R0_phys = P0 * M

    fdfd.set_gaussian_beam_source(
        B0_phys, beam_sigma, R0_phys,
        center_x=0.0, center_y=0.0
    )

    fdfd.solve()

    return fdfd#2D array of electric field values.


def simulate_kerr_newman(a, rho_Q, b_inf, ell_sign, P_min, P0=6.0,
                          n_annuli=21, wavelength=0.5, M=2.5,
                          resolution=15, N_pml=12):


    from ray_tracing_snell import build_kn_annuli#returns the edges and n_values for the annular structure of the optical Kerr-Newman black hole.

    R0 = P0 * M
    L = 60 * wavelength
    dx = wavelength / resolution

    fdfd = FDFD2D(L, L, dx, wavelength, N_pml)

    edges_dim, n_values = build_kn_annuli(a, rho_Q, b_inf, ell_sign,
                                           P_min, P0, n_annuli)
    edges_phys = edges_dim * M

    inner_eps = n_values[-1]**2 - 1j * np.pi
    fdfd.set_annular_permittivity(0.0, 0.0, edges_phys, n_values, inner_eps)

    B0_phys = b_inf * M
    beam_sigma = wavelength / 2.0
    R0_phys = P0 * M

    fdfd.set_gaussian_beam_source(
        B0_phys, beam_sigma, R0_phys
    )

    fdfd.solve()
    return fdfd