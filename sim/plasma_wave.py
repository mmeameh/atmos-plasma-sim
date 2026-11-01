import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PlasmaSimulationConfig:
    nx: int = 64              # East-West spatial grid points
    ny: int = 64              # Vertical altitude grid points
    dx: float = 1000.0        # Spatial step (m)
    dy: float = 1000.0        # Altitude step (m)
    dt: float = 0.05          # Time step (s)
    diffusion_coeff: float = 5.0 # Ambipolar diffusion coefficient (m^2/s)
    gravity: float = 9.81     # Gravity (m/s^2)
    bg_density: float = 1e11  # Background electron density (m^-3)

class PlasmaGrid2D:
    """Simulates 2D non-linear equatorial plasma depletion bubbles (Equatorial Spread-F)."""

    def __init__(self, config: PlasmaSimulationConfig):
        self.cfg = config
        # Density matrix n[y][x]
        self.density: List[List[float]] = [
            [self.cfg.bg_density for _ in range(self.cfg.nx)]
            for _ in range(self.cfg.ny)
        ]
        self._initialize_exponential_ionosphere()

    def _initialize_exponential_ionosphere(self):
        """Sets up exponential vertical density gradient with small sinusoidal perturbation."""
        scale_height = 20000.0 # 20 km
        for j in range(self.cfg.ny):
            alt = j * self.cfg.dy
            base = self.cfg.bg_density * math.exp(alt / scale_height)
            for i in range(self.cfg.nx):
                # Sinusoidal perturbation seeding Rayleigh-Taylor instability
                wave = 0.05 * math.sin(2.0 * math.pi * i / self.cfg.nx)
                self.density[j][i] = base * (1.0 + wave)

    def step(self):
        """Advances plasma density field by one time-step using explicit 2D FDTD scheme."""
        new_density = [row[:] for row in self.density]
        nx, ny = self.cfg.nx, self.cfg.ny
        dx2 = self.cfg.dx * self.cfg.dx
        dy2 = self.cfg.dy * self.cfg.dy
        diff = self.cfg.diffusion_coeff
        dt = self.cfg.dt

        for j in range(1, ny - 1):
            for i in range(nx):
                i_left = (i - 1 + nx) % nx
                i_right = (i + 1) % nx

                # 2D Laplacian for diffusion
                laplacian_x = (self.density[j][i_right] - 2 * self.density[j][i] + self.density[j][i_left]) / dx2
                laplacian_y = (self.density[j + 1][i] - 2 * self.density[j][i] + self.density[j - 1][i]) / dy2

                # Buoyancy drift velocity (Rayleigh-Taylor upward buoyant term)
                buoyancy_vy = 0.1 * math.sin(2.0 * math.pi * i / nx)
                advection_y = buoyancy_vy * (self.density[j + 1][i] - self.density[j - 1][i]) / (2.0 * self.cfg.dy)

                dn_dt = diff * (laplacian_x + laplacian_y) - advection_y
                new_density[j][i] += dt * dn_dt

        self.density = new_density

    def compute_total_electron_content(self) -> List[float]:
        """Integrates vertical column density to simulate TEC (Total Electron Content) observations."""
        tec = [0.0] * self.cfg.nx
        for i in range(self.cfg.nx):
            col_sum = sum(self.density[j][i] for j in range(self.cfg.ny))
            tec[i] = col_sum * self.cfg.dy / 1e16 # in TEC units (TECU)
        return tec
