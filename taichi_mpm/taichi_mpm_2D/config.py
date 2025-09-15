quality = 1
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

params = {
    "quality": quality,
    "n_particles": n_particles,
    "n_grid": n_grid,
    "dx": dx,
    "inv_dx": inv_dx,
    "dt": dt,
    "p_vol": p_vol,
    "p_rho": p_rho,
    "p_mass": p_mass,
    "E": E,
    "nu": nu,
    "mu_0": mu_0,
    "lambda_0": lambda_0,
}