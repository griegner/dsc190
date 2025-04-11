import numpy as np
from nilearn.surface import load_surf_mesh
from scipy.sparse import csr_matrix


def heat_kernel(d, sigma):
    """heat kernel for distance d, bandwidth sigma"""
    t = sigma
    return np.exp(-(d**2) / (4 * t))


def lb_emodes(fem, n_modes):
    """Laplace-Beltrami (LB) eigenvalues, eigenmodes"""
    evals, emodes = fem.eigs(k=n_modes + 1)
    # first emode is non-constant
    return evals[1:], emodes[:, 1:]


def heat_kernel_smoothing(evals, emodes, sigma=1, seed=0):
    """heat kernel smoothing using LB eigenvalues, eigenmodes"""
    rng = np.random.default_rng(seed=seed)
    z = rng.standard_normal(size=len(evals))
    return np.sum(np.exp(-evals * sigma) * z * emodes, axis=1)


def adjacency_matrix(surface, dtype=None):
    """adjacency matrix of a surface mesh"""
    surface = load_surf_mesh(surface)
    n = surface.coordinates.shape[0]
    faces = surface.faces
    edges = np.vstack([faces[:, [0, 1]], faces[:, [0, 2]], faces[:, [1, 2]]]).astype(np.int64)
    big = edges[:, 0] > edges[:, 1]
    code = np.concatenate([edges[big, 0] + edges[big, 1] * n, edges[~big, 1] + edges[~big, 0] * n])
    code = np.unique(code)
    u, v = code // n, code % n
    edge_vals = np.ones(code.shape, dtype=dtype) if dtype is not None else np.ones_like(code)
    return csr_matrix(
        (np.concatenate([edge_vals, edge_vals]), (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n)
    )


def iterative_smoothing(surface, surf_data, iterations=1):
    """iterative smoothing by nearest neighbors."""
    mesh = load_surf_mesh(surface)
    weight = 0.5
    A = adjacency_matrix(mesh)

    colsums = np.asarray(A.sum(axis=1)).flatten()
    A = A.multiply(weight / colsums[:, None])
    A.setdiag(weight)

    smoothed = np.asarray(surf_data)
    for _ in range(iterations):
        smoothed = A.dot(smoothed)
    smoothed = smoothed.reshape(surf_data.shape)

    return smoothed


def acf(grf, D):
    """autocorrelation function (acf) of gaussian random field"""
    grf_outer_grf = np.outer(grf, grf) / grf.var()
    idx = np.tril_indices_from(grf_outer_grf)
    prods = grf_outer_grf[idx]
    distance_values = D[idx]

    n_lags = 50
    bin_edges = np.linspace(0, n_lags, 20)
    bin_idx = np.digitize(distance_values, bin_edges) - 1

    binned_sum = np.bincount(bin_idx, weights=prods)
    binned_count = np.bincount(bin_idx)
    avg_corr = np.divide(binned_sum, binned_count, out=np.full_like(binned_sum, np.nan), where=binned_count != 0)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, avg_corr[:-1]
