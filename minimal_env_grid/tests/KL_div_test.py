import numpy as np
import torch
from torch.distributions import MultivariateNormal, kl_divergence

def kl_gaussian_divergence(mu_prior, P_prior, mu_post, P_post, eps=1e-8):
    mu0 = np.asarray(mu_prior).reshape(-1, 1)   # prior
    mu1 = np.asarray(mu_post).reshape(-1, 1)    # post
    P0 = np.asarray(P_prior, dtype=np.float64)
    P1 = np.asarray(P_post, dtype=np.float64)

    d = P0.shape[0]
    I = np.eye(d)

    P0 = P0 + eps * I
    P1 = P1 + eps * I

    invP1 = np.linalg.inv(P1)

    sign0, logdet0 = np.linalg.slogdet(P0)
    sign1, logdet1 = np.linalg.slogdet(P1)
    if sign0 <= 0 or sign1 <= 0:
        raise ValueError("Covariance not positive definite in KL computation.")

    diff = mu1 - mu0

    div = 0.5 * (
        (logdet1 - logdet0)
        - d
        + np.trace(invP1 @ P0)
        + float((diff.T @ invP1 @ diff)[0][0])
    )
    return float(div)

def compute_kl_divergence(mean_p, cov_p, mean_q, cov_q):
        """
        Compute the Kullback–Leibler divergence D_KL(P || Q) between two multivariate Gaussians.

        Parameters
        ----------
        mean_p : np.ndarray
            Mean vector of distribution P (n,)
        cov_p : np.ndarray
            Covariance matrix of distribution P (n x n)
        mean_q : np.ndarray
            Mean vector of distribution Q (n,)
        cov_q : np.ndarray
            Covariance matrix of distribution Q (n x n)

        Returns
        -------
        float
            The KL divergence D_KL(P || Q)
        """
        n = mean_p.shape[0]

        # Ensure inputs are NumPy arrays
        mean_p = np.atleast_1d(mean_p)
        mean_q = np.atleast_1d(mean_q)
        cov_p = np.atleast_2d(cov_p)
        cov_q = np.atleast_2d(cov_q)

        # Compute inverses and determinants
        inv_cov_q = np.linalg.inv(cov_q)
        det_cov_p = np.linalg.det(cov_p)
        det_cov_q = np.linalg.det(cov_q)

        # Compute trace term
        trace_term = np.trace(inv_cov_q @ cov_p)

        # Mean difference
        mean_diff = mean_q - mean_p
        mean_term = mean_diff.T @ inv_cov_q @ mean_diff

        # Log determinant ratio term
        log_det_term = np.log(det_cov_q / det_cov_p)

        # Combine terms (0.5 * [log|Σq|/|Σp| - n + Tr(invΣq Σp) + (μq - μp)^T invΣq (μq - μp)])
        d_kl = 0.5 * (log_det_term - n + trace_term + mean_term)

        return float(d_kl)

def torch_gaussian_kl(mu_p, P_p, mu_q, P_q):
    p = MultivariateNormal(
        loc=torch.tensor(mu_p, dtype=torch.float64),
        covariance_matrix=torch.tensor(P_p, dtype=torch.float64),
    )
    q = MultivariateNormal(
        loc=torch.tensor(mu_q, dtype=torch.float64),
        covariance_matrix=torch.tensor(P_q, dtype=torch.float64),
    )
    return float(kl_divergence(p, q))

def renyi_mc_from_p(mu_p, P_p, mu_q, P_q, alpha=0.5, n=200000, seed=0):
    torch.manual_seed(seed)

    p = MultivariateNormal(
        loc=torch.tensor(mu_p, dtype=torch.float64),
        covariance_matrix=torch.tensor(P_p, dtype=torch.float64),
    )
    q = MultivariateNormal(
        loc=torch.tensor(mu_q, dtype=torch.float64),
        covariance_matrix=torch.tensor(P_q, dtype=torch.float64),
    )

    x = p.sample((n,))
    log_p = p.log_prob(x)
    log_q = q.log_prob(x)

    # Since x ~ p:
    # integral p^alpha q^(1-alpha) dx
    # = E_p[exp((alpha-1) log p + (1-alpha) log q)]
    vals = torch.exp((alpha - 1.0) * log_p + (1.0 - alpha) * log_q)
    est = torch.log(vals.mean()) / (alpha - 1.0)
    return float(est)

def renyi_gaussian_divergence(mu_prior, P_prior, mu_post, P_post, alpha=0.5, eps=1e-8, logging=False):
    """
    Rényi divergence D_alpha(post || prior) 
    Uses position mean/covariance
    """
    #print(type(alpha), alpha)
    if alpha <= 0.0 or np.isclose(alpha, 1.0):
        raise ValueError("alpha must be > 0 and != 1")

    mu0 = np.asarray(mu_prior).reshape(-1, 1)
    mu1 = np.asarray(mu_post).reshape(-1, 1)
    P0 = np.asarray(P_prior)
    P1 = np.asarray(P_post)

    d = P0.shape[0]
    I = np.eye(d)

    P0 = P0 + eps * I
    P1 = P1 + eps * I

    P_alpha = alpha * P1 + (1.0 - alpha) * P0
    P_alpha = P_alpha + eps * I

    sign0, logdet0 = np.linalg.slogdet(P0)
    sign1, logdet1 = np.linalg.slogdet(P1)
    signa, logdeta = np.linalg.slogdet(P_alpha)

    if logging:
        print(f"Rényi divergence computation (alpha={alpha}):")
        print(f"P_prior:\n{P0}")
        print(f"P_post:\n{P1}")
        print(f"P_alpha:\n{P_alpha}")
        print(f"logdet(P_prior)={logdet0}, logdet(P_post)={logdet1}, logdet(P_alpha)={logdeta}")

    if sign0 <= 0 or sign1 <= 0 or signa <= 0:
        raise ValueError("Covariance not positive definite in Rényi computation.")

    diff = mu1 - mu0
    quad = float((diff.T @ np.linalg.inv(P_alpha) @ diff)[0, 0])

    term = (
        logdeta
        - alpha * logdet1
        - (1.0 - alpha) * logdet0
        + alpha * (1.0 - alpha) * quad
    )

    div = 0.5 * term / (1.0 - alpha)

    if logging:
        print(f"quadratic term: {quad}")
        print(f"Rényi divergence: {div}")

    return float(div)

"""
# Example:
mu_prior = np.array([1.0, 2.0])
P_prior  = np.array([[2.0, 0.3],
                     [0.3, 1.5]])

mu_post  = np.array([1.1, 1.9])
P_post   = np.array([[1.2, 0.1],
                     [0.1, 1.0]])

kl_ref = torch_gaussian_kl(mu_prior, P_prior, mu_post, P_post)
print("Torch  KL:", kl_ref)
kl_ours = kl_gaussian_divergence(mu_prior, P_prior, mu_post, P_post)
print("Our    KL:", kl_ours)
kl_linh = compute_kl_divergence(mu_prior, P_prior, mu_post, P_post)
print("Linh's KL:", kl_linh)

kl_ref = torch_gaussian_kl(mu_post, P_post, mu_post, P_post)
print("Torch KL:", kl_ref)
kl_ours = kl_gaussian_divergence(mu_post, P_post, mu_post, P_post)
print("Our   KL:", kl_ours)

print("========================================")
alpha = 0.5
renyi_ref = renyi_mc_from_p(mu_prior, P_prior, mu_post, P_post, alpha=alpha)
print(f"Rényi (MC)   with alpha={alpha}:", renyi_ref)
renyi_ours = renyi_gaussian_divergence(mu_post, P_post, mu_prior, P_prior, alpha=alpha)
print(f"Rényi (ours) with alpha={alpha}:", renyi_ours)
"""

def compare_2KL(mean_p, cov_p, mean_q, cov_q):
    kl = kl_gaussian_divergence(mean_p, cov_p, mean_q, cov_q)
    kl_linh = compute_kl_divergence(mean_p, cov_p, mean_q, cov_q)

    print(f"KL (Amaury) : {kl:.6f}")
    print(f"KL (Linh)   : {kl_linh:.6f}")

# Example 1
mean_1 = np.array([1, 1, 1, 1])
mean_2 = np.array([4, 2, 6, 9])

cov_1 = np.array([
	[4.0, 1.0, 0.5, 0.2],
	[1.0, 3.0, 0.3, 0.1],
	[0.5, 0.3, 2.5, 0.4],
	[0.2, 0.1, 0.4, 2.0]
])

cov_2 = np.array([
	[5.0, 0.8, 0.2, 0.1],
	[0.8, 4.0, 0.6, 0.3],
	[0.2, 0.6, 3.5, 0.5],
	[0.1, 0.3, 0.5, 2.5]
])

compare_2KL(mean_1, cov_1, mean_2, cov_2)
#KL-div = 15.316510112332585

# Example 2
mean_1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
mean_2 = np.array([2, 1, 4, 3, 6, 5, 8, 7])

cov_1 = np.array([
    [5.0, 1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0],
    [1.0, 4.5, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0],
    [0.5, 0.3, 4.0, 0.6, 0.2, 0.1, 0.1, 0.0],
    [0.2, 0.2, 0.6, 3.8, 0.5, 0.2, 0.1, 0.1],
    [0.1, 0.1, 0.2, 0.5, 3.5, 0.4, 0.2, 0.1],
    [0.0, 0.1, 0.1, 0.2, 0.4, 3.2, 0.3, 0.2],
    [0.0, 0.0, 0.1, 0.1, 0.2, 0.3, 3.0, 0.4],
    [0.0, 0.0, 0.0, 0.1, 0.1, 0.2, 0.4, 2.8]
])

cov_2 = np.array([
    [6.0, 0.8, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0],
    [0.8, 5.5, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0],
    [0.4, 0.5, 5.0, 0.7, 0.3, 0.2, 0.1, 0.0],
    [0.2, 0.3, 0.7, 4.5, 0.6, 0.3, 0.2, 0.1],
    [0.1, 0.2, 0.3, 0.6, 4.2, 0.5, 0.3, 0.2],
    [0.0, 0.1, 0.2, 0.3, 0.5, 3.8, 0.4, 0.3],
    [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 3.5, 0.5],
    [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 3.2]
])

compare_2KL(mean_1, cov_1, mean_2, cov_2)
#KL-div = 1.171588342655546

# Example 3
mean_1 = np.array([1, 1, 1])
mean_2 = np.array([2, 2, 2])

cov_1 = np.array([
	[4.0, 1.0, 0.5],
	[1.0, 3.0, 0.3],
	[0.5, 0.3, 2.5]
])

cov_2 = np.array([
	[4.0, 1.0, 0.5],
	[1.0, 3.0, 0.3],
	[0.5, 0.3, 2.5]
])

compare_2KL(mean_1, cov_1, mean_2, cov_2)
#KL-div = 0.3683027351067816

# Example 4
mean_1 = np.array([0])
mean_2 = np.array([0])

cov_1 = np.array([1])
cov_2 = np.array([1])

compare_2KL(mean_1, cov_1, mean_2, cov_2)
#KL-div = 0.0