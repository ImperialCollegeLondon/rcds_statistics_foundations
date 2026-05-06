# ECRI Statistics foundations
# Chapter 8: Evidence
# Jesús Urtasun Elizari: RCDS 2026



# Exercise 3:

# 1. Suppose we observe the following sequence of coin tosses: x = (1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0) with 1 = heads and 0 = tails.

# 2. Compute the number of heads $k$ and total number of trials $n$, and compute the posterior distribution for \theta (probability of H) given the observed x.

# 3. Plot the prior and posterior distributions of \theta.

# 4. Discuss how the observed data influence the updated belief about $\theta$.



# Import libraries ............................................................

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta



# 1. Prepare coin example ........................................................

# Prior beliefs
alpha_prior = 50 # prior "pseudo-count" of heads
beta_prior = 50  # prior "pseudo-count" of tails

# Observed data
data = [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]  # 1 = head, 0 = tail
heads = sum(data)
tails = len(data) - heads

# Print summary
print("Prior mean:", alpha_prior / (alpha_prior + beta_prior))
print(f"Observed data: {data}")
print(f"Heads: {heads}, Tails: {tails}")



# 2. Posterior update .........................................................

# Conjugate posterior
alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

# Print summary
print("Posterior mean:", alpha_post / (alpha_post + beta_post))



# 3. Plot distributions .......................................................

# Prepare for plot
x = np.linspace(0, 1, 200)
prior_pdf = beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = beta.pdf(x, alpha_post, beta_post)

# Plot prior / posterior
plt.figure()
plt.plot(x, prior_pdf, label = "Prior", linestyle = "--")
plt.plot(x, posterior_pdf, label = "Posterior")
plt.title("Bayesian Update for Coin Bias")
plt.xlabel("\u03B8 (probability of heads)")
plt.ylabel("Density")
plt.legend()
# plt.savefig("bayesian_posterior.png", dpi = 300, bbox_inches = "tight")
plt.show()
