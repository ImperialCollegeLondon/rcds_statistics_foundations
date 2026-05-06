# ECRI Statistics foundations
# Chapter 8: Evidence
# Jesús Urtasun Elizari: RCDS 2026



# Exercise 2:

# 1. Simulate an example where two dice are given, one fair with P(6) = 1/6 and other biased P(6) = 1/2.

# 2. Compute the Bayesian probability of choosing the biased one, if i) 1st roll gives a 6, and ii) 2nd roll also gives a 6

# 1. Simulate 50 rolls and compute the total updated probability P(bised | E) and P(fair | E) given the results E

# 2. Plot both probabilities in terms of the number of rolls



# Import libraries ............................................................

import numpy as np
import matplotlib.pyplot as plt

# Random seed
np.random.seed(123)

# Function applying Bayes' Theorem
def bayes_theorem(prior, likelihood, marginal):
    return (likelihood * prior) / marginal

# Function computing probability of evidence
def total_probability(likelihoods, priors):
    return np.sum(np.array(likelihoods) * np.array(priors))



# Prepare dice example ........................................................

print("Prepare dice example:\n")

# Simulation parameters
n_rolls = 50 # Number of dice rolls to simulate
p_6_fair = 1/6 # Probability of rolling a 6 with a fair die
p_6_biased = 1/2 # Probability of rolling a 6 with the biased die

# Initial priors
P_H1 = 0.5  # Prior: the die is biased
P_H2 = 0.5  # Prior: the die is fair

# Assume we picked the biased die, with of rolling a 6
true_prob = p_6_biased

# Simulate the number of 6's in n_rolls trials using a binomial distribution
six_counts = np.random.binomial(1, true_prob, n_rolls)

# Lists to track the evolution of posterior probabilities
posterior_H1_values = [P_H1]
posterior_H2_values = [P_H2]



# Simulate multiple dice rolls ................................................

print("Simulate multiple dice rolls:\n")

# Perform Bayesian updating for each roll
for i, outcome in enumerate(six_counts):

    # Store roll outcome
    if outcome == 1:
        roll_result = 6
    else:
        roll_result = "not 6"
    
    print(f"Roll {i+1}: {roll_result}")
    
    # Likelihood of the evidence (whether it's a 6 or not)
    if roll_result == 6:
        P_E_given_H1 = p_6_biased # Likelihood of rolling 6 on biased die
        P_E_given_H2 = p_6_fair # Likelihood of rolling 6 on fair die
    else:
        P_E_given_H1 = 1 - p_6_biased # Likelihood of NOT rolling a 6 on biased die
        P_E_given_H2 = 1 - p_6_fair # Likelihood of NOT rolling a 6 on fair die

    # Total probability of the evidence
    P_E = total_probability([P_E_given_H1, P_E_given_H2], [P_H1, P_H2])

    # Update posterior probabilities using Bayes' Theorem
    P_H1 = bayes_theorem(P_H1, P_E_given_H1, P_E)
    P_H2 = 1 - P_H1

    # Store the posterior probability for H1
    posterior_H1_values.append(P_H1)
    posterior_H2_values.append(P_H2)

    print(f"Updated Probability that the die is biased: {P_H1:.4f}\n")



# Plot evolution of posterior probability .....................................

plt.figure()
plt.plot(posterior_H1_values, marker = "o", linestyle = "-", color = "b", label = "P(H1 | Evidence)")
plt.plot(posterior_H2_values, marker = "o", linestyle = "-", color = "r", label = "P(H2 | Evidence)")
plt.title("Evolution of posterior probabilities")
plt.xlabel("Number of rolls")
plt.ylabel("Posterior probability")
plt.legend()
plt.grid(True)
# plt.savefig("evolution_posterior.png", dpi = 300, bbox_inches = "tight")
plt.show()
