# ECRI Statistics foundations
# Chapter 8: Evidence
# Jesús Urtasun Elizari: RCDS 2026



# Exercise 1:

# 1. Simulate an example where two dice are given, one fair with P(6) = 1/6 and other biased P(6) = 1/2.

# 2. Compute the Bayesian probability of choosing the biased one, if i) 1st roll gives a 6, and ii) 2nd roll also gives a 6

# 3. Simulate an example of a very rare disease, present only in 5% of population, and a test with 99% accuracy.

# 4. Compute the Bayesian probability of actually having the disease, if i) 1st test gives a +, and ii) 2nd test also gives a +



# Import libraries ............................................................

import numpy as np
import matplotlib.pyplot as plt

# Function applying Bayes' Theorem
def bayes_theorem(prior, likelihood, marginal):
    return (likelihood * prior) / marginal

# Function computing marginal probability of evidence
def marginal_probability(likelihoods, priors):
    return np.sum(np.array(likelihoods) * np.array(priors))



# Example 1: rolling dice .....................................................

print("Example 1: rolling dice:\n")

# Initial priors
P_H1 = 0.5  # Prior: choosing biased die
P_H2 = 0.5  # Prior: choosing fair die
print(f"Prior probabilities:\nDie is biased, P(H1) = {P_H1}\nDie is fair, P(H2) = {P_H2}\n")

# First roll: rolled a 6
P_E_given_H1 = 1/2  # Likelihood of rolling 6 on biased die
P_E_given_H2 = 1/6  # Likelihood of rolling 6 on fair die

# Total probability of evidence
P_E = marginal_probability([P_E_given_H1, P_E_given_H2], [P_H1, P_H2])

# Posterior probability after first roll
P_H1_given_E1 = bayes_theorem(P_H1, P_E_given_H1, P_E)
print(f"After 1st roll of 6: Probability that the die is biased: {P_H1_given_E1:.2f}")



# Update priors for second roll ...............................................

P_H1 = P_H1_given_E1
P_H2 = 1 - P_H1

# Second roll: rolled another 6
P_E2_given_H1 = 0.5  # Likelihood of rolling another 6 on biased die
P_E2_given_H2 = 1/6  # Likelihood of rolling another 6 on fair die

# Total probability of second evidence
P_E2 = marginal_probability([P_E2_given_H1, P_E2_given_H2], [P_H1, P_H2])

# Posterior probability after second roll
P_H1_given_E2_E1 = bayes_theorem(P_H1, P_E2_given_H1, P_E2)
print(f"After 2nd roll of 6: Probability that the die is biased: {P_H1_given_E2_E1:.2f}\n")



# Example 2: medical test .....................................................

print("Example 2: medical test:\n")

# Initial priors
P_D = 0.05  # Prior: having the disease
P_not_D = 0.95  # Prior: not having the disease
print(f"Prior probabilities:\nHave disease, P(H1) = {P_D}\nNot have disease, P(H2) = {P_not_D}\n")

# First test: tested positive
P_E_given_D = 0.99  # Likelihood of testing positive given disease
P_E_given_not_D = 0.01  # Likelihood of testing positive without disease

# Total probability of evidence (first test result)
P_E = marginal_probability([P_E_given_D, P_E_given_not_D], [P_D, P_not_D])

# Posterior probability after first test
P_D_given_E1 = bayes_theorem(P_D, P_E_given_D, P_E)
print(f"After 1st positive test: Probability of having the disease: {P_D_given_E1:.2f}")



# Update priors for second test ...............................................
P_D = P_D_given_E1
P_not_D = 1 - P_D

# Second test: tested positive again
P_E2_given_D = 0.99  # Likelihood of testing positive again given disease
P_E2_given_not_D = 0.01  # Likelihood of testing positive again without disease

# Total probability of second evidence (second test result)
P_E2 = marginal_probability([P_E2_given_D, P_E2_given_not_D], [P_D, P_not_D])

# Posterior probability after second test
P_D_given_E2_E1 = bayes_theorem(P_D, P_E2_given_D, P_E2)
print(f"After 2nd positive test: Probability of having the disease: {P_D_given_E2_E1:.2f}\n")
