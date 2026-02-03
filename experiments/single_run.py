import numpy as np

def true_welfare(tau_S, tau_Y, delta):
    mask = (tau_S > 0) & (tau_Y > 0)
    return np.mean(tau_S[mask] + delta * tau_Y[mask])

def plugin_welfare(tau_S_hat, tau_Y_hat, delta):
    mask = (tau_S_hat > 0) & (tau_Y_hat > 0)
    return np.mean(tau_S_hat[mask] + delta * tau_Y_hat[mask])