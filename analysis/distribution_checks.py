import matplotlib.pyplot as plt
import scipy.stats as stats
from experiments.monte_carlo_convergence import monte_carlo

gaps = monte_carlo(n=10000, R=500)

# Histogram
plt.hist(gaps, bins=30, density=True)
plt.title("Distribution of W_hat - W_true")
plt.show()

# QQ plot
stats.probplot(gaps, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()
