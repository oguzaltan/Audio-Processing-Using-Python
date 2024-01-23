import numpy as np
from numpy import histogram
import matplotlib.pyplot as plt
#matplotlib qt

x1 = np.random.uniform(size=10**6)
x2 = np.random.rand(10**6)
x3 = np.random.randn(10**6)

x2_var = x2*np.sqrt(0.5)/np.sqrt(np.var(x2))
x3_var = x3*np.sqrt(1.5)/np.sqrt(np.var(x3))

x2_var_mean2 = x2_var + (2-np.mean(x2_var))
x3_var_mean2 = x3_var + (2-np.mean(x3_var))

x1_norm = x1/(np.max(x1)/4)
x1_mean2_norm = x1_norm + (2-np.mean(x1_norm))

print("Mean of x1", np.mean(x1_mean2_norm))
print("Mean of x2", np.mean(x2_var_mean2))
print("Mean of x3", np.mean(x3_var_mean2))
print("Max of x1", np.max(x1_mean2_norm))
print("Variance of x2: ", np.var(x2_var_mean2))
print("Variance of x3: ", np.var(x3_var_mean2))

edges1 = np.arange(np.min(x1_mean2_norm), np.max(x1_mean2_norm), 0.1)
edges2 = np.arange(np.min(x2_var_mean2), np.max(x2_var_mean2), 0.1)
edges3 = np.arange(np.min(x3_var_mean2), np.max(x3_var_mean2), 0.1)

x1_hist, _ = np.histogram(x1_mean2_norm, bins=edges1)
x2_hist, _ = np.histogram(x2_var_mean2, bins=edges2)
x3_hist, _ = np.histogram(x3_var_mean2, bins=edges3)

centers1 = 0.5*(edges1[1:] + edges1[:-1])
centers2 = 0.5*(edges2[1:] + edges2[:-1])
centers3 = 0.5*(edges3[1:] + edges3[:-1])

plt.plot(centers1, x1_hist, label="x1")
plt.plot(centers2, x2_hist, label="x2")
plt.plot(centers3, x3_hist, label="x3")
plt.legend(loc="upper left")
plt.show()

p1 = x1_hist/np.sum(x1_hist)
p2 = x2_hist/np.sum(x2_hist)
p3 = x3_hist/np.sum(x3_hist)

c1 = np.cumsum(p1)
c2 = np.cumsum(p2)
c3 = np.cumsum(p3)

plt.plot(centers1, c1, label="c1")
plt.plot(centers2, c2, label="c2")
plt.plot(centers3, c3, label="c3")
plt.legend(loc="upper left")
plt.grid()
plt.show()
