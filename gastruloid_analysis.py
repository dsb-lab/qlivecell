import matplotlib.pyplot as plt
import numpy as np

X = np.array([24, 48, 72])

WT_purple = np.array([422, 1365, 1516])
WT_green  = np.array([396, 1387, 1717])
Totals_WT = WT_purple + WT_green

X_purple = np.array([187, 1344, 2579])
X_green  = np.array([411, 965, 1443])
Totals_X  = X_purple + X_green

fig, ax = plt.subplots(2,2, figsize=(15,15))
ax = ax.flatten()

ax[0].plot(X, WT_purple, marker='o',c=[175/255.0, 0, 175/255.0], label="A12-WT")
ax[0].plot(X, WT_green, marker='o', c=[0,175/255.0,0], label="F3")
ax[0].set_ylabel("# cells")
ax[0].set_xticks(X)
ax[0].set_xlabel("time (h)")
ax[0].set_title("F3 + A12-WT")
ax[0].set_ylabel("# cells")
ax[0].set_ylim(150, 2700)
ax[0].legend()

ax[1].plot(X, X_purple, marker='o', c=[175/255.0, 0, 175/255.0], label="p53 KO")
ax[1].plot(X, X_green, marker='o', c=[0,175/255.0,0], label="F3")
ax[1].set_xticks(X)
ax[1].set_ylabel("# cells")
ax[1].set_xlabel("time (h)")
ax[1].set_title("F3 + A12-8")
ax[1].set_ylim(150, 2700)
ax[1].legend()

ax[2].plot(X, WT_green, marker='o', c=[0,175/255.0,0], label="F3 with WT")
ax[2].plot(X, X_green, linestyle='--', marker='*', c=[0,175/255.0,0], label="F3 with p53-KO")
ax[2].set_xticks(X)
ax[2].set_xlabel("time (h)")
ax[2].set_title("A12-WT vs A12-8")
ax[2].set_ylabel("# cells")
ax[2].set_ylim(150, 2700)
ax[2].legend()

ax[3].plot(X, WT_green/Totals_WT, marker='o', c=[0,175/255.0,0], label="F3 with WT")
ax[3].plot(X, X_green/Totals_X, linestyle='--', marker='*', c=[0,175/255.0,0], label="F3 with p53-KO")
ax[3].set_xticks(X)
ax[3].set_xlabel("time (h)")
ax[3].set_title("A12-WT vs A12-8")
ax[3].set_ylabel("fraction of cells")
ax[3].set_ylim(0, 1)
ax[3].legend()
plt.show()