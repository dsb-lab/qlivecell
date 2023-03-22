import matplotlib.pyplot as plt

X = [24, 48, 72]

WT_purple = [422, 1365, 1516]
WT_green  = [396, 1387, 1717]

X_purple = [187, 1344, 2579]
X_green  = [411, 965, 1443]

fig, ax = plt.subplots(1,3, figsize=(15,6))

ax[0].plot(X, WT_purple, marker='o',c=[175/255.0, 0, 175/255.0], label="A12-WT")
ax[0].plot(X, WT_green, marker='o', c=[0,175/255.0,0], label="F3")
ax[0].set_ylabel("# cells")
ax[0].set_xticks(X)
ax[0].set_xlabel("time (h)")
ax[0].set_title("F3 + A12-WT")
ax[0].set_ylim(150, 2700)
ax[0].legend()

ax[1].plot(X, X_purple, marker='o', c=[175/255.0, 0, 175/255.0], label="p53 KO")
ax[1].plot(X, X_green, marker='o', c=[0,175/255.0,0], label="F3")
ax[1].set_xticks(X)
ax[1].set_xlabel("time (h)")
ax[1].set_title("F3 + A12-8")
ax[1].set_ylim(150, 2700)
ax[1].legend()

ax[2].plot(X, WT_green, marker='o', c=[0,175/255.0,0], label="F3 with WT")
ax[2].plot(X, X_green, linestyle='--', marker='*', c=[0,175/255.0,0], label="F3 with p53-KO")
ax[2].set_xticks(X)
ax[2].set_xlabel("time (h)")
ax[2].set_title("F3 + A12-8")
ax[2].set_ylim(150, 2700)
ax[2].legend()

plt.show()