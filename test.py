import matplotlib.pyplot as plt

fig, ax = plt.subplots(2,2)
ax[0,0].plot(range(10), range(10))
ax[0,1].plot(range(10), range(10))
ax[1,0].plot(range(10), range(10))
ax[1,1].plot(range(10), range(10))

plt.tight_layout()

_ax = fig.add_subplot()
_ax.xaxis.set_visible(False)
_ax.yaxis.set_visible(False)
_ax.set_zorder(1000)
_ax.patch.set_alpha(0.5)
_ax.patch.set_color('r')

plt.show()