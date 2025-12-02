import matplotlib.pyplot as plt
import numpy as np

# Adult fidelity matrix
# depths = 3,4,5,6,7,8
adult_fid = np.array([
    [0.92, 0.92, 0.96, 0.94, 0.94, 0.96],   # NN layers = 2
    [0.96, 0.86, 1.00, 0.88, 0.88, 0.88],   # NN layers = 4
    [0.96, 0.80, 0.92, 0.94, 0.90, 0.88],   # NN layers = 8
    [0.90, 0.94, 0.88, 0.86, 0.82, 0.92],   # NN layers = 16
])

# Credit fidelity matrix
# depths = 3,4,5,6,7,8
credit_fid = np.array([
    [0.92, 0.98, 1.00, 0.96, 1.00, 0.98],   # NN layers = 2
    [0.96, 0.96, 0.96, 0.92, 0.98, 0.92],   # NN layers = 4
    [0.90, 0.92, 0.94, 0.94, 0.88, 0.90],   # NN layers = 8
    [0.94, 0.94, 0.98, 0.98, 1.00, 0.98],   # NN layers = 16
])

depths = [3,4,5,6,7,8]
layers = [2,4,8,16]

fig, ax = plt.subplots(1, 2, figsize=(14,6), constrained_layout=True)

adult_plot = np.flipud(adult_fid)
credit_plot = np.flipud(credit_fid)

vmin, vmax = 0.75, 1.0

im1 = ax[0].imshow(adult_plot, cmap='YlGnBu', vmin=vmin, vmax=vmax)

ax[0].set_title("Adult Dataset — Fidelity Heatmap", fontsize=14)
ax[0].set_xticks(range(len(depths)))
ax[0].set_yticks(range(len(layers)))

ax[0].set_xticklabels(depths)
ax[0].set_yticklabels(layers[::-1])

ax[0].set_xlabel("Tree Depth")
ax[0].set_ylabel("NN Layers")

ax[0].axvline(x=2, color='red', linestyle='--', linewidth=2)

im2 = ax[1].imshow(credit_plot, cmap='YlGnBu', vmin=vmin, vmax=vmax)

ax[1].set_title("Credit Dataset — Fidelity Heatmap", fontsize=14)
ax[1].set_xticks(range(len(depths)))
ax[1].set_yticks(range(len(layers)))

ax[1].set_xticklabels(depths)
ax[1].set_yticklabels(layers[::-1])

ax[1].set_xlabel("Tree Depth")
ax[1].set_ylabel("NN Layers")

ax[1].axvline(x=2, color='red', linestyle='--', linewidth=2)

cbar = fig.colorbar(im1, ax=ax, location='right', shrink=0.8)
cbar.set_label("Tree Fidelity", fontsize=12)

plt.show()