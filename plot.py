# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from scipy.stats import linregress
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.style.use(r'./paper.mplstyle')

def load_data(name, **kwargs):
    return np.loadtxt('data/' + name + '.csv', delimiter=',', **kwargs).transpose()

def cm2in(x, y):
    return (x/2.54, y/2.54)

def max_ent(m):
    return 1 - ((m-1)/m)**(m-1)

def gray_out_forbidden(ax, m):
    y_lim = ax.get_ylim()
    x_lim = ax.get_xlim()

    threshhold = max_ent(m)

    ax.axhline(0, c = 'k', lw = 1, ls = '--', zorder = 1.9) # standard zorder for plots is at z=2, so z=1.9 ensures its always drawn behind the plots
    ax.fill_between([-10**6, 10**6], -10, 0, color = 'black', alpha = 0.25) # 10^6 should be enough

    ax.axhline(threshhold, c = 'k', lw = 1, ls = '--', zorder = 1.9) 
    ax.fill_between([-10**6, 10**6], threshhold, 2*threshhold, color = 'black', alpha = 0.25)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

LINEWIDTH_FULL = 16.3 # \linewidth for figure*
LINEWIDTH_HALF = 8.6  # \linewidth for figure
GOLDEN_RATIO = 1.61803



# %%
# =================== Random W states ===================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=cm2in(LINEWIDTH_HALF, 12))

data = load_data("Random-one-photon")
data_truncated = data[:, :41]  # up to n=200

# ======== data =========
ax1.plot(data_truncated[0], data_truncated[1])

# draw confidence interval
ax1.fill_between(data_truncated[0], data_truncated[2], data_truncated[3], alpha=0.2)

ax1.set_xscale("log")

# gray out zone. Do it manually as it changes for each x point
y_lim = ax1.get_ylim()
x_lim = ax1.get_xlim()
all_x = np.arange(data_truncated[0, 0], data_truncated[0, -1] + 1, 0.2)

ax1.axhline(0, c="k", lw=1, ls="--", zorder=1.9)
ax1.fill_between([-(10**6), 10**6], -10, 0, color="black", alpha=0.25)

ax1.plot(all_x, max_ent(all_x), c="k", lw=1, ls="--", zorder=1.9)
ax1.fill_between(all_x, max_ent(all_x), 2, color="black", alpha=0.25)

ax1.set_xlim(x_lim)
ax1.set_ylim(y_lim)

ax1.set_ylabel(r"entanglement $E_g$")

# ======== convergence =========
convergence = max_ent(data[0]) - data[1]
min_mi = 25 + 1  # data[0,25] = 50
lin_fit = linregress(np.log10(data[0, min_mi:]), np.log10(convergence[min_mi:]))

ax2.plot([], [])
ax2.plot([], [])
(line,) = ax2.plot(
    data[0],
    10**lin_fit.intercept * data[0] ** lin_fit.slope,
    lw=1.5,
    label=r"$E_{g, \mathrm{max}} - E_g$",
)
dots = ax2.scatter(
    data[0],
    convergence,
    s=6,
    label=r"fit $\propto M^{" + f"{lin_fit.slope:.3f}" + r"}$",
)

ax2.set_xscale("log")
ax2.set_yscale("log")

ax2.legend(handles=[line, dots], loc="upper right", bbox_to_anchor=(0.97, 0.97))

ax2.set_xlabel(r"mode number $M$")
ax2.set_ylabel(r"difference")

fig.savefig("plots/Random-one-photon.pdf")

plt.show()


# %%
# =================== Density Plot coin entanglement ===================

fig, ax = plt.subplots(figsize=cm2in(7.2, 6))

data = load_data("HW-ICs-Coin_Ent-Asymptotic").transpose()
theta = load_data("HW-ICs-Thetas")
phi = load_data("HW-ICs-Phis")
X, Y = np.meshgrid(phi, theta)


cm = ax.pcolormesh(X, Y, data, shading="gouraud")

ax.grid(False)

xticks_major = np.array([0, 1 / 2, 1, 3 / 2, 2]) * np.pi
xticks_minor = np.array([1 / 4, 3 / 4, 5 / 4, 7 / 4]) * np.pi
yticks_major = np.array([0, 1 / 4, 1 / 2, 3 / 4, 1]) * np.pi
yticks_minor = np.array([1 / 8, 3 / 8, 5 / 8, 7 / 8]) * np.pi

ax.set_xticks(xticks_major)
ax.set_xticks(xticks_minor, minor=True)
ax.set_yticks(yticks_major)
ax.set_yticks(yticks_minor, minor=True)

ax.set_xticklabels(
    [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"]
)
ax.set_xlabel(r"$\phi$")
ax.set_yticklabels(
    [r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"]
)
ax.set_ylabel(r"$\theta$")

cb = fig.colorbar(cm, shrink=0.7, aspect=10, ticks=[], pad=0.01)
data_min = np.min(data)
data_max = np.max(data)
cb.ax.text(-0.4, -0.11, f"${data_min:.2f}$", transform=cb.ax.transAxes)
cb.ax.text(-0.4, 1.04, f"${data_max:.2f}$", transform=cb.ax.transAxes)
cb.set_label(r"entanglement $E_g$", labelpad=8)

fig.savefig("plots/HW-diff_ICs-Coin_Entanglement.png")

plt.show()


# %%
# =================== P = 4, 8, 3&6 ===================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=cm2in(LINEWIDTH_HALF, 10))

data_P4 = load_data("HW-full-N=1-P=4-n=25")
data_P8 = load_data("HW-full-N=1-P=8-n=25")
data_P3 = load_data("HW-full-N=1-P=3-n=50")


(line4,) = ax1.plot(data_P4[0], data_P4[1], "o-", label=r"$P=4$")
(line8,) = ax1.plot(data_P8[0], data_P8[1], "o-", label=r"$P=8$")

# advance to third color in cycle
ax2.plot([], [])
ax2.plot([], [])
(line3,) = ax2.plot(
    data_P3[0], data_P3[1], "o-", label=r"$P=3$ and $P=6$"
)  # only plot P=3 as P=6 is identical

# gray out areas and set the ylim for both
ax1.set_ylim(-0.05, 0.65)
gray_out_forbidden(ax1, 8 * 2)

ax2.set_ylim(ax1.get_ylim())
gray_out_forbidden(ax2, 3 * 2)

# set y ticks for top row and remove ticklabels from second plot
yticks = np.linspace(0, 0.6, 7)
ax1.set_yticks(yticks)
ax2.set_yticks(yticks)

# set x ticks for top row
xticks = np.arange(0, 25, 4)
ax1.set_xticks(xticks)
ax1.set_xticks(xticks + 2, minor=True)

xticks = np.arange(0, 60, 10)
ax2.set_xticks(xticks)
ax2.set_xticks(xticks[:-1] + 5, minor=True)

# labels and legend
ax1.set_ylabel(r"entanglement $E_g$")
ax2.set_ylabel(r"entanglement $E_g$")
ax2.set_xlabel(r"time step $n$")

fig.legend(
    handles=[line4, line8, line3],
    loc="outside lower center",
    ncols=2,
    bbox_to_anchor=(0.06, -0.14, 1, 1),
)


fig.savefig("plots/HW-dynamics_P=3,4,6,8.pdf", bbox_inches="tight")

plt.show()


# %%
# =================== P=500 ===================
fig, ax = plt.subplots(figsize=cm2in(LINEWIDTH_FULL, 6))

data_P500 = load_data("HW-full-N=1-P=500-n=5000")

ax.plot(data_P500[0], data_P500[1], lw=0.8)

ax.set_ylim(0.629, 0.6312)

yticks = np.arange(0.6290, 0.6310, 5e-4)
ax.set_yticks(yticks)
ax.set_yticks(yticks[:-1] + 2.5e-4, minor=True)

xticks = np.arange(0, 5001, 1000)
ax.set_xticks(xticks)
ax.set_xticks(xticks[:-1] + 500, minor=True)

axin = inset_axes(
    ax,
    width="55%",
    height="50%",
    loc="lower right",
    borderpad=1.8,
)

axin.plot(data_P500[0, 50:250], data_P500[1, 50:250], lw=1.5)

axin.set_xticks(np.arange(50, 250, 25), minor=True)


ax.set_xlabel(r"time step $n$")
ax.set_ylabel(r"entanglement $E_g$")

fig.savefig("plots/HW-dynamics_P=500.pdf")

plt.show()


# %%
# =================== Density Plots for Initial conditions ===================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=cm2in(7.8, 12))

data1 = load_data("HW-ICs-N=1-P=5-n=2").transpose()
data2 = load_data("HW-ICs-N=1-P=5-n=4").transpose()
theta = load_data("HW-ICs-Thetas")
phi = load_data("HW-ICs-Phis")
X, Y = np.meshgrid(phi, theta)

# compute the great circle
# need to split it into three parts, as I switch the branch of the arctan
x1 = np.linspace(0, np.pi / 2, 100)
x2 = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
x3 = np.linspace(3 * np.pi / 2, 2 * np.pi, 100)
y1 = np.arctan(-1.0 / np.cos(x1)) + np.pi
y2 = np.arctan(-1.0 / np.cos(x2))
y3 = np.arctan(-1.0 / np.cos(x3)) + np.pi

x = np.concatenate((x1, x2[1:], x3[1:]))  # make sure we don't include some points twice
y = np.concatenate((y1, y2[1:], y3[1:]))

for ax, data in zip([ax1, ax2], [data1, data2]):

    cm = ax.pcolormesh(X, Y, data, shading="gouraud")

    (line,) = ax.plot(x, y, lw=3, c="deepskyblue")
    line.set_path_effects([pe.Stroke(linewidth=4, foreground="black"), pe.Normal()])

    ax.grid(False)

    xticks_major = np.array([0, 1 / 2, 1, 3 / 2, 2]) * np.pi
    xticks_minor = np.array([1 / 4, 3 / 4, 5 / 4, 7 / 4]) * np.pi
    yticks_major = np.array([0, 1 / 4, 1 / 2, 3 / 4, 1]) * np.pi
    yticks_minor = np.array([1 / 8, 3 / 8, 5 / 8, 7 / 8]) * np.pi

    ax.set_xticks(xticks_major)
    ax.set_xticks(xticks_minor, minor=True)
    ax.set_yticks(yticks_major)
    ax.set_yticks(yticks_minor, minor=True)

    ax.set_yticklabels(
        [
            r"$0$",
            r"$\frac{1}{4}\pi$",
            r"$\frac{1}{2}\pi$",
            r"$\frac{3}{4}\pi$",
            r"$\pi$",
        ]
    )

    ax.set_ylabel(r"$\theta$")

    cb = fig.colorbar(cm, shrink=0.7, aspect=10, ticks=[], pad=0.01)
    data_min = np.min(data)
    data_max = np.max(data)
    cb.ax.text(-0.35, -0.11, f"${data_min:.2f}$", transform=cb.ax.transAxes)
    cb.ax.text(-0.35, 1.04, f"${data_max:.2f}$", transform=cb.ax.transAxes)
    cb.set_label(r"entanglement $E_g$", labelpad=8)


ax1.set_xticklabels(["" for y in ax1.get_xticks()])
ax2.set_xticklabels(
    [r"$0$", r"$\frac{1}{2}\pi$", r"$\pi$", r"$\frac{3}{2}\pi$", r"$2\pi$"]
)

ax2.set_xlabel(r"$\phi$")

fig.savefig("plots/HW-diff_ICs-Circle.png")

plt.show()


# %%
# =================== ICs: Asymmetric to Symmetric ===================


fig, ax = plt.subplots(figsize=cm2in(LINEWIDTH_FULL, 6))

raw_data = load_data("HW-ICs-Sym-To-Asym-P=3-n=50")

data = raw_data[1:, 1:]
steps = raw_data[0, 1:]
thetas = raw_data[1:, 0]

cmap = mpl.colormaps["afmhot"].reversed()
norm = mpl.colors.Normalize(vmin=np.min(thetas), vmax=np.max(thetas))

for theta, vals in zip(thetas, data):
    ax.plot(steps, vals, c=cmap(norm(theta)), lw=0.6)

cb = fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    ax=ax,
    ticks=[0, np.pi / 8, np.pi / 4, np.pi * 3 / 8, np.pi / 2],
    aspect=13,
    pad=0.02,
)
cb.ax.set_yticklabels(
    [
        r"$0$",
        r"$\frac{1}{8}\pi$",
        r"$\frac{1}{4}\pi$",
        r"$\frac{3}{8}\pi$",
        r"$\frac{1}{2}\pi$",
    ]
)
cb.ax.set_ylabel(r"$\theta$", rotation="horizontal")

ax.set_xlabel(r"time step $n$")
ax.set_ylabel(r"entanglement $E_g$")

fig.savefig("plots/HW-diff_ICs-asym_to_sym.pdf")

plt.show()


# %%
# =================== Different partitions ===================

fig, ax = plt.subplots(figsize=cm2in(LINEWIDTH_FULL, 7))

data = load_data("HW-diff_parts-N=1-P=64-n=100")

(line_full,) = ax.plot(data[0], data[1], lw=1.5, label=r"$\mathcal{P}_F$")
(line_coin_num,) = ax.plot(
    data[0], data[2], lw=1.5, label=r"$\mathcal{P}_C$ on" + "\ncircle"
)
(line_pos,) = ax.plot(data[0], data[3], lw=1.5, label=r"$\mathcal{P}_P$")
(line_coin_ana,) = ax.plot(
    data[0], data[4], lw=1.5, label=r"$\mathcal{P}_C$ on" + "\nline"
)

ax.set_xlim((-2, 62))
ax.set_ylim((0, ax.get_ylim()[1]))

bound_1 = 0.38800335
bound_2 = 0.388003375
axin = ax.inset_axes(
    [0.3, 0.13, 0.4, 0.25], xlim=(32.5, 34.5), ylim=(bound_1, bound_2), yticklabels=[]
)
axin.plot([], [])
axin.plot(data[0, 33:35], data[2, 33:35], "-o", lw=1)
axin.plot([], [])
axin.plot(data[0, 33:35], data[4, 33:35], "-o", lw=1)
axin.set_xticks([33, 34])
axin.set_yticks([bound_1 + (bound_2 - bound_1) / 2])

inset_indicator = ax.indicate_inset_zoom(axin, edgecolor="k", alpha=1, lw=1, ls="--")
inset_indicator.connectors[0].set_visible(False)
inset_indicator.connectors[1].set_visible(True)
inset_indicator.connectors[2].set_visible(False)
inset_indicator.connectors[3].set_visible(True)

gray_out_forbidden(ax, 64 * 2)

ax.set_xlabel(r"time step $n$")
ax.set_ylabel(r"entanglement $E_g$")

fig.legend(
    handles=[line_full, line_pos, line_coin_num, line_coin_ana],
    loc="outside center right",
)

fig.savefig("plots/HW-diff_parts.pdf")

plt.show()


# %%
# =================== GME over N and M ===================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=cm2in(LINEWIDTH_FULL, 8))

# ======== GME over N =========

data = load_data("GME_over_N-Random-M=2_to_6-N=1_to_10")

lines = []

for i_m in range(5):
    lower_err = np.fmax(
        data[3 * i_m + 1] - data[3 * i_m + 2], np.zeros(len(data[3 * i_m + 1]))
    )
    upper_err = data[3 * i_m + 3] - data[3 * i_m + 1]  # should always work I think

    lines.append(
        ax1.errorbar(
            data[0] + i_m * 0.2,
            data[3 * i_m + 1],
            yerr=[lower_err, upper_err],
            fmt="o-",
            capsize=2,
            lw=1.5,
            label=r"$M =$ " + str(i_m + 2),
        )
    )

ax1.set_xticks(list(range(1, 10 + 1)))
ax1.set_yticks([0, 0.2, 0.4, 0.6])
ax1.set_yticks([0.1, 0.3, 0.5, 0.7], minor=True)

ax1.set_xlabel(r"photon number $N$")
ax1.set_ylabel(r"entanglement $G_g$")

fig.legend(
    handles=lines,
    bbox_to_anchor=(
        0,
        -0.16,
        0.59,
        1,
    ),  # cleaner way would be to get the position and width of the x axis of ax1 and use that. But for now this works.
    loc="lower center",
    ncols=3,
)


# ======== GME over M =========

data = load_data("GME_over_M-Random-M=2_to_10-N=1_to_4")

lines = []

for i_n in range(4):
    lower_err = np.fmax(
        data[3 * i_n + 1] - data[3 * i_n + 2], np.zeros(len(data[3 * i_n + 1]))
    )
    upper_err = data[3 * i_n + 3] - data[3 * i_n + 1]  # should always work I think

    lines.append(
        ax2.errorbar(
            data[0] + i_n * 0.25,
            data[3 * i_n + 1],
            yerr=[lower_err, upper_err],
            fmt="o-",
            capsize=2,
            lw=1.5,
            label=r"$N =$ " + str(i_n + 1),
        )
    )

ax2.set_ylim(ax1.get_ylim())
ax2.set_xticks(list(range(2, 10 + 1)))
ax2.set_yticks(ax1.get_yticks())
ax2.set_yticks(ax1.get_yticks(minor=True), minor=True)
ax2.set_yticklabels(["" for y in ax2.get_yticks()])

ax2.set_xlabel(r"mode number $M$")

fig.legend(
    handles=lines, bbox_to_anchor=(0.5, -0.16, 0.56, 1), loc="lower center", ncols=2
)


fig.savefig("plots/GME-Random.pdf", bbox_inches="tight")

plt.show()


# %%

print("You've reached the bottom. Congrats!")
