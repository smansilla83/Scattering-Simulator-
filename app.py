import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Region I Eigenvector Simulator", layout="wide")

st.title("Region I — Diagonalising the Constant Matrix")

st.markdown("Inside $(R < \\bar{a})$, the potential matrix splits as:")
st.latex(r"""
    \hat{V} = -\frac{V_e+V_c}{2}\,\mathbb{1} + \Delta_V\,\sigma_z + \hbar\Omega\,\sigma_x,
    \qquad \Delta_V \equiv \frac{V_e - V_c}{2}
""")
st.markdown("The mixing angle $\\theta$ is defined by:")
st.latex(r"\tan 2\theta = \frac{\hbar\Omega}{\Delta_V}")
st.markdown("This gives the dressed eigenstates $|{+}\\rangle$ and $|{-}\\rangle$.")

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameters")
Ve     = st.sidebar.slider("Ve  (eV)",      min_value=-10.0, max_value=10.0, value=2.0,  step=0.1)
Vc     = st.sidebar.slider("Vc  (eV)",      min_value=-10.0, max_value=10.0, value=-2.0, step=0.1)
hOmega = st.sidebar.slider("ℏΩ  (eV)",     min_value=0.0,   max_value=10.0, value=1.0,  step=0.05)

# ── Derived quantities ─────────────────────────────────────────────────────────
Delta_V   = (Ve - Vc) / 2.0
R         = np.sqrt(Delta_V**2 + hOmega**2)
two_theta = np.arctan2(hOmega, Delta_V)
theta     = two_theta / 2.0
cos_t     = np.cos(theta)
sin_t     = np.sin(theta)
lam_plus  = -(Ve + Vc) / 2.0 + R
lam_minus = -(Ve + Vc) / 2.0 - R

# NumPy cross-check
V_matrix = np.array([
    [-(Ve + Vc) / 2.0 + Delta_V, hOmega],
    [hOmega,                      -(Ve + Vc) / 2.0 - Delta_V],
])
evals_np, evecs_np = np.linalg.eigh(V_matrix)

# ── Layout ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Analytical results")
    st.markdown(f"""
| Quantity | Value |
|---|---|
| $\\Delta_V = (V_e - V_c)/2$ | **{Delta_V:.4f} eV** |
| $R = \\sqrt{{\\Delta_V^2 + \\hbar^2\\Omega^2}}$ | **{R:.4f} eV** |
| $2\\theta$ | **{np.degrees(two_theta):.2f}°** |
| $\\theta$ | **{np.degrees(theta):.2f}°** |
| $\\lambda_+$ | **{lam_plus:.4f} eV** |
| $\\lambda_-$ | **{lam_minus:.4f} eV** |
""")

    st.subheader("Eigenvectors")
    st.latex(
        rf"|{{+}}\rangle = \cos\theta\,|e\rangle + \sin\theta\,|c\rangle"
        rf" = {cos_t:+.4f}\,|e\rangle {sin_t:+.4f}\,|c\rangle"
    )
    st.latex(
        rf"|{{-}}\rangle = -\sin\theta\,|e\rangle + \cos\theta\,|c\rangle"
        rf" = {-sin_t:+.4f}\,|e\rangle {cos_t:+.4f}\,|c\rangle"
    )

    st.subheader("NumPy verification")
    st.markdown(f"""
| State | $|e\\rangle$ | $|c\\rangle$ | Eigenvalue |
|---|---|---|---|
| $\\lambda_-$ | `{evecs_np[0,0]:+.4f}` | `{evecs_np[1,0]:+.4f}` | `{evals_np[0]:.4f} eV` |
| $\\lambda_+$ | `{evecs_np[0,1]:+.4f}` | `{evecs_np[1,1]:+.4f}` | `{evals_np[1]:.4f} eV` |
""")

with col2:
    st.subheader("Bloch-plane view of the traceless part")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0e1117")

    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Left: (σ_z, σ_x) vector diagram
    ax = axes[0]
    ax.set_title(r"$(\sigma_z,\,\sigma_x)$ plane", color="white")
    ax.set_xlabel(r"$\Delta_V$  ($\sigma_z$ component)", color="white")
    ax.set_ylabel(r"$\hbar\Omega$  ($\sigma_x$ component)", color="white")
    ax.set_aspect("equal")
    lim = max(R * 1.5, 0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.2, lim * 1.4)
    ax.axhline(0, color="#444", lw=0.8)
    ax.axvline(0, color="#444", lw=0.8)

    ax.annotate("", xy=(Delta_V, hOmega), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=2))
    ax.text(Delta_V * 0.45 + lim * 0.04, hOmega * 0.55,
            r"$\vec{n}$, $R$", color="#4fc3f7", fontsize=11)

    arc_r = R * 0.3
    arc = mpatches.Arc((0, 0), 2 * arc_r, 2 * arc_r,
                       angle=0, theta1=0, theta2=np.degrees(two_theta),
                       color="#ffb74d", lw=1.5)
    ax.add_patch(arc)
    mid = two_theta / 2
    ax.text(arc_r * 1.3 * np.cos(mid), arc_r * 1.3 * np.sin(mid),
            r"$2\theta$", color="#ffb74d", fontsize=10)

    ax.plot([0, Delta_V], [0, 0], color="#ef9a9a", lw=1.4, ls="--")
    ax.plot([Delta_V, Delta_V], [0, hOmega], color="#a5d6a7", lw=1.4, ls="--")
    ax.text(Delta_V / 2, -lim * 0.1, r"$\Delta_V$",
            color="#ef9a9a", ha="center", fontsize=10)
    ax.text(Delta_V + lim * 0.07, hOmega / 2, r"$\hbar\Omega$",
            color="#a5d6a7", fontsize=10)

    # Right: unit circle with |+⟩ and |−⟩
    ax2 = axes[1]
    ax2.set_title("Dressed states on the unit circle", color="white")
    ax2.set_xlabel(r"$|e\rangle$ component", color="white")
    ax2.set_ylabel(r"$|c\rangle$ component", color="white")
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.add_patch(plt.Circle((0, 0), 1, color="#444", fill=False, lw=1))
    ax2.axhline(0, color="#444", lw=0.8)
    ax2.axvline(0, color="#444", lw=0.8)

    ax2.annotate("", xy=(cos_t, sin_t), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=2))
    ax2.text(cos_t * 1.15, sin_t * 1.15, r"$|{+}\rangle$",
             color="#4fc3f7", fontsize=12, ha="center", va="center")

    ax2.annotate("", xy=(-sin_t, cos_t), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#ffb74d", lw=2))
    ax2.text(-sin_t * 1.2, cos_t * 1.2, r"$|{-}\rangle$",
             color="#ffb74d", fontsize=12, ha="center", va="center")

    arc2 = mpatches.Arc((0, 0), 0.5, 0.5, angle=0,
                        theta1=0, theta2=np.degrees(theta),
                        color="#ce93d8", lw=1.5)
    ax2.add_patch(arc2)
    ax2.text(0.35 * np.cos(theta / 2), 0.35 * np.sin(theta / 2) + 0.05,
             r"$\theta$", color="#ce93d8", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Energy level diagram ───────────────────────────────────────────────────────
st.subheader("Eigenvalue spectrum")

bare_e = -(Ve + Vc) / 2.0 + Delta_V
bare_c = -(Ve + Vc) / 2.0 - Delta_V

levels = [
    (bare_e,   "bare $|e\\rangle$",    "#ef9a9a", "--"),
    (bare_c,   "bare $|c\\rangle$",    "#a5d6a7", "--"),
    (lam_plus, "$\\lambda_+$ dressed", "#4fc3f7", "-"),
    (lam_minus,"$\\lambda_-$ dressed", "#ffb74d", "-"),
]

all_vals = [v for v, *_ in levels]
ymin = min(all_vals)
ymax = max(all_vals)
pad  = max((ymax - ymin) * 0.4, 0.5)

fig2, ax3 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor("#0e1117")
ax3.set_facecolor("#0e1117")
ax3.tick_params(colors="white")
ax3.yaxis.label.set_color("white")
ax3.set_ylabel("Energy (eV)", color="white", fontsize=12)
for spine in ax3.spines.values():
    spine.set_edgecolor("#444")
ax3.xaxis.set_visible(False)
ax3.set_xlim(0, 1)
ax3.set_ylim(ymin - pad, ymax + pad)

# Draw lines and collect (y_actual, y_label, label, color) resolving overlaps
min_gap = pad * 0.35
sorted_levels = sorted(levels, key=lambda x: x[0])
placed = []   # (y_label, label, col)

for y, label, col, ls in sorted_levels:
    ax3.axhline(y, color=col, lw=2.5, ls=ls, alpha=0.9)
    # find a label y that doesn't collide with already-placed ones
    y_lbl = y
    for py, *_ in placed:
        if abs(y_lbl - py) < min_gap:
            y_lbl = py + min_gap
    placed.append((y_lbl, label, col))

for y_lbl, label, col in placed:
    ax3.text(
        0.02, y_lbl, label,
        color=col, va="center", fontsize=11, fontweight="bold",
        transform=ax3.get_yaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0e1117", edgecolor="none", alpha=0.8),
    )
    ax3.yaxis.set_tick_params(labelsize=10, labelcolor="white")

fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)
