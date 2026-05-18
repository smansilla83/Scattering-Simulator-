import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="Region I Eigenvector Simulator", layout="wide")

st.title("Region I — Diagonalising the Constant Matrix")
st.markdown(
    r"""
    Inside ($R < \bar{a}$), the potential matrix splits as
    $$\hat{V} = -\frac{V_e+V_c}{2}\,\mathbb{1} + \Delta_V\,\sigma_z + \hbar\Omega\,\sigma_x,
    \qquad \Delta_V \equiv \frac{V_e - V_c}{2}$$
    Diagonalising via the mixing angle $\theta$ defined by
    $$\tan 2\theta = \frac{\hbar\Omega}{\Delta_V}$$
    gives the dressed eigenstates $|{+}\rangle$ and $|{-}\rangle$.
    """
)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameters")

Ve = st.sidebar.slider(
    r"$V_e$  (electron well depth, eV)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1
)
Vc = st.sidebar.slider(
    r"$V_c$  (cavity well depth, eV)", min_value=-10.0, max_value=10.0, value=-2.0, step=0.1
)
hOmega = st.sidebar.slider(
    r"$\hbar\Omega$  (coupling, eV)", min_value=0.0, max_value=10.0, value=1.0, step=0.05
)

# ── Derived quantities ─────────────────────────────────────────────────────────
Delta_V = (Ve - Vc) / 2.0
R = np.sqrt(Delta_V**2 + hOmega**2)           # |n⃗|

# Mixing angle: atan2 handles Delta_V = 0 safely
two_theta = np.arctan2(hOmega, Delta_V)        # ∈ (-π/2, π/2) since hOmega ≥ 0
theta = two_theta / 2.0

cos_t = np.cos(theta)
sin_t = np.sin(theta)

lam_plus  = -(Ve + Vc) / 2.0 + R
lam_minus = -(Ve + Vc) / 2.0 - R

# ── Numerical verification via numpy ──────────────────────────────────────────
# Full V matrix (traceless block only matters for mixing; offset shifts both)
V_matrix = np.array([
    [-(Ve + Vc) / 2.0 + Delta_V,   hOmega],
    [hOmega,                        -(Ve + Vc) / 2.0 - Delta_V],
])
evals_np, evecs_np = np.linalg.eigh(V_matrix)   # ascending order

# ── Layout ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Analytical results")

    st.markdown(
        rf"""
| Quantity | Value |
|---|---|
| $\Delta_V = (V_e - V_c)/2$ | **{Delta_V:.4f} eV** |
| $R = \sqrt{{\Delta_V^2 + \hbar^2\Omega^2}}$ | **{R:.4f} eV** |
| $2\theta$ | **{np.degrees(two_theta):.2f}°** |
| $\theta$ | **{np.degrees(theta):.2f}°** |
| $\lambda_+$ | **{lam_plus:.4f} eV** |
| $\lambda_-$ | **{lam_minus:.4f} eV** |
        """
    )

    st.markdown("#### Eigenvectors")
    st.latex(
        rf"""
        |{{+}}\rangle = \cos\theta\,|e\rangle + \sin\theta\,|c\rangle
        = {cos_t:+.4f}\,|e\rangle {sin_t:+.4f}\,|c\rangle
        """
    )
    st.latex(
        rf"""
        |{{-}}\rangle = -\sin\theta\,|e\rangle + \cos\theta\,|c\rangle
        = {-sin_t:+.4f}\,|e\rangle {cos_t:+.4f}\,|c\rangle
        """
    )

    st.markdown("#### NumPy verification")
    st.markdown(
        rf"""
| | $|e\rangle$ component | $|c\rangle$ component | Eigenvalue |
|---|---|---|---|
| $\lambda_-$ (numpy) | `{evecs_np[0,0]:+.4f}` | `{evecs_np[1,0]:+.4f}` | `{evals_np[0]:.4f} eV` |
| $\lambda_+$ (numpy) | `{evecs_np[0,1]:+.4f}` | `{evecs_np[1,1]:+.4f}` | `{evals_np[1]:.4f} eV` |
        """
    )

with col2:
    st.subheader("Bloch-plane view of the traceless part")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # ── Left: vector diagram in (σ_z, σ_x) plane ──────────────────────────────
    ax = axes[0]
    ax.set_title(r"$(\sigma_z,\,\sigma_x)$ plane")
    ax.set_xlabel(r"$\Delta_V$ axis  ($\sigma_z$ component)")
    ax.set_ylabel(r"$\hbar\Omega$ axis  ($\sigma_x$ component)")
    ax.set_aspect("equal")

    lim = max(R * 1.4, 0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim * 0.3, lim * 1.3)

    ax.axhline(0, color="#444", lw=0.8)
    ax.axvline(0, color="#444", lw=0.8)

    # n-vector
    ax.annotate(
        "", xy=(Delta_V, hOmega), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=2),
    )
    ax.text(Delta_V * 0.55, hOmega * 0.55 + lim * 0.06, r"$\vec{n}$, $R$",
            color="#4fc3f7", fontsize=11)

    # 2θ arc
    arc_r = R * 0.35
    arc = mpatches.Arc(
        (0, 0), 2 * arc_r, 2 * arc_r,
        angle=0, theta1=0, theta2=np.degrees(two_theta),
        color="#ffb74d", lw=1.5,
    )
    ax.add_patch(arc)
    mid = two_theta / 2
    ax.text(arc_r * 1.15 * np.cos(mid), arc_r * 1.15 * np.sin(mid),
            r"$2\theta$", color="#ffb74d", fontsize=10)

    # component lines
    ax.plot([0, Delta_V], [0, 0], color="#ef9a9a", lw=1.4, ls="--")
    ax.plot([Delta_V, Delta_V], [0, hOmega], color="#a5d6a7", lw=1.4, ls="--")
    ax.text(Delta_V / 2, -lim * 0.1, r"$\Delta_V$", color="#ef9a9a",
            ha="center", fontsize=10)
    ax.text(Delta_V + lim * 0.06, hOmega / 2, r"$\hbar\Omega$",
            color="#a5d6a7", fontsize=10)

    # ── Right: Bloch-circle showing |+⟩ and |−⟩ ───────────────────────────────
    ax2 = axes[1]
    ax2.set_title(r"Dressed states on the unit circle")
    ax2.set_xlabel(r"$|e\rangle$ component")
    ax2.set_ylabel(r"$|c\rangle$ component")
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)

    circle = plt.Circle((0, 0), 1, color="#444", fill=False, lw=1)
    ax2.add_patch(circle)
    ax2.axhline(0, color="#444", lw=0.8)
    ax2.axvline(0, color="#444", lw=0.8)

    # |+⟩ arrow  (cos θ, sin θ)
    ax2.annotate(
        "", xy=(cos_t, sin_t), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=2),
    )
    ax2.text(cos_t * 1.12, sin_t * 1.12, r"$|{+}\rangle$",
             color="#4fc3f7", fontsize=12, ha="center")

    # |−⟩ arrow  (-sin θ, cos θ)
    ax2.annotate(
        "", xy=(-sin_t, cos_t), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#ffb74d", lw=2),
    )
    ax2.text(-sin_t * 1.18, cos_t * 1.18, r"$|{-}\rangle$",
             color="#ffb74d", fontsize=12, ha="center")

    # θ arc for |+⟩
    arc2 = mpatches.Arc(
        (0, 0), 0.55, 0.55,
        angle=0, theta1=0, theta2=np.degrees(theta),
        color="#ce93d8", lw=1.5,
    )
    ax2.add_patch(arc2)
    ax2.text(0.38 * np.cos(theta / 2), 0.38 * np.sin(theta / 2),
             r"$\theta$", color="#ce93d8", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Energy level diagram ───────────────────────────────────────────────────────
st.subheader("Eigenvalue spectrum")

fig2, ax3 = plt.subplots(figsize=(5, 3))
fig2.patch.set_facecolor("#0e1117")
ax3.set_facecolor("#0e1117")
ax3.tick_params(colors="white")
for spine in ax3.spines.values():
    spine.set_edgecolor("#444")
ax3.yaxis.label.set_color("white")
ax3.xaxis.set_visible(False)

bare_e = -(Ve + Vc) / 2.0 + Delta_V    # original |e⟩ diagonal element
bare_c = -(Ve + Vc) / 2.0 - Delta_V    # original |c⟩ diagonal element

for y, label, col in [
    (bare_e, r"bare $|e\rangle$", "#ef9a9a"),
    (bare_c, r"bare $|c\rangle$", "#a5d6a7"),
    (lam_plus, r"$\lambda_+$ (dressed)", "#4fc3f7"),
    (lam_minus, r"$\lambda_-$ (dressed)", "#ffb74d"),
]:
    ax3.axhline(y, color=col, lw=2, ls="--" if "bare" in label else "-")
    ax3.text(0.52, y, label, color=col, va="center", fontsize=10,
             transform=ax3.get_yaxis_transform())

ax3.set_ylabel("Energy (eV)", color="white")
ax3.set_xlim(0, 1)
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)
