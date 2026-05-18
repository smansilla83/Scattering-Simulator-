import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

st.set_page_config(page_title="Region I Eigenvector Simulator", layout="wide")

# ── Hero intro ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #0d1b2a 0%, #1a0533 100%);
            border-left: 5px solid #00e5ff;
            border-radius: 10px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;">
  <h1 style="color:#ffffff; margin:0 0 0.5rem 0; font-size:2rem;">
    Region I — Eigenvector Simulator
  </h1>
  <p style="color:#b0bec5; font-size:1.05rem; margin:0 0 1rem 0; max-width:800px;">
    Inside the interaction region (<span style="color:#00e5ff;">R &lt; ā</span>), the two-channel
    Hamiltonian couples an <span style="color:#ff6ec7;">electron state |e⟩</span> and a
    <span style="color:#69ff47;">cavity state |c⟩</span> through the off-diagonal coupling
    <span style="color:#ffd740;">ℏΩ</span>.
    Because the potential matrix <b style="color:#ffffff;">V̂</b> is constant inside this region,
    it can be diagonalised <em>once</em> via a mixing angle <span style="color:#ce93d8;">θ</span>,
    yielding the <b style="color:#00e5ff;">dressed eigenstates</b> |+⟩ and |−⟩ that
    decouple the radial equations.
  </p>
  <p style="color:#b0bec5; font-size:0.95rem; margin:0;">
    Use the sliders on the left to explore how the well depths V<sub>e</sub>, V<sub>c</sub>
    and the coupling ℏΩ control the mixing angle, eigenvalues, and the geometry of the
    dressed states on the Bloch circle.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameters")
st.sidebar.markdown("---")
Ve     = st.sidebar.slider("Ve  (eV)",  min_value=-10.0, max_value=10.0, value=2.0,  step=0.1)
Vc     = st.sidebar.slider("Vc  (eV)",  min_value=-10.0, max_value=10.0, value=-2.0, step=0.1)
hOmega = st.sidebar.slider("ℏΩ  (eV)", min_value=0.0,   max_value=10.0, value=1.0,  step=0.05)
a_bar  = st.sidebar.slider("ā  (nm)",   min_value=0.5,   max_value=5.0,  value=2.0,  step=0.1)

# ── Derived quantities ─────────────────────────────────────────────────────────
Delta_V   = (Ve - Vc) / 2.0
R         = np.sqrt(Delta_V**2 + hOmega**2)
two_theta = np.arctan2(hOmega, Delta_V)
theta     = two_theta / 2.0
cos_t     = np.cos(theta)
sin_t     = np.sin(theta)
lam_plus  = -(Ve + Vc) / 2.0 + R
lam_minus = -(Ve + Vc) / 2.0 - R

V_matrix = np.array([
    [-(Ve + Vc) / 2.0 + Delta_V, hOmega],
    [hOmega,                      -(Ve + Vc) / 2.0 - Delta_V],
])
evals_np, evecs_np = np.linalg.eigh(V_matrix)

bare_e = -(Ve + Vc) / 2.0 + Delta_V
bare_c = -(Ve + Vc) / 2.0 - Delta_V

# ── Analytical results — landscape ────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Analytical Results</h3>", unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex; gap:3rem; flex-wrap:nowrap; align-items:flex-start;
            background:#111827; border-radius:10px; padding:1.5rem;">

  <div style="flex:1; min-width:220px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.75rem 0;">Derived quantities</p>
    <table style="width:100%; border-collapse:collapse; color:#ffffff; font-size:0.95rem;">
      <tr><td style="padding:4px 8px; color:#b0bec5;">ΔV = (Ve − Vc)/2</td>
          <td style="padding:4px 8px; color:#00e5ff; font-weight:bold;">{Delta_V:.4f} eV</td></tr>
      <tr><td style="padding:4px 8px; color:#b0bec5;">R = √(ΔV² + ℏ²Ω²)</td>
          <td style="padding:4px 8px; color:#00e5ff; font-weight:bold;">{R:.4f} eV</td></tr>
      <tr><td style="padding:4px 8px; color:#b0bec5;">2θ</td>
          <td style="padding:4px 8px; color:#ce93d8; font-weight:bold;">{np.degrees(two_theta):.2f}°</td></tr>
      <tr><td style="padding:4px 8px; color:#b0bec5;">θ</td>
          <td style="padding:4px 8px; color:#ce93d8; font-weight:bold;">{np.degrees(theta):.2f}°</td></tr>
      <tr><td style="padding:4px 8px; color:#b0bec5;">λ₊</td>
          <td style="padding:4px 8px; color:#69ff47; font-weight:bold;">{lam_plus:.4f} eV</td></tr>
      <tr><td style="padding:4px 8px; color:#b0bec5;">λ₋</td>
          <td style="padding:4px 8px; color:#ff6ec7; font-weight:bold;">{lam_minus:.4f} eV</td></tr>
    </table>
  </div>

  <div style="flex:1.4; min-width:280px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.75rem 0;">Eigenvectors</p>
    <p style="color:#ffffff; font-size:0.95rem; line-height:2;">
      <span style="color:#69ff47;">|+⟩</span> = cosθ |e⟩ + sinθ |c⟩
      &nbsp;=&nbsp; <b style="color:#69ff47;">{cos_t:+.4f}</b> |e⟩
               &nbsp;<b style="color:#69ff47;">{sin_t:+.4f}</b> |c⟩<br>
      <span style="color:#ff6ec7;">|−⟩</span> = −sinθ |e⟩ + cosθ |c⟩
      &nbsp;=&nbsp; <b style="color:#ff6ec7;">{-sin_t:+.4f}</b> |e⟩
               &nbsp;<b style="color:#ff6ec7;">{cos_t:+.4f}</b> |c⟩
    </p>
  </div>

  <div style="flex:1.4; min-width:280px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.75rem 0;">NumPy verification</p>
    <table style="width:100%; border-collapse:collapse; color:#ffffff; font-size:0.9rem;">
      <tr style="border-bottom:1px solid #333; color:#b0bec5;">
        <th style="text-align:left; padding:4px 8px;">State</th>
        <th style="padding:4px 8px;">|e⟩</th>
        <th style="padding:4px 8px;">|c⟩</th>
        <th style="padding:4px 8px;">Eigenvalue</th>
      </tr>
      <tr>
        <td style="padding:4px 8px; color:#ff6ec7;">λ₋</td>
        <td style="padding:4px 8px; font-family:monospace;">{evecs_np[0,0]:+.4f}</td>
        <td style="padding:4px 8px; font-family:monospace;">{evecs_np[1,0]:+.4f}</td>
        <td style="padding:4px 8px; font-family:monospace;">{evals_np[0]:.4f} eV</td>
      </tr>
      <tr>
        <td style="padding:4px 8px; color:#69ff47;">λ₊</td>
        <td style="padding:4px 8px; font-family:monospace;">{evecs_np[0,1]:+.4f}</td>
        <td style="padding:4px 8px; font-family:monospace;">{evecs_np[1,1]:+.4f}</td>
        <td style="padding:4px 8px; font-family:monospace;">{evals_np[1]:.4f} eV</td>
      </tr>
    </table>
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Potential plot ─────────────────────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Potential Energy Diagram</h3>", unsafe_allow_html=True)

r = np.linspace(0, a_bar * 2.2, 1000)
inside = r < a_bar

# Bare diagonal potentials as a function of r
Ve_r = np.where(inside, -Ve, 0.0)
Vc_r = np.where(inside, -Vc, 0.0)

fig_pot, ax_pot = plt.subplots(figsize=(14, 5))
fig_pot.patch.set_facecolor("#0e1117")
ax_pot.set_facecolor("#0e1117")
ax_pot.tick_params(colors="white", labelsize=11)
ax_pot.xaxis.label.set_color("white")
ax_pot.yaxis.label.set_color("white")
for spine in ax_pot.spines.values():
    spine.set_edgecolor("#333")

# Shaded region I
ax_pot.axvspan(0, a_bar, color="#1a0533", alpha=0.6, label="Region I  (r < ā)")
ax_pot.axvline(a_bar, color="#ffd740", lw=1.5, ls="--", alpha=0.7)
ax_pot.text(a_bar + 0.04, ax_pot.get_ylim()[0] if ax_pot.get_ylim()[0] != 0 else -0.3,
            "ā", color="#ffd740", fontsize=13)

# Bare |e⟩ potential
ax_pot.plot(r, Ve_r, color="#ff6ec7", lw=2.5, label=r"$V_{ee}(r)$ — bare $|e\rangle$")
# Bare |c⟩ potential
ax_pot.plot(r, Vc_r, color="#69ff47", lw=2.5, label=r"$V_{cc}(r)$ — bare $|c\rangle$")

# Dressed eigenvalues inside region I as horizontal dashed lines
ax_pot.hlines(lam_plus,  0, a_bar, colors="#00e5ff", lw=2, ls="-.",
              label=f"λ₊ = {lam_plus:.3f} eV  (dressed |+⟩)")
ax_pot.hlines(lam_minus, 0, a_bar, colors="#ce93d8", lw=2, ls="-.",
              label=f"λ₋ = {lam_minus:.3f} eV  (dressed |−⟩)")

# Coupling arrow inside well
mid_r = a_bar / 2
ax_pot.annotate("", xy=(-Vc, mid_r), xytext=(-Ve, mid_r),
                xycoords=("data", "axes fraction"),
                textcoords=("data", "axes fraction"))

ax_pot.set_xlabel("r  (nm)", fontsize=12)
ax_pot.set_ylabel("Energy  (eV)", fontsize=12)
ax_pot.set_title("Bare and dressed potentials across Region I", color="white", fontsize=13)

yvals = np.concatenate([Ve_r, Vc_r, [lam_plus, lam_minus]])
ypad  = (yvals.max() - yvals.min()) * 0.18 + 0.3
ax_pot.set_ylim(yvals.min() - ypad, yvals.max() + ypad)
ax_pot.set_xlim(r.min(), r.max())

leg = ax_pot.legend(loc="upper right", framealpha=0.2, labelcolor="white",
                    facecolor="#0e1117", edgecolor="#444", fontsize=10)

# Label ℏΩ coupling inside well
ax_pot.annotate("",
    xy=(a_bar * 0.5, -Vc), xytext=(a_bar * 0.5, -Ve),
    arrowprops=dict(arrowstyle="<->", color="#ffd740", lw=1.8))
ax_pot.text(a_bar * 0.52, (-Ve + (-Vc)) / 2, "  ℏΩ coupling",
            color="#ffd740", fontsize=10, va="center")

fig_pot.tight_layout()
st.pyplot(fig_pot, use_container_width=True)
plt.close(fig_pot)

st.divider()

# ── Bloch-plane plots ──────────────────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Bloch-plane View of the Traceless Part</h3>",
            unsafe_allow_html=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor("#0e1117")

for ax in axes:
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

ax = axes[0]
ax.set_title(r"$(\sigma_z,\,\sigma_x)$ plane", color="white", fontsize=13)
ax.set_xlabel(r"$\Delta_V$  ($\sigma_z$ component)", color="white")
ax.set_ylabel(r"$\hbar\Omega$  ($\sigma_x$ component)", color="white")
ax.set_aspect("equal")
lim = max(R * 1.5, 0.5)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim * 0.2, lim * 1.4)
ax.axhline(0, color="#333", lw=0.8)
ax.axvline(0, color="#333", lw=0.8)

ax.annotate("", xy=(Delta_V, hOmega), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=2.5))
ax.text(Delta_V * 0.45 + lim * 0.04, hOmega * 0.55,
        r"$\vec{n}$, $R$", color="#00e5ff", fontsize=12)

arc_r = R * 0.3
arc = mpatches.Arc((0, 0), 2 * arc_r, 2 * arc_r,
                   angle=0, theta1=0, theta2=np.degrees(two_theta),
                   color="#ffd740", lw=2)
ax.add_patch(arc)
mid = two_theta / 2
ax.text(arc_r * 1.35 * np.cos(mid), arc_r * 1.35 * np.sin(mid),
        r"$2\theta$", color="#ffd740", fontsize=11)

ax.plot([0, Delta_V], [0, 0], color="#ff6ec7", lw=1.6, ls="--")
ax.plot([Delta_V, Delta_V], [0, hOmega], color="#69ff47", lw=1.6, ls="--")
ax.text(Delta_V / 2, -lim * 0.1, r"$\Delta_V$",
        color="#ff6ec7", ha="center", fontsize=11)
ax.text(Delta_V + lim * 0.07, hOmega / 2, r"$\hbar\Omega$",
        color="#69ff47", fontsize=11)

ax2 = axes[1]
ax2.set_title("Dressed states on the unit circle", color="white", fontsize=13)
ax2.set_xlabel(r"$|e\rangle$ component", color="white")
ax2.set_ylabel(r"$|c\rangle$ component", color="white")
ax2.set_aspect("equal")
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.add_patch(plt.Circle((0, 0), 1, color="#333", fill=False, lw=1.5))
ax2.axhline(0, color="#333", lw=0.8)
ax2.axvline(0, color="#333", lw=0.8)

ax2.annotate("", xy=(cos_t, sin_t), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", color="#69ff47", lw=2.5))
ax2.text(cos_t * 1.18, sin_t * 1.18, r"$|{+}\rangle$",
         color="#69ff47", fontsize=13, ha="center", va="center")

ax2.annotate("", xy=(-sin_t, cos_t), xytext=(0, 0),
             arrowprops=dict(arrowstyle="->", color="#ff6ec7", lw=2.5))
ax2.text(-sin_t * 1.22, cos_t * 1.22, r"$|{-}\rangle$",
         color="#ff6ec7", fontsize=13, ha="center", va="center")

arc2 = mpatches.Arc((0, 0), 0.5, 0.5, angle=0,
                    theta1=0, theta2=np.degrees(theta),
                    color="#ce93d8", lw=2)
ax2.add_patch(arc2)
ax2.text(0.35 * np.cos(theta / 2), 0.35 * np.sin(theta / 2) + 0.05,
         r"$\theta$", color="#ce93d8", fontsize=12)

fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Eigenvalue spectrum ────────────────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Eigenvalue Spectrum</h3>", unsafe_allow_html=True)

levels = [
    (bare_e,    "bare |e⟩",    "#ff6ec7", "--"),
    (bare_c,    "bare |c⟩",    "#69ff47", "--"),
    (lam_plus,  "λ₊ dressed",  "#00e5ff", "-"),
    (lam_minus, "λ₋ dressed",  "#ce93d8", "-"),
]

all_vals = [v for v, *_ in levels]
ymin = min(all_vals)
ymax = max(all_vals)
pad  = max((ymax - ymin) * 0.4, 0.5)

fig2, ax3 = plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor("#0e1117")
ax3.set_facecolor("#0e1117")
ax3.tick_params(colors="white", labelsize=11)
ax3.yaxis.label.set_color("white")
ax3.set_ylabel("Energy (eV)", color="white", fontsize=12)
for spine in ax3.spines.values():
    spine.set_edgecolor("#333")
ax3.xaxis.set_visible(False)
ax3.set_xlim(0, 1)
ax3.set_ylim(ymin - pad, ymax + pad)

min_gap = pad * 0.35
sorted_levels = sorted(levels, key=lambda x: x[0])
placed = []

for y, label, col, ls in sorted_levels:
    ax3.axhline(y, color=col, lw=2.5, ls=ls, alpha=0.95)
    y_lbl = y
    for py, *_ in placed:
        if abs(y_lbl - py) < min_gap:
            y_lbl = py + min_gap
    placed.append((y_lbl, label, col))

for y_lbl, label, col in placed:
    ax3.text(0.02, y_lbl, label, color=col, va="center", fontsize=11, fontweight="bold",
             transform=ax3.get_yaxis_transform(),
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#0e1117", edgecolor="none", alpha=0.85))

fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)
