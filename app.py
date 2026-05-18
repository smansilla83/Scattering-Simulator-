import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Region I Eigenvector Simulator", layout="wide")

# ── Hero intro ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #0d1b2a 0%, #1a0533 100%);
            border-left: 5px solid #00e5ff;
            border-radius: 10px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.5rem;">
  <h1 style="color:#ffffff; margin:0 0 1rem 0; font-size:2rem;">
    Region I — Eigenvector Simulator
  </h1>
  <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
    <span style="color:#00e5ff;">⟐ Diagonalise V̂ → dressed states |+⟩ and |−⟩</span>
    <span style="color:#ce93d8;">⟐ Compute mixing angle θ</span>
    <span style="color:#69ff47;">⟐ Get eigenvalues λ±</span>
    <span style="color:#ffd740;">⟐ Compute wavenumbers k± for the full Hamiltonian</span>
    <span style="color:#ff6ec7;">⟐ Identify propagating vs evanescent channels</span>
    <span style="color:#aaaaff;">⟐ Visualise the Bloch-plane geometry</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("Parameters")

st.sidebar.markdown("**Potential** *(Region I, r < ā)*")
Ve     = st.sidebar.slider("Ve  (eV) — electron well depth",  min_value=-10.0, max_value=10.0, value=2.0,  step=0.1)
Vc     = st.sidebar.slider("Vc  (eV) — cavity well depth",   min_value=-10.0, max_value=10.0, value=-2.0, step=0.1)
hOmega = st.sidebar.slider("ℏΩ  (eV) — Rabi coupling",       min_value=0.0,   max_value=10.0, value=1.0,  step=0.05)
a_bar  = st.sidebar.slider("ā  (nm) — region boundary",       min_value=0.5,   max_value=5.0,  value=2.0,  step=0.1)
st.sidebar.markdown("**Full Hamiltonian**")
E_tot  = st.sidebar.slider("E  (eV) — total scattering energy", min_value=-10.0, max_value=10.0, value=0.5, step=0.1)
m_r    = st.sidebar.slider("mᵣ / mₑ — reduced mass",            min_value=0.05,  max_value=2.0,  value=0.5, step=0.05)

with st.sidebar.expander("What does each parameter do?"):
    st.markdown("""
**Ve** — depth of the potential well in the bare electron channel |e⟩ inside Region I.
Sets the diagonal element $V_{ee} = -V_e$.

**Vc** — depth of the well in the bare cavity channel |c⟩.
Sets $V_{cc} = -V_c$.

**ℏΩ** — vacuum Rabi coupling; the off-diagonal element that mixes |e⟩ and |c⟩.
Controls how strongly the atom and cavity exchange energy.
*Only ΔV = (Ve−Vc)/2 and ℏΩ determine the mixing angle θ and eigenvectors —
the common offset −(Ve+Vc)/2 shifts eigenvalues only.*

**ā** — radius of the interaction region. Outside this boundary the potential drops to zero (free propagation).

**E** — total scattering energy. Determines which dressed channels are open (E > λ±, propagating)
vs closed (E < λ±, evanescent).

**mᵣ** — reduced mass of the atom–cavity system, sets the kinetic energy scale
$k_\\pm = \\sqrt{2m_r(E-\\lambda_\\pm)}/\\hbar$.

---
**Why no field dependence?**
V̂ is taken as *constant* inside Region I — the central approximation of this section.
Field dependence (e.g. a spatially varying laser or cavity mode) would make Ω = Ω(r),
breaking the analytic diagonalisation and requiring a numerical approach.

**For what atom?**
The |e⟩/|c⟩ basis and ℏΩ coupling are cavity QED language: |e⟩ is an excited atomic state,
|c⟩ is a single cavity photon. This is a generic two-channel model — commonly applied to
**hydrogen, rubidium, or caesium** in cavity QED scattering problems.
""")


# ── Physical constant: ℏ²/2mₑ in eV·nm² ──────────────────────────────────────
hbar2_over_2me = 0.038100  # eV·nm²

# ── Derived quantities ─────────────────────────────────────────────────────────
Delta_V   = (Ve - Vc) / 2.0
offset    = -(Ve + Vc) / 2.0
R         = np.sqrt(Delta_V**2 + hOmega**2)
two_theta = np.arctan2(hOmega, Delta_V)
theta     = two_theta / 2.0
cos_t     = np.cos(theta)
sin_t     = np.sin(theta)
lam_plus  = offset + R
lam_minus = offset - R

bare_e = offset + Delta_V   # = -Vc
bare_c = offset - Delta_V   # = -Ve

V_matrix = np.array([
    [bare_e, hOmega],
    [hOmega, bare_c],
])
evals_np, evecs_np = np.linalg.eigh(V_matrix)

# Wavenumbers for full Hamiltonian  k = sqrt(2 mᵣ (E−λ) / ℏ²)
def wavenumber(E, lam, mr):
    arg = (E - lam) * mr / hbar2_over_2me
    if arg >= 0:
        return np.sqrt(arg), "propagating", "#69ff47"
    else:
        return np.sqrt(-arg), "evanescent", "#ff6ec7"

k_plus,  k_plus_type,  k_plus_col  = wavenumber(E_tot, lam_plus,  m_r)
k_minus, k_minus_type, k_minus_col = wavenumber(E_tot, lam_minus, m_r)

# ── Analytical results — landscape ────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Analytical Results</h3>", unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex; gap:2.5rem; flex-wrap:nowrap; align-items:flex-start;
            background:#111827; border-radius:10px; padding:1.5rem;">

  <div style="flex:1; min-width:210px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.75rem 0;">Derived quantities</p>
    <table style="width:100%; border-collapse:collapse; color:#ffffff; font-size:0.92rem;">
      <tr><td style="padding:4px 6px; color:#b0bec5;">offset = −(Ve+Vc)/2</td>
          <td style="padding:4px 6px; color:#aaaaff; font-weight:bold;">{offset:.4f} eV</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">ΔV = (Ve−Vc)/2</td>
          <td style="padding:4px 6px; color:#00e5ff; font-weight:bold;">{Delta_V:.4f} eV</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">R = √(ΔV²+ℏ²Ω²)</td>
          <td style="padding:4px 6px; color:#00e5ff; font-weight:bold;">{R:.4f} eV</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">2θ</td>
          <td style="padding:4px 6px; color:#ce93d8; font-weight:bold;">{np.degrees(two_theta):.2f}°</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">θ</td>
          <td style="padding:4px 6px; color:#ce93d8; font-weight:bold;">{np.degrees(theta):.2f}°</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">λ₊ = offset + R</td>
          <td style="padding:4px 6px; color:#69ff47; font-weight:bold;">{lam_plus:.4f} eV</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">λ₋ = offset − R</td>
          <td style="padding:4px 6px; color:#ff6ec7; font-weight:bold;">{lam_minus:.4f} eV</td></tr>
    </table>
  </div>

  <div style="flex:1.3; min-width:260px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.75rem 0;">Eigenvectors of V̂</p>
    <p style="color:#aaaaaa; font-size:0.82rem; margin:0 0 0.5rem 0;">
      Depend only on ΔV and ℏΩ (not the offset).
    </p>
    <p style="color:#ffffff; font-size:0.93rem; line-height:2.2;">
      <span style="color:#69ff47;">|+⟩</span> = cosθ |e⟩ + sinθ |c⟩
      &nbsp;=&nbsp;<b style="color:#69ff47;">{cos_t:+.4f}</b> |e⟩
      &nbsp;<b style="color:#69ff47;">{sin_t:+.4f}</b> |c⟩<br>
      <span style="color:#ff6ec7;">|−⟩</span> = −sinθ |e⟩ + cosθ |c⟩
      &nbsp;=&nbsp;<b style="color:#ff6ec7;">{-sin_t:+.4f}</b> |e⟩
      &nbsp;<b style="color:#ff6ec7;">{cos_t:+.4f}</b> |c⟩
    </p>
    <p style="color:#aaaaaa; font-size:0.82rem; margin:0.5rem 0 0 0;">
      NumPy check — λ₋: [{evecs_np[0,0]:+.4f}, {evecs_np[1,0]:+.4f}] &nbsp;|&nbsp;
      λ₊: [{evecs_np[0,1]:+.4f}, {evecs_np[1,1]:+.4f}]
    </p>
  </div>

  <div style="flex:1.3; min-width:260px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.75rem 0;">Full Hamiltonian eigenstates</p>
    <p style="color:#aaaaaa; font-size:0.82rem; margin:0 0 0.5rem 0;">
      Spin structure = same |±⟩. Each channel gets a wavenumber k± from E and m<sub>r</sub>.
    </p>
    <table style="width:100%; border-collapse:collapse; color:#ffffff; font-size:0.9rem;">
      <tr style="border-bottom:1px solid #333; color:#b0bec5;">
        <th style="text-align:left; padding:4px 6px;">Channel</th>
        <th style="padding:4px 6px;">λ (eV)</th>
        <th style="padding:4px 6px;">k (nm⁻¹)</th>
        <th style="padding:4px 6px;">Type</th>
      </tr>
      <tr>
        <td style="padding:5px 6px; color:#69ff47;">|+⟩</td>
        <td style="padding:5px 6px; font-family:monospace;">{lam_plus:.4f}</td>
        <td style="padding:5px 6px; font-family:monospace;">{k_plus:.4f}</td>
        <td style="padding:5px 6px; color:{k_plus_col}; font-weight:bold;">{k_plus_type}</td>
      </tr>
      <tr>
        <td style="padding:5px 6px; color:#ff6ec7;">|−⟩</td>
        <td style="padding:5px 6px; font-family:monospace;">{lam_minus:.4f}</td>
        <td style="padding:5px 6px; font-family:monospace;">{k_minus:.4f}</td>
        <td style="padding:5px 6px; color:{k_minus_col}; font-weight:bold;">{k_minus_type}</td>
      </tr>
    </table>
    <p style="color:#aaaaaa; font-size:0.82rem; margin:0.75rem 0 0 0;">
      ψ±(r) = (A± e<sup>ik±r</sup> + B± e<sup>−ik±r</sup>) |±⟩ &nbsp;[propagating]<br>
      ψ±(r) = (A± e<sup>κ±r</sup> + B± e<sup>−κ±r</sup>) |±⟩ &nbsp;[evanescent, k→iκ]
    </p>
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Potential plot ─────────────────────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Potential Energy Diagram</h3>", unsafe_allow_html=True)

r = np.linspace(0, a_bar * 2.2, 1000)
inside = r < a_bar

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

ax_pot.axvspan(0, a_bar, color="#1a0533", alpha=0.6, label="Region I  (r < ā)")
ax_pot.axvline(a_bar, color="#ffd740", lw=1.5, ls="--", alpha=0.7)
ax_pot.text(a_bar * 1.01, 0, "ā", color="#ffd740", fontsize=13)

ax_pot.plot(r, Ve_r, color="#ff6ec7", lw=2.5, label=r"$V_{ee}(r)$ — bare $|e\rangle$")
ax_pot.plot(r, Vc_r, color="#69ff47", lw=2.5, label=r"$V_{cc}(r)$ — bare $|c\rangle$")

ax_pot.hlines(lam_plus,  0, a_bar, colors="#00e5ff", lw=2, ls="-.",
              label=f"λ₊ = {lam_plus:.3f} eV  (dressed |+⟩)")
ax_pot.hlines(lam_minus, 0, a_bar, colors="#ce93d8", lw=2, ls="-.",
              label=f"λ₋ = {lam_minus:.3f} eV  (dressed |−⟩)")

ax_pot.axhline(E_tot, color="#ffd740", lw=1.5, ls=":", alpha=0.85,
               label=f"E = {E_tot:.2f} eV  (total energy)")

yvals = np.concatenate([Ve_r, Vc_r, [lam_plus, lam_minus, E_tot]])
ypad  = (yvals.max() - yvals.min()) * 0.2 + 0.4
ax_pot.set_ylim(yvals.min() - ypad, yvals.max() + ypad)
ax_pot.set_xlim(r.min(), r.max())

if abs(-Ve - (-Vc)) > 0.15:
    mid_r = a_bar * 0.5
    ax_pot.annotate("", xy=(mid_r, -Vc), xytext=(mid_r, -Ve),
                    arrowprops=dict(arrowstyle="<->", color="#ffd740", lw=1.8))
    ax_pot.text(mid_r + a_bar * 0.04, (-Ve + (-Vc)) / 2,
                "  ℏΩ coupling", color="#ffd740", fontsize=10, va="center")

ax_pot.set_xlabel("r  (nm)", fontsize=12)
ax_pot.set_ylabel("Energy  (eV)", fontsize=12)
ax_pot.set_title("Bare and dressed potentials — Region I", color="white", fontsize=13)
ax_pot.legend(loc="upper right", framealpha=0.2, labelcolor="white",
              facecolor="#0e1117", edgecolor="#444", fontsize=10)

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
ax.text(Delta_V / 2, -lim * 0.1, r"$\Delta_V$", color="#ff6ec7", ha="center", fontsize=11)
ax.text(Delta_V + lim * 0.07, hOmega / 2, r"$\hbar\Omega$", color="#69ff47", fontsize=11)

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

all_vals = [v for v, *_ in levels] + [E_tot]
ymin = min(all_vals)
ymax = max(all_vals)
pad  = max((ymax - ymin) * 0.4, 0.5)

fig2, ax3 = plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor("#0e1117")
ax3.set_facecolor("#0e1117")
ax3.tick_params(colors="white", labelsize=11)
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

ax3.axhline(E_tot, color="#ffd740", lw=1.8, ls=":", alpha=0.9)
ax3.text(0.02, E_tot, f"E = {E_tot:.2f} eV", color="#ffd740", va="center", fontsize=11,
         fontweight="bold", transform=ax3.get_yaxis_transform(),
         bbox=dict(boxstyle="round,pad=0.25", facecolor="#0e1117", edgecolor="none", alpha=0.85))

fig2.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close(fig2)

st.divider()

# ── Numerical solution ─────────────────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Numerical Solution — Coupled Schrödinger ODE</h3>",
            unsafe_allow_html=True)
st.markdown("""
<p style="color:#b0bec5; font-size:0.93rem; margin:0 0 1rem 0;">
Solves <b style="color:#ffffff;">u″ = (2mᵣ/ℏ²)(V̂ − E) u</b> numerically via RK45,
starting with a pure |e⟩ injection at r = 0 [u(0) = 0, u′(0) = (1, 0)].
Dashed lines show the analytical dressed-state decomposition for comparison.
</p>
""", unsafe_allow_html=True)

factor = 2.0 * m_r / hbar2_over_2me   # nm⁻² eV⁻¹

def ode(r, y):
    ue, uc, pue, puc = y
    dpue = factor * ((bare_e - E_tot) * ue + hOmega * uc)
    dpuc = factor * ((bare_c - E_tot) * uc + hOmega * ue)
    return [pue, puc, dpue, dpuc]

r_end  = a_bar * 1.5
r_eval = np.linspace(0.0, r_end, 2000)
sol    = solve_ivp(ode, [0.0, r_end], [0.0, 0.0, 1.0, 0.0],
                   t_eval=r_eval, method="RK45", rtol=1e-10, atol=1e-12)

ue_num = sol.y[0]
uc_num = sol.y[1]
r_num  = sol.t

# ── Analytical solution with same BCs ─────────────────────────────────────────
# In dressed basis: u±(r) = A± * f±(r), with f = sin(k r) or sinh(κ r)
# BCs: u(0)=0 → no cosine/cosh; u'_e(0)=1, u'_c(0)=0
# cos_t·A+·k+ − sin_t·A-·k- = 1
# sin_t·A+·k+ + cos_t·A-·k- = 0  → A-·k- = −sin_t, A+·k+ = cos_t

def dressed_wf(r, A, k, evanescent):
    if evanescent:
        return A * np.sinh(k * r)
    return A * np.sin(k * r)

# channel +
if k_plus > 1e-8:
    Ap = cos_t / k_plus
    evan_p = (k_plus_type == "evanescent")
else:
    Ap, evan_p = cos_t * 1e8, False

# channel -
if k_minus > 1e-8:
    Am = -sin_t / k_minus
    evan_m = (k_minus_type == "evanescent")
else:
    Am, evan_m = -sin_t * 1e8, False

fp = dressed_wf(r_num, Ap, k_plus,  evan_p)
fm = dressed_wf(r_num, Am, k_minus, evan_m)

ue_ana =  cos_t * fp - sin_t * fm
uc_ana =  sin_t * fp + cos_t * fm

# dressed-channel amplitudes
fp_raw = dressed_wf(r_num, Ap, k_plus,  evan_p)
fm_raw = dressed_wf(r_num, Am, k_minus, evan_m)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
fig3.patch.set_facecolor("#0e1117")

for ax in axes3.flat:
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=10)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.axvline(a_bar, color="#ffd740", lw=1.2, ls="--", alpha=0.6)

# Top-left: bare channels numerical vs analytical
ax = axes3[0, 0]
ax.set_title("Bare channels — numerical vs analytical", color="white")
ax.plot(r_num, ue_num, color="#ff6ec7", lw=2,   label="uₑ  numerical")
ax.plot(r_num, uc_num, color="#69ff47", lw=2,   label="u_c  numerical")
ax.plot(r_num, ue_ana, color="#ff6ec7", lw=1.2, ls="--", alpha=0.6, label="uₑ  analytical")
ax.plot(r_num, uc_ana, color="#69ff47", lw=1.2, ls="--", alpha=0.6, label="u_c  analytical")
ax.axhline(0, color="#444", lw=0.6)
ax.set_xlabel("r  (nm)")
ax.set_ylabel("amplitude")
ax.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

# Top-right: dressed channel amplitudes
ax2 = axes3[0, 1]
ax2.set_title("Dressed channels u±(r)", color="white")
ax2.plot(r_num, fp_raw, color="#00e5ff", lw=2,
         label=f"|+⟩  ({'prop.' if not evan_p else 'evan.'}, k={k_plus:.3f} nm⁻¹)")
ax2.plot(r_num, fm_raw, color="#ce93d8", lw=2,
         label=f"|−⟩  ({'prop.' if not evan_m else 'evan.'}, k={k_minus:.3f} nm⁻¹)")
ax2.axhline(0, color="#444", lw=0.6)
ax2.set_xlabel("r  (nm)")
ax2.set_ylabel("amplitude")
ax2.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

# Bottom-left: residual (numerical − analytical)
ax3b = axes3[1, 0]
ax3b.set_title("Residual  |numerical − analytical|", color="white")
ax3b.semilogy(r_num[1:], np.abs(ue_num[1:] - ue_ana[1:]) + 1e-16,
              color="#ff6ec7", lw=1.5, label="|uₑ error|")
ax3b.semilogy(r_num[1:], np.abs(uc_num[1:] - uc_ana[1:]) + 1e-16,
              color="#69ff47", lw=1.5, label="|u_c error|")
ax3b.set_xlabel("r  (nm)")
ax3b.set_ylabel("absolute error")
ax3b.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")
ax3b.tick_params(axis="y", colors="white")

# Bottom-right: probability densities |u|²
ax4 = axes3[1, 1]
ax4.set_title("Probability density  |u(r)|²", color="white")
ax4.plot(r_num, ue_num**2, color="#ff6ec7", lw=2, label="|uₑ|²  electron channel")
ax4.plot(r_num, uc_num**2, color="#69ff47", lw=2, label="|u_c|²  cavity channel")
ax4.plot(r_num, ue_num**2 + uc_num**2, color="#ffd740", lw=1.5, ls="--", label="total |u|²")
ax4.axhline(0, color="#444", lw=0.6)
ax4.set_xlabel("r  (nm)")
ax4.set_ylabel("|u|²")
ax4.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

fig3.suptitle("Numerical solution inside Region I  (u′(0) = |e⟩ injection)",
              color="white", fontsize=13, y=1.01)
fig3.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)
