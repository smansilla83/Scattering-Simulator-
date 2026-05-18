import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

st.set_page_config(page_title="Feshbach Scattering — Lange et al. 2009", layout="wide")

# ── Physical constants ─────────────────────────────────────────────────────────
a0_nm       = 0.0529177210903
muB_eVperG  = 5.78838e-9
hbar2_2me   = 0.038099          # eV·nm²  (ħ²/2mₑ)
h_eVs       = 4.135668e-15      # eV·s
u_to_me     = 1822.888

# ── Cesium (Lange et al. PRA 2009, Table I) ───────────────────────────────────
m_Cs        = 132.905
m_r_me      = (m_Cs / 2.0) * u_to_me
hbar2_2mr   = hbar2_2me / m_r_me   # eV·nm²

a_bar_cs    = 95.7  * a0_nm    # nm
a_bg        = 1875.0 * a0_nm   # nm

TABLE = {
    's-wave': {'B0': -11.1,  'Bstar': 18.1,   'Bc': 19.7,   'dmu': 2.50, 'Gamma': 11.6},
    'd-wave': {'B0': 47.78,  'Bstar': 47.944,  'Bc': 47.962, 'dmu': 1.15, 'Gamma': 0.065},
    'g-wave': {'B0': 53.449, 'Bstar': 53.457,  'Bc': 53.458, 'dmu': 1.50, 'Gamma': 0.0042},
}

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a,#1a0533);
            border-left:5px solid #00e5ff; border-radius:10px;
            padding:1.4rem 2rem; margin-bottom:1.2rem;">
  <h1 style="color:#fff; margin:0 0 0.6rem 0; font-size:1.9rem;">
    Feshbach Resonance Simulator — Cs atoms
  </h1>
  <p style="color:#b0bec5; font-size:0.9rem; margin:0 0 0.5rem 0;">
    <b style="color:#fff;">Lange et al., PRA 79, 013622 (2009)</b> — two-channel square-well model
  </p>
  <div style="display:flex; gap:1.4rem; flex-wrap:wrap;">
    <span style="color:#00e5ff;">⟐ Interactive Region I eigenvectors</span>
    <span style="color:#69ff47;">⟐ Bloch-plane geometry</span>
    <span style="color:#ffd740;">⟐ Numerical RK45 wavefunction</span>
    <span style="color:#ff6ec7;">⟐ Scattering length a(B)  [Fig. 4]</span>
    <span style="color:#ce93d8;">⟐ Binding energy E_b(B)  [Fig. 3]</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Interactive Simulator")
st.sidebar.markdown("**Region I potential** *(r < ā)*")
Ve     = st.sidebar.slider("Ve  (eV) — open-channel depth",    -10.0, 10.0,  2.0,  0.1)
Vc     = st.sidebar.slider("Vc  (eV) — closed-channel depth",  -10.0, 10.0, -2.0,  0.1)
hOmega = st.sidebar.slider("ℏΩ  (eV) — interchannel coupling",   0.0, 10.0,  1.0,  0.05)
a_bar  = st.sidebar.slider("ā  (nm) — region boundary",           0.5,  5.0,  2.0,  0.1)

with st.sidebar.expander("Parameter guide"):
    st.markdown("""
**Ve** — well depth of the bare open channel |e⟩. Diagonal element $V_{ee} = -V_e$.

**Vc** — well depth of the bare closed channel |c⟩. Diagonal element $V_{cc} = -V_c$.

**ℏΩ** — off-diagonal coupling. Only ΔV and ℏΩ determine the mixing angle θ and
eigenvectors; the common offset shifts eigenvalues only.

**ā** — radius of Region I. Outside this, the potential is zero.
""")

st.sidebar.markdown("---")
st.sidebar.header("Cs paper (Lange et al.)")
st.sidebar.markdown(f"""
| | B₀ (G) | B* (G) | Bc (G) | δμ (μB) | Γ/h (MHz) |
|---|---|---|---|---|---|
| s | −11.1 | 18.1 | 19.7 | 2.50 | 11.6 |
| d | 47.78 | 47.944 | 47.962 | 1.15 | 0.065 |
| g | 53.449 | 53.457 | 53.458 | 1.50 | 0.0042 |

ā = 95.7 a₀  |  a_bg = 1875 a₀
""")
B_probe = st.sidebar.slider("B (G) — probe field", 20.0, 62.0, 30.0, 0.1)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "⟐ Interactive Simulator",
    "⟐ Paper Results (Lange et al.)",
    "⟐ Theory & Assumptions",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Interactive simulator
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<h2 style='color:#00e5ff; margin-top:0.5rem;'>Region I — Interactive Eigenvector Simulator</h2>",
                unsafe_allow_html=True)

    bare_e    = -Ve
    bare_c    = -Vc
    Delta_V   = (Ve - Vc) / 2.0
    offset    = -(Ve + Vc) / 2.0
    R         = np.sqrt(Delta_V**2 + hOmega**2)
    two_theta = np.arctan2(hOmega, Delta_V)
    theta     = two_theta / 2.0
    cos_t     = np.cos(theta)
    sin_t     = np.sin(theta)
    lam_plus  = offset + R
    lam_minus = offset - R
    E_tot     = 0.0

    V_matrix           = np.array([[bare_e, hOmega], [hOmega, bare_c]])
    evals_np, evecs_np = np.linalg.eigh(V_matrix)

    def wavenumber(lam):
        arg = (E_tot - lam) * m_r_me / hbar2_2me
        if arg >= 0:
            return np.sqrt(arg), "propagating", "#69ff47"
        return np.sqrt(-arg), "evanescent", "#ff6ec7"

    k_plus,  k_plus_type,  k_plus_col  = wavenumber(lam_plus)
    k_minus, k_minus_type, k_minus_col = wavenumber(lam_minus)

    # Analytical results
    st.markdown("<h3 style='color:#ffd740;'>Analytical Results</h3>", unsafe_allow_html=True)
    st.markdown(f"""
<div style="display:flex; gap:2.5rem; flex-wrap:nowrap; align-items:flex-start;
            background:#111827; border-radius:10px; padding:1.5rem;">

  <div style="flex:1; min-width:210px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.75rem 0;">Derived quantities</p>
    <table style="width:100%; border-collapse:collapse; color:#fff; font-size:0.92rem;">
      <tr><td style="padding:4px 6px; color:#b0bec5;">offset = −(Ve+Vc)/2</td>
          <td style="padding:4px 6px; color:#aaaaff; font-weight:bold;">{offset:.4f} eV</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">ΔV = (Ve−Vc)/2</td>
          <td style="padding:4px 6px; color:#00e5ff; font-weight:bold;">{Delta_V:.4f} eV</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">R = √(ΔV²+ℏ²Ω²)</td>
          <td style="padding:4px 6px; color:#00e5ff; font-weight:bold;">{R:.4f} eV</td></tr>
      <tr><td style="padding:4px 6px; color:#b0bec5;">2θ = arctan(ℏΩ/ΔV)</td>
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
    <p style="color:#aaa; font-size:0.82rem; margin:0 0 0.5rem 0;">
      Depend only on ΔV and ℏΩ (not the offset).
    </p>
    <p style="color:#fff; font-size:0.93rem; line-height:2.2;">
      <span style="color:#69ff47;">|+⟩</span> = cosθ |e⟩ + sinθ |c⟩
      &nbsp;=&nbsp;<b style="color:#69ff47;">{cos_t:+.4f}</b> |e⟩
      &nbsp;<b style="color:#69ff47;">{sin_t:+.4f}</b> |c⟩<br>
      <span style="color:#ff6ec7;">|−⟩</span> = −sinθ |e⟩ + cosθ |c⟩
      &nbsp;=&nbsp;<b style="color:#ff6ec7;">{-sin_t:+.4f}</b> |e⟩
      &nbsp;<b style="color:#ff6ec7;">{cos_t:+.4f}</b> |c⟩
    </p>
    <p style="color:#aaa; font-size:0.82rem; margin:0.5rem 0 0 0;">
      NumPy check — λ₋: [{evecs_np[0,0]:+.4f}, {evecs_np[1,0]:+.4f}] &nbsp;|&nbsp;
      λ₊: [{evecs_np[0,1]:+.4f}, {evecs_np[1,1]:+.4f}]
    </p>
  </div>

  <div style="flex:1.3; min-width:260px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.75rem 0;">Full Hamiltonian  (E = 0, Cs m_r)</p>
    <table style="width:100%; border-collapse:collapse; color:#fff; font-size:0.9rem;">
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
    <p style="color:#aaa; font-size:0.82rem; margin:0.75rem 0 0 0;">
      ψ±(r) = (A± e<sup>ik±r</sup> + B± e<sup>−ik±r</sup>) |±⟩  [propagating]<br>
      ψ±(r) = (A± e<sup>κ±r</sup> + B± e<sup>−κ±r</sup>) |±⟩  [evanescent]
    </p>
  </div>

</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Potential diagram
    st.markdown("<h3 style='color:#ffd740;'>Potential Energy Diagram</h3>", unsafe_allow_html=True)
    r_pot  = np.linspace(0, a_bar * 2.2, 1000)
    inside = r_pot < a_bar
    Ve_r   = np.where(inside, -Ve, 0.0)
    Vc_r   = np.where(inside, -Vc, 0.0)

    fig_pot, ax_pot = plt.subplots(figsize=(14, 5))
    fig_pot.patch.set_facecolor("#0e1117"); ax_pot.set_facecolor("#0e1117")
    ax_pot.tick_params(colors="white", labelsize=11)
    ax_pot.xaxis.label.set_color("white"); ax_pot.yaxis.label.set_color("white")
    for sp in ax_pot.spines.values(): sp.set_edgecolor("#333")
    ax_pot.axvspan(0, a_bar, color="#1a0533", alpha=0.6, label="Region I  (r < ā)")
    ax_pot.axvline(a_bar, color="#ffd740", lw=1.5, ls="--", alpha=0.7)
    ax_pot.text(a_bar * 1.01, 0, "ā", color="#ffd740", fontsize=13)
    ax_pot.plot(r_pot, Ve_r, color="#ff6ec7", lw=2.5, label=r"$V_{ee}$ — bare |e⟩")
    ax_pot.plot(r_pot, Vc_r, color="#69ff47", lw=2.5, label=r"$V_{cc}$ — bare |c⟩")
    ax_pot.hlines(lam_plus,  0, a_bar, colors="#00e5ff", lw=2, ls="-.", label=f"λ₊ = {lam_plus:.3f} eV")
    ax_pot.hlines(lam_minus, 0, a_bar, colors="#ce93d8", lw=2, ls="-.", label=f"λ₋ = {lam_minus:.3f} eV")
    ax_pot.axhline(E_tot, color="#ffd740", lw=1.5, ls=":", alpha=0.85, label=f"E = {E_tot:.2f} eV")
    if abs(-Ve - (-Vc)) > 0.15:
        mid_r = a_bar * 0.5
        ax_pot.annotate("", xy=(mid_r, -Vc), xytext=(mid_r, -Ve),
                        arrowprops=dict(arrowstyle="<->", color="#ffd740", lw=1.8))
        ax_pot.text(mid_r + a_bar * 0.04, (-Ve + (-Vc)) / 2, "  ℏΩ",
                    color="#ffd740", fontsize=10, va="center")
    yvals = np.concatenate([Ve_r, Vc_r, [lam_plus, lam_minus, E_tot]])
    ypad  = (yvals.max() - yvals.min()) * 0.2 + 0.4
    ax_pot.set_ylim(yvals.min() - ypad, yvals.max() + ypad)
    ax_pot.set_xlim(r_pot.min(), r_pot.max())
    ax_pot.set_xlabel("r  (nm)", fontsize=12); ax_pot.set_ylabel("Energy  (eV)", fontsize=12)
    ax_pot.set_title("Bare and dressed potentials — Region I", color="white", fontsize=13)
    ax_pot.legend(loc="upper right", framealpha=0.2, labelcolor="white",
                  facecolor="#0e1117", edgecolor="#444", fontsize=10)
    fig_pot.tight_layout(); st.pyplot(fig_pot, use_container_width=True); plt.close(fig_pot)

    # Bloch-plane
    st.markdown("<h3 style='color:#ffd740;'>Bloch-plane View</h3>", unsafe_allow_html=True)
    fig_bl, axes_bl = plt.subplots(1, 2, figsize=(18, 9))
    fig_bl.patch.set_facecolor("#0e1117")
    for ax in axes_bl:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes_bl[0]
    ax.set_title(r"$(\sigma_z,\,\sigma_x)$ plane", color="white", fontsize=13)
    ax.set_xlabel(r"$\Delta_V$  ($\sigma_z$)", color="white")
    ax.set_ylabel(r"$\hbar\Omega$  ($\sigma_x$)", color="white")
    ax.set_aspect("equal")
    lim = max(R * 1.5, 0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.2, lim * 1.4)
    ax.axhline(0, color="#333", lw=0.8); ax.axvline(0, color="#333", lw=0.8)
    ax.annotate("", xy=(Delta_V, hOmega), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=2.5))
    ax.text(Delta_V * 0.45 + lim * 0.04, hOmega * 0.55, r"$\vec{n}$, $R$",
            color="#00e5ff", fontsize=12)
    arc_r = R * 0.3
    arc   = mpatches.Arc((0,0), 2*arc_r, 2*arc_r, angle=0,
                         theta1=0, theta2=np.degrees(two_theta), color="#ffd740", lw=2)
    ax.add_patch(arc)
    mid = two_theta / 2
    ax.text(arc_r*1.35*np.cos(mid), arc_r*1.35*np.sin(mid), r"$2\theta$", color="#ffd740", fontsize=11)
    ax.plot([0, Delta_V], [0, 0], color="#ff6ec7", lw=1.6, ls="--")
    ax.plot([Delta_V, Delta_V], [0, hOmega], color="#69ff47", lw=1.6, ls="--")
    ax.text(Delta_V/2, -lim*0.1, r"$\Delta_V$", color="#ff6ec7", ha="center", fontsize=11)
    ax.text(Delta_V+lim*0.07, hOmega/2, r"$\hbar\Omega$", color="#69ff47", fontsize=11)

    ax2 = axes_bl[1]
    ax2.set_title("Dressed states on the unit circle", color="white", fontsize=13)
    ax2.set_xlabel(r"$|e\rangle$ component", color="white")
    ax2.set_ylabel(r"$|c\rangle$ component", color="white")
    ax2.set_aspect("equal"); ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5)
    ax2.add_patch(plt.Circle((0,0), 1, color="#333", fill=False, lw=1.5))
    ax2.axhline(0, color="#333", lw=0.8); ax2.axvline(0, color="#333", lw=0.8)
    ax2.annotate("", xy=(cos_t, sin_t), xytext=(0,0),
                 arrowprops=dict(arrowstyle="->", color="#69ff47", lw=2.5))
    ax2.text(cos_t*1.18, sin_t*1.18, r"$|{+}\rangle$",
             color="#69ff47", fontsize=13, ha="center", va="center")
    ax2.annotate("", xy=(-sin_t, cos_t), xytext=(0,0),
                 arrowprops=dict(arrowstyle="->", color="#ff6ec7", lw=2.5))
    ax2.text(-sin_t*1.22, cos_t*1.22, r"$|{-}\rangle$",
             color="#ff6ec7", fontsize=13, ha="center", va="center")
    arc2 = mpatches.Arc((0,0), 0.5, 0.5, angle=0,
                        theta1=0, theta2=np.degrees(theta), color="#ce93d8", lw=2)
    ax2.add_patch(arc2)
    ax2.text(0.35*np.cos(theta/2), 0.35*np.sin(theta/2)+0.05, r"$\theta$",
             color="#ce93d8", fontsize=12)
    fig_bl.tight_layout(); st.pyplot(fig_bl, use_container_width=True); plt.close(fig_bl)

    # Eigenvalue spectrum
    st.markdown("<h3 style='color:#ffd740;'>Eigenvalue Spectrum</h3>", unsafe_allow_html=True)
    levels = [
        (bare_e,   "bare |e⟩",   "#ff6ec7", "--"),
        (bare_c,   "bare |c⟩",   "#69ff47", "--"),
        (lam_plus, "λ₊ dressed", "#00e5ff", "-"),
        (lam_minus,"λ₋ dressed", "#ce93d8", "-"),
    ]
    all_vals = [v for v, *_ in levels] + [E_tot]
    ymin = min(all_vals); ymax = max(all_vals)
    pad  = max((ymax - ymin) * 0.4, 0.5)

    fig_sp, ax_sp = plt.subplots(figsize=(12, 5))
    fig_sp.patch.set_facecolor("#0e1117"); ax_sp.set_facecolor("#0e1117")
    ax_sp.tick_params(colors="white", labelsize=11)
    ax_sp.set_ylabel("Energy (eV)", color="white", fontsize=12)
    for sp in ax_sp.spines.values(): sp.set_edgecolor("#333")
    ax_sp.xaxis.set_visible(False)
    ax_sp.set_xlim(0, 1); ax_sp.set_ylim(ymin - pad, ymax + pad)
    min_gap = pad * 0.35
    sorted_levels = sorted(levels, key=lambda x: x[0])
    placed = []
    for y, label, col, ls in sorted_levels:
        ax_sp.axhline(y, color=col, lw=2.5, ls=ls, alpha=0.95)
        y_lbl = y
        for py, *_ in placed:
            if abs(y_lbl - py) < min_gap:
                y_lbl = py + min_gap
        placed.append((y_lbl, label, col))
    for y_lbl, label, col in placed:
        ax_sp.text(0.02, y_lbl, label, color=col, va="center", fontsize=11, fontweight="bold",
                   transform=ax_sp.get_yaxis_transform(),
                   bbox=dict(boxstyle="round,pad=0.25", facecolor="#0e1117", edgecolor="none", alpha=0.85))
    ax_sp.axhline(E_tot, color="#ffd740", lw=1.8, ls=":", alpha=0.9)
    ax_sp.text(0.02, E_tot, f"E = {E_tot:.2f} eV", color="#ffd740", va="center", fontsize=11,
               fontweight="bold", transform=ax_sp.get_yaxis_transform(),
               bbox=dict(boxstyle="round,pad=0.25", facecolor="#0e1117", edgecolor="none", alpha=0.85))
    fig_sp.tight_layout(); st.pyplot(fig_sp, use_container_width=True); plt.close(fig_sp)

    # Numerical ODE
    st.markdown("<h3 style='color:#ffd740;'>Numerical Solution — Coupled Schrödinger ODE</h3>",
                unsafe_allow_html=True)
    st.markdown("""<p style="color:#b0bec5; font-size:0.93rem; margin:0 0 1rem 0;">
Solves <b style="color:#fff;">u″ = (2mᵣ/ℏ²)(V̂ − E) u</b> via RK45,
starting with pure |e⟩ injection at r=0. Dashed lines show the analytical
dressed-state decomposition.</p>""", unsafe_allow_html=True)

    factor_sim = m_r_me / hbar2_2me

    def ode_sim(r, y):
        ue, uc, pue, puc = y
        dpue = factor_sim * ((bare_e - E_tot) * ue + hOmega * uc)
        dpuc = factor_sim * ((bare_c - E_tot) * uc + hOmega * ue)
        return [pue, puc, dpue, dpuc]

    r_end_sim  = a_bar * 1.5
    r_eval_sim = np.linspace(0.0, r_end_sim, 2000)
    sol_sim    = solve_ivp(ode_sim, [0.0, r_end_sim], [0.0, 0.0, 1.0, 0.0],
                           t_eval=r_eval_sim, method="RK45", rtol=1e-10, atol=1e-12)
    ue_num = sol_sim.y[0]; uc_num = sol_sim.y[1]; r_num = sol_sim.t

    def dressed_wf(r, A, k, evanescent):
        return A * (np.sinh(k * r) if evanescent else np.sin(k * r))

    Ap     = cos_t  / k_plus  if k_plus  > 1e-8 else cos_t  * 1e8
    Am     = -sin_t / k_minus if k_minus > 1e-8 else -sin_t * 1e8
    evan_p = (k_plus_type  == "evanescent")
    evan_m = (k_minus_type == "evanescent")
    fp     = dressed_wf(r_num, Ap, k_plus,  evan_p)
    fm     = dressed_wf(r_num, Am, k_minus, evan_m)
    ue_ana = cos_t * fp - sin_t * fm
    uc_ana = sin_t * fp + cos_t * fm

    fig_ode, axes_ode = plt.subplots(2, 2, figsize=(16, 9))
    fig_ode.patch.set_facecolor("#0e1117")
    for ax in axes_ode.flat:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.axvline(a_bar, color="#ffd740", lw=1.2, ls="--", alpha=0.6)

    ax = axes_ode[0, 0]
    ax.set_title("Bare channels — numerical vs analytical", color="white")
    ax.plot(r_num, ue_num, color="#ff6ec7", lw=2, label="uₑ  num")
    ax.plot(r_num, uc_num, color="#69ff47", lw=2, label="u_c  num")
    ax.plot(r_num, ue_ana, color="#ff6ec7", lw=1.2, ls="--", alpha=0.6, label="uₑ  ana")
    ax.plot(r_num, uc_ana, color="#69ff47", lw=1.2, ls="--", alpha=0.6, label="u_c  ana")
    ax.axhline(0, color="#444", lw=0.6); ax.set_xlabel("r (nm)"); ax.set_ylabel("amplitude")
    ax.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    ax2 = axes_ode[0, 1]
    ax2.set_title("Dressed channels u±(r)", color="white")
    ax2.plot(r_num, fp, color="#00e5ff", lw=2,
             label=f"|+⟩  {'prop.' if not evan_p else 'evan.'}, k={k_plus:.3f} nm⁻¹")
    ax2.plot(r_num, fm, color="#ce93d8", lw=2,
             label=f"|−⟩  {'prop.' if not evan_m else 'evan.'}, k={k_minus:.3f} nm⁻¹")
    ax2.axhline(0, color="#444", lw=0.6); ax2.set_xlabel("r (nm)"); ax2.set_ylabel("amplitude")
    ax2.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    ax3b = axes_ode[1, 0]
    ax3b.set_title("Residual  |numerical − analytical|", color="white")
    ax3b.semilogy(r_num[1:], np.abs(ue_num[1:]-ue_ana[1:])+1e-16, color="#ff6ec7", lw=1.5, label="|uₑ error|")
    ax3b.semilogy(r_num[1:], np.abs(uc_num[1:]-uc_ana[1:])+1e-16, color="#69ff47", lw=1.5, label="|u_c error|")
    ax3b.set_xlabel("r (nm)"); ax3b.set_ylabel("absolute error")
    ax3b.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")
    ax3b.tick_params(axis="y", colors="white")

    ax4 = axes_ode[1, 1]
    ax4.set_title("Probability density  |u(r)|²", color="white")
    ax4.plot(r_num, ue_num**2, color="#ff6ec7", lw=2, label="|uₑ|²")
    ax4.plot(r_num, uc_num**2, color="#69ff47", lw=2, label="|u_c|²")
    ax4.plot(r_num, ue_num**2 + uc_num**2, color="#ffd740", lw=1.5, ls="--", label="total")
    ax4.axhline(0, color="#444", lw=0.6); ax4.set_xlabel("r (nm)"); ax4.set_ylabel("|u|²")
    ax4.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    fig_ode.suptitle("RK45 solution — u′(0) = |e⟩ injection", color="white", fontsize=13, y=1.01)
    fig_ode.tight_layout(); st.pyplot(fig_ode, use_container_width=True); plt.close(fig_ode)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Lange et al. paper results
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<h2 style='color:#00e5ff;'>Lange et al. Paper Results — Cs atoms</h2>",
                unsafe_allow_html=True)

    def a_of_B(B):
        B = np.asarray(B, dtype=float); result = np.ones_like(B)
        for r in TABLE.values():
            denom = B - r['B0']; numer = B - r['Bstar']
            result = np.where(np.abs(denom) > 0.005, result * numer / denom, np.nan)
        return a_bg * result

    def Eb_kHz(a_nm_arr):
        a  = np.asarray(a_nm_arr, dtype=float); Eb = np.full_like(a, np.nan)
        m  = a > 2 * a_bar_cs
        disc  = 1 - 2 * a_bar_cs / a[m]
        kappa = (1 - np.sqrt(np.clip(disc, 0, None))) / a_bar_cs
        Eb[m] = -hbar2_2mr * kappa**2 / (h_eVs * 1e3)
        m2    = (a > 0) & (a <= 2 * a_bar_cs)
        Eb[m2]= -hbar2_2mr / a[m2]**2 / (h_eVs * 1e3)
        return Eb

    B_arr    = np.linspace(20.0, 62.0, 8000)
    a_arr    = a_of_B(B_arr)
    Eb_arr   = Eb_kHz(a_arr)
    a_probe  = a_of_B(B_probe)
    Eb_probe = Eb_kHz(np.array([a_probe]))[0]

    # Fig 3
    st.markdown("<h3 style='color:#00e5ff;'>Fig. 3 — Molecular Binding Energy  E_b/h (kHz)</h3>",
                unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(13, 6))
    fig3.patch.set_facecolor("#0e1117"); ax3.set_facecolor("#0e1117")
    ax3.tick_params(colors="white", labelsize=11)
    ax3.xaxis.label.set_color("white"); ax3.yaxis.label.set_color("white")
    for sp in ax3.spines.values(): sp.set_edgecolor("#333")
    ax3.plot(B_arr, Eb_arr, color="#e74c3c", lw=2, label="Model (Eq. 13, effective-range)")
    for name, r in TABLE.items():
        ax3.axvline(r['B0'], color="#ffd740", lw=1.2, ls="--", alpha=0.6)
        ax3.text(r['B0']+0.2, -5, name, color="#ffd740", fontsize=9)
    if not np.isnan(Eb_probe):
        ax3.scatter([B_probe], [Eb_probe], color="#00e5ff", s=80, zorder=6)
        ax3.text(B_probe+0.3, Eb_probe-8,
                 f"B={B_probe:.1f}G\nEb/h={Eb_probe:.1f}kHz", color="#00e5ff", fontsize=9)
    ax3.set_xlim(20, 62); ax3.set_ylim(-360, 10)
    ax3.set_xlabel("Magnetic field (G)", fontsize=12)
    ax3.set_ylabel("Binding energy  Eᵦ/h  (kHz)", fontsize=12)
    ax3.set_title("Binding energy of Cs₂  [cf. Fig. 3, Lange et al.]", color="white", fontsize=12)
    ax3.legend(fontsize=10, framealpha=0.2, labelcolor="white", facecolor="#0e1117", edgecolor="#444")
    ax3.axhline(0, color="#555", lw=0.8, ls="--")
    fig3.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close(fig3)

    # Fig 4
    st.markdown("<h3 style='color:#00e5ff;'>Fig. 4 — Scattering Length  a(B)</h3>",
                unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(13, 5))
    fig4.patch.set_facecolor("#0e1117"); ax4.set_facecolor("#0e1117")
    ax4.tick_params(colors="white", labelsize=11)
    ax4.xaxis.label.set_color("white"); ax4.yaxis.label.set_color("white")
    for sp in ax4.spines.values(): sp.set_edgecolor("#333")
    a_plot = np.clip(a_arr / a0_nm, -8000, 8000)
    ax4.plot(B_arr, a_plot, color="#00e5ff", lw=2)
    ax4.axhline(0, color="#555", lw=0.8, ls="--")
    ax4.axhline(a_bg/a0_nm, color="#aaaaff", lw=1, ls=":", alpha=0.7,
                label=f"a_bg = {a_bg/a0_nm:.0f} a₀")
    for name, r in TABLE.items():
        ax4.axvline(r['B0'],    color="#ffd740", lw=1.2, ls="--", alpha=0.5)
        ax4.axvline(r['Bstar'], color="#ff6ec7", lw=1.0, ls=":",  alpha=0.5)
    ax4.scatter([B_probe], [np.clip(a_probe/a0_nm,-8000,8000)], color="#00e5ff", s=80, zorder=6)
    ax4.text(B_probe+0.3, np.clip(a_probe/a0_nm,-8000,8000)+200,
             f"a={a_probe/a0_nm:.0f} a₀", color="#00e5ff", fontsize=9)
    ax4.set_xlim(20, 62); ax4.set_ylim(-8000, 8000)
    ax4.set_xlabel("Magnetic field (G)", fontsize=12)
    ax4.set_ylabel("Scattering length  a  (a₀)", fontsize=12)
    ax4.set_title("Scattering length a(B)  [cf. Fig. 4, Lange et al.]", color="white", fontsize=12)
    ax4.legend(fontsize=10, framealpha=0.2, labelcolor="white", facecolor="#0e1117", edgecolor="#444")
    ax4.text(18.1, 500, "B*_s", color="#ff6ec7", fontsize=8, ha="center")
    fig4.tight_layout(); st.pyplot(fig4, use_container_width=True); plt.close(fig4)

    # Region I at B_probe
    st.markdown(f"<h3 style='color:#00e5ff;'>Region I Analysis at B = {B_probe:.1f} G</h3>",
                unsafe_allow_html=True)
    delta = {nm: r['dmu'] * muB_eVperG * (B_probe - r['Bc']) for nm, r in TABLE.items()}

    def abg_from_V(V_eV):
        k  = np.sqrt(2 * m_r_me * V_eV / hbar2_2me)
        ka = k * a_bar_cs
        if abs(np.cos(ka)) < 1e-12: return np.inf
        return a_bar_cs * (1 - np.tan(ka) / ka)

    V_pole_1  = (np.pi/2)**2 * hbar2_2me / (2 * m_r_me * a_bar_cs**2)
    V_open_eV = brentq(lambda V: abg_from_V(V) - a_bg, V_pole_1*1.001, V_pole_1*5.0)

    delta_s_eV = delta['s-wave']
    hGamma_s   = TABLE['s-wave']['Gamma'] * 1e6 * h_eVs
    W_eV       = np.sqrt(hGamma_s * hbar2_2mr / a_bar_cs**2)
    V_oo_cs    = -V_open_eV
    V_cc_cs    = delta_s_eV
    DV_cs      = (V_oo_cs - V_cc_cs) / 2
    off_cs     = (V_oo_cs + V_cc_cs) / 2
    R_cs       = np.sqrt(DV_cs**2 + W_eV**2)
    th_cs      = np.arctan2(W_eV, DV_cs) / 2
    lp_cs      = off_cs + R_cs; lm_cs = off_cs - R_cs

    st.markdown(f"""
<div style="display:flex; gap:2rem; flex-wrap:nowrap; align-items:flex-start;
            background:#111827; border-radius:10px; padding:1.3rem;">
  <div style="flex:1; min-width:200px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.6rem 0;">Cs Region I</p>
    <table style="color:#fff; font-size:0.88rem; border-collapse:collapse; width:100%;">
      <tr><td style="color:#aaa;padding:3px 6px;">V_open</td>
          <td style="padding:3px 6px;color:#69ff47;">{V_open_eV*1e9:.4f} neV</td></tr>
      <tr><td style="color:#aaa;padding:3px 6px;">δ_s at B={B_probe:.1f}G</td>
          <td style="padding:3px 6px;color:#ffd740;">{delta_s_eV*1e9:.4f} neV</td></tr>
      <tr><td style="color:#aaa;padding:3px 6px;">W (coupling)</td>
          <td style="padding:3px 6px;color:#ff6ec7;">{W_eV*1e9:.4f} neV</td></tr>
      <tr><td style="color:#aaa;padding:3px 6px;">a(B)</td>
          <td style="padding:3px 6px;color:#00e5ff;">{a_probe/a0_nm:.1f} a₀</td></tr>
      <tr><td style="color:#aaa;padding:3px 6px;">E_b/h</td>
          <td style="padding:3px 6px;color:#69ff47;">{Eb_probe:.2f} kHz</td></tr>
    </table>
  </div>
  <div style="flex:1; min-width:220px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.6rem 0;">Dressed eigenstates</p>
    <table style="color:#fff; font-size:0.88rem; border-collapse:collapse; width:100%;">
      <tr><td style="color:#aaa;padding:3px 6px;">θ</td>
          <td style="padding:3px 6px;color:#ce93d8;">{np.degrees(th_cs):.4f}°</td></tr>
      <tr><td style="color:#aaa;padding:3px 6px;">λ₊</td>
          <td style="padding:3px 6px;color:#69ff47;">{lp_cs*1e9:.4f} neV</td></tr>
      <tr><td style="color:#aaa;padding:3px 6px;">λ₋</td>
          <td style="padding:3px 6px;color:#ff6ec7;">{lm_cs*1e9:.4f} neV</td></tr>
    </table>
    <p style="color:#fff; font-size:0.88rem; margin:0.8rem 0 0 0; line-height:2;">
      <span style="color:#69ff47;">|+⟩</span> = {np.cos(th_cs):+.4f} |open⟩ {np.sin(th_cs):+.4f} |closed⟩<br>
      <span style="color:#ff6ec7;">|−⟩</span> = {-np.sin(th_cs):+.4f} |open⟩ {np.cos(th_cs):+.4f} |closed⟩
    </p>
  </div>
  <div style="flex:1; min-width:200px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.6rem 0;">Magnetic detunings</p>
    <table style="color:#fff; font-size:0.88rem; border-collapse:collapse; width:100%;">
      {"".join(f'<tr><td style="color:#aaa;padding:3px 6px;">{nm}</td>'
               f'<td style="padding:3px 6px;color:#ffd740;">{dv*1e9:.4f} neV</td></tr>'
               for nm, dv in delta.items())}
    </table>
  </div>
</div>
""", unsafe_allow_html=True)

    # Cs ODE wavefunction
    st.markdown(f"<h3 style='color:#00e5ff;'>Numerical Wavefunction at B = {B_probe:.1f} G  (E = 0)</h3>",
                unsafe_allow_html=True)
    factor_cs = 2.0 * m_r_me / hbar2_2me

    def ode_cs(t, y):
        uo, uc, puo, puc = y
        return [puo, puc,
                factor_cs * (V_oo_cs * uo + W_eV * uc),
                factor_cs * (V_cc_cs * uc + W_eV * uo)]

    r_end_cs  = a_bar_cs * 3.0
    r_eval_cs = np.linspace(1e-6, r_end_cs, 3000)
    sol_cs    = solve_ivp(ode_cs, [1e-6, r_end_cs], [1e-6, 0., 1., 0.],
                          t_eval=r_eval_cs, method="RK45", rtol=1e-10, atol=1e-12)
    uo_num    = sol_cs.y[0]; uc_num_cs = sol_cs.y[1]; r_cs = sol_cs.t
    uo_end    = sol_cs.y[0,-1]; puo_end = sol_cs.y[2,-1]
    a_num     = r_end_cs - uo_end/puo_end if abs(puo_end) > 1e-12 else np.nan

    fig5, axes5 = plt.subplots(1, 2, figsize=(16, 5))
    fig5.patch.set_facecolor("#0e1117")
    for ax in axes5:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.axvline(a_bar_cs, color="#ffd740", lw=1.2, ls="--", alpha=0.5, label="ā boundary")
    ax = axes5[0]
    ax.plot(r_cs, uo_num,    color="#69ff47", lw=2, label="u_open(r)")
    ax.plot(r_cs, uc_num_cs, color="#ff6ec7", lw=2, label="u_closed(r)")
    ax.axhline(0, color="#444", lw=0.6)
    ax.set_xlabel("r (nm)"); ax.set_ylabel("u(r) [arb.]")
    ax.set_title(f"E=0 wavefunction → a_num={a_num/a0_nm:.1f} a₀  (analytic: {a_probe/a0_nm:.1f} a₀)",
                 color="white")
    ax.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")
    ax2 = axes5[1]
    ax2.plot(r_cs, uo_num**2,    color="#69ff47", lw=2, label="|u_open|²")
    ax2.plot(r_cs, uc_num_cs**2, color="#ff6ec7", lw=2, label="|u_closed|²")
    ax2.plot(r_cs, uo_num**2+uc_num_cs**2, color="#ffd740", lw=1.5, ls="--", label="total")
    ax2.axhline(0, color="#444", lw=0.6)
    ax2.set_xlabel("r (nm)"); ax2.set_ylabel("|u(r)|²")
    ax2.set_title("Probability density", color="white")
    ax2.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")
    fig5.tight_layout(); st.pyplot(fig5, use_container_width=True); plt.close(fig5)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Theory & Assumptions
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<h2 style='color:#00e5ff;'>Theory & Assumptions</h2>", unsafe_allow_html=True)
    st.markdown("""
<p style="color:#b0bec5;">
All calculations follow <b style="color:#fff;">Lange et al., PRA 79, 013622 (2009)</b>.
The model describes two Cs atoms scattering near Feshbach resonances using a
two-channel square-well potential.
</p>
""", unsafe_allow_html=True)

    # ── 1. Physical setup
    st.markdown("### 1. Physical Setup")
    st.markdown("""
Two ultracold Cs atoms in state |F=3, m_F=3⟩ collide at low energy.
The relative motion is governed by a two-channel (open + closed) radial Schrödinger equation.
A magnetic field **B** tunes the energy of the closed channel relative to the open one
via the differential magnetic moment δμ.
""")
    st.markdown("**Reduced mass:**")
    st.latex(r"m_r = \frac{m_{\mathrm{Cs}}}{2} = \frac{132.905\,\mathrm{amu}}{2}")

    # ── 2. Region I potential matrix
    st.markdown("### 2. Two-Channel Square-Well Potential (Region I,  r < ā)")
    st.markdown("""
Inside the interaction range ā (mean scattering length), the potential is **constant**,
allowing exact analytic diagonalisation. The bare-channel potential matrix is:
""")
    st.latex(r"\hat{V} = \begin{pmatrix} V_\mathrm{oo} & W \\ W & V_\mathrm{cc}(B) \end{pmatrix}")
    st.markdown("""
- **V_oo = −V_open** — open-channel well depth (attractive). Fixed by requiring the single-channel
  scattering length equals a_bg.
- **V_cc(B) = δμ(B − Bc)** — closed-channel threshold, shifted by the magnetic detuning.
- **W** — interchannel coupling (off-diagonal), related to the resonance width Γ.
""")
    st.markdown("The magnetic detuning for each resonance i:")
    st.latex(r"\delta_i(B) = \delta\mu_i \cdot \mu_B \cdot (B - B_{c,i})")

    # ── 3. Coupled Schrödinger equation
    st.markdown("### 3. Coupled-Channel Schrödinger Equation")
    st.markdown("The radial wavefunction vector **u** = (u_open, u_closed)ᵀ satisfies:")
    st.latex(r"-\frac{\hbar^2}{2m_r}\frac{d^2\mathbf{u}}{dr^2} + \hat{V}\,\mathbf{u} = E\,\mathbf{u}")
    st.markdown("Rearranged for numerical integration:")
    st.latex(r"\frac{d^2\mathbf{u}}{dr^2} = \frac{2m_r}{\hbar^2}\bigl(\hat{V} - E\bigr)\,\mathbf{u}")
    st.markdown("""
This is solved numerically via **RK45** (scipy `solve_ivp`) with initial conditions
u(0) = 0 and u′(0) = (1, 0) (pure open-channel injection).
The scattering length is extracted from the asymptotic slope:
""")
    st.latex(r"a = r_\mathrm{end} - \frac{u_\mathrm{open}(r_\mathrm{end})}{u'_\mathrm{open}(r_\mathrm{end})}")

    # ── 4. Diagonalisation
    st.markdown("### 4. Analytical Diagonalisation of V̂")
    st.markdown("""
Because V̂ is **uniform** inside Region I, it can be diagonalised exactly.
Writing the traceless part with ΔV = (V_oo − V_cc)/2 and coupling W:
""")
    st.latex(r"R = \sqrt{\Delta V^2 + W^2}")
    st.latex(r"\tan 2\theta = \frac{W}{\Delta V}")
    st.markdown("The dressed (adiabatic) eigenvalues are:")
    st.latex(r"\lambda_\pm = \frac{V_\mathrm{oo}+V_\mathrm{cc}}{2} \pm R")
    st.markdown("The dressed eigenstates (mixing open and closed channels):")
    st.latex(r"|{+}\rangle = \cos\theta\,|\mathrm{open}\rangle + \sin\theta\,|\mathrm{closed}\rangle")
    st.latex(r"|{-}\rangle = -\sin\theta\,|\mathrm{open}\rangle + \cos\theta\,|\mathrm{closed}\rangle")
    st.markdown("""
Inside Region I the wavefunction propagates (or decays) in each dressed channel
with wavenumber k± = √(2m_r(E − λ±))/ħ. A channel is **propagating** when E > λ±
and **evanescent** when E < λ±.
""")

    # ── 5. Scattering length
    st.markdown("### 5. Multi-Resonance Scattering Length  a(B)")
    st.markdown("""
Lange et al. fit the full coupled-channel calculation to a product formula over
three Feshbach resonances (s-, d-, g-wave):
""")
    st.latex(r"a(B) = a_\mathrm{bg}\prod_{i}\frac{B - B^*_i}{B - B_{0,i}}")
    st.markdown("""
| Symbol | Meaning |
|--------|---------|
| a_bg = 1875 a₀ | background scattering length |
| B0_i | pole of a(B) — resonance position |
| B*_i | zero of a(B) — where open meets closed without coupling |

The formula gives a(B) → ±∞ at each B0_i and a(B) = 0 at each B*_i.
""")

    # ── 6. Binding energy
    st.markdown("### 6. Molecular Binding Energy  E_b(B)")
    st.markdown("""
Near threshold the bound-state energy is related to the scattering length via the
**universal formula with effective-range correction** (r_eff ≈ ā):
""")
    st.latex(r"\kappa = \frac{1 - \sqrt{1 - 2\bar{a}/a}}{\bar{a}}")
    st.latex(r"E_b = -\frac{\hbar^2\kappa^2}{2m_r}")
    st.markdown("""
This is valid when a > 2ā. For a ≤ 2ā the simple universal relation
E_b = −ħ²/(2m_r a²) is used instead.
The mean scattering length **ā = 95.7 a₀** sets the effective range scale.
""")

    # ── 7. Open-channel well depth from a_bg
    st.markdown("### 7. Finding V_open from a_bg")
    st.markdown("""
The open-channel well depth V_open is fixed by requiring the **single-channel** square-well
scattering length to equal a_bg:
""")
    st.latex(r"a_\mathrm{bg} = \bar{a}\left(1 - \frac{\tan(k\bar{a})}{k\bar{a}}\right), \quad k = \sqrt{\frac{2m_r V_\mathrm{open}}{\hbar^2}}")
    st.markdown("""
This equation has multiple branches (poles whenever k·ā = (n+½)π).
The first pole occurs at:
""")
    st.latex(r"V_\mathrm{pole,1} = \frac{\pi^2}{8}\frac{\hbar^2}{m_r\bar{a}^2} \approx 15.1\;\mathrm{neV}")
    st.markdown("""
Since a_bg = 1875 a₀ ≫ ā, the well is near-resonant and V_open sits just above
V_pole,1 in the first branch. The root is found numerically with `scipy.optimize.brentq`.
""")

    # ── 8. Assumptions
    st.markdown("### 8. Key Assumptions")
    st.markdown("""
| Assumption | Consequence |
|------------|-------------|
| **Square-well potential** — V̂ is piecewise constant | Allows exact analytic diagonalisation inside Region I |
| **Constant V̂ inside ā** | Wavefunction has sinusoidal/hyperbolic form; mixing angle θ is position-independent |
| **s-wave dominance** | Only ℓ=0 partial wave contributes at ultracold temperatures (kT ≪ centrifugal barrier) |
| **E ≈ 0 (threshold)** | Scattering length fully characterises low-energy cross section |
| **Three independent resonances** | Product formula for a(B) treats s-, d-, g-wave poles as independent; cross-terms neglected |
| **Effective range r_eff = ā** | Simplifies binding energy formula; valid near a broad resonance |
| **Single detuning parameter per resonance** | Linear B-dependence of δ(B) = δμ·μB·(B − Bc) assumed |
| **No three-body losses** | Model is purely two-body; ignores Efimov physics and inelastic processes |
""")

    st.markdown("### 9. Cs Parameters (Table I, Lange et al. 2009)")
    st.markdown(f"""
| Quantity | Value |
|----------|-------|
| ā (mean scattering length) | 95.7 a₀ = {a_bar_cs/a0_nm:.1f} a₀ |
| a_bg (background scattering length) | 1875 a₀ |
| m_r (reduced mass) | m_Cs/2 = {m_r_me:.0f} mₑ |
| ħ²/(2m_r) | {hbar2_2mr*1e7:.4f} × 10⁻⁷ eV·nm² |
| μB | 5.788 × 10⁻⁹ eV/G |

**Resonance parameters:**

| | B₀ (G) | B* (G) | Bc (G) | δμ (μB) | Γ/h (MHz) |
|---|---|---|---|---|---|
| s-wave | −11.1 | 18.1 | 19.7 | 2.50 | 11.6 |
| d-wave | 47.78 | 47.944 | 47.962 | 1.15 | 0.065 |
| g-wave | 53.449 | 53.457 | 53.458 | 1.50 | 0.0042 |
""")
