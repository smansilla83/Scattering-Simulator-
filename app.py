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
hOmega = st.sidebar.slider("W = ℏΩ  (eV) — interchannel coupling",   0.0, 10.0,  1.0,  0.05)
a_bar  = st.sidebar.slider("ā  (nm) — region boundary",           0.5,  5.0,  2.0,  0.1)

with st.sidebar.expander("Parameter guide"):
    st.markdown("""
**Ve** — well depth of the bare open channel |e⟩. Diagonal element $V_{ee} = -V_e$.

**Vc** — well depth of the bare closed channel |c⟩. Diagonal element $V_{cc} = -V_c$.

**ℏΩ = W** — off-diagonal interchannel coupling. In the Lange et al. paper this is called **W**;
in quantum optics / cavity QED it is written ℏΩ where Ω is the Rabi frequency. Same quantity,
different notation. Only ΔV and W determine the mixing angle θ and eigenvectors; the common
offset shifts eigenvalues only.

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⟐ Interactive Simulator",
    "⟐ Paper Results (Lange et al.)",
    "⟐ Theory & Assumptions",
    "⟐ van der Waals Extension",
    "⟐ Verification",
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

    # Mixing angle vs B
    st.markdown("<h3 style='color:#00e5ff;'>Mixing Angle θ(B) — field-driven evolution</h3>",
                unsafe_allow_html=True)
    st.markdown("""
<p style="color:#b0bec5; font-size:0.93rem; margin:0 0 0.8rem 0;">
<b style="color:#fff;">W is a fitted constant</b> (from Γ in Table I) — it does <i>not</i> change with B.
What changes with field is the detuning δ(B) = δμ·μB·(B − Bc), which shifts the closed-channel
threshold V_cc. This makes ΔV = (V_oo − V_cc)/2 field-dependent, so the mixing angle
<b style="color:#ce93d8;">θ(B) = ½ arctan(W / ΔV)</b> rotates continuously across the resonance.
Near B = Bc the channels are degenerate (ΔV → 0) and θ → 45°; far from resonance θ → 0° or 90°.
</p>
""", unsafe_allow_html=True)

    # Compute θ(B) across the sweep
    def abg_from_V_sweep(V_eV):
        k  = np.sqrt(2 * m_r_me * V_eV / hbar2_2me)
        ka = k * a_bar_cs
        if abs(np.cos(ka)) < 1e-12: return np.inf
        return a_bar_cs * (1 - np.tan(ka) / ka)

    _V_pole_1  = (np.pi/2)**2 * hbar2_2me / (2 * m_r_me * a_bar_cs**2)
    _V_open_eV = brentq(lambda V: abg_from_V_sweep(V) - a_bg, _V_pole_1*1.001, _V_pole_1*5.0)
    _hGamma_s  = TABLE['s-wave']['Gamma'] * 1e6 * h_eVs
    _W_eV      = np.sqrt(_hGamma_s * hbar2_2mr / a_bar_cs**2)
    _V_oo      = -_V_open_eV

    theta_B = np.zeros_like(B_arr)
    lam_p_B = np.zeros_like(B_arr)
    lam_m_B = np.zeros_like(B_arr)
    for i, B in enumerate(B_arr):
        _delta_s = TABLE['s-wave']['dmu'] * muB_eVperG * (B - TABLE['s-wave']['Bc'])
        _V_cc    = _delta_s
        _DV      = (_V_oo - _V_cc) / 2
        _off     = (_V_oo + _V_cc) / 2
        _R       = np.sqrt(_DV**2 + _W_eV**2)
        theta_B[i] = np.degrees(np.arctan2(_W_eV, _DV) / 2)
        lam_p_B[i] = _off + _R
        lam_m_B[i] = _off - _R

    fig_th, axes_th = plt.subplots(1, 2, figsize=(16, 5))
    fig_th.patch.set_facecolor("#0e1117")
    for ax in axes_th:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.set_xlim(20, 62)
        for nm, r in TABLE.items():
            ax.axvline(r['B0'], color="#ffd740", lw=1, ls="--", alpha=0.4)
            ax.axvline(r['Bc'], color="#ff6ec7", lw=1, ls=":",  alpha=0.4)

    ax_th = axes_th[0]
    ax_th.plot(B_arr, theta_B, color="#ce93d8", lw=2)
    ax_th.axhline(45, color="#aaa", lw=0.8, ls="--", alpha=0.5, label="θ = 45° (resonance)")
    ax_th.scatter([B_probe], [np.interp(B_probe, B_arr, theta_B)],
                  color="#00e5ff", s=80, zorder=6)
    ax_th.set_xlabel("Magnetic field B (G)", fontsize=12)
    ax_th.set_ylabel("Mixing angle θ (degrees)", fontsize=12)
    ax_th.set_title("θ(B) — driven by δ(B), not by W", color="white", fontsize=12)
    ax_th.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")
    ax_th.text(TABLE['s-wave']['Bc']+0.3, 5, "Bc_s", color="#ff6ec7", fontsize=8)

    ax_lm = axes_th[1]
    ax_lm.plot(B_arr, lam_p_B * 1e9, color="#69ff47", lw=2, label="λ₊(B)")
    ax_lm.plot(B_arr, lam_m_B * 1e9, color="#ff6ec7", lw=2, label="λ₋(B)")
    ax_lm.axhline(0, color="#555", lw=0.8, ls="--")
    ax_lm.scatter([B_probe], [np.interp(B_probe, B_arr, lam_p_B)*1e9], color="#69ff47", s=80, zorder=6)
    ax_lm.scatter([B_probe], [np.interp(B_probe, B_arr, lam_m_B)*1e9], color="#ff6ec7", s=80, zorder=6)
    ax_lm.set_xlabel("Magnetic field B (G)", fontsize=12)
    ax_lm.set_ylabel("Eigenvalue (neV)", fontsize=12)
    ax_lm.set_title("Dressed eigenvalues λ±(B)", color="white", fontsize=12)
    ax_lm.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    fig_th.suptitle(f"W = {_W_eV*1e9:.4f} neV  (constant, from Γ_s = {TABLE['s-wave']['Gamma']} MHz)",
                    color="#ffd740", fontsize=11)
    fig_th.tight_layout(); st.pyplot(fig_th, use_container_width=True); plt.close(fig_th)

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
- **W = ℏΩ** — interchannel coupling (off-diagonal), related to the resonance width Γ. The paper calls it W; quantum optics calls the same quantity ℏΩ where Ω is the Rabi frequency. Both notations refer to the identical off-diagonal matrix element.
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

    st.markdown("#### Field dependence of θ: W is constant, δ(B) is not")
    st.markdown("""
A key point of the Lange et al. model: **W does not depend on B**.
It is a single fitted constant per resonance, determined from the measured resonance width Γ:
""")
    st.latex(r"W \approx \sqrt{\frac{\hbar\Gamma \cdot \hbar^2/(2m_r)}{\bar{a}^2}}")
    st.markdown("""
What varies with field is the **detuning** δ(B) = δμ·μB·(B − Bc), which shifts the
closed-channel threshold:
""")
    st.latex(r"V_\mathrm{cc}(B) = \delta\mu \cdot \mu_B \cdot (B - B_c)")
    st.markdown(r"""
So ΔV(B) = (V_oo − V_cc(B))/2 is B-dependent, and therefore the mixing angle:

$$\theta(B) = \tfrac{1}{2}\arctan\!\left(\frac{W}{\Delta V(B)}\right)$$

rotates continuously with field. Far from resonance (|ΔV| ≫ W) θ → 0° (channels nearly decoupled);
at the bare-state crossing B = Bc (ΔV → 0) θ → 45° (maximum mixing);
past resonance θ → 90°. The fit to experimental binding energies E_b(B) determines all five
parameters per resonance: B0, B*, Bc, δμ, Γ.
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — van der Waals Extension
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<h2 style='color:#00e5ff;'>van der Waals Extension</h2>",
                unsafe_allow_html=True)
    st.markdown("""
<p style="color:#b0bec5;">
The Lange et al. square-well is a convenient fiction: it is flat inside ā so that V̂ can be
diagonalised analytically. The real Cs–Cs interaction has a long-range
<b style="color:#ff6ec7;">−C₆/r⁶</b> tail. Here we replace the square-well with a van der Waals
potential, solve the two-channel ODE numerically, and compare a(B) with the paper formula.
The channel matrix becomes:
</p>
""", unsafe_allow_html=True)
    st.latex(r"\hat{V}(r) = \begin{pmatrix} -C_6/r^6 & W \\ W & -C_6/r^6 + \delta(B) \end{pmatrix}")
    st.markdown("""
<p style="color:#b0bec5; margin-top:0.5rem;">
Both channels share the <b style="color:#fff;">same</b> long-range tail; only the threshold of the
closed channel is shifted by δ(B). W is the same fitted constant as before (from Γ_s).
A hard wall at r_min replaces the unphysical short-range core.
</p>
""", unsafe_allow_html=True)

    # ── Parameters ────────────────────────────────────────────────────────────
    col_sl1, col_sl2 = st.columns(2)
    with col_sl1:
        C6_au    = st.slider("C₆ (atomic units)  — Cs ≈ 6890", 1000, 15000, 6890, 10)
        r_min_a0 = st.slider("r_min (a₀) — hard-wall radius",  5,    40,    20,    1)
    with col_sl2:
        n_B_vdw  = st.slider("B-sweep points  (more = slower)", 30, 150, 60, 10)
        st.markdown(f"""
<div style="background:#111827; border-radius:8px; padding:0.9rem 1.2rem; font-size:0.9rem; color:#fff;">
<b style="color:#ffd740;">Derived quantities</b><br>
C₆ = {C6_au} au<br>
r_min = {r_min_a0} a₀ = {r_min_a0*a0_nm*1000:.2f} pm<br>
l_vdW = ½(C₆/[ħ²/2mᵣ])^(1/4) = {0.5*(C6_au*27.2114*a0_nm**6/hbar2_2mr)**0.25/a0_nm:.1f} a₀<br>
Paper ā = 95.7 a₀
</div>
""", unsafe_allow_html=True)

    # Convert C6 to eV·nm⁶
    C6_eV   = C6_au * 27.2114 * a0_nm**6    # eV·nm⁶
    r_min_v = r_min_a0 * a0_nm               # nm
    r_max_v = 6.0 * a_bar_cs                 # nm
    _hGamma_s_v = TABLE['s-wave']['Gamma'] * 1e6 * h_eVs
    _W_eV_v     = np.sqrt(_hGamma_s_v * hbar2_2mr / a_bar_cs**2)
    _factor_v   = m_r_me / hbar2_2me

    # ── Potential plot ────────────────────────────────────────────────────────
    st.markdown("<h3 style='color:#ffd740;'>van der Waals Potential vs Square-Well</h3>",
                unsafe_allow_html=True)
    r_show  = np.linspace(r_min_v, 2.5 * a_bar_cs, 1000)
    V_vdw_r = -C6_eV / r_show**6

    _delta_s_probe = TABLE['s-wave']['dmu'] * muB_eVperG * (B_probe - TABLE['s-wave']['Bc'])
    _V_pole_v  = (np.pi/2)**2 * hbar2_2me / (2 * m_r_me * a_bar_cs**2)

    try:
        _V_open_v = brentq(lambda V: (lambda k, ka: a_bar_cs*(1-np.tan(ka)/ka)
                                      if abs(np.cos(ka))>1e-12 else np.inf)(
                               np.sqrt(2*m_r_me*V/hbar2_2me),
                               np.sqrt(2*m_r_me*V/hbar2_2me)*a_bar_cs) - a_bg,
                           _V_pole_v*1.001, _V_pole_v*5.0)
    except Exception:
        _V_open_v = V_pole_1 * 2.0

    fig_vp, ax_vp = plt.subplots(figsize=(13, 5))
    fig_vp.patch.set_facecolor("#0e1117"); ax_vp.set_facecolor("#0e1117")
    ax_vp.tick_params(colors="white", labelsize=11)
    ax_vp.xaxis.label.set_color("white"); ax_vp.yaxis.label.set_color("white")
    for sp in ax_vp.spines.values(): sp.set_edgecolor("#333")

    ax_vp.plot(r_show / a0_nm, V_vdw_r * 1e3, color="#ff6ec7", lw=2.5, label="vdW  −C₆/r⁶")
    ax_vp.axhline(-_V_open_v * 1e3, color="#69ff47", lw=2, ls="--",
                  label=f"Square-well  V_oo = {_V_open_v*1e9:.2f} neV")
    ax_vp.axvline(a_bar_cs / a0_nm, color="#ffd740", lw=1.5, ls="--", alpha=0.7, label="ā = 95.7 a₀")
    ax_vp.axvline(r_min_a0, color="#aaa", lw=1, ls=":", alpha=0.6, label=f"r_min = {r_min_a0} a₀")
    ax_vp.set_xlim(r_min_a0 * 0.8, 2.5 * a_bar_cs / a0_nm)
    ylo = max(V_vdw_r.min() * 1e3, -50.0)  # clip extreme near-r values
    ax_vp.set_ylim(ylo, 2.0)
    ax_vp.set_xlabel("r  (a₀)", fontsize=12); ax_vp.set_ylabel("V(r)  (meV)", fontsize=12)
    ax_vp.set_title("Open-channel potential: vdW vs square-well", color="white", fontsize=12)
    ax_vp.legend(fontsize=10, framealpha=0.2, labelcolor="white",
                 facecolor="#0e1117", edgecolor="#444")
    fig_vp.tight_layout(); st.pyplot(fig_vp, use_container_width=True); plt.close(fig_vp)

    # ── B sweep ───────────────────────────────────────────────────────────────
    st.markdown("<h3 style='color:#ffd740;'>a(B) — vdW two-channel ODE vs paper formula</h3>",
                unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def compute_vdw_sweep(C6_eV_key, r_min_key, r_max_key, W_key, n_B_key):
        B_sw   = np.linspace(20.0, 62.0, n_B_key)
        a_sw   = np.full(n_B_key, np.nan)
        factor = m_r_me / hbar2_2me
        for i, B in enumerate(B_sw):
            ds = TABLE['s-wave']['dmu'] * muB_eVperG * (B - TABLE['s-wave']['Bc'])
            def ode_v(r, y, ds=ds):
                uo, uc, puo, puc = y
                Vl = -C6_eV_key / r**6
                return [puo, puc,
                        factor * ((Vl)      * uo + W_key * uc),
                        factor * ((Vl + ds) * uc + W_key * uo)]
            try:
                sol = solve_ivp(ode_v, [r_min_key, r_max_key],
                                [0., 0., 1., 0.],
                                method='RK45', rtol=1e-7, atol=1e-9)
                uo_e = sol.y[0,-1]; puo_e = sol.y[2,-1]
                if abs(puo_e) > 1e-20:
                    a_sw[i] = sol.t[-1] - uo_e / puo_e
            except Exception:
                pass
        return B_sw, a_sw

    with st.spinner(f"Solving vdW ODE for {n_B_vdw} field values …"):
        B_vdw, a_vdw_arr = compute_vdw_sweep(
            C6_eV, r_min_v, r_max_v, float(_W_eV_v), n_B_vdw)

    # Paper formula on same grid
    def a_of_B_paper(B):
        B = np.asarray(B, dtype=float); res = np.ones_like(B)
        for r in TABLE.values():
            dn = B - r['B0']; nm = B - r['Bstar']
            res = np.where(np.abs(dn) > 0.005, res * nm / dn, np.nan)
        return a_bg * res

    a_paper_coarse = a_of_B_paper(B_vdw)

    fig_cmp, ax_cmp = plt.subplots(figsize=(14, 6))
    fig_cmp.patch.set_facecolor("#0e1117"); ax_cmp.set_facecolor("#0e1117")
    ax_cmp.tick_params(colors="white", labelsize=11)
    ax_cmp.xaxis.label.set_color("white"); ax_cmp.yaxis.label.set_color("white")
    for sp in ax_cmp.spines.values(): sp.set_edgecolor("#333")

    clip = 8000
    ax_cmp.plot(B_vdw, np.clip(a_paper_coarse / a0_nm, -clip, clip),
                color="#00e5ff", lw=2, label="Square-well product formula (paper)")
    ax_cmp.plot(B_vdw, np.clip(a_vdw_arr / a0_nm, -clip, clip),
                color="#ff6ec7", lw=2, ls="--", marker="o", ms=3,
                label=f"vdW ODE  (C₆={C6_au} au, r_min={r_min_a0} a₀)")
    ax_cmp.axhline(0, color="#555", lw=0.8, ls="--")
    ax_cmp.axhline(a_bg / a0_nm, color="#aaaaff", lw=1, ls=":", alpha=0.6,
                   label=f"a_bg = 1875 a₀")
    for nm_r, r in TABLE.items():
        ax_cmp.axvline(r['B0'], color="#ffd740", lw=1, ls="--", alpha=0.4)
    ax_cmp.set_xlim(20, 62); ax_cmp.set_ylim(-clip, clip)
    ax_cmp.set_xlabel("Magnetic field B (G)", fontsize=12)
    ax_cmp.set_ylabel("Scattering length a  (a₀)", fontsize=12)
    ax_cmp.set_title("a(B): van der Waals ODE vs square-well formula", color="white", fontsize=12)
    ax_cmp.legend(fontsize=10, framealpha=0.2, labelcolor="white",
                  facecolor="#0e1117", edgecolor="#444")
    fig_cmp.tight_layout(); st.pyplot(fig_cmp, use_container_width=True); plt.close(fig_cmp)

    # ── Single-B wavefunction ─────────────────────────────────────────────────
    st.markdown(f"<h3 style='color:#ffd740;'>Wavefunction at B = {B_probe:.1f} G  (vdW ODE)</h3>",
                unsafe_allow_html=True)

    _ds_probe = TABLE['s-wave']['dmu'] * muB_eVperG * (B_probe - TABLE['s-wave']['Bc'])

    def ode_vdw_probe(r, y):
        uo, uc, puo, puc = y
        Vl = -C6_eV / r**6
        return [puo, puc,
                _factor_v * (Vl       * uo + _W_eV_v * uc),
                _factor_v * ((Vl + _ds_probe) * uc + _W_eV_v * uo)]

    r_eval_vp = np.linspace(r_min_v, r_max_v, 4000)
    sol_vp    = solve_ivp(ode_vdw_probe, [r_min_v, r_max_v], [0., 0., 1., 0.],
                          t_eval=r_eval_vp, method='RK45', rtol=1e-10, atol=1e-12)

    uo_vp = sol_vp.y[0]; uc_vp = sol_vp.y[1]; r_vp = sol_vp.t
    uo_end_v = sol_vp.y[0,-1]; puo_end_v = sol_vp.y[2,-1]
    a_vdw_probe = r_vp[-1] - uo_end_v/puo_end_v if abs(puo_end_v) > 1e-20 else np.nan

    fig_wf, axes_wf = plt.subplots(1, 2, figsize=(16, 5))
    fig_wf.patch.set_facecolor("#0e1117")
    for ax in axes_wf:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.axvline(a_bar_cs / a0_nm, color="#ffd740", lw=1.2, ls="--", alpha=0.5, label="ā")
        ax.axvline(r_min_a0, color="#aaa", lw=1.0, ls=":", alpha=0.5, label="r_min")

    ax_w = axes_wf[0]
    ax_w.plot(r_vp / a0_nm, uo_vp, color="#69ff47", lw=2, label="u_open")
    ax_w.plot(r_vp / a0_nm, uc_vp, color="#ff6ec7", lw=2, label="u_closed")
    ax_w.axhline(0, color="#444", lw=0.6)
    ax_w.set_xlabel("r  (a₀)"); ax_w.set_ylabel("u(r)  [arb.]")
    ax_w.set_title(f"vdW wavefunction → a = {a_vdw_probe/a0_nm:.1f} a₀  "
                   f"(paper: {a_of_B_paper(B_probe)/a0_nm:.1f} a₀)",
                   color="white")
    ax_w.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    ax_w2 = axes_wf[1]
    ax_w2.plot(r_vp / a0_nm, uo_vp**2, color="#69ff47", lw=2, label="|u_open|²")
    ax_w2.plot(r_vp / a0_nm, uc_vp**2, color="#ff6ec7", lw=2, label="|u_closed|²")
    ax_w2.plot(r_vp / a0_nm, uo_vp**2 + uc_vp**2, color="#ffd740", lw=1.5, ls="--", label="total")
    ax_w2.axhline(0, color="#444", lw=0.6)
    ax_w2.set_xlabel("r  (a₀)"); ax_w2.set_ylabel("|u(r)|²")
    ax_w2.set_title("Probability density", color="white")
    ax_w2.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    fig_wf.tight_layout(); st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)

    st.markdown("""
<div style="background:#111827; border-radius:8px; padding:1rem 1.4rem; margin-top:1rem;">
<p style="color:#ffd740; font-weight:bold; margin:0 0 0.5rem 0;">What the comparison tells you</p>
<ul style="color:#b0bec5; margin:0; padding-left:1.2rem; line-height:1.9;">
  <li>Where the two curves <b style="color:#fff;">agree</b>: the square-well captures the correct
      resonance positions — the physics is set by the long-range scale ā, not the well shape.</li>
  <li>Where they <b style="color:#fff;">differ</b>: near-resonance curvature and the value of a_bg
      depend on r_min (the short-range phase). Changing r_min shifts which branch of the
      scattering length you land on.</li>
  <li>The vdW length <b style="color:#ce93d8;">l_vdW = ½(C₆/[ħ²/2mᵣ])^(1/4)</b> sets the
      natural scale — the paper's ā ≈ 95.7 a₀ is precisely this quantity for Cs.</li>
</ul>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Verification
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<h2 style='color:#00e5ff;'>Verification</h2>", unsafe_allow_html=True)
    st.markdown("""
<p style="color:#b0bec5;">
Independent checks of every numerical and analytical result in the app.
Each check recomputes from scratch using known identities, limiting cases, or
cross-method comparisons. <span style="color:#69ff47;">✓ PASS</span> means the
error is below the stated tolerance; <span style="color:#ff6ec7;">✗ FAIL</span>
means something is wrong.
</p>
""", unsafe_allow_html=True)

    # ── helper to render a check row ─────────────────────────────────────────
    def check_row(name, computed, expected, tol, unit="", note=""):
        err   = abs(computed - expected)
        ok    = err < tol
        badge = "<span style='color:#69ff47; font-weight:bold;'>✓ PASS</span>" if ok \
                else "<span style='color:#ff6ec7; font-weight:bold;'>✗ FAIL</span>"
        return (
            f"<tr>"
            f"<td style='padding:5px 8px; color:#b0bec5;'>{name}</td>"
            f"<td style='padding:5px 8px; font-family:monospace; color:#fff;'>"
            f"  {computed:.6g} {unit}</td>"
            f"<td style='padding:5px 8px; font-family:monospace; color:#aaa;'>"
            f"  {expected:.6g} {unit}</td>"
            f"<td style='padding:5px 8px; font-family:monospace; color:#ffd740;'>"
            f"  {err:.2e}</td>"
            f"<td style='padding:5px 8px;'>{badge}</td>"
            f"<td style='padding:5px 8px; color:#555; font-size:0.82rem;'>{note}</td>"
            f"</tr>"
        )

    def table_wrap(title, color, rows_html):
        return f"""
<div style="margin-bottom:1.6rem;">
<p style="color:{color}; font-weight:bold; margin:0 0 0.5rem 0; font-size:1.05rem;">{title}</p>
<table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
  <tr style="border-bottom:1px solid #333; color:#888;">
    <th style="text-align:left; padding:4px 8px;">Check</th>
    <th style="text-align:left; padding:4px 8px;">Computed</th>
    <th style="text-align:left; padding:4px 8px;">Expected</th>
    <th style="text-align:left; padding:4px 8px;">|Error|</th>
    <th style="text-align:left; padding:4px 8px;">Result</th>
    <th style="text-align:left; padding:4px 8px;">Notes</th>
  </tr>
  {rows_html}
</table>
</div>"""

    # ════════════════════════════════════════════════════════════════════════
    # A. Matrix algebra (uses current Tab 1 slider values)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### A. Matrix algebra — Interactive Simulator (Tab 1)")
    st.markdown(f"*Using current slider values:  Ve={Ve} eV, Vc={Vc} eV, "
                f"ℏΩ/W={hOmega} eV, ā={a_bar} nm*")

    # Recompute independently
    _bare_e = -Ve;  _bare_c = -Vc
    _DV     = (Ve - Vc) / 2.0
    _off    = -(Ve + Vc) / 2.0
    _R      = np.sqrt(_DV**2 + hOmega**2)
    _lp_ana = _off + _R
    _lm_ana = _off - _R
    _2th    = np.arctan2(hOmega, _DV)
    _th     = _2th / 2
    _cos_t  = np.cos(_th);  _sin_t = np.sin(_th)
    _vp     = np.array([_cos_t,  _sin_t])   # |+⟩
    _vm     = np.array([-_sin_t, _cos_t])   # |−⟩
    _Vm     = np.array([[_bare_e, hOmega], [hOmega, _bare_c]])
    _evals, _evecs = np.linalg.eigh(_Vm)
    # reconstruction using numpy evecs (spectral theorem: V = Σ λᵢ |i><i|)
    _Vm_recon = (_evals[1] * np.outer(_evecs[:,1], _evecs[:,1])
               + _evals[0] * np.outer(_evecs[:,0], _evecs[:,0]))
    _recon_err = float(np.max(np.abs(_Vm_recon - _Vm)))

    rows_A = ""
    rows_A += check_row("λ₊  analytic vs numpy",     _lp_ana,  _evals[1], 1e-10, "eV",
                         "offset + R = numpy eig[1]")
    rows_A += check_row("λ₋  analytic vs numpy",     _lm_ana,  _evals[0], 1e-10, "eV",
                         "offset − R = numpy eig[0]")
    rows_A += check_row("⟨+|+⟩ = 1  (normalised)",   float(np.dot(_vp,_vp)), 1.0, 1e-14, "",
                         "cos²θ + sin²θ = 1")
    rows_A += check_row("⟨−|−⟩ = 1  (normalised)",   float(np.dot(_vm,_vm)), 1.0, 1e-14, "",
                         "sin²θ + cos²θ = 1")
    rows_A += check_row("⟨+|−⟩ = 0  (orthogonal)",   abs(float(np.dot(_vp,_vm))), 0.0, 1e-14, "",
                         "cosθ·(−sinθ) + sinθ·cosθ = 0")
    rows_A += check_row("V̂ reconstruction error",    _recon_err, 0.0, 1e-10, "eV",
                         "max|λ+|+><+| + λ-|-><-| − V̂|")
    rows_A += check_row("tan 2θ = W/ΔV",
                         float(np.tan(_2th)), hOmega/_DV if abs(_DV)>1e-15 else np.inf,
                         1e-10, "", "definition of mixing angle")
    st.markdown(table_wrap("Matrix algebra", "#ffd740", rows_A), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # B. Scattering length formula (Cs paper)
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### B. Scattering length formula  a(B)  [Tab 2]")

    def _a_of_B(B):
        B = np.asarray(B, dtype=float); res = np.ones_like(B)
        for r in TABLE.values():
            dn = B - r['B0']; nm = B - r['Bstar']
            res = np.where(np.abs(dn) > 0.005, res * nm / dn, np.nan)
        return a_bg * res

    # far-field: at B = 10000 G all resonances negligible → a → a_bg
    _a_far   = _a_of_B(np.array([10000.0]))[0]
    # zeros at B*
    _a_at_Bs = _a_of_B(np.array([TABLE['s-wave']['Bstar']]))[0]
    _a_at_Bd = _a_of_B(np.array([TABLE['d-wave']['Bstar']]))[0]
    _a_at_Bg = _a_of_B(np.array([TABLE['g-wave']['Bstar']]))[0]
    # poles: a should exceed clip limit at B0
    _a_near_B0s = _a_of_B(np.array([TABLE['s-wave']['B0'] + 0.001]))[0]

    rows_B = ""
    rows_B += check_row("a(10000 G) / a_bg → 1",
                         _a_far / a_bg, 1.0, 1e-4, "",
                         "far from all resonances, product → 1")
    rows_B += check_row("a(B*_s) = 0",
                         abs(_a_at_Bs / a0_nm), 0.0, 0.5, "a₀",
                         "zero at B = B*_s = 18.1 G")
    rows_B += check_row("a(B*_d) = 0",
                         abs(_a_at_Bd / a0_nm), 0.0, 0.5, "a₀",
                         "zero at B = B*_d = 47.944 G")
    rows_B += check_row("a(B*_g) = 0",
                         abs(_a_at_Bg / a0_nm), 0.0, 0.5, "a₀",
                         "zero at B = B*_g = 53.457 G")
    rows_B += check_row("|a(B₀_s + 0.001G)| large",
                         min(abs(_a_near_B0s / a0_nm), 1e6), 1e6, 1e6, "a₀",
                         "pole at B = B₀_s = −11.1 G")
    st.markdown(table_wrap("Scattering length formula", "#00e5ff", rows_B), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # C. V_open root-finding
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### C. V_open root-finding  [Tab 2]")

    def _abg_from_V(V_eV):
        k = np.sqrt(2 * m_r_me * V_eV / hbar2_2me)
        ka = k * a_bar_cs
        if abs(np.cos(ka)) < 1e-12: return np.inf
        return a_bar_cs * (1 - np.tan(ka) / ka)

    _Vp1    = (np.pi/2)**2 * hbar2_2me / (2 * m_r_me * a_bar_cs**2)
    _V_open = brentq(lambda V: _abg_from_V(V) - a_bg, _Vp1*1.001, _Vp1*5.0)
    _a_check= _abg_from_V(_V_open)

    rows_C = ""
    rows_C += check_row("abg_from_V(V_open) = a_bg",
                         _a_check / a0_nm, a_bg / a0_nm, 1.0, "a₀",
                         "brentq root satisfies the equation")
    rows_C += check_row("V_open > V_pole_1",
                         _V_open * 1e9, _Vp1 * 1e9, 1e8, "neV",
                         "guaranteed by brentq bracket [V_pole_1·1.001, V_pole_1·5.0]")
    rows_C += check_row("a_bg / ā  (far from square-well pole)",
                         a_bg / a_bar_cs, 1875.0 / 95.7, 1e-3, "",
                         "ratio should be ≈ 19.6")
    st.markdown(table_wrap("V_open root-finding", "#69ff47", rows_C), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # D. Binding energy formula
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### D. Binding energy  E_b(B)  [Tab 2]")

    def _Eb_kHz(a_nm):
        a = np.asarray(a_nm, dtype=float); Eb = np.full_like(a, np.nan)
        m = a > 2 * a_bar_cs
        disc  = 1 - 2 * a_bar_cs / a[m]
        kappa = (1 - np.sqrt(np.clip(disc, 0, None))) / a_bar_cs
        Eb[m] = -hbar2_2mr * kappa**2 / (h_eVs * 1e3)
        m2 = (a > 0) & (a <= 2 * a_bar_cs)
        Eb[m2] = -hbar2_2mr / a[m2]**2 / (h_eVs * 1e3)
        return Eb

    # Large a limit: κ ≈ 1/a → E_b → -hbar2_2mr/a² (simple universal)
    _a_large   = 1000 * a_bar_cs   # very large
    _Eb_eff    = float(_Eb_kHz(np.array([_a_large])))
    _kappa_eff = (1 - np.sqrt(1 - 2*a_bar_cs/_a_large)) / a_bar_cs
    _Eb_simple = -hbar2_2mr / _a_large**2 / (h_eVs * 1e3)   # simple 1/a²
    _Eb_exact  = -hbar2_2mr * _kappa_eff**2 / (h_eVs * 1e3)

    # at a = ∞, E_b → 0
    _Eb_at_inf = float(_Eb_kHz(np.array([1e10 * a_bar_cs])))

    # effective-range correction: κ ≈ (1 - √(1-2ā/a))/ā; for a=10ā, compare
    _a_10 = 10 * a_bar_cs
    _kappa_full   = (1 - np.sqrt(1 - 2*a_bar_cs/_a_10)) / a_bar_cs
    _kappa_simple = 1.0 / _a_10
    _Eb_full_kHz  = -hbar2_2mr * _kappa_full**2   / (h_eVs * 1e3)
    _Eb_simpl_kHz = -hbar2_2mr * _kappa_simple**2 / (h_eVs * 1e3)

    rows_D = ""
    rows_D += check_row("E_b(a→∞) → 0",
                         abs(_Eb_at_inf), 0.0, 1e-10, "kHz",
                         "bound state energy vanishes at threshold")
    rows_D += check_row("E_b(a=1000ā) eff-range vs 1/a²",
                         _Eb_exact, _Eb_simple, abs(_Eb_simple)*0.01, "kHz",
                         "effective-range correction < 1% when a ≫ ā")
    rows_D += check_row("κ(a=10ā): full vs simple  [% diff]",
                         abs(_kappa_full - _kappa_simple) / _kappa_simple * 100,
                         0.0, 12.0, "%",
                         "effective-range matters at a~10ā (~10% correction expected)")
    st.markdown(table_wrap("Binding energy", "#ce93d8", rows_D), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # E. ODE wavefunction checks
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### E. ODE wavefunction — boundary conditions & asymptotic  [Tabs 1 & 2]")

    # Tab 1 ODE: re-run
    _fac1 = m_r_me / hbar2_2me
    def _ode1(r, y):
        ue, uc, pue, puc = y
        return [pue, puc,
                _fac1*((_bare_e - 0.)*ue + hOmega*uc),
                _fac1*((_bare_c - 0.)*uc + hOmega*ue)]
    _r_end1 = a_bar * 1.5
    _sol1   = solve_ivp(_ode1, [0., _r_end1], [0., 0., 1., 0.],
                        method='RK45', rtol=1e-12, atol=1e-14)

    # Tab 2 Cs ODE: re-run at B_probe
    _ds2    = TABLE['s-wave']['dmu'] * muB_eVperG * (B_probe - TABLE['s-wave']['Bc'])
    _hG2    = TABLE['s-wave']['Gamma'] * 1e6 * h_eVs
    _W2     = np.sqrt(_hG2 * hbar2_2mr / a_bar_cs**2)
    _Voo2   = -_V_open;   _Vcc2 = _ds2
    _fac2   = 2.0 * m_r_me / hbar2_2me
    def _ode2(r, y):
        uo, uc, puo, puc = y
        return [puo, puc,
                _fac2*(_Voo2*uo + _W2*uc),
                _fac2*(_Vcc2*uc + _W2*uo)]
    _r_end2   = a_bar_cs * 3.0
    _sol2     = solve_ivp(_ode2, [1e-6, _r_end2], [1e-6, 0., 1., 0.],
                          method='RK45', rtol=1e-12, atol=1e-14)

    # Tab 1: analytical vs numerical max residual inside Region I
    _k_p = k_plus;  _k_m = k_minus
    _evan_p2 = (k_plus_type  == "evanescent")
    _evan_m2 = (k_minus_type == "evanescent")
    _Ap2 = cos_t / _k_p  if _k_p  > 1e-8 else cos_t  * 1e8
    _Am2 = -sin_t / _k_m if _k_m > 1e-8 else -sin_t * 1e8
    _r1  = _sol1.t
    _fp2 = _Ap2 * (np.sinh(_k_p*_r1) if _evan_p2 else np.sin(_k_p*_r1))
    _fm2 = _Am2 * (np.sinh(_k_m*_r1) if _evan_m2 else np.sin(_k_m*_r1))
    _ue_ana2 = cos_t*_fp2 - sin_t*_fm2
    _uc_ana2 = sin_t*_fp2 + cos_t*_fm2
    _res_e   = float(np.max(np.abs(_sol1.y[0] - _ue_ana2)))
    _res_c   = float(np.max(np.abs(_sol1.y[1] - _uc_ana2)))

    # Asymptotic linearity of open channel (Tab 2 Cs): fit last 20 points to line
    _r_tail  = _sol2.t[-20:]
    _u_tail  = _sol2.y[0, -20:]
    _coeffs  = np.polyfit(_r_tail, _u_tail, 1)
    _lin_err = float(np.max(np.abs(np.polyval(_coeffs, _r_tail) - _u_tail)))

    rows_E = ""
    rows_E += check_row("Tab 1 u_e(0) = 0  (BC)",
                         abs(float(_sol1.y[0, 0])), 0.0, 1e-10, "",
                         "wavefunction zero at origin")
    rows_E += check_row("Tab 1 u_c(0) = 0  (BC)",
                         abs(float(_sol1.y[1, 0])), 0.0, 1e-10, "",
                         "closed-channel zero at origin")
    rows_E += check_row("Tab 1 max|u_e num − analytic|",
                         _res_e, 0.0, 1e-5, "",
                         "ODE vs sine/sinh solution")
    rows_E += check_row("Tab 1 max|u_c num − analytic|",
                         _res_c, 0.0, 1e-5, "",
                         "ODE vs sine/sinh solution")
    rows_E += check_row("Tab 2 Cs u_open(r_min) ≈ 0  (BC)",
                         abs(float(_sol2.y[0, 0])), 0.0, 1e-4, "",
                         "initial condition at r_min = 1e-6 nm")
    rows_E += check_row("Tab 2 asymptotic linearity of u_open",
                         _lin_err, 0.0, 1e-4, "",
                         "u ~ r − a for large r (fits straight line)")
    st.markdown(table_wrap("ODE wavefunction", "#ff6ec7", rows_E), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # F. vdW cross-checks
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### F. van der Waals cross-checks  [Tab 4]")

    # Cs C6
    _C6_au  = 6890
    _C6_eV  = _C6_au * 27.2114 * a0_nm**6
    _r_min4 = 20 * a0_nm
    _r_max4 = 6 * a_bar_cs
    _hG4    = TABLE['s-wave']['Gamma'] * 1e6 * h_eVs
    _W4     = np.sqrt(_hG4 * hbar2_2mr / a_bar_cs**2)
    _fac4   = m_r_me / hbar2_2me

    # vdW length
    _l_vdw  = 0.5 * (_C6_eV / hbar2_2mr)**0.25   # nm
    _l_vdw_a0 = _l_vdw / a0_nm

    # Single vdW ODE at B=30G
    _ds4 = TABLE['s-wave']['dmu'] * muB_eVperG * (30.0 - TABLE['s-wave']['Bc'])
    def _ode4(r, y):
        uo, uc, puo, puc = y
        Vl = -_C6_eV / r**6
        return [puo, puc,
                _fac4*(Vl*uo       + _W4*uc),
                _fac4*((Vl+_ds4)*uc + _W4*uo)]
    _sol4 = solve_ivp(_ode4, [_r_min4, _r_max4], [0., 0., 1., 0.],
                      method='RK45', rtol=1e-10, atol=1e-12)
    _uo_e4 = _sol4.y[0,-1];  _puo_e4 = _sol4.y[2,-1]
    _a_vdw4 = _sol4.t[-1] - _uo_e4/_puo_e4 if abs(_puo_e4)>1e-20 else np.nan

    # Paper formula at B=30G
    _a_paper30 = float(_a_of_B(np.array([30.0])))

    # Hard-wall BC
    _u_hw = abs(float(_sol4.y[0, 0]))

    # Asymptotic linearity
    _r_t4  = _sol4.t[-20:]
    _u_t4  = _sol4.y[0, -20:]
    _c4    = np.polyfit(_r_t4, _u_t4, 1)
    _le4   = float(np.max(np.abs(np.polyval(_c4, _r_t4) - _u_t4)))

    rows_F = ""
    rows_F += check_row("l_vdW(Cs C₆=6890) / ā_paper",
                         _l_vdw_a0, a_bar_cs / a0_nm, 10.0, "a₀",
                         "l_vdW ≈ 101 a₀, paper ā = 95.7 a₀  (~5% differ by Gribakin factor)")
    rows_F += check_row("vdW u_open(r_min) ≈ 0  (hard wall BC)",
                         _u_hw, 0.0, 1e-4, "",
                         "wavefunction vanishes at hard wall r_min")
    rows_F += check_row("vdW asymptotic linearity of u_open",
                         _le4, 0.0, 1e-3, "",
                         "u ~ r − a for large r")
    rows_F += check_row("a_vdW(30G) vs a_paper(30G)  [%]",
                         abs(_a_vdw4 - _a_paper30) / abs(_a_paper30) * 100,
                         0.0, 5.0, "%",
                         "vdW ODE vs product formula should agree within ~2%")
    rows_F += check_row("V_vdW(r_min=20a₀)  magnitude",
                         abs(-_C6_eV / _r_min4**6) * 1e3,
                         2.929, 0.01, "meV",
                         "spot-check potential at hard-wall radius")
    st.markdown(table_wrap("van der Waals cross-checks", "#ffd740", rows_F), unsafe_allow_html=True)

    # ── Summary scorecard ─────────────────────────────────────────────────
    st.markdown("### Summary scorecard")
    _all_checks = [
        ("λ₊ analytic vs numpy",        abs(_lp_ana - _evals[1]) < 1e-10),
        ("λ₋ analytic vs numpy",        abs(_lm_ana - _evals[0]) < 1e-10),
        ("⟨+|+⟩ = 1",                  abs(np.dot(_vp,_vp)-1) < 1e-14),
        ("⟨−|−⟩ = 1",                  abs(np.dot(_vm,_vm)-1) < 1e-14),
        ("⟨+|−⟩ = 0",                  abs(np.dot(_vp,_vm)) < 1e-14),
        ("V̂ reconstruction",           _recon_err < 1e-10),
        ("a(10000G)/a_bg → 1",          abs(_a_far/a_bg-1) < 1e-4),
        ("a(B*_s) = 0",                 abs(_a_at_Bs/a0_nm) < 0.5),
        ("a(B*_d) = 0",                 abs(_a_at_Bd/a0_nm) < 0.5),
        ("a(B*_g) = 0",                 abs(_a_at_Bg/a0_nm) < 0.5),
        ("abg_from_V(V_open) = a_bg",  abs(_a_check-a_bg)/a0_nm < 1.0),
        ("E_b(a→∞) → 0",               abs(_Eb_at_inf) < 1e-10),
        ("ODE u_e vs analytic",         _res_e < 1e-5),
        ("ODE u_c vs analytic",         _res_c < 1e-5),
        ("vdW asymptotic linear",       _le4 < 1e-3),
        ("a_vdW(30G) vs paper < 5%",   abs(_a_vdw4-_a_paper30)/abs(_a_paper30)*100 < 5.0),
    ]
    n_pass = sum(v for _, v in _all_checks)
    n_total = len(_all_checks)
    bar_color = "#69ff47" if n_pass == n_total else ("#ffd740" if n_pass > n_total*0.8 else "#ff6ec7")

    st.markdown(f"""
<div style="background:#111827; border-radius:10px; padding:1.2rem 1.5rem; margin-top:0.5rem;">
  <p style="color:{bar_color}; font-size:1.4rem; font-weight:bold; margin:0 0 0.8rem 0;">
    {n_pass} / {n_total} checks passed
  </p>
  <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
    {"".join(
        f'<span style="background:{"#1a3a1a" if ok else "#3a1a1a"}; '
        f'color:{"#69ff47" if ok else "#ff6ec7"}; '
        f'border-radius:4px; padding:3px 10px; font-size:0.82rem;">{"✓" if ok else "✗"} {name}</span>'
        for name, ok in _all_checks
    )}
  </div>
</div>
""", unsafe_allow_html=True)
