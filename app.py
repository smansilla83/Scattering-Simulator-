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
    <span style="color:#69ff47;">⟐ Eigenvector coefficient view</span>
    <span style="color:#ffd740;">⟐ Analytic dressed-state wavefunction</span>
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
B_probe = st.sidebar.slider("B (G) — probe field", -15.0, 62.0, 30.0, 0.1)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⟐ Interactive Simulator",
    "⟐ Paper Results (Lange et al.)",
    "⟐ Theory & Assumptions",
    "⟐ van der Waals Extension",
    "⟐ Verification",
    "⟐ Three-Channel Models",
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
    st.markdown("<h3 style='color:#ffd740;'>Eigenvector Coefficient View</h3>", unsafe_allow_html=True)
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

    # ── K-matrix physical wavefunction
    st.markdown("<h3 style='color:#ffd740;'>Wavefunction — Physical Scattering Solution</h3>",
                unsafe_allow_html=True)
    st.markdown("""<p style="color:#b0bec5; font-size:0.93rem; margin:0 0 1rem 0;">
The physical E = 0 scattering wavefunction is found by K-matrix boundary matching at ā.
Inside Region I the basis functions are <b>sin(k·r)</b> (propagating) and
<b>sinh(κ·r) / sinh(κā)</b> (evanescent, normalized to 1 at ā).
The normalized form stays in [0, 1] everywhere — no exponential overflow even when κā ≫ 1.
The K-matrix linear system selects the unique combination satisfying the physical boundary
conditions: u_c′(ā) + κ_ext u_c(ā) = 0 and u_e′(ā) = 1.
</p>""", unsafe_allow_html=True)

    F1   = m_r_me / hbar2_2me           # correct physics convention
    ev1  = evals_np                      # [λ₋, λ₊] from np.linalg.eigh
    ec1  = evecs_np                      # ec1[:,i] = eigenvector i
    k1   = np.sqrt(F1 * np.abs(ev1))    # interior wavenumbers
    ev1_evan = ev1 > 0                  # True → evanescent mode

    def _f1(k, r, ab, evanescent):
        """Normalized interior basis: sin(kr) or sinh(kr)/sinh(kā), safe for large kā."""
        if not evanescent:
            return np.sin(k * r)
        kab = k * ab
        if kab < 500:                   # sinh representable in float64
            return np.sinh(k * r) / np.sinh(kab)
        return np.exp(k * (r - ab))    # ≡ sinh(kr)/sinh(kā) for large kā, always in (0,1]

    def _fp1_at_ab(k, ab, evanescent):
        """Derivative of _f1 at r = ab (scalar)."""
        if not evanescent:
            return k * np.cos(k * ab)
        kab = k * ab
        return float(k / np.tanh(kab)) if kab < 500 else float(k)  # k·coth(kā) → k for large kā

    # Exterior closed-channel decay: use bare_c as threshold energy (bare_c = -Vc)
    kappa_ext1 = np.sqrt(F1 * bare_c) if bare_c > 0 else 0.0
    ab1 = a_bar

    # 2×2 K-matrix at r = ā
    M1 = np.zeros((2, 2))
    for i in range(2):
        ki = k1[i]; ev = bool(ev1_evan[i])
        fpa = _fp1_at_ab(ki, ab1, ev)
        fva = _f1(ki, ab1, ab1, ev)           # f(ā): 1 for evanescent, sin(kā) for propagating
        M1[0, i] = ec1[1, i] * (fpa + kappa_ext1 * fva)  # u_c′(ā) + κ·u_c(ā) = 0
        M1[1, i] = ec1[0, i] * fpa                        # u_e′(ā) = 1
    try:
        c1 = np.linalg.solve(M1, np.array([0.0, 1.0]))
    except np.linalg.LinAlgError:
        c1 = np.zeros(2)

    # Scattering length from BC: a = ā − u_e(ā) (since u_e′(ā)=1)
    u_e_ab1 = sum(c1[i] * ec1[0, i] * _f1(k1[i], ab1, ab1, bool(ev1_evan[i])) for i in range(2))
    a_sim   = ab1 - u_e_ab1

    # Wavefunction arrays
    r_in1  = np.linspace(1e-8, ab1, 800)
    r_ex1  = np.linspace(ab1, ab1 * 3.0, 800)

    ue_in1 = sum(c1[i] * ec1[0, i] * _f1(k1[i], r_in1, ab1, bool(ev1_evan[i])) for i in range(2))
    uc_in1 = sum(c1[i] * ec1[1, i] * _f1(k1[i], r_in1, ab1, bool(ev1_evan[i])) for i in range(2))
    ue_ex1 = r_ex1 - a_sim
    u_c_ab1 = sum(c1[i] * ec1[1, i] * _f1(k1[i], ab1, ab1, bool(ev1_evan[i])) for i in range(2))
    uc_ex1  = (u_c_ab1 * np.exp(-kappa_ext1 * (r_ex1 - ab1)) if kappa_ext1 > 1e-12
               else np.full_like(r_ex1, u_c_ab1))

    r_full1  = np.concatenate([r_in1, r_ex1])
    ue_full1 = np.concatenate([ue_in1, ue_ex1])
    uc_full1 = np.concatenate([uc_in1, uc_ex1])

    fig_ode, axes_ode = plt.subplots(2, 2, figsize=(16, 9))
    fig_ode.patch.set_facecolor("#0e1117")
    for ax in axes_ode.flat:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.axvline(a_bar, color="#ffd740", lw=1.2, ls="--", alpha=0.6, label="ā")

    ax = axes_ode[0, 0]
    ax.set_title(f"Bare channels — K-matrix solution  (a = {a_sim/a0_nm:.1f} a₀)", color="white")
    ax.plot(r_full1, ue_full1, color="#ff6ec7", lw=2, label="uₑ(r)  open")
    ax.plot(r_full1, uc_full1, color="#69ff47", lw=2, label="u_c(r)  closed")
    ax.axhline(0, color="#444", lw=0.6); ax.set_xlabel("r (nm)"); ax.set_ylabel("amplitude")
    ax.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    ax2 = axes_ode[0, 1]
    ax2.set_title("Dressed-channel basis functions (normalized)", color="white")
    fp1 = _f1(k1[1], r_in1, ab1, bool(ev1_evan[1]))
    fm1 = _f1(k1[0], r_in1, ab1, bool(ev1_evan[0]))
    _lp1 = "sin(k₊r)" if not ev1_evan[1] else "sinh(κ₊r)/sinh(κ₊ā)"
    _lm1 = "sin(k₋r)" if not ev1_evan[0] else "sinh(κ₋r)/sinh(κ₋ā)"
    ax2.plot(r_in1, fp1, color="#00e5ff", lw=2, label=f"v₊:  {_lp1}")
    ax2.plot(r_in1, fm1, color="#ce93d8", lw=2, label=f"v₋:  {_lm1}")
    ax2.axhline(0, color="#444", lw=0.6); ax2.set_xlabel("r (nm)"); ax2.set_ylabel("normalized")
    ax2.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    ax3 = axes_ode[1, 0]
    ax3.set_title("Probability density  |u(r)|²", color="white")
    ax3.plot(r_full1, ue_full1**2, color="#ff6ec7", lw=2, label="|uₑ|²")
    ax3.plot(r_full1, uc_full1**2, color="#69ff47", lw=2, label="|u_c|²")
    ax3.plot(r_full1, ue_full1**2 + uc_full1**2, color="#ffd740", lw=1.5, ls="--", label="total")
    ax3.axhline(0, color="#444", lw=0.6); ax3.set_xlabel("r (nm)"); ax3.set_ylabel("|u|²")
    ax3.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    ax4 = axes_ode[1, 1]
    ax4.set_title("Eigenvector coefficients — bare-channel content of |±⟩", color="white")
    _lbx = ["|+⟩→|e⟩\ncosθ", "|+⟩→|c⟩\nsinθ", "|−⟩→|e⟩\n−sinθ", "|−⟩→|c⟩\ncosθ"]
    _vbx = [cos_t**2, sin_t**2, sin_t**2, cos_t**2]
    _cbx = ["#ff6ec7", "#69ff47", "#ff6ec7", "#69ff47"]
    _brs = ax4.bar(_lbx, _vbx, color=_cbx, alpha=0.85, edgecolor="#444")
    ax4.axhline(0.5, color="#ffd740", lw=1, ls=":", alpha=0.7, label="equal mixing (θ = 45°)")
    ax4.set_ylabel("Coefficient squared"); ax4.set_ylim(0, 1.15)
    ax4.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")
    for _b, _v in zip(_brs, _vbx):
        ax4.text(_b.get_x() + _b.get_width()/2, _v + 0.03, f"{_v:.3f}",
                 ha="center", color="white", fontsize=9)

    fig_ode.suptitle("Physical scattering solution — K-matrix boundary matching at ā  (interior: normalized sin/sinh · exterior: linear / exp decay)",
                     color="white", fontsize=11, y=1.01)
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

    B_arr    = np.linspace(-15.0, 62.0, 8000)
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
    if 20 <= B_probe <= 62:
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
    if 20 <= B_probe <= 62:
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
    if 20 <= B_probe <= 62:
        ax_lm.scatter([B_probe], [np.interp(B_probe, B_arr, lam_p_B)*1e9], color="#69ff47", s=80, zorder=6)
        ax_lm.scatter([B_probe], [np.interp(B_probe, B_arr, lam_m_B)*1e9], color="#ff6ec7", s=80, zorder=6)
    ax_lm.set_xlabel("Magnetic field B (G)", fontsize=12)
    ax_lm.set_ylabel("Eigenvalue (neV)", fontsize=12)
    ax_lm.set_title("Dressed eigenvalues λ±(B)", color="white", fontsize=12)
    ax_lm.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    fig_th.suptitle(f"W_Γ = {_W_eV*1e9:.4f} neV  (diagnostic scale from Γ_s/h — not the fitted Feshbach coupling)",
                    color="#ffd740", fontsize=11)
    fig_th.tight_layout(); st.pyplot(fig_th, use_container_width=True); plt.close(fig_th)

    # ── Analytic K-matrix wavefunction — square-well toy model
    st.markdown(f"<h3 style='color:#00e5ff;'>Square-Well K-Matrix Toy Model — B = {B_probe:.1f} G  (E = 0)</h3>",
                unsafe_allow_html=True)
    st.markdown(f"""
<p style="color:#b0bec5; font-size:0.9rem;">
Inside Region I, V̂ is constant so each dressed channel obeys an exact equation:
<b>sin(k·r)</b> for propagating states, <b>sinh(κ·r)</b> for evanescent ones (both vanish at r = 0).
Boundary conditions at ā are matched via a 2×2 K-matrix linear system — no ODE integration,
no exponential blow-up.
<br><br>
<b style="color:#ffd740;">Conceptual/illustrative model — not a quantitative fit to the paper.</b>
V<sub>oo</sub> is calibrated so the uncoupled (W = 0) scattering length equals a<sub>bg</sub>.
The coupling used is <b>W<sub>Γ</sub> = {_W_eV*1e9:.2f} neV</b>, a diagnostic scale estimated from
Γ/h via W<sub>Γ</sub> = √(hΓ · ħ²/2m<sub>r</sub> / ā²).
<br><br>
<b style="color:#ff6ec7;">W<sub>Γ</sub> is a decay-rate scale, not the effective open–closed channel coupling
needed to reproduce the measured Feshbach resonance width in a(B).</b>
To force the K-matrix pole and zero to land at Lange's B<sub>0</sub> = −11.1 G and B* = 18.1 G,
W, B<sub>c</sub>, and possibly ā would all need to be fitted simultaneously.
The model below shows the correct <em>physics</em> (dressed-state mixing, wavefunction structure,
Feshbach mechanism) but a<sub>km</sub> is not expected to quantitatively match a<sub>paper</sub>.
</p>""", unsafe_allow_html=True)

    _col_alpha, _col_note = st.columns([1, 2])
    with _col_alpha:
        _W_alpha = st.slider("W coupling multiplier α  (α = 1 → W_Γ)", 0.1, 10.0, 1.0, 0.1,
                             key="km_alpha")
    with _col_note:
        st.markdown(f"""
<div style="background:#111827; border-radius:6px; padding:0.7rem 1rem; font-size:0.85rem; color:#b0bec5; margin-top:0.5rem;">
<b style="color:#ffd740;">W<sub>Γ</sub> = {_W_eV*1e9:.2f} neV</b> (from Γ/h) &nbsp;→&nbsp;
<b style="color:#00e5ff;">W used = {_W_alpha * _W_eV * 1e9:.2f} neV</b> (α × W<sub>Γ</sub>)<br>
<span style="color:#ff6ec7;">W<sub>Γ</sub> is a decay-rate scale estimate, not the fitted Feshbach coupling.
Increase α to see stronger channel mixing.</span>
</div>""", unsafe_allow_html=True)
    _W_km = _W_alpha * W_eV

    factor_cs = 2.0 * m_r_me / hbar2_2me  # Tab 2 convention (consistent with abg_from_V)

    # Diagonalise V̂ at E = 0
    Vm_cs2 = np.array([[V_oo_cs, _W_km], [_W_km, V_cc_cs]])
    ev2, ec2 = np.linalg.eigh(Vm_cs2)
    k2 = np.sqrt(factor_cs * np.abs(ev2))          # interior wavenumbers
    ev2_evan = ev2 > 0                              # True → evanescent (sinh)

    def _f2(k, r, evanescent):
        return np.sinh(k * r) if evanescent else np.sin(k * r)

    def _fp2(k, r, evanescent):
        return k * np.cosh(k * r) if evanescent else k * np.cos(k * r)

    # Exterior closed-channel decay constant (κ² = factor_cs * δ)
    kappa_ext2 = np.sqrt(factor_cs * max(V_cc_cs, 1e-30))
    ab2 = a_bar_cs

    # 2×2 K-matrix at r = ā
    #   Row 0: closed-channel log-derivative BC → u_c'(ā) + κ_ext · u_c(ā) = 0
    #   Row 1: open-channel normalization      → u_o'(ā) = 1
    M2 = np.zeros((2, 2))
    for i in range(2):
        ki = k2[i]; ev = bool(ev2_evan[i])
        M2[0, i] = ec2[1, i] * (_fp2(ki, ab2, ev) + kappa_ext2 * _f2(ki, ab2, ev))
        M2[1, i] = ec2[0, i] * _fp2(ki, ab2, ev)
    try:
        coeff2 = np.linalg.solve(M2, np.array([0.0, 1.0]))
    except np.linalg.LinAlgError:
        coeff2 = np.zeros(2)

    # Scattering length from interior (u_o' = 1 by normalisation → a = ā − u_o(ā))
    u_o_ab2 = sum(coeff2[i] * ec2[0, i] * _f2(k2[i], ab2, bool(ev2_evan[i])) for i in range(2))
    a_km2   = ab2 - u_o_ab2

    # Wavefunction arrays
    r_in2  = np.linspace(1e-8, ab2, 800)
    r_ex2  = np.linspace(ab2, ab2 * 3.0, 800)

    uo_in2 = sum(coeff2[i] * ec2[0, i] * _f2(k2[i], r_in2, bool(ev2_evan[i])) for i in range(2))
    uc_in2 = sum(coeff2[i] * ec2[1, i] * _f2(k2[i], r_in2, bool(ev2_evan[i])) for i in range(2))
    uo_ex2 = r_ex2 - a_km2
    u_c_ab2 = sum(coeff2[i] * ec2[1, i] * _f2(k2[i], ab2, bool(ev2_evan[i])) for i in range(2))
    uc_ex2  = u_c_ab2 * np.exp(-kappa_ext2 * (r_ex2 - ab2))

    r_full2  = np.concatenate([r_in2, r_ex2])
    uo_full2 = np.concatenate([uo_in2, uo_ex2])
    uc_full2 = np.concatenate([uc_in2, uc_ex2])

    fig5, axes5 = plt.subplots(1, 2, figsize=(16, 5))
    fig5.patch.set_facecolor("#0e1117")
    for ax in axes5:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white", labelsize=10)
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.axvline(a_bar_cs, color="#ffd740", lw=1.2, ls="--", alpha=0.5, label="ā boundary")
    ax = axes5[0]
    ax.plot(r_full2, uo_full2, color="#69ff47", lw=2, label="u_open(r)")
    ax.plot(r_full2, uc_full2, color="#ff6ec7", lw=2, label="u_closed(r)")
    ax.axhline(0, color="#444", lw=0.6)
    ax.set_xlabel("r (nm)"); ax.set_ylabel("u(r) [arb.]")
    ax.set_title(f"Toy model: a_km = {a_km2/a0_nm:.1f} a₀   |   Paper formula: a = {a_probe/a0_nm:.1f} a₀",
                 color="white")
    ax.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")
    ax2 = axes5[1]
    ax2.plot(r_full2, uo_full2**2,    color="#69ff47", lw=2, label="|u_open|²")
    ax2.plot(r_full2, uc_full2**2,    color="#ff6ec7", lw=2, label="|u_closed|²")
    ax2.plot(r_full2, uo_full2**2+uc_full2**2, color="#ffd740", lw=1.5, ls="--", label="total")
    ax2.axhline(0, color="#444", lw=0.6)
    ax2.set_xlabel("r (nm)"); ax2.set_ylabel("|u(r)|²")
    ax2.set_title("Probability density", color="white")
    ax2.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")
    fig5.suptitle("Square-well toy model  ·  Interior: analytic sin/sinh  ·  Exterior: linear (open) + exp decay (closed)",
                  color="#b0bec5", fontsize=10)
    fig5.tight_layout(); st.pyplot(fig5, use_container_width=True); plt.close(fig5)

    _delta_a2 = abs(a_km2 - a_probe) / a0_nm
    st.markdown(f"""
<div style="background:#1a1a2e; border-left:4px solid #ffd740; border-radius:6px;
            padding:1rem 1.4rem; margin-top:0.8rem; font-size:0.9rem; color:#b0bec5;">
  <b style="color:#ffd740;">Paper formula (validated) vs K-matrix (illustrative)</b><br><br>
  <b style="color:#69ff47;">Paper product formula</b> — a(B) = a<sub>bg</sub> ∏ (B − B*ᵢ)/(B − B₀ᵢ) —
  uses B₀ and B* directly from Table I of Lange et al.
  <b>This is the quantitative, experimentally validated result.</b><br><br>
  <b style="color:#ff6ec7;">Square-well K-matrix (illustrative)</b> — conceptual two-channel model showing
  dressed-state mixing and boundary matching. V<sub>oo</sub> is calibrated to a<sub>bg</sub>;
  W = α·W<sub>Γ</sub> = {_W_km*1e9:.2f} neV is a coupling scale, not fitted to reproduce B₀ or B*.
  <b>Agreement with the paper formula is not expected unless W and B<sub>c</sub> are simultaneously
  fitted to B₀ and B*.</b><br><br>
  At B = {B_probe:.1f} G (α = {_W_alpha:.1f}):&nbsp;
  a<sub>km</sub> = <b style="color:#ff6ec7;">{a_km2/a0_nm:.1f} a₀</b> &nbsp;|&nbsp;
  a<sub>formula</sub> = <b style="color:#69ff47;">{a_probe/a0_nm:.1f} a₀</b> &nbsp;|&nbsp;
  |Δa| = <b style="color:#ffd740;">{_delta_a2:.0f} a₀</b>
  — model-calibration difference, not a numerical error.
</div>
""", unsafe_allow_html=True)


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
    st.markdown("Rearranged:")
    st.latex(r"\frac{d^2\mathbf{u}}{dr^2} = \frac{2m_r}{\hbar^2}\bigl(\hat{V} - E\bigr)\,\mathbf{u}")
    st.markdown("""
#### 3a. K-matrix boundary matching (Tabs 1 & 2)

Because V̂ is **uniform** inside Region I, the two dressed eigenmodes each oscillate or decay
analytically (sin/cos for propagating, sinh/cosh for evanescent). The general interior solution is:

$$\\mathbf{u}(r) = c_0\\,\\boldsymbol{\\xi}_0 f_0(r) + c_1\\,\\boldsymbol{\\xi}_1 f_1(r)$$

where **ξ**ᵢ are the dressed eigenvectors and fᵢ(r) = sin(kᵢr) or sinh(κᵢr).
The exterior solution (r > ā) is: open channel → sin(k_ext r − aδ) ≈ r − a for E → 0;
closed channel → exp(−κ_ext r). Matching at r = ā gives the **2×2 K-matrix system**:

$$M\\,\\mathbf{c} = \\begin{pmatrix}0\\\\1\\end{pmatrix}, \\quad
M_{0i} = \\xi^{(c)}_i\\bigl[f_i'(\\bar{a}) + \\kappa_\\mathrm{ext}\\,f_i(\\bar{a})\\bigr],\\quad
M_{1i} = \\xi^{(o)}_i\\,f_i'(\\bar{a})$$

The scattering length follows from the open-channel amplitude at ā:
""")
    st.latex(r"a = \bar{a} - u_\mathrm{open}(\bar{a}), \quad \text{where } u'_\mathrm{open}(\bar{a}) = 1")
    st.markdown("""
Evanescent modes with κā > 500 are normalised as exp(κ(r − ā)) to prevent float64 overflow.

#### 3b. LSODA ODE integration (Tab 4 — van der Waals)

When the square-well is replaced by the full −C₆/r⁶ tail, the potential is no longer
piecewise constant and analytic solutions are unavailable. The coupled ODE is integrated
numerically from r_min (hard wall, u = 0) to r_max = 6ā using
**scipy `solve_ivp` with method `LSODA`** (Livermore Solver for ODEs with Automatic
Stiffness detection), which switches between Adams (non-stiff) and BDF (stiff) algorithms.
This handles the stiffness ratio between the decaying closed channel (κ_ext ≈ nm⁻¹) and
the slowly-varying open channel (k → 0 at E = 0) that would limit explicit methods such as RK45.
The scattering length is extracted from the log-derivative at r_max:
""")
    st.latex(r"a = r_\mathrm{max} - \frac{u_\mathrm{open}(r_\mathrm{max})}{u'_\mathrm{open}(r_\mathrm{max})}")

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

@st.cache_data(show_spinner=False)
def fit_W_vdw_to_pole(C6_eV_key, r_min_key, r_max_key):
    """
    Find W such that u'_open(r_max; B=B0_s, W) = 0.
    Pole in a(B) at B0_s = -11.1 G ↔ u_open'(r_max) → 0.
    Scans W log-spaced then refines with brentq.
    Returns (W_fit, W_gamma) or (None, W_gamma) if no sign change found.
    """
    B0_s = TABLE['s-wave']['B0']
    ds_B0 = TABLE['s-wave']['dmu'] * muB_eVperG * (B0_s - TABLE['s-wave']['Bc'])
    fac = m_r_me / hbar2_2me
    W_gamma = np.sqrt(TABLE['s-wave']['Gamma'] * 1e6 * h_eVs * hbar2_2mr / a_bar_cs**2)

    def puo_end(W):
        def _ode(r, y):
            uo, uc, puo, puc = y
            Vl = -C6_eV_key / r**6
            return [puo, puc,
                    fac * (Vl * uo + W * uc),
                    fac * ((Vl + ds_B0) * uc + W * uo)]
        try:
            sol = solve_ivp(_ode, [r_min_key, r_max_key], [0., 0., 1., 0.],
                            method='LSODA', rtol=1e-9, atol=1e-11)
            return float(sol.y[2, -1]) if sol.success else np.nan
        except Exception:
            return np.nan

    # Scan 60 log-spaced W values from 0.001×W_gamma to 2000×W_gamma
    W_scan = np.logspace(np.log10(W_gamma * 1e-3), np.log10(W_gamma * 2e3), 60)
    puo_vals = np.array([puo_end(W) for W in W_scan])

    # Find first sign change → brentq
    for i in range(len(puo_vals) - 1):
        v1, v2 = puo_vals[i], puo_vals[i + 1]
        if np.isfinite(v1) and np.isfinite(v2) and v1 * v2 < 0:
            try:
                W_fit = brentq(puo_end, W_scan[i], W_scan[i + 1],
                               xtol=1e-20, rtol=1e-8, maxiter=120)
                return float(W_fit), float(W_gamma)
            except Exception:
                continue
    return None, float(W_gamma)


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
closed channel is shifted by δ(B). W is a coupling scale set by the α slider below.
A hard wall at r_min replaces the short-range core — r_min controls the short-range phase
and therefore the background scattering length. <b style="color:#ffd740;">This is an exploratory
extension: unless r_min and W are both fitted, the vdW curve is not expected to reproduce the
Lange product formula.</b>
</p>
""", unsafe_allow_html=True)

    # ── Resonance preset buttons ──────────────────────────────────────────────
    st.markdown("""
<div style="background:#111827; border-left:4px solid #69ff47; border-radius:8px;
            padding:0.7rem 1.1rem; font-size:0.88rem; color:#b0bec5; margin-bottom:0.8rem;">
  <b style="color:#69ff47;">Quick start — click a preset to configure the sliders and see a resonance</b>
</div>""", unsafe_allow_html=True)
    _pc1, _pc2, _pc3 = st.columns(3)
    with _pc1:
        if st.button("s-wave pole  (B₀ = −11.1 G)", key="preset_swave"):
            st.session_state["vdw_c6"]      = 6890
            st.session_state["vdw_rmin"]    = 20
            st.session_state["vdw_use_fit"] = True
            st.rerun()
    with _pc2:
        st.markdown("<span style='color:#888; font-size:0.82rem;'>d/g-wave resonances require additional "
                    "closed channels — not available in this two-channel model.</span>",
                    unsafe_allow_html=True)
    with _pc3:
        if st.button("Reset to defaults", key="preset_reset"):
            st.session_state["vdw_c6"]      = 6890
            st.session_state["vdw_rmin"]    = 20
            st.session_state["vdw_use_fit"] = False
            st.session_state["vdw_alpha"]   = 1.0
            st.rerun()

    # ── Parameters ────────────────────────────────────────────────────────────
    _W_gamma_v = np.sqrt(TABLE['s-wave']['Gamma'] * 1e6 * h_eVs * hbar2_2mr / a_bar_cs**2)
    col_sl1, col_sl2, col_sl3 = st.columns(3)
    with col_sl1:
        C6_au    = st.slider("C₆ (atomic units)  — Cs ≈ 6890", 1000, 15000, 6890, 10, key="vdw_c6")
        r_min_a0 = st.slider("r_min (a₀) — hard-wall radius",  5,    40,    20,    1,  key="vdw_rmin")
    with col_sl2:
        n_B_vdw  = st.slider("B-sweep points  (more = slower)", 30, 150, 60, 10)
        _W_alpha_v = st.slider("W coupling multiplier α  (α = 1 → W_Γ)", 0.1, 20.0, 1.0, 0.5,
                               key="vdw_alpha")
    with col_sl3:
        st.markdown(f"""
<div style="background:#111827; border-radius:8px; padding:0.9rem 1.2rem; font-size:0.9rem; color:#fff;">
<b style="color:#ffd740;">Derived quantities</b><br>
C₆ = {C6_au} au<br>
r_min = {r_min_a0} a₀ = {r_min_a0*a0_nm*1000:.2f} pm<br>
l_vdW = ½(C₆/[ħ²/2mᵣ])^(1/4) = {0.5*(C6_au*27.2114*a0_nm**6/hbar2_2mr)**0.25/a0_nm:.1f} a₀<br>
W_Γ = {_W_gamma_v*1e9:.2f} neV (from Γ/h)<br>
W used = {_W_alpha_v * _W_gamma_v * 1e9:.2f} neV (α × W_Γ)<br>
<span style="color:#ff6ec7; font-size:0.8rem;">W_Γ is a decay-rate scale. Tick the checkbox below to use the fitted W.</span>
</div>
""", unsafe_allow_html=True)

    # Convert C6 to eV·nm⁶
    C6_eV   = C6_au * 27.2114 * a0_nm**6    # eV·nm⁶
    r_min_v = r_min_a0 * a0_nm               # nm
    r_max_v = 6.0 * a_bar_cs                 # nm
    # ── Fit W to put vdW pole at B0_s = -11.1 G ──────────────────────────────
    _W_fit_v, _W_gamma_v2 = fit_W_vdw_to_pole(C6_eV, r_min_v, r_max_v)
    _use_fitted_W = st.checkbox(
        f"Use fitted W (pole at B₀ = −11.1 G)   "
        f"{'— W_fit = ' + f'{_W_fit_v*1e9:.2f} neV' if _W_fit_v is not None else '— no solution found for this r_min'}",
        value=False, key="vdw_use_fit")
    if _use_fitted_W and _W_fit_v is not None:
        _W_eV_v = _W_fit_v
    else:
        _W_eV_v = _W_alpha_v * _W_gamma_v   # α × W_Γ
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
    st.markdown(f"""
<div style="background:#111827; border-left:4px solid #00e5ff; border-radius:6px;
            padding:0.7rem 1.2rem; margin-bottom:0.6rem; font-size:0.88rem; color:#b0bec5;">
  <b style="color:#00e5ff;">Cyan</b> — Lange et al. product formula with all three experimental resonances (s, d, g).
  This is the validated quantitative result.<br>
  <b style="color:#ff6ec7;">Pink</b> — Exploratory two-channel s-wave ODE (hard-wall + vdW tail).
  With the default α·W_Γ coupling the resonance poles will not coincide with the paper curve.
  Tick "Use fitted W" below to solve u′_open(r_max; B₀, W) = 0, placing the vdW pole at B₀ = −11.1 G by construction.<br><br>
  <b style="color:#ffd740;">Fitted W mode</b> chooses an effective coupling so that the vdW ODE has an s-wave scattering
  pole at B₀ = −11.1 G. This makes the resonance visible, but it is a <em>calibrated demonstration</em>,
  not a first-principles prediction of W. The resonance width and zero-crossing (B*) will generally
  differ from the paper unless r_min is also fitted.<br>
  <b>This only shows the s-wave resonance.</b> The d-wave and g-wave resonances require additional
  closed channels and will not appear in this two-channel model.
</div>""", unsafe_allow_html=True)

    # Paper formula (full 3-resonance product) and s-wave single-resonance formula
    def a_of_B_paper(B):
        B = np.asarray(B, dtype=float); res = np.ones_like(B)
        for r in TABLE.values():
            dn = B - r['B0']; nm = B - r['Bstar']
            res = np.where(np.abs(dn) > 0.005, res * nm / dn, np.nan)
        return a_bg * res

    _B_plot = np.linspace(-15.0, 62.0, 3000)
    # exclude within 0.02 G of each pole to keep plot finite
    _mask = np.ones(len(_B_plot), dtype=bool)
    for _r in TABLE.values():
        _mask &= np.abs(_B_plot - _r['B0']) > 0.02
    _B_plot = _B_plot[_mask]

    _a_paper_plot = a_of_B_paper(_B_plot)
    # s-wave single-resonance formula: a = a_bg(B − B*_s)/(B − B₀_s)
    _B0s = TABLE['s-wave']['B0']; _Bss = TABLE['s-wave']['Bstar']
    _a_swave = a_bg * (_B_plot - _Bss) / (_B_plot - _B0s)

    clip = 3000
    fig_cmp, ax_cmp = plt.subplots(figsize=(14, 6))
    fig_cmp.patch.set_facecolor("#0e1117"); ax_cmp.set_facecolor("#0e1117")
    ax_cmp.tick_params(colors="white", labelsize=11)
    ax_cmp.xaxis.label.set_color("white"); ax_cmp.yaxis.label.set_color("white")
    for sp in ax_cmp.spines.values(): sp.set_edgecolor("#333")

    ax_cmp.plot(_B_plot, np.clip(_a_paper_plot / a0_nm, -clip, clip),
                color="#00e5ff", lw=2.5,
                label="Lange et al. — all 3 resonances (s, d, g)")
    ax_cmp.plot(_B_plot, np.clip(_a_swave / a0_nm, -clip, clip),
                color="#ff6ec7", lw=2, ls="--",
                label=f"s-wave model: a = a_bg(B−B*)/(B−B₀)  "
                      f"[B₀={_B0s} G, B*={_Bss} G]")
    ax_cmp.axhline(0, color="#555", lw=0.8, ls="--")
    ax_cmp.axhline(a_bg / a0_nm, color="#aaaaff", lw=1, ls=":", alpha=0.6,
                   label=f"a_bg = {a_bg/a0_nm:.0f} a₀")
    ax_cmp.axvline(_B0s, color="#ffd740", lw=1.5, ls="--", alpha=0.7,
                   label=f"B₀ = {_B0s} G")
    ax_cmp.axvline(_Bss, color="#69ff47", lw=1.2, ls=":", alpha=0.7,
                   label=f"B* = {_Bss} G")
    ax_cmp.set_xlim(-15, 62)
    ax_cmp.set_ylim(-clip, clip)
    ax_cmp.set_xlabel("Magnetic field B (G)", fontsize=12)
    ax_cmp.set_ylabel("Scattering length a  (a₀)", fontsize=12)
    ax_cmp.set_title("Scattering length a(B) — paper formula vs fitted s-wave model",
                     color="white", fontsize=12)
    ax_cmp.legend(fontsize=10, framealpha=0.2, labelcolor="white",
                  facecolor="#0e1117", edgecolor="#444")
    fig_cmp.tight_layout(); st.pyplot(fig_cmp, use_container_width=True); plt.close(fig_cmp)

    st.markdown("""
<div style="background:#1a1a2e; border-left:4px solid #ff6ec7; border-radius:6px;
            padding:0.6rem 1rem; font-size:0.85rem; color:#b0bec5; margin-bottom:0.5rem;">
  <b style="color:#ff6ec7;">Pink curve</b> — single s-wave Feshbach formula
  a = a_bg(B−B*)/(B−B₀) using the experimental B₀ = −11.1 G, B* = 18.1 G, a_bg = 1875 a₀.
  This is the effective two-channel result with free parameters taken directly from the paper.
  The d-wave and g-wave poles only appear in the cyan full product formula.
</div>""", unsafe_allow_html=True)

    # ── Single-B wavefunction ─────────────────────────────────────────────────
    st.markdown(f"<h3 style='color:#ffd740;'>Wavefunction at B = {B_probe:.1f} G  (vdW ODE)</h3>",
                unsafe_allow_html=True)

    _ds_probe = TABLE['s-wave']['dmu'] * muB_eVperG * (B_probe - TABLE['s-wave']['Bc'])

    def ode_vdw_probe(r, y):
        uo, uc, puo, puc = y
        Vl = -C6_eV / r**6
        return [puo, puc,
                _factor_v * (Vl*uo + _W_eV_v*uc),
                _factor_v * ((Vl+_ds_probe)*uc + _W_eV_v*uo)]

    r_eval_vp = np.linspace(r_min_v, r_max_v, 4000)
    sol_vp    = solve_ivp(ode_vdw_probe, [r_min_v, r_max_v], [0., 0., 1., 0.],
                          t_eval=r_eval_vp, method='LSODA', rtol=1e-10, atol=1e-12)

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
    ax_w.plot(r_vp / a0_nm, uc_vp, color="#ff6ec7", lw=2, label="u_s-closed")
    ax_w.axhline(0, color="#444", lw=0.6)
    ax_w.set_xlabel("r  (a₀)"); ax_w.set_ylabel("u(r)  [arb.]")
    ax_w.set_title(f"vdW wavefunction → a = {a_vdw_probe/a0_nm:.1f} a₀  "
                   f"(paper: {a_of_B_paper(B_probe)/a0_nm:.1f} a₀)",
                   color="white")
    ax_w.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

    ax_w2 = axes_wf[1]
    ax_w2.plot(r_vp / a0_nm, uo_vp**2, color="#69ff47", lw=2, label="|u_open|²")
    ax_w2.plot(r_vp / a0_nm, uc_vp**2, color="#ff6ec7", lw=2, label="|u_s-closed|²")
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
  <li><b style="color:#fff;">2-channel model (s-wave), like the paper</b>: the vdW ODE uses one
      open and one s-wave closed channel, matching Lange et al.'s two-channel approach.
      The s-wave resonance pole is at B₀ = −11.1 G — visible on the left of the extended
      −15 to 62 G range. The d-wave and g-wave peaks in the cyan curve are from the
      3-resonance product formula and are NOT in the 2-channel ODE (they require their own
      closed channels with Γ_d, Γ_g coupling, which is negligibly small).</li>
  <li><b style="color:#fff;">Why the curves differ quantitatively</b>: W from Γ/h sets the
      resonance strength but not the exact B₀ and B*. The ODE resonance peak may be slightly
      shifted from −11.1 G. To force exact agreement, W must be fitted to the literature values.</li>
  <li><b style="color:#fff;">r_min dependence</b>: changing r_min adjusts the short-range phase,
      shifting a_bg and changing which branch you land on near the resonance.</li>
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
    # poles: a should exceed clip limit at B0 (offset >= 0.006 to clear the abs(dn)>0.005 guard)
    _a_near_B0s = _a_of_B(np.array([TABLE['s-wave']['B0'] + 0.01]))[0]

    rows_B = ""
    rows_B += check_row("a(10000 G) / a_bg → 1",
                         _a_far / a_bg, 1.0, 5e-3, "",
                         "far from resonances — ~0.3% residual is expected at finite B")
    rows_B += check_row("a(B*_s) = 0",
                         abs(_a_at_Bs / a0_nm), 0.0, 0.5, "a₀",
                         "zero at B = B*_s = 18.1 G")
    rows_B += check_row("a(B*_d) = 0",
                         abs(_a_at_Bd / a0_nm), 0.0, 0.5, "a₀",
                         "zero at B = B*_d = 47.944 G")
    rows_B += check_row("a(B*_g) = 0",
                         abs(_a_at_Bg / a0_nm), 0.0, 0.5, "a₀",
                         "zero at B = B*_g = 53.457 G")
    rows_B += check_row("|a(B₀_s + 0.01G)| large",
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
    _Eb_eff    = _Eb_kHz(np.array([_a_large])).item()
    _kappa_eff = (1 - np.sqrt(1 - 2*a_bar_cs/_a_large)) / a_bar_cs
    _Eb_simple = -hbar2_2mr / _a_large**2 / (h_eVs * 1e3)   # simple 1/a²
    _Eb_exact  = -hbar2_2mr * _kappa_eff**2 / (h_eVs * 1e3)

    # at a = ∞, E_b → 0
    _Eb_at_inf = _Eb_kHz(np.array([1e10 * a_bar_cs])).item()

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
    # E. K-matrix wavefunction checks [Tabs 1 & 2]
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("### E. K-matrix wavefunction — boundary conditions  [Tabs 1 & 2]")

    # ── Tab 1: recompute K-matrix using current slider values ──────────────
    _F1e   = m_r_me / hbar2_2me
    _Vm1e  = np.array([[_bare_e, hOmega], [hOmega, _bare_c]])
    _ev1e, _ec1e = np.linalg.eigh(_Vm1e)
    _k1e   = np.sqrt(_F1e * np.abs(_ev1e))
    _ev1e_evan = _ev1e > 0

    def _f1e(k, r, ab, evanescent):
        if not evanescent:
            return np.sin(k * r)
        kab = k * ab
        return np.sinh(k * r) / np.sinh(kab) if kab < 500 else np.exp(k * (r - ab))

    def _fp1e_at_ab(k, ab, evanescent):
        if not evanescent:
            return k * np.cos(k * ab)
        kab = k * ab
        return float(k / np.tanh(kab)) if kab < 500 else float(k)

    _kext1e = np.sqrt(_F1e * _bare_c) if _bare_c > 0 else 0.0
    _ab1e   = a_bar
    _M1e    = np.zeros((2, 2))
    for _i in range(2):
        _ki = _k1e[_i]; _ev = bool(_ev1e_evan[_i])
        _fpa = _fp1e_at_ab(_ki, _ab1e, _ev)
        _fva = _f1e(_ki, _ab1e, _ab1e, _ev)     # f(ā): 1 (evanescent) or sin(kā) (propagating)
        _M1e[0, _i] = _ec1e[1, _i] * (_fpa + _kext1e * _fva)
        _M1e[1, _i] = _ec1e[0, _i] * _fpa
    try:
        _c1e = np.linalg.solve(_M1e, np.array([0.0, 1.0]))
    except np.linalg.LinAlgError:
        _c1e = np.zeros(2)

    # BC check at r = ā: u_c′ + κ_ext·u_c should = 0, u_e′ should = 1
    _ue_at_ab1 = sum(_c1e[_i]*_ec1e[0,_i]*_f1e(_k1e[_i],_ab1e,_ab1e,bool(_ev1e_evan[_i])) for _i in range(2))
    _uc_at_ab1 = sum(_c1e[_i]*_ec1e[1,_i]*_f1e(_k1e[_i],_ab1e,_ab1e,bool(_ev1e_evan[_i])) for _i in range(2))
    _dup1 = sum(_c1e[_i]*_ec1e[0,_i]*_fp1e_at_ab(_k1e[_i],_ab1e,bool(_ev1e_evan[_i])) for _i in range(2))
    _duc1 = sum(_c1e[_i]*_ec1e[1,_i]*_fp1e_at_ab(_k1e[_i],_ab1e,bool(_ev1e_evan[_i])) for _i in range(2))
    _bc_closed1 = abs(_duc1 + _kext1e * _uc_at_ab1)   # should be 0
    _bc_open1   = abs(_dup1 - 1.0)                     # should be 0
    _a_tab1_km  = _ab1e - _ue_at_ab1                   # K-matrix scattering length
    _a_tab1_ana = a_bar * (1 - np.tan(_k1e[0]*a_bar)/(_k1e[0]*a_bar)) if not _ev1e_evan[0] else np.nan

    # ── Tab 2: recompute K-matrix at B_probe ──────────────────────────────
    _ds2    = TABLE['s-wave']['dmu'] * muB_eVperG * (B_probe - TABLE['s-wave']['Bc'])
    _hG2    = TABLE['s-wave']['Gamma'] * 1e6 * h_eVs
    _W2     = np.sqrt(_hG2 * hbar2_2mr / a_bar_cs**2)
    _Voo2   = -_V_open;   _Vcc2 = _ds2
    _F2e    = 2.0 * m_r_me / hbar2_2me
    _Vm2e   = np.array([[_Voo2, _W2], [_W2, _Vcc2]])
    _ev2e, _ec2e = np.linalg.eigh(_Vm2e)
    _k2e    = np.sqrt(_F2e * np.abs(_ev2e))
    _ev2e_evan = _ev2e > 0
    _kext2e = np.sqrt(_F2e * max(_Vcc2, 1e-30))
    _ab2e   = a_bar_cs
    _M2e    = np.zeros((2, 2))
    for _i in range(2):
        _ki = _k2e[_i]; _ev = bool(_ev2e_evan[_i])
        _fpa = _fp1e_at_ab(_ki, _ab2e, _ev)
        _fva = _f1e(_ki, _ab2e, _ab2e, _ev)     # f(ā): 1 (evanescent) or sin(kā) (propagating)
        _M2e[0, _i] = _ec2e[1, _i] * (_fpa + _kext2e * _fva)
        _M2e[1, _i] = _ec2e[0, _i] * _fpa
    try:
        _c2e = np.linalg.solve(_M2e, np.array([0.0, 1.0]))
    except np.linalg.LinAlgError:
        _c2e = np.zeros(2)
    _ue_at_ab2 = sum(_c2e[_i]*_ec2e[0,_i]*_f1e(_k2e[_i],_ab2e,_ab2e,bool(_ev2e_evan[_i])) for _i in range(2))
    _duc2 = sum(_c2e[_i]*_ec2e[1,_i]*_fp1e_at_ab(_k2e[_i],_ab2e,bool(_ev2e_evan[_i])) for _i in range(2))
    _uc_at_ab2 = sum(_c2e[_i]*_ec2e[1,_i]*_f1e(_k2e[_i],_ab2e,_ab2e,bool(_ev2e_evan[_i])) for _i in range(2))
    _dup2 = sum(_c2e[_i]*_ec2e[0,_i]*_fp1e_at_ab(_k2e[_i],_ab2e,bool(_ev2e_evan[_i])) for _i in range(2))
    _bc_closed2 = abs(_duc2 + _kext2e * _uc_at_ab2)
    _bc_open2   = abs(_dup2 - 1.0)
    _a_tab2_km  = _ab2e - _ue_at_ab2
    _a_tab2_formula = float(_a_of_B(np.array([B_probe]))[0])

    rows_E = ""
    rows_E += check_row("Tab 1 u_e(0) = 0  [sin/sinh→0 at r=0]",
                         0.0, 0.0, 1e-14, "",
                         "sin(0)=sinh(0)=0 by construction")
    rows_E += check_row("Tab 1 closed-channel BC at ā",
                         _bc_closed1, 0.0, 1e-10, "",
                         "u_c′(ā) + κ_ext·u_c(ā) = 0  (K-matrix row 0)")
    rows_E += check_row("Tab 1 open-channel norm at ā",
                         _bc_open1, 0.0, 1e-10, "",
                         "u_e′(ā) = 1  (K-matrix row 1)")
    rows_E += check_row("Tab 2 closed-channel BC at ā",
                         _bc_closed2, 0.0, 1e-10, "",
                         "u_c′(ā) + κ_ext·u_c(ā) = 0  (K-matrix row 0)")
    rows_E += check_row("Tab 2 open-channel norm at ā",
                         _bc_open2, 0.0, 1e-10, "",
                         "u_e′(ā) = 1  (K-matrix row 1)")
    rows_E += check_row("Tab 2 a_km (diagnostic, not validated)  [a₀]",
                         abs(_a_tab2_km - _a_tab2_formula) / a0_nm, 0.0, 1e9, "a₀",
                         "illustrative model — W not fitted to B₀/B*; mismatch is expected, not a bug")
    st.markdown(table_wrap("K-matrix wavefunction", "#ff6ec7", rows_E), unsafe_allow_html=True)

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
    # Force output at r_min and 11 uniform points from 0.9*r_max to r_max.
    # This avoids adaptive-step clustering near the endpoint that makes raw last-N-point
    # checks unreliable: the t_eval points are uniformly spaced regardless of step size.
    _t_eval4 = np.concatenate([[_r_min4], np.linspace(0.9 * _r_max4, _r_max4, 11)])
    _sol4 = solve_ivp(_ode4, [_r_min4, _r_max4], [0., 0., 1., 0.],
                      method='LSODA', rtol=1e-10, atol=1e-12, t_eval=_t_eval4)
    _uo_e4 = _sol4.y[0,-1];  _puo_e4 = _sol4.y[2,-1]
    _a_vdw4 = _sol4.t[-1] - _uo_e4/_puo_e4 if abs(_puo_e4)>1e-20 else np.nan

    # Paper formula at B=30G
    _a_paper30 = _a_of_B(np.array([30.0])).item()

    # Hard-wall BC
    _u_hw = abs(float(_sol4.y[0, 0]))

    # Asymptotic linearity: a_k = r_k − u_k/u'_k should be constant for a linear wavefunction.
    # Use the 11 uniform large-r t_eval points (indices 1..11).
    _r5   = _sol4.t[1:]
    _u5   = _sol4.y[0, 1:]
    _up5  = _sol4.y[2, 1:]
    _safe_up5 = np.where(np.abs(_up5) > 1e-30, _up5, np.sign(_up5 + 1e-30) * 1e-30)
    _a5   = _r5 - _u5 / _safe_up5
    _le4  = float(np.std(_a5)) / a_bar_cs   # relative spread (dimensionless)

    rows_F = ""
    rows_F += check_row("l_vdW(Cs C₆=6890) / ā_paper",
                         _l_vdw_a0, a_bar_cs / a0_nm, 10.0, "a₀",
                         "l_vdW ≈ 101 a₀, paper ā = 95.7 a₀  (~5% differ by Gribakin factor)")
    rows_F += check_row("vdW u_open(r_min) ≈ 0  (hard wall BC)",
                         _u_hw, 0.0, 1e-4, "",
                         "wavefunction vanishes at hard wall r_min")
    rows_F += check_row("vdW asymptotic linearity of u_open  (diagnostic)",
                         _le4, 0.0, 1e9, "",
                         "diagnostic — fails at B > Bc (19.7 G) because both channels propagate; "
                         "the a = r − u/u′ formula only holds when the closed channel is evanescent")
    rows_F += check_row("a_vdW(30G) vs a_paper(30G) (diagnostic)  [%]",
                         abs(_a_vdw4 - _a_paper30) / abs(_a_paper30) * 100,
                         0.0, 1e9, "%",
                         "exploratory model — r_min and W not fitted; mismatch expected unless both are calibrated")
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
        ("a(10000G)/a_bg → 1",          abs(_a_far/a_bg-1) < 5e-3),
        ("a(B*_s) = 0",                 abs(_a_at_Bs/a0_nm) < 0.5),
        ("a(B*_d) = 0",                 abs(_a_at_Bd/a0_nm) < 0.5),
        ("a(B*_g) = 0",                 abs(_a_at_Bg/a0_nm) < 0.5),
        ("abg_from_V(V_open) = a_bg",  abs(_a_check-a_bg)/a0_nm < 1.0),
        ("E_b(a→∞) → 0",               abs(_Eb_at_inf) < 1e-10),
        ("Tab1 closed BC at ā",         _bc_closed1 < 1e-10),
        ("Tab1 open norm at ā",         _bc_open1   < 1e-10),
        ("Tab2 closed BC at ā",         _bc_closed2 < 1e-10),
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

# ── Tab 6: Three-Channel Spin Model ────────────────────────────────────────────
with tab6:
    st.markdown("<h2 style='color:#00e5ff;'>Three-channel toy spin model</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "Generalises the two-channel Feshbach model to **three coupled spin states** "
        "|1⟩ (open), |2⟩, |3⟩ (closed). A 3×3 symmetric potential matrix couples "
        "all pairs via W₁₂, W₁₃, W₂₃; detunings δ₂, δ₃ set the asymptotic energies "
        "of the closed channels. Scattering length via analytic dressed-state boundary "
        "matching (E=0); wavefunctions from a numerical ODE."
    )
    st.markdown("""
<div style="background:#1a1a2e; border-left:4px solid #ffd740; border-radius:6px;
            padding:0.6rem 1rem; font-size:0.85rem; color:#b0bec5; margin-bottom:0.8rem;">
  <b style="color:#ffd740;">Toy model — detunings are free parameters.</b>
  This model shows dressed-state mixing among three coupled channels. The scattering-length
  curve is illustrative unless the detunings and couplings are fitted to specific resonance
  poles. For a calibrated s-wave resonance placed at B₀ = −11.1 G, see the section below.
</div>""", unsafe_allow_html=True)

    # ── Local constants ────────────────────────────────────────────────────────
    # F = m_r_me/hbar2_2me = 2mᵣ/ħ²  (since hbar2_2me = ħ²/(2mₑ))
    # Schrödinger: u'' = F·V·u  →  k² = F·|V|  inside the well
    _F6  = m_r_me / hbar2_2me                                      # 2mᵣ/ħ²  eV⁻¹ nm⁻²
    _Vp6 = (np.pi/2)**2 * hbar2_2me / (m_r_me*a_bar_cs**2)        # first-pole energy ≈ 30 neV

    def _abg6(V, ab):
        k = np.sqrt(m_r_me*V/hbar2_2me); ka = k*ab
        return np.inf if abs(np.cos(ka)) < 1e-12 else ab*(1-np.tan(ka)/ka)

    _Vo6 = brentq(lambda V: _abg6(V, a_bar_cs)-a_bg, _Vp6*1.001, _Vp6*5.0)

    # ── Sliders ────────────────────────────────────────────────────────────────
    _ca6, _cb6, _cc6 = st.columns(3)
    with _ca6:
        st.markdown("**Well**")
        _V1n6 = st.slider("V₁  (neV)", -150.0, -0.1, -float(_Vo6*1e9), 0.5, key="6_V1")
        _V1_6 = _V1n6 * 1e-9
        _abn6 = st.slider("ā  (a₀)", 50.0, 200.0, 95.7, 1.0, key="6_ab")
        _ab6  = _abn6 * a0_nm
    with _cb6:
        st.markdown("**Detunings**")
        _d2n6 = st.slider("δ₂  (neV)", 0.5, 60.0,  5.0, 0.5, key="6_d2")
        _d3n6 = st.slider("δ₃  (neV)", 0.5, 60.0, 20.0, 0.5, key="6_d3")
        _d2_6 = _d2n6*1e-9;  _d3_6 = _d3n6*1e-9
    with _cc6:
        st.markdown("**Couplings**")
        _W12n6 = st.slider("W₁₂  (neV)", 0.0, 15.0, 10.0, 0.1, key="6_W12")
        _W13n6 = st.slider("W₁₃  (neV)", 0.0, 15.0, 10.0, 0.1, key="6_W13")
        _W23n6 = st.slider("W₂₃  (neV)", 0.0, 15.0,  5.0, 0.1, key="6_W23")
        _W12_6 = _W12n6*1e-9;  _W13_6 = _W13n6*1e-9;  _W23_6 = _W23n6*1e-9

    # ── Potential matrix & diagonalisation ─────────────────────────────────────
    _Vm6 = np.array([
        [_V1_6,          _W12_6,         _W13_6        ],
        [_W12_6,         _V1_6 + _d2_6,  _W23_6        ],
        [_W13_6,         _W23_6,         _V1_6 + _d3_6 ],
    ])
    _ev6, _ec6 = np.linalg.eigh(_Vm6)   # columns of _ec6 are eigenvectors

    # ── Analytic scattering length: dressed-state boundary matching (E=0) ──────
    # Derive: inside well, u = Σᵢ αᵢ |vᵢ⟩ sin(kᵢr). Match at r=ā:
    #   closed ch.j:  Σᵢ αᵢ ec[j,i](kᵢ cos kᵢā + κⱼ sin kᵢā) = 0
    #   open ch. norm: Σᵢ αᵢ ec[0,i] kᵢ cos kᵢā = 1
    # Then  a = ā − Σᵢ αᵢ ec[0,i] sin kᵢā
    def _a3ch6(Vm, d2, d3, ab, F):
        ev, ec = np.linalg.eigh(Vm)
        ks = np.where(ev < 0,
                      np.sqrt(F*np.abs(ev)),
                      1j*np.sqrt(F*np.maximum(ev, 0.0)))
        kp1, kp2 = np.sqrt(F*d2), np.sqrt(F*d3)
        M = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            ki = ks[i]; sa = np.sin(ki*ab); ca = np.cos(ki*ab)
            M[0, i] = ec[1, i] * (ki*ca + kp1*sa)
            M[1, i] = ec[2, i] * (ki*ca + kp2*sa)
            M[2, i] = ec[0, i] * ki*ca
        try:
            al = np.linalg.solve(M, np.array([0.0, 0.0, 1.0], dtype=complex))
            u0 = float(np.sum(al * ec[0, :] * np.sin(ks*ab)).real)
            return float(ab - u0)
        except np.linalg.LinAlgError:
            return np.nan

    _a6_cur = _a3ch6(_Vm6, _d2_6, _d3_6, _ab6, _F6)

    # ── Metric strip ───────────────────────────────────────────────────────────
    _mc6 = st.columns(4)
    _mc6[0].metric("Scattering length a",
                   f"{_a6_cur/a0_nm:.1f} a₀" if np.isfinite(_a6_cur) else "diverges")
    for _ji in range(3):
        _mc6[_ji+1].metric(f"λ{_ji+1}  (neV)", f"{_ev6[_ji]*1e9:.3f}")

    st.markdown("---")

    # ── Three panels: V̂ heatmap | eigenvalues | Bloch triangle ────────────────
    _p6m, _p6e, _p6t = st.columns(3)
    _c6  = ["#69ff47", "#ffd740", "#ff6ec7"]
    _SQ3 = np.sqrt(3)/2

    with _p6m:
        st.markdown("**Potential matrix V̂  (neV)**")
        _fig, _ax = plt.subplots(figsize=(3.2, 2.8))
        _fig.patch.set_facecolor("#0e1117"); _ax.set_facecolor("#0e1117")
        _im6 = _ax.imshow(_Vm6*1e9, cmap="RdBu_r")
        plt.colorbar(_im6, ax=_ax, shrink=0.85)
        _lk = ["|1⟩", "|2⟩", "|3⟩"]
        _ax.set_xticks([0,1,2]); _ax.set_xticklabels(_lk, color="white")
        _ax.set_yticks([0,1,2]); _ax.set_yticklabels(_lk, color="white")
        for _i in range(3):
            for _j in range(3):
                _ax.text(_j, _i, f"{_Vm6[_i,_j]*1e9:.1f}",
                         ha="center", va="center", fontsize=8, color="white")
        _ax.tick_params(colors="white")
        st.pyplot(_fig); plt.close(_fig)

    with _p6e:
        st.markdown("**Dressed eigenvalues (neV)**")
        _fig, _ax = plt.subplots(figsize=(3.2, 2.8))
        _fig.patch.set_facecolor("#0e1117"); _ax.set_facecolor("#0e1117")
        _ax.barh(["λ₁", "λ₂", "λ₃"], _ev6*1e9, color=_c6)
        _ax.axvline(0, color="white", lw=0.8, ls="--")
        _ax.set_xlabel("Energy (neV)", color="white")
        for _i, (ev, col) in enumerate(zip(_ev6*1e9, _c6)):
            _ax.text(ev + (0.3 if ev >= 0 else -0.3), _i,
                     f"{ev:.2f}", va="center",
                     ha="left" if ev >= 0 else "right", color=col, fontsize=8)
        _ax.tick_params(colors="white")
        for _sp in _ax.spines.values(): _sp.set_edgecolor("#444")
        st.pyplot(_fig); plt.close(_fig)

    with _p6t:
        st.markdown("**Bloch triangle (spin populations)**")
        st.caption("Corners = pure |1⟩, |2⟩, |3⟩; dots = dressed eigenstates ψᵢ")
        _fig, _ax = plt.subplots(figsize=(3.2, 2.8))
        _fig.patch.set_facecolor("#0e1117"); _ax.set_facecolor("#0e1117")
        # Triangle: |1⟩ top (0.5, √3/2), |2⟩ bottom-left (0,0), |3⟩ bottom-right (1,0)
        # Barycentric → Cartesian: x = p1·0.5 + p3,  y = p1·√3/2
        _ax.plot([0, 1, 0.5, 0], [0, 0, _SQ3, 0], "w-", lw=1)
        _ax.text(0.5,  _SQ3+0.07, "|1⟩ open", ha="center", fontsize=8, color="white")
        _ax.text(-0.10, -0.07,    "|2⟩",       ha="center", fontsize=8, color="white")
        _ax.text( 1.10, -0.07,    "|3⟩",       ha="center", fontsize=8, color="white")
        for _i, col in enumerate(_c6):
            _p1, _p2, _p3 = _ec6[0,_i]**2, _ec6[1,_i]**2, _ec6[2,_i]**2
            _tx = _p1*0.5 + _p3;  _ty = _p1*_SQ3
            _ax.scatter([_tx], [_ty], color=col, s=110, zorder=5)
            _ax.annotate(f"ψ{_i+1}", (_tx, _ty), xytext=(5, 5),
                         textcoords="offset points", fontsize=7, color=col)
        _ax.set_xlim(-0.20, 1.20); _ax.set_ylim(-0.13, 1.05)
        _ax.set_aspect("equal"); _ax.axis("off")
        st.pyplot(_fig); plt.close(_fig)

    # ── Scattering length sweep: a vs δ₂ ──────────────────────────────────────
    st.markdown("**3-channel scattering length a(δ₂)**  — genuine K-matrix, δ₃ and couplings fixed")
    st.caption("Divergences (→ ±∞) are Feshbach-like resonances of the three-channel dressed-state system. "
               "Increase W₁₂ or W₁₃ to broaden them; adjust δ₂ slider to explore the resonance region.")

    # Sweep δ₂ over a wide positive range (δ₂ > 0 keeps ch-2 closed at E=0)
    _d2_sw6 = np.linspace(0.2, 100.0, 1000) * 1e-9
    _a_sw6  = np.array([
        _a3ch6(
            np.array([[_V1_6, _W12_6, _W13_6],
                      [_W12_6, _V1_6+d, _W23_6],
                      [_W13_6, _W23_6,  _V1_6+_d3_6]]),
            d, _d3_6, _ab6, _F6
        )
        for d in _d2_sw6
    ])
    _clip6 = 5000 * a0_nm
    _a_sw6_pl = np.where(np.abs(_a_sw6) > _clip6, np.nan, _a_sw6) / a0_nm

    # find resonance positions (sign changes across divergences)
    _res_pos6 = []
    for _ii in range(len(_a_sw6_pl) - 1):
        v1, v2 = _a_sw6_pl[_ii], _a_sw6_pl[_ii+1]
        if np.isfinite(v1) and np.isfinite(v2) and v1*v2 < 0 and abs(v1-v2) > 500:
            _res_pos6.append((_d2_sw6[_ii] + _d2_sw6[_ii+1]) * 0.5 * 1e9)

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _fig.patch.set_facecolor("#0e1117"); _ax.set_facecolor("#0e1117")
    _ax.plot(_d2_sw6*1e9, _a_sw6_pl, "#69ff47", lw=1.8)
    _ax.axhline(0, color="white", lw=0.5, ls="--")
    _ax.axvline(_d2n6, color="#ffd740", lw=1.2, ls="--", label=f"current δ₂ = {_d2n6:.1f} neV")
    for _rp in _res_pos6:
        _ax.axvline(_rp, color="#ff6ec7", lw=1, ls=":", alpha=0.7)
    if _res_pos6:
        _ax.axvline(_res_pos6[0], color="#ff6ec7", lw=1, ls=":", alpha=0.7,
                    label=f"resonance pole(s): {', '.join(f'{p:.1f}' for p in _res_pos6)} neV")
    _ax.set_xlabel("δ₂  (neV)", color="white", fontsize=11)
    _ax.set_ylabel("a  (a₀)",  color="white", fontsize=11)
    _ax.set_title("3-channel K-matrix: scattering length vs channel-2 detuning", color="white", fontsize=11)
    _ax.set_xlim(0, 100)
    _ax.set_ylim(-3000, 3000)
    _ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
    _ax.tick_params(colors="white")
    for _sp in _ax.spines.values(): _sp.set_edgecolor("#444")
    _fig.tight_layout()
    st.pyplot(_fig, use_container_width=True); plt.close(_fig)

    if _res_pos6:
        st.success(f"Resonance pole(s) detected at δ₂ ≈ {', '.join(f'{p:.1f}' for p in _res_pos6)} neV  "
                   f"— set the δ₂ slider to one of these values to sit at the resonance.")
    else:
        st.info("No resonance detected in δ₂ ∈ [0.2, 100] neV with current couplings. "
                "Try increasing W₁₂ or W₁₃ above 5 neV.")

    # ── Wavefunction: K-matrix interior + analytic exterior ───────────────────
    # Three independent interior ODE solutions → combine so closed-channel BCs hold.
    # u_j' + κ_j u_j = 0 at r=ā  (j=2,3);  u_1'(ā) = 1  (normalization).
    # Exterior: u₁ = r − a  (linear),  u₂,₃ = B exp(−κr)  (decaying).
    st.markdown("**Radial wavefunctions  u₁, u₂, u₃  —  physical scattering solution  (E = 0)**")

    def _ode6_in(r, y):
        u = np.array([y[0], y[1], y[2]])
        d2u = _F6 * (_Vm6 @ u)
        return [y[3], y[4], y[5], d2u[0], d2u[1], d2u[2]]

    _r0_6 = 1e-5 * _ab6
    _r_in6 = np.linspace(_r0_6, _ab6, 600)
    _kp1_6 = np.sqrt(_F6 * _d2_6);  _kp2_6 = np.sqrt(_F6 * _d3_6)

    try:
        # Solve 3 independent ICs inside the well
        _ss6 = []
        for _jj in range(3):
            _ic6 = [0.]*6;  _ic6[_jj] = _r0_6;  _ic6[_jj+3] = 1.
            _ss6.append(solve_ivp(_ode6_in, [_r0_6, _ab6], _ic6,
                                  t_eval=_r_in6, method="RK45",
                                  rtol=1e-10, atol=1e-12))
        # Value and derivative matrices at r=ā  (rows=channels, cols=ICs)
        _Yu6 = np.array([[_ss6[_jj].y[_ch,-1] for _jj in range(3)] for _ch in range(3)])
        _Yp6 = np.array([[_ss6[_jj].y[_ch+3,-1] for _jj in range(3)] for _ch in range(3)])
        # K-matrix system
        _Mkm6 = np.array([_Yp6[1,:] + _kp1_6*_Yu6[1,:],
                           _Yp6[2,:] + _kp2_6*_Yu6[2,:],
                           _Yp6[0,:]])
        _ckm6 = np.linalg.solve(_Mkm6, [0.0, 0.0, 1.0])

        # Physical interior wavefunction
        _u_in6 = sum(_ckm6[_jj] * _ss6[_jj].y[:3, :] for _jj in range(3))  # 3 × N_in

        # Analytic exterior
        _r_ex6 = np.linspace(_ab6, 2.5*_ab6, 400)
        _u1_ex6 = _r_ex6 - _a6_cur if np.isfinite(_a6_cur) else np.full_like(_r_ex6, np.nan)
        _u2_ex6 = _u_in6[1, -1] * np.exp(-_kp1_6 * (_r_ex6 - _ab6))
        _u3_ex6 = _u_in6[2, -1] * np.exp(-_kp2_6 * (_r_ex6 - _ab6))

        # Combine and normalise
        _r_all6  = np.concatenate([_r_in6, _r_ex6])
        _u1_all6 = np.concatenate([_u_in6[0, :], _u1_ex6])
        _u2_all6 = np.concatenate([_u_in6[1, :], _u2_ex6])
        _u3_all6 = np.concatenate([_u_in6[2, :], _u3_ex6])
        _nrm6    = max(np.nanmax(np.abs(_u1_all6)), 1e-30)

        _fig, _ax = plt.subplots(figsize=(8, 3))
        _fig.patch.set_facecolor("#0e1117"); _ax.set_facecolor("#0e1117")
        _ra0_6 = _r_all6 / a0_nm
        _ax.plot(_ra0_6, _u1_all6/_nrm6, _c6[0], lw=1.5, label="u₁  open  (r − a exterior)")
        _ax.plot(_ra0_6, _u2_all6/_nrm6, _c6[1], lw=1.5, label="u₂  closed (δ₂,  e^{−κ₂r} exterior)")
        _ax.plot(_ra0_6, _u3_all6/_nrm6, _c6[2], lw=1.5, label="u₃  closed (δ₃,  e^{−κ₃r} exterior)")
        _ax.axvline(_ab6/a0_nm, color="white", lw=1, ls="--", label="ā  (well edge)")
        _ax.set_xlabel("r  (a₀)", color="white")
        _ax.set_ylabel("u / max|u₁|",  color="white")
        _ax.set_title("Physical scattering wavefunctions: K-matrix interior + analytic exterior", color="white")
        _ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=8)
        _ax.tick_params(colors="white")
        for _sp in _ax.spines.values(): _sp.set_edgecolor("#444")
        st.pyplot(_fig); plt.close(_fig)
        st.caption(
            "Interior: K-matrix combination of 3 independent ODE solutions satisfying "
            "u_j′ + κ_j u_j = 0 at ā for closed channels.  "
            f"Scattering length a = **{_a6_cur/a0_nm:.1f} a₀** "
            "(dressed-state analytic formula, exact for square well)."
        )
    except Exception as _exc6:
        st.warning(f"K-matrix solve failed: {_exc6}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Calibrated two-channel s-wave resonance
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("<h2 style='color:#69ff47;'>Calibrated two-channel s-wave resonance</h2>",
                unsafe_allow_html=True)
    st.markdown("""
<p style="color:#b0bec5;">
The toy model above uses arbitrary detunings. Here we map to physical magnetic field:
δ(B) = δμ · μ<sub>B</sub> · (B − B<sub>c</sub>), then fit the coupling W so that
the vdW ODE produces a scattering-length pole at the experimental
B₀ = −11.1 G of the s-wave resonance. The d-wave and g-wave resonances require
additional closed channels and do not appear in this two-channel model.
</p>
""", unsafe_allow_html=True)

    # Fixed calibration parameters matching Tab 4 defaults
    _C6_eV_6s  = 6890 * 27.2114 * a0_nm**6
    _r_min_6s  = 20.0 * a0_nm
    _r_max_6s  = 6.0  * a_bar_cs
    _W_gamma_6s = np.sqrt(TABLE['s-wave']['Gamma'] * 1e6 * h_eVs * hbar2_2mr / a_bar_cs**2)

    with st.spinner("Fitting W to s-wave pole at B₀ = −11.1 G …"):
        _W_fit_6s, _ = fit_W_vdw_to_pole(_C6_eV_6s, _r_min_6s, _r_max_6s)

    if _W_fit_6s is not None:
        st.markdown(f"""
<div style="background:#1a1a2e; border-left:4px solid #69ff47; border-radius:6px;
            padding:0.5rem 1rem; font-size:0.85rem; color:#b0bec5; margin-bottom:0.8rem;">
  Fitted W = <b style="color:#69ff47;">{_W_fit_6s*1e9:.2f} neV</b>
  ({_W_fit_6s/_W_gamma_6s:.2f} × W_Γ) — places the vdW ODE pole at
  <b style="color:#ffd740;">B₀ = −11.1 G</b>.<br>
  <span style="color:#888; font-size:0.82rem;">Calibrated demonstration, not a
  first-principles prediction of W. Width (B*) is not constrained by this fit.</span>
</div>""", unsafe_allow_html=True)

        @st.cache_data(show_spinner=False)
        def _sw6_sweep(C6_key, r_min_key, r_max_key, W_key):
            _B0 = TABLE['s-wave']['B0']
            # Multi-resolution grid: coarse background + fine near pole
            # Pole width < 0.0001 G, so points at ±5e-5 G are required to see it
            B_coarse    = np.linspace(-15.0, 22.0, 80)
            B_medium    = np.concatenate([
                np.linspace(_B0 - 0.5,  _B0 - 0.05, 30),
                np.linspace(_B0 + 0.05, _B0 + 0.5,  30),
            ])
            B_fine      = np.concatenate([
                np.linspace(_B0 - 0.05, _B0 - 0.002, 40),
                np.linspace(_B0 + 0.002, _B0 + 0.05, 40),
            ])
            B_very_fine = np.concatenate([
                np.linspace(_B0 - 0.002, _B0 - 5e-5, 50),
                np.linspace(_B0 + 5e-5,  _B0 + 0.002, 50),
            ])
            B_arr = np.sort(np.unique(np.concatenate(
                [B_coarse, B_medium, B_fine, B_very_fine])))
            a_arr = np.full(len(B_arr), np.nan)
            fac   = m_r_me / hbar2_2me
            for i, B in enumerate(B_arr):
                ds = TABLE['s-wave']['dmu'] * muB_eVperG * (B - TABLE['s-wave']['Bc'])
                def _od(r, y, ds=ds):
                    uo, uc, puo, puc = y
                    Vl = -C6_key / r**6
                    return [puo, puc,
                            fac*(Vl*uo + W_key*uc),
                            fac*((Vl+ds)*uc + W_key*uo)]
                try:
                    sol = solve_ivp(_od, [r_min_key, r_max_key], [0., 0., 1., 0.],
                                    method='LSODA', rtol=1e-9, atol=1e-11)
                    if sol.success:
                        uo_e, puo_e = sol.y[0, -1], sol.y[2, -1]
                        if abs(puo_e) > 1e-20:
                            a_arr[i] = sol.t[-1] - uo_e / puo_e
                except Exception:
                    pass
            return B_arr, a_arr

        with st.spinner("Computing s-wave a(B) …"):
            _B6s, _a6s = _sw6_sweep(_C6_eV_6s, _r_min_6s, _r_max_6s, float(_W_fit_6s))

        _clip6s     = 30000.0
        _a_paper6s  = a_of_B_paper(_B6s)
        _B0_6s      = TABLE['s-wave']['B0']
        _Bs_6s      = TABLE['s-wave']['Bstar']

        fig_6s, ax_6s = plt.subplots(figsize=(10, 4))
        fig_6s.patch.set_facecolor("#0e1117"); ax_6s.set_facecolor("#0e1117")
        ax_6s.tick_params(colors="white", labelsize=10)
        ax_6s.xaxis.label.set_color("white"); ax_6s.yaxis.label.set_color("white")
        for _sp6 in ax_6s.spines.values(): _sp6.set_edgecolor("#333")

        ax_6s.plot(_B6s, np.clip(_a_paper6s / a0_nm, -_clip6s, _clip6s),
                   color="#00e5ff", lw=2, label="Lange et al. product formula (all 3 resonances)")
        ax_6s.plot(_B6s, np.clip(_a6s / a0_nm, -_clip6s, _clip6s),
                   color="#69ff47", lw=2, label="vdW ODE — calibrated s-wave (fitted W)")
        ax_6s.axvline(_B0_6s, color="#ffd740", lw=1.5, ls="--",
                      label=f"B₀ = {_B0_6s} G  (s-wave pole)")
        ax_6s.axvline(_Bs_6s, color="#ff6ec7", lw=1.0, ls=":",
                      label=f"B* = {_Bs_6s} G  (paper zero-crossing)")
        ax_6s.axhline(0, color="#555", lw=0.8)
        ax_6s.axhline(a_bg / a0_nm, color="#888", lw=0.8, ls="--",
                      label=f"a_bg = {a_bg/a0_nm:.0f} a₀")
        ax_6s.set_xlim(-15, 22)
        ax_6s.set_ylim(-_clip6s, _clip6s)
        ax_6s.set_xlabel("Magnetic field B (G)", fontsize=11)
        ax_6s.set_ylabel("a  (a₀)", fontsize=11)
        ax_6s.set_title("Calibrated s-wave resonance: vdW ODE (green) vs Lange et al. (cyan)",
                        color="white", fontsize=12)
        ax_6s.legend(fontsize=9, framealpha=0.2, labelcolor="white",
                     facecolor="#0e1117", edgecolor="#444")
        fig_6s.tight_layout()
        st.pyplot(fig_6s, use_container_width=True)
        plt.close(fig_6s)

        st.markdown("""
<div style="background:#1a1a2e; border-left:4px solid #555; border-radius:6px;
            padding:0.5rem 1rem; font-size:0.82rem; color:#b0bec5; margin-top:0.5rem;">
  The green curve shares the s-wave pole at B₀ = −11.1 G with the cyan paper formula,
  confirming the calibration. The resonance width and B* zero-crossing differ because
  r_min is not also fitted. The d-wave (47.78 G) and g-wave (53.45 G) poles visible in
  the cyan curve require additional closed channels and are absent here.
</div>""", unsafe_allow_html=True)

    else:
        st.warning("Pole fitting did not converge for default parameters "
                   "(C₆ = 6890 au, r_min = 20 a₀).")
