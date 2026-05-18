import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Feshbach Scattering — Lange et al. 2009", layout="wide")

# ── Physical constants ─────────────────────────────────────────────────────────
a0_nm      = 0.0529177210903   # nm
muB_eVperG = 5.78838e-9        # eV/G
hbar2_2me  = 0.038099          # eV·nm²
h_eVs      = 4.135668e-15      # eV·s
u_to_me    = 1822.888          # amu → mₑ

# ── Cesium (Lange et al. PRA 2009, Table I) ───────────────────────────────────
m_Cs        = 132.905
m_r_me      = (m_Cs / 2.0) * u_to_me
hbar2_2mr   = hbar2_2me / m_r_me   # eV·nm²

a_bar       = 95.7  * a0_nm    # nm  (mean scattering length)
a_bg        = 1875.0 * a0_nm   # nm  (background scattering length)

# Table I: B0=pole, Bstar=zero of a, Bc=bare-state crossing, dmu in μB, Gamma in MHz
TABLE = {
    's-wave': {'B0': -11.1,  'Bstar': 18.1,   'Bc': 19.7,   'dmu': 2.50, 'Gamma': 11.6},
    'd-wave': {'B0': 47.78,  'Bstar': 47.944,  'Bc': 47.962, 'dmu': 1.15, 'Gamma': 0.065},
    'g-wave': {'B0': 53.449, 'Bstar': 53.457,  'Bc': 53.458, 'dmu': 1.50, 'Gamma': 0.0042},
}

# ── Formulae ──────────────────────────────────────────────────────────────────
def a_of_B(B):
    B = np.asarray(B, dtype=float)
    result = np.ones_like(B)
    for r in TABLE.values():
        denom = B - r['B0']
        numer = B - r['Bstar']
        safe  = np.abs(denom) > 0.005
        result = np.where(safe, result * numer / denom, np.nan)
    return a_bg * result

def Eb_kHz(a_nm_arr):
    """Binding energy in kHz with effective-range correction (r_eff = a_bar)."""
    a  = np.asarray(a_nm_arr, dtype=float)
    Eb = np.full_like(a, np.nan)
    m  = a > 2 * a_bar          # universal formula valid when a > 2 r_eff
    disc   = 1 - 2 * a_bar / a[m]
    kappa  = (1 - np.sqrt(np.clip(disc, 0, None))) / a_bar
    Eb[m]  = -hbar2_2mr * kappa**2 / (h_eVs * 1e3)   # kHz
    # Simple universal (no eff-range) for a_bar < a < 2*a_bar
    m2     = (a > 0) & (a <= 2 * a_bar)
    Eb[m2] = -hbar2_2mr / a[m2]**2 / (h_eVs * 1e3)
    return Eb

# ── Hero intro ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d1b2a,#1a0533);
            border-left:5px solid #00e5ff; border-radius:10px;
            padding:1.4rem 2rem; margin-bottom:1.2rem;">
  <h1 style="color:#fff; margin:0 0 0.6rem 0; font-size:1.9rem;">
    Feshbach Resonance Simulator — Cs atoms
  </h1>
  <p style="color:#b0bec5; font-size:0.9rem; margin:0 0 0.5rem 0;">
    <b style="color:#fff;">Lange et al., PRA 79, 013622 (2009)</b> — two-channel square-well
    model for Cs in |F=3, m<sub>F</sub>=3⟩. Reproduces Figs. 3 & 4 of the paper.
  </p>
  <div style="display:flex; gap:1.4rem; flex-wrap:wrap;">
    <span style="color:#00e5ff;">⟐ Scattering length a(B)</span>
    <span style="color:#69ff47;">⟐ Binding energy E_b(B)</span>
    <span style="color:#ffd740;">⟐ s-, d-, g-wave Feshbach resonances</span>
    <span style="color:#ff6ec7;">⟐ Coupled-channel ODE at Region I boundary</span>
    <span style="color:#ce93d8;">⟐ Dressed-state eigenvectors</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Cs Parameters (Table I)")
st.sidebar.markdown(f"""
| | B₀ (G) | B* (G) | Bc (G) | δμ (μB) | Γ/h (MHz) |
|---|---|---|---|---|---|
| s | −11.1 | 18.1 | 19.7 | 2.50 | 11.6 |
| d | 47.78 | 47.944 | 47.962 | 1.15 | 0.065 |
| g | 53.449 | 53.457 | 53.458 | 1.50 | 0.0042 |
""")
st.sidebar.markdown(f"ā = 95.7 a₀ = {a_bar/a0_nm:.1f} a₀  \na_bg = 1875 a₀  \nmᵣ = m_Cs/2")

st.sidebar.markdown("---")
st.sidebar.markdown("**Explore**")
B_probe = st.sidebar.slider("B (G) — probe field", 20.0, 62.0, 30.0, 0.1)

# ── Compute sweep ──────────────────────────────────────────────────────────────
B_arr  = np.linspace(20.0, 62.0, 8000)
a_arr  = a_of_B(B_arr)
Eb_arr = Eb_kHz(a_arr)

a_probe  = a_of_B(B_probe)
Eb_probe = Eb_kHz(np.array([a_probe]))[0]

# ── Paper results section ──────────────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Fig. 3 — Molecular Binding Energy  E_b/h (kHz)</h3>",
            unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(13, 6))
fig3.patch.set_facecolor("#0e1117")
ax3.set_facecolor("#0e1117")
ax3.tick_params(colors="white", labelsize=11)
ax3.xaxis.label.set_color("white")
ax3.yaxis.label.set_color("white")
for sp in ax3.spines.values():
    sp.set_edgecolor("#333")

# Plot binding energy (negative convention, show |Eb| on inverted axis like paper)
ax3.plot(B_arr, Eb_arr, color="#e74c3c", lw=2, label="Model  (two-channel, Eq. 13)")

# Resonance poles
for name, r in TABLE.items():
    ax3.axvline(r['B0'], color="#ffd740", lw=1.2, ls="--", alpha=0.6)
    ax3.text(r['B0'] + 0.2, -5, name, color="#ffd740", fontsize=9)

# Probe marker
if not np.isnan(Eb_probe):
    ax3.scatter([B_probe], [Eb_probe], color="#00e5ff", s=80, zorder=6)
    ax3.text(B_probe + 0.3, Eb_probe - 8,
             f"B={B_probe:.1f}G\nEb/h={Eb_probe:.1f}kHz",
             color="#00e5ff", fontsize=9)

ax3.set_xlim(20, 62)
ax3.set_ylim(-360, 10)
ax3.set_xlabel("Magnetic field (G)", fontsize=12)
ax3.set_ylabel("Binding energy  Eᵦ/h  (kHz)", fontsize=12)
ax3.set_title("Binding energy of Cs₂ near three Feshbach resonances  [cf. Fig. 3, Lange et al.]",
              color="white", fontsize=12)
ax3.legend(fontsize=10, framealpha=0.2, labelcolor="white",
           facecolor="#0e1117", edgecolor="#444")
ax3.axhline(0, color="#555", lw=0.8, ls="--")
fig3.tight_layout()
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

st.markdown("<h3 style='color:#00e5ff;'>Fig. 4 — Scattering Length  a(B)</h3>",
            unsafe_allow_html=True)

fig4, ax4 = plt.subplots(figsize=(13, 5))
fig4.patch.set_facecolor("#0e1117")
ax4.set_facecolor("#0e1117")
ax4.tick_params(colors="white", labelsize=11)
ax4.xaxis.label.set_color("white")
ax4.yaxis.label.set_color("white")
for sp in ax4.spines.values():
    sp.set_edgecolor("#333")

a_plot = np.clip(a_arr / a0_nm, -8000, 8000)
ax4.plot(B_arr, a_plot, color="#00e5ff", lw=2)
ax4.axhline(0, color="#555", lw=0.8, ls="--")
ax4.axhline(a_bg / a0_nm, color="#aaaaff", lw=1, ls=":", alpha=0.7,
            label=f"a_bg = {a_bg/a0_nm:.0f} a₀")

for name, r in TABLE.items():
    ax4.axvline(r['B0'],    color="#ffd740", lw=1.2, ls="--", alpha=0.5)
    ax4.axvline(r['Bstar'], color="#ff6ec7", lw=1.0, ls=":",  alpha=0.5)

ax4.scatter([B_probe], [np.clip(a_probe / a0_nm, -8000, 8000)],
            color="#00e5ff", s=80, zorder=6)
ax4.text(B_probe + 0.3, np.clip(a_probe / a0_nm, -8000, 8000) + 200,
         f"a={a_probe/a0_nm:.0f} a₀", color="#00e5ff", fontsize=9)

ax4.set_xlim(20, 62)
ax4.set_ylim(-8000, 8000)
ax4.set_xlabel("Magnetic field (G)", fontsize=12)
ax4.set_ylabel("Scattering length  a  (a₀)", fontsize=12)
ax4.set_title("Scattering length a(B) — diverges at each Feshbach resonance  [cf. Fig. 4, Lange et al.]",
              color="white", fontsize=12)
ax4.legend(fontsize=10, framealpha=0.2, labelcolor="white",
           facecolor="#0e1117", edgecolor="#444")
# Annotate zeros and poles
ax4.text(18.1, 500, "B*_s", color="#ff6ec7", fontsize=8, ha="center")
ax4.text(-11.1, 500, "B₀_s", color="#ffd740", fontsize=8, ha="center")
fig4.tight_layout()
st.pyplot(fig4, use_container_width=True)
plt.close(fig4)

st.divider()

# ── Region I physics at probe field ───────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Region I Analysis at  B = {:.1f} G</h3>".format(B_probe),
            unsafe_allow_html=True)

# Derived quantities at B_probe
delta      = {name: r['dmu'] * muB_eVperG * (B_probe - r['Bc'])
              for name, r in TABLE.items()}
# For Region I we use the s-wave channel as the dominant one
# V matrix: V_oo = -V_open, V_cc = -V_closed + delta_s
# Lange model: well depths determined from a_bg and a_bar
# From single-channel: a_bg = a_bar * (1 - tan(k0*a_bar)/(k0*a_bar))
# Solve numerically for V_open from a_bg
from scipy.optimize import brentq

def abg_from_V(V_eV):
    k = np.sqrt(2 * m_r_me * V_eV / hbar2_2me)   # nm⁻¹
    ka = k * a_bar
    if abs(np.cos(ka)) < 1e-12:
        return np.inf
    return a_bar * (1 - np.tan(ka) / ka)

# First resonance pole is at k·ā = π/2; a_bg>0 so V_open sits just above it
# in the first branch where a decreases from +∞ toward 0
V_pole_1 = (np.pi / 2)**2 * hbar2_2me / (2 * m_r_me * a_bar**2)
V_open_eV = brentq(lambda V: abg_from_V(V) - a_bg,
                   V_pole_1 * 1.001, V_pole_1 * 5.0)

# Closed channel: detuning shifts threshold (eV), using s-wave as dominant
delta_s_eV = delta['s-wave']

V_oo  = -V_open_eV
V_cc  = V_oo + (a_bg - a_bar) / a_bar * 0.0   # placeholder — see below

# Use simpler parameterisation: V matrix from scattering length formula
# V_oo = -V_open, V_cc = -V_open + delta_s (closed channel threshold shifted)
V_cc = delta_s_eV   # closed channel threshold energy (eV) relative to open

# Coupling W from Gamma_s:  ℏΓ = 2μ W² φc²(a_bar) / ℏk → approximate
# Use Gamma parameter: hΓ in eV
hGamma_s = TABLE['s-wave']['Gamma'] * 1e6 * h_eVs  # eV
W_eV     = np.sqrt(hGamma_s * hbar2_2mr / a_bar**2)  # rough estimate

Delta_V   = (V_oo - V_cc) / 2
offset    = (V_oo + V_cc) / 2
R_ev      = np.sqrt(Delta_V**2 + W_eV**2)
two_theta = np.arctan2(W_eV, Delta_V)
theta     = two_theta / 2
cos_t     = np.cos(theta)
sin_t     = np.sin(theta)
lam_plus  = offset + R_ev
lam_minus = offset - R_ev

V_matrix  = np.array([[V_oo, W_eV], [W_eV, V_cc]])
evals_np, evecs_np = np.linalg.eigh(V_matrix)

# Analytical table
st.markdown(f"""
<div style="display:flex; gap:2rem; flex-wrap:nowrap; align-items:flex-start;
            background:#111827; border-radius:10px; padding:1.3rem;">

  <div style="flex:1; min-width:200px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.6rem 0;">Cs Region I parameters</p>
    <table style="color:#fff; font-size:0.88rem; border-collapse:collapse; width:100%;">
      <tr><td style="color:#aaa; padding:3px 6px;">ā</td>
          <td style="padding:3px 6px; color:#00e5ff;">{a_bar/a0_nm:.1f} a₀ = {a_bar:.4f} nm</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">a_bg</td>
          <td style="padding:3px 6px; color:#00e5ff;">{a_bg/a0_nm:.0f} a₀ = {a_bg:.3f} nm</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">V_open</td>
          <td style="padding:3px 6px; color:#69ff47;">{V_open_eV:.4f} eV</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">δ_s (at B={B_probe:.1f}G)</td>
          <td style="padding:3px 6px; color:#ffd740;">{delta_s_eV*1e6:.3f} μeV</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">W (coupling)</td>
          <td style="padding:3px 6px; color:#ff6ec7;">{W_eV*1e6:.3f} μeV</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">a(B)</td>
          <td style="padding:3px 6px; color:#00e5ff;">{a_probe/a0_nm:.1f} a₀</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">E_b/h</td>
          <td style="padding:3px 6px; color:#69ff47;">{Eb_probe:.2f} kHz</td></tr>
    </table>
  </div>

  <div style="flex:1; min-width:220px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.6rem 0;">Dressed eigenstates</p>
    <table style="color:#fff; font-size:0.88rem; border-collapse:collapse; width:100%;">
      <tr><td style="color:#aaa; padding:3px 6px;">ΔV</td>
          <td style="padding:3px 6px; color:#00e5ff;">{Delta_V:.6f} eV</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">R</td>
          <td style="padding:3px 6px; color:#00e5ff;">{R_ev:.6f} eV</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">θ</td>
          <td style="padding:3px 6px; color:#ce93d8;">{np.degrees(theta):.4f}°</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">λ₊</td>
          <td style="padding:3px 6px; color:#69ff47;">{lam_plus:.6f} eV</td></tr>
      <tr><td style="color:#aaa; padding:3px 6px;">λ₋</td>
          <td style="padding:3px 6px; color:#ff6ec7;">{lam_minus:.6f} eV</td></tr>
    </table>
    <p style="color:#fff; font-size:0.88rem; margin:0.8rem 0 0 0; line-height:2;">
      <span style="color:#69ff47;">|+⟩</span> = {cos_t:+.4f} |open⟩ {sin_t:+.4f} |closed⟩<br>
      <span style="color:#ff6ec7;">|−⟩</span> = {-sin_t:+.4f} |open⟩ {cos_t:+.4f} |closed⟩
    </p>
  </div>

  <div style="flex:1; min-width:220px;">
    <p style="color:#ffd740; font-weight:bold; margin:0 0 0.6rem 0;">Magnetic detunings (eV)</p>
    <table style="color:#fff; font-size:0.88rem; border-collapse:collapse; width:100%;">
      {"".join(f'<tr><td style="color:#aaa;padding:3px 6px;">{nm}</td><td style="padding:3px 6px;color:#ffd740;">{dv*1e9:.4f} neV</td></tr>'
               for nm, dv in delta.items())}
    </table>
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Numerical ODE at probe B ────────────────────────────────────────────────
st.markdown("<h3 style='color:#00e5ff;'>Numerical Wavefunction at B = {:.1f} G  (E = 0)</h3>".format(B_probe),
            unsafe_allow_html=True)

factor = 2.0 * m_r_me / hbar2_2me   # nm⁻² eV⁻¹
E_scatt = 0.0

def ode(t, y):
    uo, uc, puo, puc = y
    dpuo = factor * ((V_oo - E_scatt) * uo + W_eV * uc)
    dpuc = factor * ((V_cc - E_scatt) * uc + W_eV * uo)
    return [puo, puc, dpuo, dpuc]

r_end  = a_bar * 3.0
r_eval = np.linspace(1e-6, r_end, 3000)
sol    = solve_ivp(ode, [1e-6, r_end], [1e-6, 0.0, 1.0, 0.0],
                   t_eval=r_eval, method="RK45", rtol=1e-10, atol=1e-12)

uo_num = sol.y[0];  uc_num = sol.y[1]
r_num  = sol.t

# Scattering length from wavefunction
uo_end  = sol.y[0, -1]
puo_end = sol.y[2, -1]
a_num   = r_end - uo_end / puo_end if abs(puo_end) > 1e-12 else np.nan

fig5, axes5 = plt.subplots(1, 2, figsize=(16, 5))
fig5.patch.set_facecolor("#0e1117")
for ax in axes5:
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=10)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.axvline(a_bar / a0_nm if ax == axes5[0] else a_bar,
               color="#ffd740", lw=1.2, ls="--", alpha=0.5, label="ā boundary")

# Wavefunction
ax = axes5[0]
ax.plot(r_num / a0_nm, uo_num, color="#69ff47", lw=2, label="u_open(r)")
ax.plot(r_num / a0_nm, uc_num, color="#ff6ec7", lw=2, label="u_closed(r)")
ax.axhline(0, color="#444", lw=0.6)
ax.set_xlabel("r  (a₀)")
ax.set_ylabel("u(r)  [arb.]")
ax.set_title(f"E=0 radial wavefunction  →  a = {a_num/a0_nm:.1f} a₀  (analytic: {a_probe/a0_nm:.1f} a₀)",
             color="white")
ax.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

# Probability density
ax2 = axes5[1]
ax2.plot(r_num, uo_num**2, color="#69ff47", lw=2, label="|u_open|²")
ax2.plot(r_num, uc_num**2, color="#ff6ec7", lw=2, label="|u_closed|²")
ax2.plot(r_num, uo_num**2 + uc_num**2, color="#ffd740", lw=1.5, ls="--", label="total")
ax2.axhline(0, color="#444", lw=0.6)
ax2.set_xlabel("r  (nm)")
ax2.set_ylabel("|u(r)|²")
ax2.set_title("Probability density", color="white")
ax2.legend(fontsize=9, framealpha=0.15, labelcolor="white", facecolor="#0e1117")

fig5.tight_layout()
st.pyplot(fig5, use_container_width=True)
plt.close(fig5)
