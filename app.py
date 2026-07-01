import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
from pathlib import Path

st.set_page_config(
    page_title="Engine Health Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR    = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

  * { font-family: 'Inter', sans-serif !important; }

  .stApp, [data-testid="stAppViewContainer"] {
    background-color: #080d1a !important;
  }
  [data-testid="stHeader"] { background-color: #080d1a !important; }
  [data-testid="stDecoration"] { display: none; }

  /* ── Header ── */
  .hdr-wrap {
    background: linear-gradient(135deg, #0f1f3d 0%, #0d1b35 60%, #0a1628 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hdr-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #3b82f6, #60a5fa, transparent);
  }
  .hdr-title {
    font-size: 1.75rem; font-weight: 800; color: #f1f5f9;
    letter-spacing: -0.02em; margin-bottom: 4px;
  }
  .hdr-sub {
    font-size: 0.85rem; color: #64748b; font-weight: 400;
  }
  .hdr-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.25);
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.75rem; font-weight: 600; color: #60a5fa;
    margin-top: 10px;
  }
  .hdr-dot { width: 6px; height: 6px; border-radius: 50%; background: #10b981;
             box-shadow: 0 0 6px #10b981; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  /* ── Metric pills (top row) ── */
  .metric-pill {
    background: #0f1f3d;
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
                  letter-spacing: 0.1em; color: #475569; margin-bottom: 6px; }
  .metric-value { font-size: 1.6rem; font-weight: 800; line-height: 1; }
  .metric-sub   { font-size: 0.72rem; color: #475569; margin-top: 4px; }

  /* ── RUL card ── */
  .rul-card {
    border-radius: 18px; padding: 28px 20px; text-align: center;
    position: relative; overflow: hidden;
  }
  .rul-safe   { background: linear-gradient(145deg,#052e16,#064e3b);
                border: 1px solid rgba(16,185,129,0.3); }
  .rul-warn   { background: linear-gradient(145deg,#1c1003,#292009);
                border: 1px solid rgba(245,158,11,0.3); }
  .rul-crit   { background: linear-gradient(145deg,#1f0707,#2d0a0a);
                border: 1px solid rgba(239,68,68,0.3); }

  .rul-label  { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.15em; color: #64748b; margin-bottom: 8px; }
  .rul-number { font-size: 5rem; font-weight: 900; line-height: 1;
                letter-spacing: -0.04em; }
  .rul-unit   { font-size: 0.85rem; color: #64748b; margin-top: 4px; }
  .safe-num   { color: #10b981; text-shadow: 0 0 30px rgba(16,185,129,0.3); }
  .warn-num   { color: #f59e0b; text-shadow: 0 0 30px rgba(245,158,11,0.3); }
  .crit-num   { color: #ef4444; text-shadow: 0 0 30px rgba(239,68,68,0.3); }

  .status-chip {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 30px; padding: 6px 18px;
    font-size: 0.82rem; font-weight: 700; margin-top: 14px;
  }
  .chip-safe { background: rgba(16,185,129,0.15); color: #10b981;
               border: 1px solid rgba(16,185,129,0.3); }
  .chip-warn { background: rgba(245,158,11,0.15); color: #f59e0b;
               border: 1px solid rgba(245,158,11,0.3); }
  .chip-crit { background: rgba(239,68,68,0.15); color: #ef4444;
               border: 1px solid rgba(239,68,68,0.3); }

  /* ── Info boxes ── */
  .ibox {
    background: #0f1829; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 16px 18px; margin-bottom: 10px;
  }
  .ibox-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: 0.1em; color: #475569; margin-bottom: 6px; }
  .ibox-value { font-size: 1.8rem; font-weight: 800; line-height: 1.1; }
  .ibox-sub   { font-size: 0.72rem; color: #475569; margin-top: 3px; }

  /* ── Progress bar ── */
  .prog-wrap {
    background: #0f1829; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 16px 20px;
  }
  .prog-title { font-size: 0.82rem; font-weight: 600; color: #94a3b8; margin-bottom: 10px; }
  .prog-track { background: rgba(255,255,255,0.06); border-radius: 999px;
                height: 10px; overflow: hidden; }
  .prog-fill  { border-radius: 999px; height: 10px; transition: width 0.3s; }
  .prog-labels { display: flex; justify-content: space-between;
                 font-size: 0.68rem; color: #334155; margin-top: 6px; }

  /* ── Section titles ── */
  .sec-title {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.12em; color: #3b82f6;
    border-left: 3px solid #3b82f6; padding-left: 12px;
    margin: 28px 0 14px 0;
  }

  /* ── Control buttons ── */
  div[data-testid="stButton"] button {
    background: #0f1829 !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
  }
  div[data-testid="stButton"] button:hover {
    background: #1e3a5f !important;
    border-color: rgba(59,130,246,0.4) !important;
    color: #60a5fa !important;
  }

  /* ── Slider ── */
  .stSlider label { font-size: 0.8rem !important; font-weight: 600 !important;
                    color: #64748b !important; text-transform: uppercase;
                    letter-spacing: 0.08em !important; }
  .stSlider [data-baseweb="slider"] { margin-top: 4px; }

  /* ── Select slider ── */
  .stSelectSlider label { color: #475569 !important; font-size: 0.75rem !important; }

  /* ── Footer ── */
  .footer {
    text-align: center; color: #1e293b;
    font-size: 0.72rem; margin-top: 32px;
    border-top: 1px solid rgba(255,255,255,0.04);
    padding-top: 16px; letter-spacing: 0.05em;
  }
  .footer span { color: #334155; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_preds():
    df = pd.read_csv(RESULTS_DIR / "meta_ensemble_predictions.csv")
    df.columns = ["cycle", "true_RUL", "pred", "std", "ci_low", "ci_high", "abs_error"]
    return df

df        = load_preds()
min_cycle = int(df["cycle"].min())
max_cycle = int(df["cycle"].max())

if "cycle"   not in st.session_state: st.session_state.cycle   = min_cycle
if "playing" not in st.session_state: st.session_state.playing = False

DARK = dict(
    paper_bgcolor="#0a0f1e",
    plot_bgcolor="#0a0f1e",
    font=dict(color="#94a3b8", family="Inter", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", showline=False,
               zeroline=False, tickfont=dict(color="#475569")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", showline=False,
               zeroline=False, tickfont=dict(color="#475569")),
    margin=dict(l=55, r=30, t=40, b=55),
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr-wrap">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
    <div>
      <div class="hdr-title">⚙️ Engine Health Monitor</div>
      <div class="hdr-sub">WTA³ Ensemble · NASA C-MAPSS FD001 · Engine #52</div>
      <div class="hdr-badge"><span class="hdr-dot"></span> Live Prognostics Dashboard</div>
    </div>
    <div style="display:flex; gap:10px; flex-wrap:wrap;">
      <div class="metric-pill">
        <div class="metric-label">RMSE</div>
        <div class="metric-value" style="color:#3b82f6;">3.73</div>
        <div class="metric-sub">cycles</div>
      </div>
      <div class="metric-pill">
        <div class="metric-label">MAE</div>
        <div class="metric-value" style="color:#3b82f6;">3.19</div>
        <div class="metric-sub">cycles</div>
      </div>
      <div class="metric-pill">
        <div class="metric-label">R²</div>
        <div class="metric-value" style="color:#10b981;">0.9929</div>
        <div class="metric-sub">score</div>
      </div>
      <div class="metric-pill">
        <div class="metric-label">±10 Cycle</div>
        <div class="metric-value" style="color:#10b981;">100%</div>
        <div class="metric-sub">accuracy</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl_col, speed_col, reset_col = st.columns([1, 2, 1])

with ctrl_col:
    if st.session_state.playing:
        if st.button("⏸  Pause", use_container_width=True):
            st.session_state.playing = False
            st.rerun()
    else:
        if st.button("▶  Play", use_container_width=True):
            st.session_state.playing = True
            st.rerun()

with speed_col:
    speed = st.select_slider(
        "Speed",
        options=["Slow", "Normal", "Fast", "Very Fast"],
        value="Normal",
        label_visibility="collapsed",
    )
    speed_map = {"Slow": 0.30, "Normal": 0.15, "Fast": 0.07, "Very Fast": 0.02}
    delay = speed_map[speed]

with reset_col:
    if st.button("↺  Reset", use_container_width=True):
        st.session_state.cycle   = min_cycle
        st.session_state.playing = False
        st.rerun()

selected_cycle = st.slider(
    "CURRENT CYCLE",
    min_value=min_cycle,
    max_value=max_cycle,
    value=st.session_state.cycle,
    step=1,
    key="slider_val",
)
if selected_cycle != st.session_state.cycle:
    st.session_state.cycle = selected_cycle

cycle = st.session_state.cycle

# ── Derived values ────────────────────────────────────────────────────────────
row      = df[df["cycle"] == cycle].iloc[0]
rul_pred = float(row["pred"])
ci_low   = float(row["ci_low"])
ci_high  = float(row["ci_high"])
rul_std  = float(row["std"])
ci_width = ci_high - ci_low
pct_used = (cycle - min_cycle) / (max_cycle - min_cycle)

if rul_pred > 80:
    zone, card, num_cl, badge, chip_cl, accent = (
        "SAFE",     "rul-safe", "safe-num", "✓  Engine is Safe",   "chip-safe", "#10b981")
elif rul_pred > 30:
    zone, card, num_cl, badge, chip_cl, accent = (
        "MONITOR",  "rul-warn", "warn-num", "⚠  Needs Monitoring", "chip-warn", "#f59e0b")
else:
    zone, card, num_cl, badge, chip_cl, accent = (
        "CRITICAL", "rul-crit", "crit-num", "✕  Action Required",  "chip-crit", "#ef4444")

# ══════════════════════════════════════════════════════════════════════════════
main_placeholder = st.empty()

with main_placeholder.container():

    # ── Row 1 ──────────────────────────────────────────────────────────────────
    g_col, rul_col, info_col = st.columns([1.2, 1, 1.6], gap="large")

    with g_col:
        max_rul = 220
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rul_pred,
            number=dict(font=dict(size=48, color=accent, family="Inter"), suffix=" cyc"),
            delta=dict(
                reference=float(df[df["cycle"] == min_cycle].iloc[0]["pred"]),
                decreasing=dict(color="#ef4444"),
                increasing=dict(color="#10b981"),
                font=dict(size=14),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, max_rul],
                    tickwidth=1, tickcolor="rgba(255,255,255,0.1)",
                    tickfont=dict(size=10, color="#475569"),
                    nticks=6,
                ),
                bar=dict(color=accent, thickness=0.22),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,  30],  color="rgba(239,68,68,0.08)"),
                    dict(range=[30, 80],  color="rgba(245,158,11,0.08)"),
                    dict(range=[80, max_rul], color="rgba(16,185,129,0.06)"),
                ],
                threshold=dict(
                    line=dict(color=accent, width=2),
                    thickness=0.75, value=rul_pred,
                ),
            ),
            title=dict(text="Remaining Useful Life",
                       font=dict(size=12, color="#475569", family="Inter")),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0a0f1e",
            font=dict(family="Inter"),
            height=300,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{cycle}")

    with rul_col:
        st.markdown(f"""
        <div class="rul-card {card}">
          <div class="rul-label">Predicted RUL</div>
          <div class="rul-number {num_cl}">{rul_pred:.0f}</div>
          <div class="rul-unit">cycles remaining</div>
          <div><span class="status-chip {chip_cl}">{badge}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with info_col:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="ibox">
              <div class="ibox-label">Best Case</div>
              <div class="ibox-value" style="color:#3b82f6;">{ci_high:.0f}</div>
              <div class="ibox-sub">upper bound (cycles)</div>
            </div>
            <div class="ibox">
              <div class="ibox-label">Uncertainty</div>
              <div class="ibox-value" style="color:#a78bfa;">{ci_width:.0f}</div>
              <div class="ibox-sub">range width (cycles)</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="ibox">
              <div class="ibox-label">Worst Case</div>
              <div class="ibox-value" style="color:#f87171;">{ci_low:.0f}</div>
              <div class="ibox-sub">lower bound (cycles)</div>
            </div>
            <div class="ibox">
              <div class="ibox-label">CI Coverage</div>
              <div class="ibox-value" style="color:#10b981;">90%</div>
              <div class="ibox-sub">confidence interval</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prog-wrap">
          <div class="prog-title">Engine Lifecycle &nbsp;·&nbsp; Cycle {cycle} / {max_cycle} &nbsp;({pct_used*100:.0f}% consumed)</div>
          <div class="prog-track">
            <div class="prog-fill" style="width:{pct_used*100:.1f}%; background:linear-gradient(90deg,{accent}99,{accent});"></div>
          </div>
          <div class="prog-labels"><span>Start</span><span>End of Life</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Main chart ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Remaining Useful Life — Prediction Timeline</div>',
                unsafe_allow_html=True)

    past   = df[df["cycle"] <= cycle]
    future = df[df["cycle"] >= cycle]

    fig = go.Figure()

    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(239,68,68,0.04)",  layer="below", line_width=0)
    fig.add_hrect(y0=30, y1=80,  fillcolor="rgba(245,158,11,0.04)", layer="below", line_width=0)
    fig.add_hrect(y0=80, y1=220, fillcolor="rgba(16,185,129,0.03)", layer="below", line_width=0)

    for y, label, color in [(15,  "CRITICAL", "rgba(239,68,68,0.5)"),
                             (55,  "MONITOR",  "rgba(245,158,11,0.5)"),
                             (115, "SAFE",     "rgba(16,185,129,0.5)")]:
        fig.add_annotation(x=max_cycle, y=y, text=label, showarrow=False,
                           font=dict(color=color, size=10, family="Inter"),
                           xanchor="right")

    fig.add_trace(go.Scatter(
        x=pd.concat([future["cycle"], future["cycle"][::-1]]),
        y=pd.concat([future["ci_high"], future["ci_low"][::-1]]),
        fill="toself", fillcolor="rgba(59,130,246,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Confidence Band", hoverinfo="skip",
    ))
    for c_col in ["ci_high", "ci_low"]:
        fig.add_trace(go.Scatter(
            x=future["cycle"], y=future[c_col], mode="lines",
            line=dict(color="rgba(59,130,246,0.25)", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

    fig.add_trace(go.Scatter(
        x=past["cycle"], y=past["pred"], mode="lines",
        line=dict(color="rgba(59,130,246,0.2)", width=1.5),
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=future["cycle"], y=future["pred"], mode="lines",
        name="Predicted RUL",
        line=dict(color="#3b82f6", width=2.5),
        hovertemplate="Cycle %{x}<br><b>RUL: %{y:.0f} cycles</b><br>Range: %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>",
        customdata=future[["ci_low", "ci_high"]].values,
    ))

    fig.add_vline(x=cycle, line_width=1.5, line_dash="solid",
                  line_color="rgba(255,255,255,0.2)",
                  annotation_text=f"  Cycle {cycle}",
                  annotation_font=dict(color="#64748b", size=11, family="Inter"),
                  annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=[cycle], y=[rul_pred], mode="markers",
        marker=dict(size=14, color=accent,
                    line=dict(color="#080d1a", width=3),
                    symbol="circle"),
        name=f"Now · {rul_pred:.0f} cycles",
        hovertemplate=f"<b>Cycle {cycle}</b><br>RUL: {rul_pred:.0f} cycles<br>Range: {ci_low:.0f}–{ci_high:.0f}<extra></extra>",
    ))

    fig.update_layout(
        **DARK, height=420,
        xaxis_title="Engine Cycle",
        yaxis_title="Cycles Remaining Until Failure",
        legend=dict(bgcolor="rgba(15,24,41,0.8)", bordercolor="rgba(255,255,255,0.08)",
                    borderwidth=1, font=dict(size=12, color="#94a3b8"),
                    x=0.01, y=0.99),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"main_{cycle}")

    # ── Bottom charts ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Uncertainty Analysis</div>',
                unsafe_allow_html=True)

    bl, br = st.columns(2, gap="large")
    ci_widths = df["ci_high"] - df["ci_low"]

    with bl:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["cycle"], y=df["std"], mode="lines", fill="tozeroy",
            fillcolor="rgba(167,139,250,0.06)", line=dict(color="#a78bfa", width=2),
            name="σ",
            hovertemplate="Cycle %{x}<br>σ = ±%{y:.1f} cycles<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=[cycle], y=[rul_std], mode="markers",
            marker=dict(size=12, color=accent, line=dict(color="#080d1a", width=2)),
            name=f"Now",
            hovertemplate=f"Cycle {cycle}<br>σ = ±{rul_std:.1f}<extra></extra>",
        ))
        fig2.add_vline(x=cycle, line_width=1, line_dash="dash",
                       line_color="rgba(255,255,255,0.1)")
        fig2.update_layout(
            **DARK, height=280,
            title=dict(text="Model Disagreement (Standard Deviation)",
                       font=dict(size=12, color="#475569", family="Inter")),
            xaxis_title="Engine Cycle",
            yaxis_title="σ (cycles)",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"std_{cycle}")
        st.markdown('<p style="font-size:0.72rem;color:#334155;margin-top:-10px;">Lower σ = stronger model consensus = higher confidence</p>', unsafe_allow_html=True)

    with br:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["cycle"], y=ci_widths, mode="lines", fill="tozeroy",
            fillcolor="rgba(59,130,246,0.06)", line=dict(color="#3b82f6", width=2),
            name="CI width",
            hovertemplate="Cycle %{x}<br>Width: %{y:.0f} cycles<extra></extra>",
        ))
        fig3.add_trace(go.Scatter(
            x=[cycle], y=[ci_width], mode="markers",
            marker=dict(size=12, color=accent, line=dict(color="#080d1a", width=2)),
            name="Now",
            hovertemplate=f"Cycle {cycle}<br>{ci_width:.0f} cycles wide<extra></extra>",
        ))
        fig3.add_vline(x=cycle, line_width=1, line_dash="dash",
                       line_color="rgba(255,255,255,0.1)")
        fig3.add_hline(y=ci_widths.mean(), line_dash="dot",
                       line_color="rgba(255,255,255,0.08)",
                       annotation_text=f"Avg: {ci_widths.mean():.0f}",
                       annotation_font=dict(color="#334155", size=10))
        fig3.update_layout(
            **DARK, height=280,
            title=dict(text="90% Confidence Interval Width",
                       font=dict(size=12, color="#475569", family="Inter")),
            xaxis_title="Engine Cycle",
            yaxis_title="CI Width (cycles)",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True, key=f"ci_{cycle}")
        st.markdown('<p style="font-size:0.72rem;color:#334155;margin-top:-10px;">Narrower band = more precise prediction</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
      <span>RF · XGB · SVR · LSTM · CNN · Transformer</span> &nbsp;·&nbsp;
      100 experiments &nbsp;·&nbsp; 90% CI calibrated on 195,900 validation samples &nbsp;·&nbsp;
      <span>WTA³ MAD-robust ensemble weighting</span>
    </div>
    """, unsafe_allow_html=True)

# ── Animation loop ────────────────────────────────────────────────────────────
if st.session_state.playing:
    time.sleep(delay)
    if st.session_state.cycle < max_cycle:
        st.session_state.cycle += 1
    else:
        st.session_state.playing = False
    st.rerun()
