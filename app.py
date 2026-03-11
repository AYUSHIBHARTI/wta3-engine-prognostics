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
  .stApp, [data-testid="stAppViewContainer"] { background-color: #f5f7fa !important; }
  [data-testid="stHeader"] { background-color: #f5f7fa !important; }

  .stSlider label { font-size: 1rem !important; font-weight: 600 !important; color: #1a1a2e !important; }

  .rul-card { border-radius: 20px; padding: 30px 24px; text-align: center;
              box-shadow: 0 4px 20px rgba(0,0,0,0.10); }
  .rul-green  { background: #e8f5e9; border: 2px solid #43a047; }
  .rul-yellow { background: #fff8e1; border: 2px solid #ffa000; }
  .rul-red    { background: #ffebee; border: 2px solid #e53935; }

  .rul-top    { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.12em;
                text-transform: uppercase; color: #666; margin-bottom: 6px; }
  .rul-number { font-size: 4.8rem; font-weight: 800; line-height: 1; }
  .green-num  { color: #2e7d32; }
  .yellow-num { color: #e65100; }
  .red-num    { color: #c62828; }
  .rul-bottom { font-size: 0.9rem; color: #666; margin-top: 6px; }

  .status-badge { display:inline-block; border-radius:30px; padding:6px 20px;
                  font-size:0.95rem; font-weight:700; margin-top:12px; }
  .badge-green  { background:#43a047; color:white; }
  .badge-yellow { background:#ffa000; color:white; }
  .badge-red    { background:#e53935; color:white; }

  .info-box { background:white; border-radius:14px; padding:18px 20px;
              box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-bottom:12px; }
  .info-title { font-size:0.72rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.1em; color:#999; margin-bottom:4px; }
  .info-value { font-size:1.9rem; font-weight:800; color:#1a1a2e; line-height:1.1; }
  .info-sub   { font-size:0.8rem; color:#888; margin-top:3px; }

  .prog-wrap  { background:white; border-radius:14px; padding:18px 22px;
                box-shadow:0 2px 10px rgba(0,0,0,0.07); }
  .prog-title { font-size:0.88rem; font-weight:700; color:#333; margin-bottom:8px; }
  .prog-track { background:#e0e0e0; border-radius:999px; height:18px; position:relative; }
  .prog-fill  { border-radius:999px; height:18px; transition: width 0.3s; }
  .prog-label { display:flex; justify-content:space-between;
                font-size:0.75rem; color:#aaa; margin-top:5px; }

  .ctrl-btn { text-align:center; }

  .sec-title { font-size:1.05rem; font-weight:700; color:#1a1a2e;
               border-left:4px solid #1565c0; padding-left:12px;
               margin:20px 0 10px 0; }

  /* Play button styling */
  div[data-testid="stButton"] button {
    border-radius: 30px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 10px 28px !important;
    border: none !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_preds():
    df = pd.read_csv(RESULTS_DIR / "meta_ensemble_predictions.csv")
    df.columns = ["cycle", "true_RUL", "pred", "std", "ci_low", "ci_high", "abs_error"]
    return df

df        = load_preds()
min_cycle = int(df["cycle"].min())
max_cycle = int(df["cycle"].max())

# ── Session state ─────────────────────────────────────────────────────────────
if "cycle"   not in st.session_state: st.session_state.cycle   = min_cycle
if "playing" not in st.session_state: st.session_state.playing = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:white; border-radius:16px; padding:20px 28px;
            box-shadow:0 2px 10px rgba(0,0,0,0.07); margin-bottom:20px;">
  <div style="font-size:1.5rem; font-weight:800; color:#1a1a2e;">⚙️ Engine Health Monitor</div>
  <div style="font-size:0.88rem; color:#888; margin-top:3px;">
    Drag the cycle slider — or press <b>▶ Play</b> to simulate the engine aging in real time
  </div>
</div>
""", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
ctrl_col, speed_col, reset_col = st.columns([1, 2, 1])

with ctrl_col:
    if st.session_state.playing:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.playing = False
            st.rerun()
    else:
        if st.button("▶ Play", use_container_width=True):
            st.session_state.playing = True
            st.rerun()

with speed_col:
    speed = st.select_slider(
        "Animation Speed",
        options=["Slow", "Normal", "Fast", "Very Fast"],
        value="Normal",
        label_visibility="collapsed",
    )
    speed_map = {"Slow": 0.30, "Normal": 0.15, "Fast": 0.07, "Very Fast": 0.02}
    delay = speed_map[speed]

with reset_col:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.cycle   = min_cycle
        st.session_state.playing = False
        st.rerun()

# ── Slider (synced with session state) ───────────────────────────────────────
selected_cycle = st.slider(
    "Current Engine Cycle",
    min_value=min_cycle,
    max_value=max_cycle,
    value=st.session_state.cycle,
    step=1,
    key="slider_val",
)
# If user drags slider manually, sync
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
    zone, card, num_cl, badge, badge_cl, bar_color, gauge_color = (
        "SAFE",     "rul-green",  "green-num",  "✅ Engine is Safe",    "badge-green",  "#43a047", "#43a047")
elif rul_pred > 30:
    zone, card, num_cl, badge, badge_cl, bar_color, gauge_color = (
        "MONITOR",  "rul-yellow", "yellow-num", "⚠️ Needs Monitoring",  "badge-yellow", "#ffa000", "#ffa000")
else:
    zone, card, num_cl, badge, badge_cl, bar_color, gauge_color = (
        "CRITICAL", "rul-red",    "red-num",    "🚨 Action Required",   "badge-red",    "#e53935", "#e53935")

LIGHT = dict(
    paper_bgcolor="white", plot_bgcolor="#fafbfc",
    font=dict(color="#333", family="Arial", size=13),
    xaxis=dict(gridcolor="#eeeeee", showline=True, linecolor="#cccccc", zeroline=False),
    yaxis=dict(gridcolor="#eeeeee", showline=True, linecolor="#cccccc", zeroline=False),
    margin=dict(l=55, r=30, t=40, b=55),
)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT (all inside placeholders so animation updates them)
# ══════════════════════════════════════════════════════════════════════════════
main_placeholder = st.empty()

with main_placeholder.container():

    # ── Row 1: Gauge + RUL card + info boxes ──────────────────────────────────
    g_col, rul_col, info_col = st.columns([1.2, 1, 1.6], gap="large")

    # Speedometer gauge
    with g_col:
        max_rul = 220
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rul_pred,
            number=dict(font=dict(size=52, color=gauge_color), suffix=" cycles"),
            delta=dict(
                reference=float(df[df["cycle"] == min_cycle].iloc[0]["pred"]),
                decreasing=dict(color="#e53935"),
                increasing=dict(color="#43a047"),
                font=dict(size=16),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, max_rul],
                    tickwidth=1,
                    tickcolor="#ccc",
                    tickfont=dict(size=11, color="#888"),
                    nticks=8,
                ),
                bar=dict(color=gauge_color, thickness=0.25),
                bgcolor="white",
                borderwidth=0,
                steps=[
                    dict(range=[0,   30],  color="#ffebee"),
                    dict(range=[30,  80],  color="#fff8e1"),
                    dict(range=[80,  max_rul], color="#e8f5e9"),
                ],
                threshold=dict(
                    line=dict(color="#333", width=3),
                    thickness=0.75,
                    value=rul_pred,
                ),
            ),
            title=dict(text="Remaining Useful Life", font=dict(size=14, color="#555")),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="white",
            font=dict(family="Arial"),
            height=300,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{cycle}")

    # RUL text card
    with rul_col:
        st.markdown(f"""
        <div class="rul-card {card}">
          <div class="rul-top">Status</div>
          <div class="rul-number {num_cl}">{rul_pred:.0f}</div>
          <div class="rul-bottom">cycles remaining</div>
          <div><span class="status-badge {badge_cl}">{badge}</span></div>
        </div>
        """, unsafe_allow_html=True)

    # Info boxes
    with info_col:
        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="info-box">
              <div class="info-title">Best Case</div>
              <div class="info-value" style="color:#1565c0;">{ci_high:.0f}</div>
              <div class="info-sub">cycles (upper bound)</div>
            </div>
            <div class="info-box">
              <div class="info-title">Uncertainty Range</div>
              <div class="info-value" style="color:#6a1b9a;">{ci_width:.0f}</div>
              <div class="info-sub">cycles wide</div>
            </div>
            """, unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="info-box">
              <div class="info-title">Worst Case</div>
              <div class="info-value" style="color:#c62828;">{ci_low:.0f}</div>
              <div class="info-sub">cycles (lower bound)</div>
            </div>
            <div class="info-box">
              <div class="info-title">Confidence</div>
              <div class="info-value" style="color:#2e7d32;">90%</div>
              <div class="info-sub">range coverage</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prog-wrap">
          <div class="prog-title">Engine Lifecycle — Cycle {cycle} of {max_cycle} &nbsp;({pct_used*100:.0f}% used)</div>
          <div class="prog-track">
            <div class="prog-fill" style="width:{pct_used*100:.1f}%; background:{bar_color};"></div>
          </div>
          <div class="prog-label"><span>Start</span><span>End of Life</span></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Main prediction chart ──────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📈 Remaining Useful Life Over Time</div>',
                unsafe_allow_html=True)

    past   = df[df["cycle"] <= cycle]
    future = df[df["cycle"] >= cycle]

    fig = go.Figure()

    # Zone bands
    fig.add_hrect(y0=0,   y1=30,  fillcolor="rgba(229,57,53,0.07)",  layer="below", line_width=0)
    fig.add_hrect(y0=30,  y1=80,  fillcolor="rgba(255,160,0,0.07)",  layer="below", line_width=0)
    fig.add_hrect(y0=80,  y1=220, fillcolor="rgba(67,160,71,0.05)",  layer="below", line_width=0)
    for y, label, color in [(15, "Critical", "rgba(229,57,53,0.45)"),
                             (55, "Warning",  "rgba(255,160,0,0.45)"),
                             (115,"Safe",     "rgba(67,160,71,0.45)")]:
        fig.add_annotation(x=max_cycle, y=y, text=label, showarrow=False,
                           font=dict(color=color, size=11), xanchor="right")

    # CI band — future only
    fig.add_trace(go.Scatter(
        x=pd.concat([future["cycle"], future["cycle"][::-1]]),
        y=pd.concat([future["ci_high"], future["ci_low"][::-1]]),
        fill="toself", fillcolor="rgba(21,101,192,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% Confidence Range", hoverinfo="skip",
    ))
    for c_col in ["ci_high", "ci_low"]:
        fig.add_trace(go.Scatter(
            x=future["cycle"], y=future[c_col], mode="lines",
            line=dict(color="rgba(21,101,192,0.35)", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

    # Past prediction (faded)
    fig.add_trace(go.Scatter(
        x=past["cycle"], y=past["pred"], mode="lines",
        line=dict(color="rgba(21,101,192,0.3)", width=2),
        hoverinfo="skip", showlegend=False,
    ))

    # Future prediction (bold)
    fig.add_trace(go.Scatter(
        x=future["cycle"], y=future["pred"], mode="lines",
        name="Predicted RUL",
        line=dict(color="#1565c0", width=3),
        hovertemplate="Cycle %{x}<br><b>Predicted: %{y:.0f} cycles left</b><br>Range: %{customdata[0]:.0f}–%{customdata[1]:.0f}<extra></extra>",
        customdata=future[["ci_low", "ci_high"]].values,
    ))

    # Current position
    fig.add_vline(x=cycle, line_width=2, line_dash="solid", line_color="#555",
                  annotation_text=f"  Now (Cycle {cycle})",
                  annotation_font=dict(color="#333", size=11),
                  annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=[cycle], y=[rul_pred], mode="markers",
        marker=dict(size=16, color=bar_color, line=dict(color="white", width=3)),
        name=f"Now: {rul_pred:.0f} cycles left",
        hovertemplate=f"<b>Cycle {cycle}</b><br>Predicted: {rul_pred:.0f} cycles left<br>Range: {ci_low:.0f}–{ci_high:.0f}<extra></extra>",
    ))

    fig.update_layout(
        **LIGHT, height=400,
        xaxis_title="Engine Cycle (time)",
        yaxis_title="Cycles Remaining Until Failure",
        legend=dict(bgcolor="white", bordercolor="#ddd", borderwidth=1,
                    font=dict(size=12), x=0.01, y=0.99),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"main_{cycle}")

    # ── Bottom charts ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">📊 How Uncertainty Changes Over Time</div>',
                unsafe_allow_html=True)

    bl, br = st.columns(2, gap="large")
    ci_widths = df["ci_high"] - df["ci_low"]

    with bl:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["cycle"], y=df["std"], mode="lines", fill="tozeroy",
            fillcolor="rgba(106,27,154,0.08)", line=dict(color="#6a1b9a", width=2.5),
            name="Model disagreement (σ)",
            hovertemplate="Cycle %{x}<br>Disagreement: ±%{y:.1f} cycles<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=[cycle], y=[rul_std], mode="markers",
            marker=dict(size=14, color=bar_color, line=dict(color="white", width=2)),
            name=f"Now: ±{rul_std:.1f} cycles",
        ))
        fig2.add_vline(x=cycle, line_width=1.5, line_dash="dash", line_color="#999")
        fig2.update_layout(
            **LIGHT, height=280,
            title=dict(text="How much do the 6 models disagree with each other?",
                       font=dict(size=12, color="#666")),
            xaxis_title="Engine Cycle",
            yaxis_title="Standard Deviation (cycles)",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True, key=f"std_{cycle}")
        st.caption("Lower = models agree more = more confident prediction")

    with br:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["cycle"], y=ci_widths, mode="lines", fill="tozeroy",
            fillcolor="rgba(21,101,192,0.08)", line=dict(color="#1565c0", width=2.5),
            name="CI width",
            hovertemplate="Cycle %{x}<br>Range: %{y:.0f} cycles wide<extra></extra>",
        ))
        fig3.add_trace(go.Scatter(
            x=[cycle], y=[ci_width], mode="markers",
            marker=dict(size=14, color=bar_color, line=dict(color="white", width=2)),
            name=f"Now: {ci_width:.0f} cycles wide",
        ))
        fig3.add_vline(x=cycle, line_width=1.5, line_dash="dash", line_color="#999")
        fig3.add_hline(y=ci_widths.mean(), line_dash="dot", line_color="#bbb",
                       annotation_text=f"Average: {ci_widths.mean():.0f}",
                       annotation_font=dict(color="#aaa", size=11))
        fig3.update_layout(
            **LIGHT, height=280,
            title=dict(text="How wide is the uncertainty window?",
                       font=dict(size=12, color="#666")),
            xaxis_title="Engine Cycle",
            yaxis_title="Uncertainty Range Width (cycles)",
            showlegend=False,
        )
        st.plotly_chart(fig3, use_container_width=True, key=f"ci_{cycle}")
        st.caption("Narrower = more precise prediction")

    st.markdown("""
    <div style="text-align:center; color:#ccc; font-size:0.75rem;
                margin-top:20px; border-top:1px solid #e8e8e8; padding-top:12px;">
      6 models combined (RF · XGB · SVR · LSTM · CNN · Transformer) ·
      100 experiments · 90% confidence interval calibrated on 195,900 validation samples
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
