"""StatArb Bot — Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py

Three pages (sidebar nav):
    Backtest   — run any of the 4 benchmark levels, see NAV + metrics
    Signals    — live z-scores heatmap and active positions
    Risk       — kill switch status, drawdown gauge, exposure breakdown
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make repo root importable when running from any working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.engine import BacktestConfig, BacktestEngine
from config.universe import TICKERS
from risk.config import load_risk_params
from risk.kill_switch import KillSwitch
from signals.generator import generate_signals

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StatArb Bot",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LEVEL_LABELS = {
    1: "Level 1 — Buy & Hold SPY",
    2: "Level 2 — Equal-Weight Sector",
    3: "Level 3 — Classic PCA Stat Arb",
    4: "Level 4 — Adaptive PCA + Macro Filter",
}

_PARQUET_PATH = "db/prices.parquet"


@st.cache_data(show_spinner=False)
def _load_real_returns(start: str, end: str) -> pd.DataFrame | None:
    """Load log-returns from the parquet store, or return None if unavailable."""
    if not os.path.exists(_PARQUET_PATH):
        return None
    try:
        from data.loader import load_returns
        from datetime import date as _date
        df = load_returns(TICKERS, _date.fromisoformat(start), _date.fromisoformat(end))
        # Strip UTC timezone so the engine can slice with tz-naive timestamps.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def _make_synthetic_returns(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Correlated synthetic log-returns that vaguely mimic equity behaviour."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n, k = len(dates), len(tickers)
    # One common factor + idiosyncratic noise → non-trivial PCA structure.
    market = rng.normal(0.0003, 0.010, size=(n, 1))
    betas = rng.uniform(0.5, 1.5, size=(1, k))
    idio = rng.normal(0.0, 0.008, size=(n, k))
    returns = market * betas + idio
    return pd.DataFrame(returns, index=dates, columns=tickers)


def _synthetic_spy(start: str, end: str, seed: int = 0) -> pd.Series:
    """Synthetic SPY log-return series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    r = rng.normal(0.0004, 0.010, size=len(dates))
    return pd.Series(r, index=dates, name="SPY")


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_x(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}x"


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Backtest", "Signals", "Risk Monitor"],
    index=0,
)

st.sidebar.markdown("---")
has_real_data = os.path.exists(_PARQUET_PATH)
if has_real_data:
    st.sidebar.success("Données réelles chargées")
else:
    st.sidebar.warning("Mode démo — données synthétiques")

# ---------------------------------------------------------------------------
# ─────────────────────────────  BACKTEST PAGE  ──────────────────────────────
# ---------------------------------------------------------------------------

if page == "Backtest":
    st.title("Backtest — 4 niveaux de benchmark")

    # ── Controls ──────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Paramètres")

        selected_levels = st.multiselect(
            "Niveaux à comparer",
            options=[1, 2, 3, 4],
            default=[1, 3, 4],
            format_func=lambda v: _LEVEL_LABELS[v],
        )

        start_date = st.date_input("Date de début", value=pd.Timestamp("2021-01-01"))
        end_date = st.date_input("Date de fin", value=pd.Timestamp("2026-06-16"))

        with st.expander("Paramètres avancés"):
            cost_bps = st.slider("Coût de transaction (bps)", 0, 20, 8)
            rebal_freq = st.selectbox("Fréquence de rebalancement", ["weekly", "monthly", "daily"], index=0)
            pca_window = st.slider("Fenêtre PCA (jours)", 30, 252, 60)
            n_components = st.slider("Composantes PCA", 2, 10, 5)
            entry_thr = st.slider("Seuil d'entrée |z|", 1.0, 3.5, 2.0, step=0.1)
            risk_free = st.slider("Taux sans risque (%)", 0.0, 5.0, 2.0, step=0.5) / 100

        run_btn = st.button("Lancer le backtest", type="primary", use_container_width=True)

    with col_r:
        if not run_btn:
            st.info("Sélectionne les paramètres et clique sur **Lancer le backtest**.")
            st.stop()

        if not selected_levels:
            st.warning("Sélectionne au moins un niveau.")
            st.stop()

        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        real_returns = _load_real_returns(start_str, end_str)
        demo_mode = real_returns is None

        if real_returns is not None:
            returns_df = real_returns
            # SPY proxy = equal-weight average of all tickers (market factor)
            spy_series = returns_df.mean(axis=1).rename("SPY_proxy")
            # VIX proxy = rolling 21-day realized vol of the portfolio × √252 × 100
            vix_series = (
                spy_series.rolling(21).std() * np.sqrt(252) * 100
            ).bfill()
            # Credit spread proxy: rough approximation from vol regime
            spread_series = (vix_series / 20.0).clip(0.5, 6.0)
        else:
            all_tickers = TICKERS[:10]
            returns_df = _make_synthetic_returns(all_tickers, start_str, end_str)
            spy_series = _synthetic_spy(start_str, end_str)
            vix_series = pd.Series(18.0, index=returns_df.index)
            spread_series = pd.Series(1.5, index=returns_df.index)
            st.info("Mode démo : données synthétiques (lance `data.refresh` pour les vraies données).")

        results: dict[int, object] = {}
        progress = st.progress(0)

        for i, lvl in enumerate(sorted(selected_levels)):
            cfg = BacktestConfig(
                level=lvl,
                start=start_date,
                end=end_date,
                initial_capital=100_000.0,
                transaction_cost_bps=cost_bps,
                rebalance_freq=rebal_freq,
                pca_window=pca_window,
                n_components=n_components,
                entry_threshold=entry_thr,
            )
            engine = BacktestEngine(cfg)

            if lvl == 1:
                r = engine.run(returns_df, spy_returns=spy_series)
            elif lvl == 4:
                r = engine.run(
                    returns_df,
                    vix_series=vix_series,
                    spread_series=spread_series,
                )
            else:
                r = engine.run(returns_df)

            results[lvl] = r
            progress.progress((i + 1) / len(selected_levels))

        progress.empty()

    # ── NAV chart ─────────────────────────────────────────────────────────
    fig = go.Figure()
    colors = {1: "#636EFA", 2: "#EF553B", 3: "#00CC96", 4: "#AB63FA"}

    for lvl, res in results.items():
        nav = res.portfolio_value
        fig.add_trace(go.Scatter(
            x=nav.index,
            y=nav.values,
            name=_LEVEL_LABELS[lvl],
            line=dict(color=colors[lvl], width=2),
        ))

    fig.update_layout(
        title="NAV normalisée (base 100k$)",
        xaxis_title="Date",
        yaxis_title="Valeur du portefeuille ($)",
        hovermode="x unified",
        height=420,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics table ──────────────────────────────────────────────────────
    st.subheader("Métriques de performance")
    rows = []
    for lvl, res in results.items():
        s = res.summary()
        sharpe = res.sharpe_ratio(risk_free_rate=risk_free)
        rows.append({
            "Niveau": _LEVEL_LABELS[lvl],
            "Rendement total": _fmt_pct(s["total_return"]),
            "CAGR": _fmt_pct(s["annualized_return"]),
            "Volatilité ann.": _fmt_pct(s["annualized_volatility"]),
            "Sharpe": f"{sharpe:.2f}" + (" ⚠️" if sharpe > 3.0 else ""),
            "Max Drawdown": _fmt_pct(s["max_drawdown"]),
            "Calmar": f"{s['calmar_ratio']:.2f}",
            "Turnover ann.": _fmt_x(s["turnover"]),
            "Hit rate": _fmt_pct(s["hit_rate"]),
            "Frais totaux ($)": f"{s['total_transaction_costs_dollars']:,.0f}",
            "Nb trades": s["n_trades"],
        })

    st.dataframe(pd.DataFrame(rows).set_index("Niveau"), use_container_width=True)

    # ── Drawdown chart ─────────────────────────────────────────────────────
    with st.expander("Drawdown par niveau"):
        fig_dd = go.Figure()
        for lvl, res in results.items():
            pv = res.portfolio_value
            peak = pv.cummax()
            dd = (pv - peak) / peak * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                name=_LEVEL_LABELS[lvl],
                line=dict(color=colors[lvl], width=1.5),
                fill="tozeroy",
                fillcolor=colors[lvl].replace(")", ", 0.08)").replace("rgb", "rgba") if "rgb" in colors[lvl] else colors[lvl],
            ))
        fig_dd.update_layout(
            title="Drawdown (%)",
            yaxis_title="Drawdown (%)",
            height=280,
            hovermode="x unified",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_dd, use_container_width=True)

# ---------------------------------------------------------------------------
# ──────────────────────────────  SIGNALS PAGE  ──────────────────────────────
# ---------------------------------------------------------------------------

elif page == "Signals":
    st.title("Signaux — Z-scores & positions actives")

    col_s, col_r = st.columns([1, 2])

    with col_s:
        st.subheader("Paramètres")
        level = st.selectbox(
            "Niveau de modèle",
            [3, 4],
            format_func=lambda v: _LEVEL_LABELS[v],
        )
        lookback_days = st.slider("Historique utilisé (jours)", 60, 504, 252)
        pca_w = st.slider("Fenêtre PCA", 30, 120, 60)
        z_window = st.slider("Fenêtre z-score", 10, 60, 21)
        entry_thr_s = st.slider("Seuil d'entrée |z|", 1.0, 3.5, 2.0, step=0.1)
        vix_val = st.slider("VIX (niveau actuel)", 10.0, 60.0, 18.0, step=0.5)
        spread_val = st.slider("Spread crédit (pp)", 0.5, 6.0, 1.5, step=0.1)

        compute_btn = st.button("Calculer les signaux", type="primary", use_container_width=True)

    with col_r:
        if not compute_btn:
            st.info("Ajuste les paramètres et clique sur **Calculer les signaux**.")
            st.stop()

        end_s = pd.Timestamp.today().normalize()
        start_s = end_s - pd.offsets.BDay(lookback_days)
        real_ret = _load_real_returns(start_s.date().isoformat(), end_s.date().isoformat())
        demo_sig = real_ret is None

        tickers_sig = TICKERS[:15]
        ret_sig = (
            real_ret[tickers_sig] if real_ret is not None and all(t in real_ret for t in tickers_sig)
            else _make_synthetic_returns(tickers_sig, start_s.date().isoformat(), end_s.date().isoformat())
        )

        if demo_sig:
            st.info("Mode démo : données synthétiques.")

        with st.spinner("Calcul des signaux…"):
            try:
                signals = generate_signals(
                    ret_sig,
                    level=level,
                    window=pca_w,
                    n_components=min(5, len(tickers_sig) - 1),
                    zscore_window=z_window,
                    entry_threshold=entry_thr_s,
                    vix=vix_val if level == 4 else None,
                    credit_spread=spread_val if level == 4 else None,
                )
            except Exception as e:
                st.error(f"Erreur lors du calcul des signaux : {e}")
                st.stop()

        # ── Macro filter status ────────────────────────────────────────
        if level == 4:
            col_v, col_c = st.columns(2)
            vix_color = "red" if vix_val >= 30 else ("orange" if vix_val >= 25 else "green")
            spread_color = "red" if spread_val >= 3.0 else ("orange" if spread_val >= 2.5 else "green")
            col_v.metric("VIX", f"{vix_val:.1f}", delta=None)
            col_c.metric("Spread crédit", f"{spread_val:.2f} pp", delta=None)
            if vix_val >= 30 or spread_val >= 3.0:
                st.error("Filtre macro actif — trading suspendu.")
            elif vix_val >= 25 or spread_val >= 2.5:
                st.warning("Filtre macro : poids réduits (zone de rampe).")
            else:
                st.success("Filtre macro inactif — trading autorisé.")

        # ── Active signals ─────────────────────────────────────────────
        st.subheader(f"{len(signals)} signal(s) actif(s)")
        if signals:
            rows_sig = []
            for ticker, info in sorted(signals.items(), key=lambda x: abs(x[1]["z_score"]), reverse=True):
                rows_sig.append({
                    "Ticker": ticker,
                    "Direction": ("🟢 Long" if info["direction"] == "long" else "🔴 Short"),
                    "Z-score": round(info["z_score"], 3),
                    "Poids cible": _fmt_pct(info["weight"]),
                })
            df_sig = pd.DataFrame(rows_sig).set_index("Ticker")
            st.dataframe(df_sig, use_container_width=True)
        else:
            st.info("Aucun signal actif au-dessus du seuil.")

        # ── Z-score bar chart ──────────────────────────────────────────
        if signals:
            z_vals = {t: info["z_score"] for t, info in signals.items()}
            colors_bar = ["#00CC96" if v < 0 else "#EF553B" for v in z_vals.values()]
            fig_z = go.Figure(go.Bar(
                x=list(z_vals.keys()),
                y=list(z_vals.values()),
                marker_color=colors_bar,
            ))
            fig_z.add_hline(y=entry_thr_s, line_dash="dot", line_color="gray", annotation_text="seuil +")
            fig_z.add_hline(y=-entry_thr_s, line_dash="dot", line_color="gray", annotation_text="seuil −")
            fig_z.update_layout(
                title="Z-scores des positions actives",
                yaxis_title="Z-score",
                height=300,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_z, use_container_width=True)

# ---------------------------------------------------------------------------
# ────────────────────────────  RISK MONITOR PAGE  ───────────────────────────
# ---------------------------------------------------------------------------

elif page == "Risk Monitor":
    st.title("Risk Monitor")

    params = load_risk_params()

    # ── Kill switch state ──────────────────────────────────────────────────
    st.subheader("Kill Switch")

    if "kill_switch" not in st.session_state:
        st.session_state.kill_switch = KillSwitch(params)
    if "nav_history" not in st.session_state:
        st.session_state.nav_history: list[float] = [100_000.0]

    ks: KillSwitch = st.session_state.kill_switch
    current_nav = st.session_state.nav_history[-1]

    col_ks1, col_ks2, col_ks3 = st.columns(3)

    status_label = "DÉCLENCHÉ" if ks.is_triggered else "Actif"
    status_color = "🔴" if ks.is_triggered else "🟢"
    col_ks1.metric("Statut", f"{status_color} {status_label}")
    col_ks2.metric(
        "Drawdown actuel",
        f"{ks.current_drawdown(current_nav) * 100:.2f}%" if ks.peak_value > 0 else "—",
    )
    col_ks3.metric(
        "Seuil kill switch",
        _fmt_pct(params["drawdown"]["kill_switch_threshold"]),
    )

    if ks.is_triggered:
        st.error("Kill switch déclenché — aucun nouveau trade autorisé. Réinitialiser manuellement.")
        if st.button("Reset Kill Switch", type="primary"):
            ks.reset()
            st.success("Kill switch réinitialisé.")
            st.rerun()
    else:
        st.success("Kill switch inactif — trading autorisé.")

    # ── Simulate NAV update ────────────────────────────────────────────────
    with st.expander("Simuler une mise à jour NAV"):
        new_nav = st.number_input(
            "Nouvelle valeur du portefeuille ($)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=float(st.session_state.nav_history[-1]),
            step=1000.0,
        )
        if st.button("Mettre à jour la NAV"):
            st.session_state.nav_history.append(new_nav)
            ks.check(new_nav)
            st.rerun()

    # ── NAV mini-chart ─────────────────────────────────────────────────────
    if len(st.session_state.nav_history) > 1:
        nav_s = pd.Series(st.session_state.nav_history)
        fig_nav = px.line(
            nav_s,
            labels={"index": "Tick", "value": "NAV ($)"},
            title="Évolution NAV (simulation)",
            color_discrete_sequence=["#636EFA"],
        )
        fig_nav.add_hline(
            y=st.session_state.nav_history[0] * (1 - params["drawdown"]["kill_switch_threshold"]),
            line_dash="dot",
            line_color="red",
            annotation_text="seuil kill switch",
        )
        fig_nav.add_hline(
            y=st.session_state.nav_history[0] * (1 - params["drawdown"]["warning_threshold"]),
            line_dash="dot",
            line_color="orange",
            annotation_text="alerte drawdown",
        )
        fig_nav.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig_nav, use_container_width=True)

    # ── Risk limits recap ──────────────────────────────────────────────────
    st.subheader("Limites de risque (config/risk_params.yaml)")

    col_p, col_d, col_sz = st.columns(3)
    with col_p:
        st.markdown("**Position**")
        st.write(f"Max par ticker : {_fmt_pct(params['position']['max_weight_per_ticker'])}")
        st.write(f"Exposition brute max : {_fmt_x(params['position']['max_gross_exposure'])}")
        st.write(f"Exposition nette max : {_fmt_pct(params['position']['max_net_exposure'])}")

    with col_d:
        st.markdown("**Drawdown**")
        st.write(f"Kill switch : {_fmt_pct(params['drawdown']['kill_switch_threshold'])}")
        st.write(f"Alerte : {_fmt_pct(params['drawdown']['warning_threshold'])}")

    with col_sz:
        st.markdown("**Sizing**")
        st.write(f"Fraction investie : {_fmt_pct(params['sizing']['portfolio_fraction'])}")
        st.write(f"Max positions : {params['sizing']['max_positions']}")
        st.write(f"Coût transaction : {params['transaction']['cost_bps']} bps")
