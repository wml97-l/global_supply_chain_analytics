"""
Build the README insight charts and print the headline figures.

    pip install pandas matplotlib scikit-learn imbalanced-learn xgboost
    python docs/make_charts.py

Writes light and dark PNGs to docs/charts/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "charts"
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", ink2="#59636e", grid="#d8dee4",
                  main="#2a78d6", accent="#eb6834", muted="#b7c0ca"),
    "dark": dict(bg="#0d1117", ink="#f0f6fc", ink2="#9198a1", grid="#30363d",
                 main="#3987e5", accent="#d95926", muted="#4a535e"),
}


def style(ax, t, grid_axis="x"):
    ax.set_facecolor(t["bg"])
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], length=0, labelsize=10.5)
    ax.tick_params(axis="y", colors=t["ink"], labelsize=11)
    ax.grid(axis=grid_axis, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def heading(fig, t, head, sub):
    fig.text(0.02, 0.965, head, fontsize=15, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.02, 0.885, sub, fontsize=11, color=t["ink2"], va="top")


def label(ax, t, x, y, text, bold=True, dx=6):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", va="center", fontsize=10.5,
                color=t["ink"], fontweight="bold" if bold else "normal",
                bbox=dict(facecolor=t["bg"], edgecolor="none", pad=1.5), zorder=4)


def save(fig, name, mode):
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def chart_weather(df, t, mode):
    order = ["Clear", "Rain", "Fog", "Storm", "Hurricane"]
    g = df.groupby("Weather_Condition").agg(lead=("Lead_Time_Days", "mean"),
                                            dis=("Disruption_Occurred", "mean")).loc[order]
    share = df.groupby("Weather_Condition").Lead_Time_Days.sum() / df.Lead_Time_Days.sum()
    n_share = df.Weather_Condition.value_counts(normalize=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), facecolor=t["bg"])
    fig.subplots_adjust(left=0.1, right=0.97, top=0.73, bottom=0.1, wspace=0.45)
    cols = [t["accent"] if w == "Hurricane" else t["main"] for w in order]
    ax = axes[0]
    ax.barh(order, g.lead, color=cols, height=0.58)
    for i, v in enumerate(g.lead):
        label(ax, t, v, i, f"{v:.1f} days")
    ax.set_xlim(0, g.lead.max() * 1.35)
    ax.set_title("Average lead time", loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=8)
    style(ax, t)
    ax = axes[1]
    ax.barh(order, g.dis * 100, color=cols, height=0.58)
    for i, v in enumerate(g.dis):
        label(ax, t, v * 100, i, f"{v:.0%}")
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(mt.PercentFormatter())
    ax.set_title("Shipments disrupted", loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=8)
    style(ax, t)
    heading(fig, t, f"Hurricanes are {n_share['Hurricane']:.0%} of shipments but {share['Hurricane']:.0%} of all lead-time days",
            "By weather condition at time of shipment (5,000 shipments)")
    save(fig, "weather-impact", mode)
    return g, share["Hurricane"], n_share["Hurricane"]


def chart_mode(df, t, mode):
    g = df.groupby("Transport_Mode").agg(lead=("Lead_Time_Days", "mean"),
                                         dis=("Disruption_Occurred", "mean")).sort_values("lead")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=t["bg"])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.70, bottom=0.11, wspace=0.4)
    ax = axes[0]
    ax.barh(g.index, g.lead, color=t["main"], height=0.58)
    for i, v in enumerate(g.lead):
        label(ax, t, v, i, f"{v:.1f} days")
    ax.set_xlim(0, g.lead.max() * 1.35)
    ax.set_title("Average lead time", loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=8)
    style(ax, t)
    ax = axes[1]
    ax.barh(g.index, g.dis * 100, color=t["muted"], height=0.58)
    for i, v in enumerate(g.dis):
        label(ax, t, v * 100, i, f"{v:.0%}")
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(mt.PercentFormatter())
    ax.set_title("Shipments disrupted", loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=8)
    style(ax, t)
    heading(fig, t, "Transport mode changes speed, not disruption risk",
            f"Sea takes {g.lead['Sea']/g.lead['Air']:.0f}x longer than air, yet every mode is disrupted about 61% of the time")
    save(fig, "transport-mode", mode)
    return g


def chart_risk(df, t, mode):
    bands = pd.cut(df.Geopolitical_Risk_Score, [-0.1, 2.5, 5, 7.5, 10], labels=["Low\n(0 to 2.5)", "Moderate\n(2.5 to 5)", "High\n(5 to 7.5)", "Severe\n(7.5 to 10)"])
    g = df.groupby(bands, observed=True).Disruption_Occurred.mean()
    fig, ax = plt.subplots(figsize=(10, 4.0), facecolor=t["bg"])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.17)
    shades = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab"] if mode == "light" else ["#184f95", "#256abf", "#3987e5", "#86b6ef"]
    x = np.arange(len(g))
    ax.bar(x, g * 100, color=shades, width=0.56)
    for i, v in enumerate(g):
        ax.text(i, v * 100 + 2.5, f"{v:.0%}", ha="center", fontsize=12, fontweight="bold", color=t["ink"])
    ax.set_xticks(x, g.index, color=t["ink"], fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mt.PercentFormatter())
    style(ax, t, grid_axis="y")
    ax.tick_params(axis="x", colors=t["ink"])
    heading(fig, t, f"Disruption rises from {g.iloc[0]:.0%} to {g.iloc[-1]:.0%} as geopolitical risk increases",
            "Share of shipments disrupted, by geopolitical risk score of the route")
    save(fig, "geopolitical-risk", mode)
    return g


def model(df):
    feats = ["Distance_km", "Weight_MT", "Fuel_Price_Index", "Geopolitical_Risk_Score",
             "Carrier_Reliability_Score", "Weather_Condition", "Transport_Mode"]
    X = pd.get_dummies(df[feats], columns=["Weather_Condition", "Transport_Mode"], drop_first=True)
    y = df.Disruption_Occurred
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    Xs, ys = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced").fit(Xs, ys)
    xg = XGBClassifier(random_state=42, eval_metric="logloss").fit(Xs, ys)
    scores = {}
    for name, m in [("Random Forest", rf), ("XGBoost", xg)]:
        p, pr = m.predict(Xte), m.predict_proba(Xte)[:, 1]
        scores[name] = dict(acc=accuracy_score(yte, p), f1=f1_score(yte, p, average="macro"),
                            auc=roc_auc_score(yte, pr),
                            recall=((p == 1) & (yte == 1)).sum() / (yte == 1).sum(),
                            precision=((p == 1) & (yte == 1)).sum() / (p == 1).sum())
    # grouped permutation importance (drop in ROC-AUC when a whole factor is shuffled)
    groups = {
        "Weather": [c for c in X.columns if c.startswith("Weather_")],
        "Geopolitical risk": ["Geopolitical_Risk_Score"],
        "Carrier reliability": ["Carrier_Reliability_Score"],
        "Transport mode": [c for c in X.columns if c.startswith("Transport_")],
        "Distance": ["Distance_km"], "Weight": ["Weight_MT"], "Fuel price": ["Fuel_Price_Index"],
    }
    base = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
    rng = np.random.default_rng(42)
    imp = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(20):
            Xp = Xte.copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xte[cols].values[idx]
            drops.append(base - roc_auc_score(yte, rf.predict_proba(Xp)[:, 1]))
        imp[g] = float(np.mean(drops))
    return scores, pd.Series(imp).sort_values(), (yte == 1).mean()


def chart_importance(imp, t, mode):
    fig, ax = plt.subplots(figsize=(10, 4.0), facecolor=t["bg"])
    fig.subplots_adjust(left=0.2, right=0.95, top=0.76, bottom=0.08)
    top2 = set(imp.index[-2:])
    cols = [t["main"] if i in top2 else t["muted"] for i in imp.index]
    vals = imp.clip(lower=0)
    ax.barh(imp.index, vals, color=cols, height=0.58)
    for i, v in enumerate(imp):
        label(ax, t, max(v, 0), i, f"{v:.3f}" if v >= 0.002 else "no measurable effect", bold=imp.index[i] in top2)
    ax.set_xlim(0, vals.max() * 1.25)
    style(ax, t)
    heading(fig, t, "Weather and geopolitical risk drive the disruption model",
            "Drop in Random Forest ROC-AUC when each factor is shuffled (higher = more important)")
    save(fig, "feature-importance", mode)


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "global_supply_chain_risk_2026.csv")
    scores, imp, base_rate = model(df)
    for mode, t in THEMES.items():
        w, hshare, hn = chart_weather(df, t, mode)
        m = chart_mode(df, t, mode)
        r = chart_risk(df, t, mode)
        chart_importance(imp, t, mode)
    print(f"shipments {len(df):,}  disrupted {df.Disruption_Occurred.mean():.1%}  mean lead {df.Lead_Time_Days.mean():.1f}  median {df.Lead_Time_Days.median():.1f}")
    print(w.round(3)); print(f"hurricane share of shipments {hn:.1%}, of lead-time days {hshare:.1%}")
    print(m.round(3)); print(r.round(3))
    print(f"majority-class baseline accuracy {base_rate:.1%}")
    for k, v in scores.items():
        print(k, {a: round(b, 3) for a, b in v.items()})
    print(imp.round(4))
