"""Tomorrow's Air — a public PM2.5 forecast tool for New Zealand.

Answers one question: will tomorrow's PM2.5 exceed the WHO 24-hour guideline
of 15 µg/m³?  Uses the tuned XGBoost classifier from airQualityForecasting.ipynb.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
THRESHOLD = 15  # µg/m³, WHO 24-hour guideline
DECISION_CUTOFF = 0.5  # the notebook's final model uses predict() at 0.5

st.set_page_config(page_title="Tomorrow's Air", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------- styling --
st.markdown(
    """
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
.hero {border-radius: 16px; padding: 1.4rem 1.6rem; border: 1px solid var(--edge);
       background: var(--tint);}
.hero .label {font-size: .8rem; letter-spacing: .02em; opacity: .75;}
.hero .big {font-size: 3.6rem; font-weight: 700; line-height: 1.05; color: var(--ink);}
.hero .band {font-size: 1.25rem; font-weight: 650; color: var(--ink); margin-bottom: .4rem;}
.hero p {margin: .3rem 0 0 0;}
.meter {position: relative; height: 12px; border-radius: 6px; margin: 1rem 0 .3rem 0;
        background: linear-gradient(90deg, #2e9d6a 0%, #2e9d6a 20%, #e0a82e 20%, #e0a82e 50%,
                    #e0742e 50%, #e0742e 75%, #c8372d 75%, #c8372d 100%);}
.meter .pin {position: absolute; top: -5px; width: 4px; height: 22px; border-radius: 2px;
             background: currentColor; box-shadow: 0 0 0 2px var(--pinring);}
.ticks {display:flex; justify-content: space-between; font-size: .72rem; opacity: .65;}
.muted {opacity: .7; font-size: .9rem;}
@media (prefers-color-scheme: dark) {.hero {--pinring: #0e1117;}}
@media (prefers-color-scheme: light) {.hero {--pinring: #ffffff;}}
</style>
""",
    unsafe_allow_html=True,
)

BANDS = [  # (upper bound, name, ink, tint, edge, headline, advice)
    (0.20, "Low risk", "#1f7a50", "rgba(46,157,106,.12)", "rgba(46,157,106,.45)",
     "Tomorrow should stay within the WHO guideline.",
     "Nothing special needed — a good day to ventilate the house."),
    (0.50, "Elevated", "#a97a0c", "rgba(224,168,46,.14)", "rgba(224,168,46,.5)",
     "Tomorrow could go either way.",
     "Worth checking the forecast again tomorrow morning, especially if anyone at home has asthma or a heart condition."),
    (0.75, "High risk", "#b8561a", "rgba(224,116,46,.14)", "rgba(224,116,46,.5)",
     "Tomorrow is likely to exceed the WHO guideline.",
     "Consider shorter or gentler outdoor exercise, keep windows closed in the evening, and keep any reliever medication handy."),
    (1.01, "Very high risk", "#b02a22", "rgba(200,55,45,.14)", "rgba(200,55,45,.5)",
     "Tomorrow is very likely to exceed the WHO guideline.",
     "Limit strenuous outdoor activity, keep windows closed, and check on neighbours who are older or have lung or heart conditions."),
]


def band_for(p: float):
    return next(b for b in BANDS if p < b[0])


# ---------------------------------------------------------------- model ----
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(str(ROOT / "model" / "air_quality_model.json"))
    features = json.loads((ROOT / "model" / "feature_columns.json").read_text())
    ratios = {int(k): v for k, v in json.loads((ROOT / "data" / "derivation_ratios.json").read_text()).items()}
    return model, features, ratios


@st.cache_data
def load_history():
    df = pd.read_csv(ROOT / "data" / "history.csv.gz", parse_dates=["date"])
    return df


MODEL, FEATURES, RATIOS = load_model()
HISTORY = load_history()
TRAINED_SITES = ["Anzac Square", "Rangiora", "Hampstead", "Masterton West (Pownall Street)"]


def season_of(month: int) -> int:  # dataset convention: 1 Summer … 4 Spring (Dec–Feb = summer)
    return 1 if month in (12, 1, 2) else 2 if month in (3, 4, 5) else 3 if month in (6, 7, 8) else 4


def build_features(a: dict) -> dict:
    """Turn the ~10 public answers into the 19 model features."""
    month = a["date"].month
    season = season_of(month)
    r = RATIOS[season]
    pm25 = a["pm25"]
    # A 7-day maximum can't be lower than today's reading, and a day over 15 counts as an exceedance.
    max7 = max(a["max7"], pm25)
    days_over = max(a["days_over"], 1 if pm25 > THRESHOLD else 0)
    pm10 = a["pm10"] if a.get("pm10") else max(pm25 * r["pm10_per_pm25"], pm25)
    return {
        "pm10_ugm3": pm10,
        "pm25_ugm3": pm25,
        "pm25_rollmean_7d": a["avg7"],
        "pm25_rollmean_14d": a["avg14"],
        "exceedance_count_14d": float(days_over),
        "pm25_rollmax_7d": max7,
        "pm10_mean7": a["avg7"] * r["pm10_mean7_per_pm25_mean7"],
        "pm10_max7": max7 * r["pm10_max7_per_pm25_max7"],
        "season_num": season,
        "cold_day": int(a["tmax"] > 16),  # same definition as the notebook
        "pressure_change_1d": a["pdelta"],
        "pm25_fraction": pm25 / pm10 if pm10 > 0 else 0.5,
        "month": month,
        "temp_max_c": a["tmax"],
        "temp_min_c": a["tmin"],
        "wind_max_kmh": a["wind"],
        "precip_mm": a["rain"],
        "pressure_hpa": a["pressure"],
        "humidity_pct": a["humidity"],
    }


def predict(rows: list[dict]) -> list[float]:
    X = pd.DataFrame(rows, columns=FEATURES).astype(float)
    return MODEL.predict_proba(X)[:, 1].tolist()


GROUPS = {
    "Recent PM2.5 and PM10 levels": ["pm10_ugm3", "pm25_ugm3", "pm25_rollmean_7d", "pm25_rollmean_14d",
                                     "exceedance_count_14d", "pm25_rollmax_7d", "pm10_mean7", "pm10_max7",
                                     "pm25_fraction"],
    "Temperature": ["temp_max_c", "temp_min_c", "cold_day"],
    "Wind, rain and humidity": ["wind_max_kmh", "precip_mm", "humidity_pct"],
    "Air pressure": ["pressure_hpa", "pressure_change_1d"],
    "Time of year": ["month", "season_num"],
}


def explain(row: dict) -> pd.DataFrame:
    """Per-feature push (log-odds) from XGBoost, rolled up into plain-language groups."""
    X = pd.DataFrame([row], columns=FEATURES).astype(float)
    contribs = MODEL.get_booster().predict(xgb.DMatrix(X, feature_names=FEATURES), pred_contribs=True)[0]
    by_feat = dict(zip(FEATURES, contribs[:-1]))
    return pd.DataFrame(
        [{"group": g, "push": sum(by_feat[f] for f in fs)} for g, fs in GROUPS.items()]
    ).sort_values("push")


# ---------------------------------------------------------------- inputs ---
DEFAULTS = dict(
    date=date.today(), pm25=8.0, avg7=8.0, max7=12.0, avg14=8.0, days_over=0,
    tmax=12.0, tmin=3.0, wind=15.0, humidity=80,
    rain=0.0, pressure=1010.0, pdelta=0.0, use_pm10=False, pm10=15.0,
)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def load_real_day():
    site, day = st.session_state["ex_site"], pd.Timestamp(st.session_state["ex_date"])
    row = HISTORY[(HISTORY.site_name == site) & (HISTORY.date == day)]
    if row.empty:
        st.session_state["ex_msg"] = "No reading for that site on that date — try another day."
        return
    r = row.iloc[0]
    need = ["pm25_ugm3", "pm10_ugm3", "pm25_rollmean_7d", "pm25_rollmean_14d", "pm25_rollmax_7d",
            "exceedance_count_14d", "temp_max_c", "temp_min_c", "wind_max_kmh", "humidity_pct",
            "precip_mm", "pressure_hpa"]
    if r[need].isna().any():
        st.session_state["ex_msg"] = "That day has missing readings — try another day."
        return
    clip = lambda v, lo, hi: float(min(max(v, lo), hi))
    st.session_state.update(
        date=day.date(), pm25=clip(r.pm25_ugm3, 0, 80), avg7=clip(r.pm25_rollmean_7d, 0, 60),
        max7=clip(r.pm25_rollmax_7d, 0, 80), avg14=clip(r.pm25_rollmean_14d, 0, 60),
        days_over=int(r.exceedance_count_14d), tmax=clip(r.temp_max_c, -5, 35),
        tmin=clip(r.temp_min_c, -10, 25), wind=clip(r.wind_max_kmh, 0, 100),
        humidity=int(clip(r.humidity_pct, 20, 100)), rain=clip(r.precip_mm, 0, 100),
        pressure=clip(r.pressure_hpa, 950, 1040),
        pdelta=clip(0 if pd.isna(r.pressure_change_1d) else r.pressure_change_1d, -25, 25),
        use_pm10=True, pm10=clip(r.pm10_ugm3, 0, 200),
        ex_msg="",
        ex_actual=None if pd.isna(r.target_exceeds_tomorrow) else int(r.target_exceeds_tomorrow),
    )


def reset_all():
    st.session_state.update(DEFAULTS)
    st.session_state["ex_msg"] = ""
    st.session_state["ex_actual"] = None


# ---------------------------------------------------------------- layout ---
st.title("Tomorrow's Air")
st.markdown(
    "Will **tomorrow's PM2.5** go over the **WHO guideline of 15 µg/m³**? "
    "Answer a few questions about today and the past week and the model will estimate the risk. "
    "Slide things around to see what would change the answer."
)

tab_forecast, tab_explore, tab_about = st.tabs(["Forecast", "Explore the data", "About the model"])

with tab_forecast:
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Your answers")
        st.date_input("1 · What's today's date?", key="date",
                      help="Used for the time of year — winter heating is the big driver of PM2.5 in NZ.")

        st.markdown("**Today and the recent past**  \n<span class='muted'>Your local monitor on LAWA or the council air-quality page shows these.</span>",
                    unsafe_allow_html=True)
        st.slider("2 · Today's PM2.5 (µg/m³)", 0.0, 80.0, key="pm25", step=0.5)
        st.slider("3 · Average PM2.5 over the last 7 days", 0.0, 60.0, key="avg7", step=0.5)
        st.slider("4 · Highest daily PM2.5 in the last 7 days", 0.0, 80.0, key="max7", step=0.5)
        st.slider("5 · Average PM2.5 over the last 14 days", 0.0, 60.0, key="avg14", step=0.5)
        st.slider("6 · Days over 15 µg/m³ in the last 14", 0, 14, key="days_over")

        st.markdown("**Today's weather**")
        c1, c2 = st.columns(2)
        c1.slider("7 · Highest temp (°C)", -5.0, 35.0, key="tmax", step=0.5)
        c2.slider("8 · Lowest temp (°C)", -10.0, 25.0, key="tmin", step=0.5)
        c1.slider("9 · Top wind speed (km/h)", 0.0, 100.0, key="wind", step=1.0)
        c2.slider("10 · Humidity (%)", 20, 100, key="humidity")

        with st.expander("Fine-tune (optional — sensible defaults are used)"):
            st.number_input("Rain today (mm)", 0.0, 100.0, key="rain", step=0.5)
            st.number_input("Air pressure (hPa)", 950.0, 1040.0, key="pressure", step=1.0)
            st.number_input("Pressure change since yesterday (hPa)", -25.0, 25.0, key="pdelta", step=0.5,
                            help="Positive = rising, usually settled weather.")
            st.checkbox("I know today's PM10 reading", key="use_pm10")
            if st.session_state["use_pm10"]:
                st.number_input("Today's PM10 (µg/m³)", 0.0, 200.0, key="pm10", step=0.5)

        with st.expander("Try a real day from the data"):
            sites = sorted(HISTORY.site_name.unique())
            st.selectbox("Monitoring site", sites, key="ex_site",
                         index=sites.index("Anzac Square"))
            site_dates = HISTORY[HISTORY.site_name == st.session_state["ex_site"]].date
            st.date_input("Day", value=pd.Timestamp("2024-07-10").date(), key="ex_date",
                          min_value=site_dates.min().date(), max_value=site_dates.max().date())
            st.button("Load this day", on_click=load_real_day, width="stretch")
            if st.session_state.get("ex_msg"):
                st.warning(st.session_state["ex_msg"])
        st.button("Reset all answers", on_click=reset_all)

    # ----- compute
    s = st.session_state
    answers = dict(
        date=s["date"], pm25=s["pm25"], avg7=s["avg7"], max7=s["max7"], avg14=s["avg14"],
        days_over=s["days_over"], tmax=s["tmax"], tmin=s["tmin"], wind=s["wind"],
        humidity=s["humidity"], rain=s["rain"], pressure=s["pressure"], pdelta=s["pdelta"],
        pm10=s["pm10"] if s["use_pm10"] else None,
    )
    row = build_features(answers)
    prob = predict([row])[0]
    name, ink, tint, edge, headline, advice = band_for(prob)[1:]

    with right:
        st.subheader("Tomorrow's forecast")
        st.markdown(
            f"""
<div class="hero" style="--ink:{ink}; --tint:{tint}; --edge:{edge}; color:{ink};">
  <div class="label">Risk that tomorrow exceeds {THRESHOLD} µg/m³</div>
  <div class="big">{prob:.0%}</div>
  <div class="band">{name}</div>
  <div class="meter"><div class="pin" style="left: calc({min(prob, 1) * 100:.1f}% - 2px);"></div></div>
  <div class="ticks"><span>0%</span><span>20%</span><span>50%</span><span>75%</span><span>100%</span></div>
  <p style="color: inherit;"><b>{headline}</b></p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(f"<p class='muted' style='margin-top:.8rem'>{advice} "
                    "<em>General information only, not medical advice.</em></p>", unsafe_allow_html=True)

        actual = s.get("ex_actual")
        if actual is not None and s.get("use_pm10"):
            st.info(
                f"**What actually happened:** on the day after this reading, PM2.5 "
                f"{'did exceed' if actual else 'did not exceed'} {THRESHOLD} µg/m³. "
                f"The model said {'exceed' if prob >= DECISION_CUTOFF else 'no exceedance'}."
            )

        # ----- what's driving it
        st.markdown("#### What's pushing the risk")
        drivers = explain(row)
        fig = go.Figure(go.Bar(
            x=drivers.push, y=drivers.group, orientation="h",
            marker_color=["#c8372d" if v > 0 else "#2e9d6a" for v in drivers.push],
            hovertemplate="%{y}<extra></extra>",
        ))
        fig.update_layout(height=250, margin=dict(l=0, r=10, t=10, b=30), showlegend=False,
                          xaxis=dict(title="← lowers risk     raises risk →", zeroline=True,
                                     showticklabels=False),
                          yaxis=dict(automargin=True))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        top = drivers.iloc[-1]
        bottom = drivers.iloc[0]
        st.caption(
            f"Biggest push up: **{top.group}**. Biggest push down: **{bottom.group}**. "
            "Bars show the model's internal log-odds contributions, so compare sizes, not absolute values."
        )

    # ----- what-if curve
    st.divider()
    st.subheader("What if…?")
    st.caption("Change one thing and hold everything else at your answers above.")
    VARY = {
        "Today's PM2.5": ("pm25", 0.0, 60.0, "µg/m³"),
        "Lowest temperature tonight": ("tmin", -8.0, 20.0, "°C"),
        "Highest temperature today": ("tmax", 0.0, 30.0, "°C"),
        "Wind speed": ("wind", 0.0, 80.0, "km/h"),
        "Humidity": ("humidity", 30, 100, "%"),
    }
    choice = st.radio("Vary", list(VARY), horizontal=True, label_visibility="collapsed")
    key, lo, hi, unit = VARY[choice]
    steps = 60
    xs = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    probs = predict([build_features({**answers, key: x}) for x in xs])
    fig2 = go.Figure()
    fig2.add_hrect(y0=DECISION_CUTOFF, y1=1, fillcolor="#c8372d", opacity=0.07, line_width=0)
    fig2.add_trace(go.Scatter(x=xs, y=probs, mode="lines", line=dict(width=3, color="#2a7fb8"),
                              hovertemplate=f"%{{x:.1f}} {unit}<br>risk %{{y:.0%}}<extra></extra>"))
    fig2.add_trace(go.Scatter(x=[answers[key]], y=[prob], mode="markers",
                              marker=dict(size=13, color=ink, line=dict(width=2, color="white")),
                              hovertemplate="Your answer<extra></extra>"))
    fig2.update_layout(height=320, margin=dict(l=0, r=10, t=10, b=10), showlegend=False,
                       xaxis_title=f"{choice} ({unit})", yaxis=dict(title="Risk", tickformat=".0%", range=[0, 1]))
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})
    st.caption("Shaded area: where the model would call tomorrow an exceedance (50% or more). "
               "Because the recent-average questions are held fixed, the curve shows this one factor only — real days move together.")

# ---------------------------------------------------------------- explore --
with tab_explore:
    st.subheader("What the monitors have recorded")
    sites = sorted(HISTORY.site_name.unique())
    c1, c2 = st.columns([2, 1])
    site = c1.selectbox("Site", sites, index=sites.index("Anzac Square"), key="explore_site")
    sdf = HISTORY[HISTORY.site_name == site].dropna(subset=["pm25_ugm3"])
    years = sorted(sdf.date.dt.year.unique())
    yr = c2.select_slider("Years", options=years, value=(years[-3] if len(years) > 3 else years[0], years[-1]))
    view = sdf[(sdf.date.dt.year >= yr[0]) & (sdf.date.dt.year <= yr[1])]
    town = view.town.iloc[0] if len(view) else ""
    if site not in TRAINED_SITES:
        st.caption(f"{site} ({town}) was **not** one of the four sites the model learned from.")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=view.date, y=view.pm25_ugm3, mode="lines", line=dict(width=1.2, color="#2a7fb8"),
                              name="PM2.5", hovertemplate="%{x|%d %b %Y}: %{y:.1f} µg/m³<extra></extra>"))
    fig3.add_hline(y=THRESHOLD, line_dash="dash", line_color="#c8372d",
                   annotation_text="WHO guideline 15", annotation_position="top left")
    fig3.update_layout(height=340, margin=dict(l=0, r=10, t=20, b=10), showlegend=False,
                       yaxis_title="PM2.5 (µg/m³)")
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

    exc = sdf.assign(over=sdf.pm25_ugm3 > THRESHOLD).groupby(sdf.date.dt.month)["over"].mean()
    exc = exc.reindex(range(1, 13)).fillna(0)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig4 = go.Figure(go.Bar(x=months, y=exc.values, marker_color="#2a7fb8",
                            hovertemplate="%{x}: %{y:.0%} of days<extra></extra>"))
    fig4.update_layout(height=280, margin=dict(l=0, r=10, t=10, b=10),
                       yaxis=dict(title="Days over 15 µg/m³", tickformat=".0%"))
    st.markdown("**When do exceedances happen?** Share of days over the guideline, by month, across all years at this site.")
    st.plotly_chart(fig4, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------- about ----
with tab_about:
    st.subheader("How this works")
    st.markdown(
        f"""
**The question.** Will tomorrow's average PM2.5 be above **{THRESHOLD} µg/m³**, the WHO 24-hour guideline?
PM2.5 particles are small enough to reach deep into the lungs and bloodstream, and in NZ they spike on cold, still
winter nights when wood burners are in use.

**The model.** An XGBoost classifier trained on daily readings from four LAWA monitoring sites (Timaru, Ashburton,
Rangiora and Masterton), combining recent PM2.5 and PM10 levels, weather and time of year. It was tuned for **recall**,
so it would rather raise a false alarm than miss a bad day.

**How well it did on days it had never seen (2024 onwards):**
"""
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Recall", "85%", help="Of the days that really exceeded, the share the model flagged.")
    m2.metric("Precision", "61%", help="Of the days the model flagged, the share that really exceeded.")
    m3.metric("Accuracy", "88%")
    st.markdown(
        """
In plain terms: it catches most bad days, but roughly 4 in 10 warnings are false alarms.

**Things to keep in mind**
- The percentage is the model's **risk score**, not a calibrated probability — it was trained to favour catching exceedances.
- It learned from four sites. Other towns, especially ones with different heating habits or geography, may behave differently.
- It doesn't know about traffic, industry, fires or burning on the night.
- Treat it as a guide alongside your council's air-quality alerts, not a replacement.

**Questions you aren't asked.** The model uses 19 inputs. Your 10 answers cover most of them; the rest are filled in:
- *Month and season* come from the date.
- *Cold day* uses the notebook's definition (highest temperature above 16 °C).
- *PM10 today, 7-day mean PM10, 7-day max PM10* are estimated from your PM2.5 answers using the typical PM10-to-PM2.5
  ratio for that season in the dataset (unless you enter PM10 yourself).
- *PM2.5 fraction* is PM2.5 divided by PM10.
- *Rain, pressure and pressure change* default to a dry, settled day (change the Fine-tune box if you know them).
- A 7-day maximum is never allowed to sit below today's reading, and a day above 15 counts toward the exceedance days.
"""
    )
