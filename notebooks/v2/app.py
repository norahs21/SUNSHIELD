import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------- Settings ----------
DATA_DIR = "final_outputs"                      # the unzipped final_outputs.zip (Notebook 5), next to app.py
FILES_DIR = os.path.join(DATA_DIR, "final_files")
STAGES = {                                      # Version A files (plain 0/1), best k from training
    "Training":   "2_LSTM_Training_Output_Thresholding.csv",
    "Validation": "4_LSTM_Validation_Output_Thresholding.csv",
    "Testing":    "6_LSTM_Testing_Output_Anomaly.csv",
}

st.set_page_config(page_title="SUNSHIELD Anomaly Dashboard", layout="wide")
st.title("SUNSHIELD - PV Anomaly Detection Dashboard")
st.caption(
    "SUNSHIELD v2. The LSTM (trained on healthy readings) predicts the AC power from the weather. "
    "For each reading, actual − expected power is compared with μ ± k·σ of that inverter (median and 1.4826·MAD, training days). "
    "Actual 0/1: expected = physical PVWatts power. Predicted 0/1: expected = LSTM line. "
    "Readings with irradiation < 0.2 kW/m² are not evaluated. Version A: plain 0/1, best k on Training."
)

if not os.path.isdir(FILES_DIR):
    st.error(f"Folder '{FILES_DIR}' not found. Unzip final_outputs.zip into a folder named 'final_outputs' next to app.py.")
    st.stop()


@st.cache_data
def load_stage(stage):
    df = pd.read_csv(os.path.join(FILES_DIR, STAGES[stage]), parse_dates=["DATE_TIME"])
    return df.sort_values(["PLANT_ID", "SOURCE_KEY", "DATE_TIME"]).reset_index(drop=True)


@st.cache_data
def load_per_inverter():
    return pd.read_csv(os.path.join(DATA_DIR, "metrics_per_inverter.csv"))


def scores(d):
    """TP / FP / FN / TN from the result column of the final files, then precision, recall, F1."""
    tp = int((d["result"] == "TP").sum())
    fp = int((d["result"] == "FP").sum())
    fn = int((d["result"] == "FN").sum())
    tn = int((d["result"] == "TN").sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f1, 3)}


# ---------- Sidebar ----------
st.sidebar.header("Filters")
stage = st.sidebar.selectbox("Stage", list(STAGES))
df = load_stage(stage)

plant = st.sidebar.selectbox("Plant", sorted(df["PLANT_ID"].unique()))
inverters = sorted(df[df["PLANT_ID"] == plant]["SOURCE_KEY"].unique())
inverter = st.sidebar.selectbox("Inverter", inverters)
days = sorted(df[(df["PLANT_ID"] == plant) & (df["SOURCE_KEY"] == inverter)]["DATE_TIME"].dt.date.unique())
day_choice = st.sidebar.selectbox("Day", ["All days"] + [str(d) for d in days])

st.sidebar.markdown(f"**k** = {df['k'].iloc[0]}")
st.sidebar.markdown("**Version** = A (plain 0/1)")

# ---------- Results per stage ----------
st.subheader("Results per stage (all plants and inverters)")
rows = []
for s in STAGES:
    d = load_stage(s)
    rows.append({"Stage": s, "Windows": len(d),
                 "Actual anomalies": int(d["actual_anomaly"].sum()),
                 "Predicted anomalies": int(d["predicted_anomaly"].sum()),
                 **scores(d)})
st.dataframe(pd.DataFrame(rows).set_index("Stage"), use_container_width=True)

# ---------- Per-inverter F1 ----------
st.subheader(f"F1 of every inverter, {stage}")
pi = load_per_inverter()
pi = pi[pi["Split"] == stage].sort_values(["PLANT_ID", "F1"], ascending=[True, False])
st.dataframe(pi[["PLANT_ID", "SOURCE_KEY", "Precision", "Recall", "F1"]].reset_index(drop=True),
             use_container_width=True, height=280)

# ---------- Selected inverter ----------
subset = df[(df["PLANT_ID"] == plant) & (df["SOURCE_KEY"] == inverter)]
if day_choice != "All days":
    subset = subset[subset["DATE_TIME"].dt.date.astype(str) == day_choice]

st.subheader(f"{stage} - plant {plant}, inverter {inverter}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Windows", len(subset))
c2.metric("Actual anomalies", int(subset["actual_anomaly"].sum()))
c3.metric("Predicted anomalies", int(subset["predicted_anomaly"].sum()))
sc = scores(subset) if len(subset) else {"Precision": 0, "Recall": 0, "F1": 0}
has_events = int(subset["actual_anomaly"].sum()) + int(subset["predicted_anomaly"].sum()) > 0
c4.metric("F1 (this selection)", sc["F1"] if has_events else "n/a")
st.caption(f"Precision {sc['Precision']}  |  Recall {sc['Recall']}")

# Empty rows for nights, so the lines are not joined across them
plot_df = subset.set_index("DATE_TIME")
if len(plot_df):
    full = pd.date_range(plot_df.index.min(), plot_df.index.max(), freq="15min")
    plot_df = plot_df.reindex(full)

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ACTUAL_AC_POWER_kW"], mode="lines",
                         name="Actual", connectgaps=False, line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["PREDICTED_AC_POWER_kW"], mode="lines",
                         name="Predicted (LSTM)", connectgaps=False, line=dict(color="#ff7f0e")))
if "EXPECTED_PHYS_kW" in plot_df:
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EXPECTED_PHYS_kW"], mode="lines", name="Expected (PVWatts)",
                             connectgaps=False, line=dict(color="#7f7f7f", dash="dash", width=1)))
a_hit = plot_df[plot_df["actual_anomaly"] == 1]
fig.add_trace(go.Scatter(x=a_hit.index, y=a_hit["ACTUAL_AC_POWER_kW"], mode="markers",
                         name="Actual anomaly", marker=dict(color="red", size=8)))
p_hit = plot_df[plot_df["predicted_anomaly"] == 1]
fig.add_trace(go.Scatter(x=p_hit.index, y=p_hit["PREDICTED_AC_POWER_kW"], mode="markers",
                         name="Predicted anomaly", marker=dict(color="black", size=9, symbol="diamond-open")))
fig.update_layout(xaxis_title="Time", yaxis_title="AC power (kW)", height=450, legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# ---------- Table ----------
st.subheader("Anomaly readings (TP, FP, FN)")
table = subset[~subset["result"].isin(["TN", "not evaluated"])][
    ["DATE_TIME", "ACTUAL_AC_POWER_kW", "PREDICTED_AC_POWER_kW", "actual_anomaly", "predicted_anomaly", "result"]
]
st.dataframe(table, use_container_width=True)
st.download_button("Download this table (CSV)", table.to_csv(index=False), file_name="anomaly_readings.csv")
