import json, os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="HumanFlow AI", page_icon="🧠", layout="wide")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, "employees.csv")
    return pd.read_csv(path)

@st.cache_resource
def train_models(df):
    feature_cols = ["attendance_rate","performance_score","workload_hours",
                    "engagement_score","skill_match","feedback_score","tenure_years"]
    X = df[feature_cols]
    y = df["attrition"]
    clf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    iso = IsolationForest(contamination=0.08, random_state=42)
    iso.fit(X)
    return clf, iso, feature_cols

def risk_label(p):
    if p >= .70: return "HIGH"
    if p >= .40: return "MODERATE"
    return "LOW"

def explanation(row, probability):
    reasons = []
    if row.attendance_rate < 85: reasons.append("attendance has declined")
    if row.performance_score < 65: reasons.append("performance is below the internal benchmark")
    if row.workload_hours > 48: reasons.append("workload is elevated")
    if row.engagement_score < 60: reasons.append("engagement is low")
    if row.skill_match < 65: reasons.append("current project skill match is limited")
    if row.feedback_score < 60: reasons.append("recent feedback signals are weak")
    if not reasons: reasons.append("the employee's combined pattern resembles higher-risk historical cases")
    return reasons

df = load_data()
clf, iso, feature_cols = train_models(df)

df["risk_probability"] = clf.predict_proba(df[feature_cols])[:,1]
df["risk"] = df.risk_probability.map(risk_label)
df["anomaly"] = iso.predict(df[feature_cols]) == -1

st.title("🧠 HumanFlow AI")
st.caption("Multi-source workforce intelligence • prediction • explanation • what-if simulation")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Employees", len(df))
c2.metric("High Risk", int((df.risk=="HIGH").sum()))
c3.metric("Moderate Risk", int((df.risk=="MODERATE").sum()))
c4.metric("Anomalies", int(df.anomaly.sum()))

tab1, tab2, tab3, tab4 = st.tabs(["Workforce Overview","Employee Intelligence","What-if Simulator","Ask the Organization"])

with tab1:
    left,right = st.columns(2)
    with left:
        counts = df["risk"].value_counts().rename_axis("Risk").reset_index(name="Employees")
        st.plotly_chart(px.bar(counts, x="Risk", y="Employees", title="Workforce Risk"), use_container_width=True)
    with right:
        dept = df.groupby("department")["risk_probability"].mean().sort_values(ascending=False).reset_index()
        st.plotly_chart(px.bar(dept, x="department", y="risk_probability", title="Average Risk by Department"), use_container_width=True)
    st.subheader("Emerging signals")
    signals = []
    if df.workload_hours.mean() > 45: signals.append("Average workload is elevated.")
    if df.engagement_score.mean() < 70: signals.append("Average engagement is below 70.")
    if df.skill_match.mean() < 75: signals.append("Skill alignment indicates a workforce-wide gap.")
    if df.anomaly.sum() > 0: signals.append(f"{int(df.anomaly.sum())} employees show unusual multi-factor patterns.")
    for s in signals or ["No major aggregate signal detected in the demo dataset."]:
        st.write("• " + s)

with tab2:
    selected = st.selectbox("Select employee", df.employee_id.astype(str).tolist())
    row = df[df.employee_id.astype(str)==selected].iloc[0]
    p = float(row.risk_probability)
    st.subheader(f"Employee {selected} — {row.name if False else row.employee_name}")
    a,b,c,d = st.columns(4)
    a.metric("Risk", risk_label(p))
    b.metric("Risk probability", f"{p:.0%}")
    c.metric("Performance", f"{row.performance_score:.0f}")
    d.metric("Skill match", f"{row.skill_match:.0f}%")

    st.markdown("### Why?")
    for reason in explanation(row,p):
        st.write("• " + reason)

    st.markdown("### Evidence")
    ev = pd.DataFrame({
        "Signal":["Attendance","Performance","Workload","Engagement","Skill match","Feedback","Tenure"],
        "Value":[f"{row.attendance_rate:.1f}%",f"{row.performance_score:.1f}",f"{row.workload_hours:.1f} hrs",
                 f"{row.engagement_score:.1f}",f"{row.skill_match:.1f}%",f"{row.feedback_score:.1f}",f"{row.tenure_years:.1f} yrs"]
    })
    st.dataframe(ev, use_container_width=True, hide_index=True)

    st.markdown("### Recommended investigation")
    actions=[]
    if row.workload_hours > 48: actions.append("Review workload and project allocation.")
    if row.skill_match < 65: actions.append("Review targeted upskilling or project-skill alignment.")
    if row.performance_score < 65: actions.append("Review goals, blockers and recent performance feedback.")
    if row.engagement_score < 60: actions.append("Schedule a manager check-in focused on engagement and blockers.")
    if not actions: actions.append("Continue monitoring the employee's trend and validate the model signal with human review.")
    for a in actions: st.write("• " + a)

with tab3:
    selected = st.selectbox("Employee for simulation", df.employee_id.astype(str).tolist(), key="sim")
    row = df[df.employee_id.astype(str)==selected].iloc[0]
    reduction = st.slider("Reduce workload by", 0, 50, 20, 5)
    simulated = row.copy()
    simulated["workload_hours"] = max(20, row.workload_hours*(1-reduction/100))
    base_p = float(clf.predict_proba(pd.DataFrame([row[feature_cols].values], columns=feature_cols))[:,1][0])
    sim_p = float(clf.predict_proba(pd.DataFrame([simulated[feature_cols].values], columns=feature_cols))[:,1][0])
    x,y = st.columns(2)
    x.metric("Current risk", f"{base_p:.0%}")
    y.metric("Model-simulated risk", f"{sim_p:.0%}", delta=f"{(sim_p-base_p):.0%}")
    st.info("This is a model-based scenario simulation, not a guaranteed real-world outcome.")

with tab4:
    q = st.text_input("Ask a workforce question", placeholder="Which department has the highest average risk?")
    if q:
        ql=q.lower()
        if "department" in ql and ("risk" in ql or "highest" in ql):
            g=df.groupby("department").risk_probability.mean().sort_values(ascending=False)
            st.write(f"Highest average modeled risk in this demo dataset: **{g.index[0]}** ({g.iloc[0]:.0%}).")
        elif "high risk" in ql or "risk" in ql:
            st.write(f"There are **{int((df.risk=='HIGH').sum())}** employees in the high-risk band.")
        elif "skill" in ql:
            st.write(f"Average skill match is **{df.skill_match.mean():.1f}%**; {int((df.skill_match<65).sum())} employees are below 65%.")
        elif "workload" in ql:
            st.write(f"Average workload is **{df.workload_hours.mean():.1f} hours/week**.")
        else:
            st.write("For this prototype, ask about risk, departments, skills, workload, performance, or anomalies.")

st.divider()
st.caption("HumanFlow AI is a prototype. Predictions support human review; they should not be used as the sole basis for employment decisions.")
