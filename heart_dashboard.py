#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib
import altair as alt
from langchain_community.llms import Ollama
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent import AgentExecutor

# Load ML Models
ef_model = joblib.load("ef_model.pkl")
vo2_model = joblib.load("vo2_model.pkl")

# Page Configuration
st.set_page_config(page_title="Heart Digital Twin", layout="wide")
st.title("💓 Digital Twin Dashboard – Heart EF & VO₂")

# Sidebar Input
st.sidebar.header("Enter Patient Vitals")
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", value=85)
sbp = st.sidebar.number_input("Systolic BP (mmHg)", value=120)
dbp = st.sidebar.number_input("Diastolic BP (mmHg)", value=80)
spo2 = st.sidebar.number_input("SpO₂ (%)", value=96)
hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", value=14.0)
pao2 = st.sidebar.number_input("PaO₂ (mmHg)", value=85.0)

# Create DataFrame
input_df = pd.DataFrame([{
    "Heart Rate": heart_rate,
    "Non Invasive Systolic BP": sbp,
    "Non Invasive Diastolic BP": dbp,
    "SpO2": spo2,
    "Hemoglobin": hemoglobin,
    "PaO2": pao2
}])

# Predict EF and VO2
input_df["EF_percent"] = ef_model.predict(input_df[["Heart Rate", "Non Invasive Systolic BP", "Non Invasive Diastolic BP"]])
input_df["VO2_ml_per_min"] = vo2_model.predict(input_df[["Heart Rate", "SpO2", "Hemoglobin", "PaO2"]])

ef = input_df["EF_percent"].iloc[0]
vo2 = input_df["VO2_ml_per_min"].iloc[0]

# Layout
col1, col2 = st.columns([1, 2])
with col1:
    st.image("C:/Users/Madayanthikaa ramesh/Downloads/Screenshot 2025-05-21 224125.jpg", caption="Cardiac Contraction Pattern", use_column_width=True)

with col2:
    st.markdown("### Predicted EF and VO₂")
    st.dataframe(input_df[["EF_percent", "VO2_ml_per_min"]].style.format(precision=2))

    st.markdown("### EF/VO₂ Alerts")
    if ef < 40:
        st.error(f"❗ EF is {ef:.2f}% — possible heart failure (HFrEF).")
    elif ef > 75:
        st.warning(f"⚠️ EF is {ef:.2f}% — possible hyperdynamic function.")
    else:
        st.success(f"✅ EF is normal: {ef:.2f}%")

    if vo2 < 250:
        st.error(f"❗ VO₂ is low: {vo2:.2f} ml/min — possible poor oxygen delivery.")
    elif vo2 > 400:
        st.warning(f"⚠️ VO₂ is elevated: {vo2:.2f} ml/min — check for stress/exertion.")
    else:
        st.success(f"✅ VO₂ is within normal range: {vo2:.2f} ml/min")

# Altair Chart
chart_data = pd.DataFrame({
    "Metric": ["EF_percent", "VO2_ml_per_min"],
    "Value": [ef, vo2]
})

chart = alt.Chart(chart_data).mark_bar().encode(
    x=alt.X("Metric", sort=None),
    y="Value",
    color=alt.Color("Metric", legend=None),
    tooltip=["Metric", "Value"]
).properties(
    width=500,
    height=300,
    title="EF and VO₂ Comparison Chart"
)

st.altair_chart(chart)

# LLM + Agent
st.markdown("### Mistral AI Interpretation")

llm = Ollama(model="mistral")  # Only this is needed

# ✅ MUST PASS allow_dangerous_code=True
agent = create_pandas_dataframe_agent(
    llm,
    input_df,
    verbose=False,
    allow_dangerous_code=True
)

agent_executor = AgentExecutor(agent=agent.agent, tools=agent.tools, handle_parsing_errors=True)

query = f"""
Analyze the predicted values:
- Ejection Fraction (EF): {ef:.2f}%
- VO₂: {vo2:.2f} ml/min

Explain what they suggest about cardiovascular performance.
Respond in clinical terms without giving direct medical advice.
"""

with st.spinner("AI analyzing your heart data..."):
    try:
        response = agent_executor.invoke({"input": query})
        st.success("AI Analysis Complete:")
        st.write(response)
    except Exception as e:
        st.error(f"Agent error: {str(e)}")

# Download Button
st.download_button(
    "Download EF/VO₂ Report",
    data=input_df.to_csv(index=False),
    file_name="heart_report.csv"
)


# In[ ]:




