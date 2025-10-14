import streamlit as st
from vitality_score import synergy_score, toy_T50, toy_T95, toy_HRV50

st.title("Vitality Score (beta)")
age = st.number_input("Age", 18, 90, 45)
sex = st.selectbox("Sex", ["male","female"])
T = st.number_input("Total T (ng/dL)", 100, 1500, 600)
sleep_h = st.number_input("Avg sleep (h)", 0.0, 12.0, 8.5, step=0.25)
train_d = st.slider("Training days/week", 0, 7, 5)
bmi = st.number_input("BMI", 15.0, 40.0, 23.0)
rhr = st.number_input("Resting HR (bpm)", 30, 120, 60)
hrv = st.number_input("HRV (RMSSD, ms)", 10, 150, 45)

out = synergy_score(T, age, sex, toy_T50, toy_T95,
                    training_days_per_week=train_d,
                    avg_sleep_hours=sleep_h, BMI=bmi,
                    rhr_bpm=rhr, hrv_ms=hrv, HRV50_fn=toy_HRV50)

label = ("Peak" if out["Score"]>=90 else
         "Strong" if out["Score"]>=75 else
         "Good" if out["Score"]>=60 else
         "Below Avg" if out["Score"]>=40 else "Low")

st.metric("Vitality Score", f"{out['Score']:.1f}/100", label)
st.caption(f"Hormone H: {out['H']:.2f} | Lifestyle L: {out['L']:.2f}")
