import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap
from fpdf import FPDF
import tempfile
import os

st.set_page_config(
    page_title="Heart Health Dashboard",
    page_icon="❤️",
    layout="wide"
)

logo_col1, logo_col2, logo_col3 = st.columns([1, 1.2, 1])
with logo_col2:
    st.image("logo.png", use_container_width=True)
    st.markdown("<p style='text-align:center; color:#5DADE2; font-size:13px;'>Developed by Petronilda Biyanwila | Yoobee College of Creative Innovation | MBI908 Capstone</p>", unsafe_allow_html=True)

with st.expander("What do these terms mean?"):
    st.write("**Likelihood estimate**: How closely your profile matches patterns linked to heart disease in this dataset. This is not a diagnosis or a guaranteed future outcome.")
    st.write("**Contributing factor**: Shows which of your answers had the biggest effect on your result, and whether each one pushed it up or down.")
    st.write("**Illustrative model scenario**: A 'what-if' example showing how the estimate would change if one factor changed. It reflects patterns in the data, not a promise about your real health.")
    st.write("**Percentile**: Shows where your result sits compared to everyone else in this dataset, not a clinical category.")

model = joblib.load('model_ebm.pkl')
X_train = pd.read_csv('X_train.csv')

st.markdown("""
<style>
div.stButton > button {
    background-color: white;
    color: black;
    font-weight: bold;
    border: 3px solid #4A9EFF;
    padding: 12px;
}
div.stButton > button:hover {
    background-color: #f0f0f0;
    color: black;
    border: 3px solid #F39C12;
}
</style>
""", unsafe_allow_html=True)

st.write("---")

GREEN = '#27AE60'
RED = '#E74C3C'
AMBER = '#F39C12'
BLUE = '#5DADE2'
BOX_BG = '#1B3A5C'

main_left, main_right = st.columns([1, 1.2])

with main_left:
    st.header("Your Information")

    with st.container(border=True):
        st.subheader("Step 1: Required Information")
        st.caption("These need a Yes/No answer, since guessing could give a misleading result.")
        high_bp = st.radio("High blood pressure diagnosed by a doctor?", ["No", "Yes"])
        high_chol = st.radio("High cholesterol diagnosed by a doctor?", ["No", "Yes"])
        chol_check = st.radio("Cholesterol checked in the past 5 years?", ["No", "Yes"])
        stroke = st.radio("Ever told you had a stroke?", ["No", "Yes"])
        diabetes = st.selectbox("Do you have diabetes?", ["No", "Pre-diabetes/borderline", "Yes"])

    with st.container(border=True):
        st.subheader("Step 2: About You")
        st.caption("If unsure about income or education, select 'Unsure'.")
        sex = st.radio("Sex", ["Female", "Male"])
        age_group = st.selectbox("Age group", [
            "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
            "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
        ])
        education = st.selectbox("Highest education level", [
            "Unsure", "None or only kindergarten", "Grades 1-8", "Grades 9-11",
            "Grade 12 or GED", "Some college (1-3 years)", "College graduate (4+ years)"
        ])
        income = st.selectbox("Annual household income", [
            "Unsure", "Under $10,000", "$10,000-$14,999", "$15,000-$19,999",
            "$20,000-$24,999", "$25,000-$34,999", "$35,000-$49,999",
            "$50,000-$74,999", "$75,000 or more"
        ])

    with st.container(border=True):
        st.subheader("Step 3: Lifestyle")
        st.caption("These directly shape your recommendations, so please answer all.")
        smoker = st.radio(
            "In your entire life, have you smoked at least 100 cigarettes (about 5 packs)? This includes past smoking, even if you've since quit.",
            ["No", "Yes"]
        )
        activity_days_per_week = st.number_input("How many days per week do you do physical activity (outside of your regular job)?", min_value=0, max_value=7, value=0)
        fruits = st.radio("Eat fruit 1+ times per day?", ["No", "Yes"])
        veggies = st.radio("Eat vegetables 1+ times per day?", ["No", "Yes"])
        drinks_per_week = st.number_input("On average, how many alcoholic drinks do you have per week?", min_value=0, max_value=100, value=0)

    with st.container(border=True):
        st.subheader("Step 4: Body Measurements")
        height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        height_inches = height_cm / 2.54
        st.caption(f"≈ {height_inches:.1f} inches ({int(height_inches // 12)} ft {height_inches % 12:.0f} in)")

        weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
        weight_lb = weight_kg * 2.20462
        st.caption(f"≈ {weight_lb:.1f} lb")

        bmi_calculated = weight_kg / ((height_cm / 100) ** 2)
        st.write(f"Your calculated BMI: **{bmi_calculated:.1f}**")

    with st.container(border=True):
        st.subheader("Step 5: Additional Health Information")
        st.caption("These are also used by the model and directly affect the accuracy of your result.")
        gen_health = st.selectbox("How would you rate your general health?", ["Excellent", "Very good", "Good", "Fair", "Poor"])
        ment_hlth_days = st.number_input("In the past 30 days, how many days was your mental health not good?", min_value=0, max_value=30, value=0)
        phys_hlth_days = st.number_input("In the past 30 days, how many days was your physical health not good?", min_value=0, max_value=30, value=0)
        diff_walk = st.radio("Do you have serious difficulty walking or climbing stairs?", ["No", "Yes"])
        any_healthcare = st.radio("Do you have any form of health care coverage?", ["Yes", "No"])
        no_doc_cost = st.radio("In the past 12 months, was there a time you needed to see a doctor but couldn't because of cost?", ["No", "Yes"])

    get_result = st.button("Get My Result", use_container_width=True)
    left_result_placeholder = st.container()

with main_right:
    st.header("Live Snapshot")
    st.caption("Updates instantly as you answer.")

    with st.container(border=True):
        hvy_alcohol_live = (drinks_per_week > 14) if sex == "Male" else (drinks_per_week > 7)
        phys_activity_live = activity_days_per_week > 0

        snapshot_labels = ["High BP", "High Chol", "Stroke", "Smoker", "Active", "Fruits", "Veggies", "Alcohol"]
        snapshot_values = [
            1 if high_bp == "Yes" else 0,
            1 if high_chol == "Yes" else 0,
            1 if stroke == "Yes" else 0,
            1 if smoker == "Yes" else 0,
            1 if phys_activity_live else 0,
            1 if fruits == "Yes" else 0,
            1 if veggies == "Yes" else 0,
            1 if hvy_alcohol_live else 0,
        ]
        protective_when_yes = {"Active", "Fruits", "Veggies"}
        snapshot_colors = []
        for label, val in zip(snapshot_labels, snapshot_values):
            if label in protective_when_yes:
                snapshot_colors.append(GREEN if val == 1 else RED)
            else:
                snapshot_colors.append(RED if val == 1 else GREEN)

        fig_snap, ax_snap = plt.subplots(figsize=(5, 2.8))
        fig_snap.patch.set_facecolor(BOX_BG)
        ax_snap.set_facecolor(BOX_BG)
        ax_snap.barh(snapshot_labels[::-1], [1]*8, color=snapshot_colors[::-1])
        ax_snap.set_xlim(0, 1)
        ax_snap.set_xticks([])
        ax_snap.tick_params(colors='white')
        for spine in ax_snap.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_snap, use_container_width=True)

        bmi_status = "Healthy range" if 18.5 <= bmi_calculated <= 24.9 else "Outside healthy range"
        bmi_color = "green" if 18.5 <= bmi_calculated <= 24.9 else "orange"
        st.markdown(f"BMI: **{bmi_calculated:.1f}** — :{bmi_color}[{bmi_status}]")

        st.write("---")
        st.markdown("**Your Healthy Margins**")
        smoking_margin_text = "0 cigarettes lifetime is the only fully risk-free level - any smoking history counts."
        if sex == "Male":
            alcohol_margin_text = "Under 14 drinks/week (men) is the healthy margin used in this model."
        else:
            alcohol_margin_text = "Under 7 drinks/week (women) is the healthy margin used in this model."
        st.caption(f"🚬 {smoking_margin_text}")
        st.caption(f"🍷 {alcohol_margin_text}")
        st.caption("🏃 CDC/WHO recommend 150 min/week of moderate activity (≈30 min, 5 days/week) for genuine health benefit - this model's 'active' question is a lower bar (any activity at all).")

    right_result_placeholder = st.container()

if get_result:
    if sex == "Male":
        hvy_alcohol = "Yes" if drinks_per_week > 14 else "No"
    else:
        hvy_alcohol = "Yes" if drinks_per_week > 7 else "No"
    phys_activity = "Yes" if activity_days_per_week > 0 else "No"

    gen_health_map = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    gen_health_value = gen_health_map[gen_health]

    defaults = X_train.mean()
    person_row = defaults.copy()

    person_row['HighBP'] = 1.0 if high_bp == "Yes" else 0.0
    person_row['HighChol'] = 1.0 if high_chol == "Yes" else 0.0
    person_row['CholCheck'] = 1.0 if chol_check == "Yes" else 0.0
    person_row['Stroke'] = 1.0 if stroke == "Yes" else 0.0
    person_row['Diabetes'] = {"No": 0.0, "Pre-diabetes/borderline": 1.0, "Yes": 2.0}[diabetes]

    person_row['Sex'] = 1.0 if sex == "Male" else 0.0
    age_map = {"18-24":1, "25-29":2, "30-34":3, "35-39":4, "40-44":5, "45-49":6,
               "50-54":7, "55-59":8, "60-64":9, "65-69":10, "70-74":11, "75-79":12, "80+":13}
    person_row['Age'] = age_map[age_group]

    person_row['Smoker'] = 1.0 if smoker == "Yes" else 0.0
    person_row['PhysActivity'] = 1.0 if phys_activity == "Yes" else 0.0
    person_row['Fruits'] = 1.0 if fruits == "Yes" else 0.0
    person_row['Veggies'] = 1.0 if veggies == "Yes" else 0.0
    person_row['HvyAlcoholConsump'] = 1.0 if hvy_alcohol == "Yes" else 0.0
    person_row['BMI'] = bmi_calculated

    person_row['GenHlth'] = gen_health_value
    person_row['MentHlth'] = ment_hlth_days
    person_row['PhysHlth'] = phys_hlth_days
    person_row['DiffWalk'] = 1.0 if diff_walk == "Yes" else 0.0
    person_row['AnyHealthcare'] = 1.0 if any_healthcare == "Yes" else 0.0
    person_row['NoDocbcCost'] = 1.0 if no_doc_cost == "Yes" else 0.0

    education_map = {"None or only kindergarten":1, "Grades 1-8":2, "Grades 9-11":3,
                      "Grade 12 or GED":4, "Some college (1-3 years)":5, "College graduate (4+ years)":6}
    if education != "Unsure":
        person_row['Education'] = education_map[education]

    income_map = {"Under $10,000":1, "$10,000-$14,999":2, "$15,000-$19,999":3, "$20,000-$24,999":4,
                  "$25,000-$34,999":5, "$35,000-$49,999":6, "$50,000-$74,999":7, "$75,000 or more":8}
    if income != "Unsure":
        person_row['Income'] = income_map[income]

    person_data = pd.DataFrame([person_row])[X_train.columns]
    likelihood = model.predict_proba(person_data)[0][1]

    if likelihood >= 0.30:
        tier_label = "🔴 HIGHER LIKELIHOOD"
        tier_color = "red"
    elif likelihood >= 0.10:
        tier_label = "🟡 MODERATE LIKELIHOOD"
        tier_color = "orange"
    else:
        tier_label = "🟢 LOWER LIKELIHOOD"
        tier_color = "green"

    all_train_probas = model.predict_proba(X_train)[:, 1]
    percentile = (all_train_probas < likelihood).mean() * 100

    with left_result_placeholder:
        st.header("Your Result")
        with st.container(border=True):
            st.metric("Likelihood Estimate", f"{likelihood:.1%}")
            st.markdown(f":{tier_color}[**{tier_label}**]")
            st.caption(f"Higher than {percentile:.0f}% of people in this dataset. Dataset-based estimate, not a clinical risk score.")

            fig_gauge, ax_gauge = plt.subplots(figsize=(3.6, 2.2), subplot_kw={'projection': 'polar'})
            fig_gauge.patch.set_alpha(0)
            gauge_colors = [GREEN, AMBER, RED]
            bounds = [0, 0.10, 0.30, 1.0]
            for i in range(3):
                theta1 = np.pi * (1 - bounds[i])
                theta2 = np.pi * (1 - bounds[i+1])
                ax_gauge.bar(x=(theta1+theta2)/2, height=1, width=abs(theta1-theta2), bottom=2,
                       color=gauge_colors[i], edgecolor=BOX_BG, linewidth=2)
            needle_angle = np.pi * (1 - likelihood)
            ax_gauge.plot([needle_angle, needle_angle], [0, 2.3], color='white', linewidth=3)
            ax_gauge.set_theta_zero_location('W')
            ax_gauge.set_theta_direction(1)
            ax_gauge.set_thetamin(0)
            ax_gauge.set_thetamax(180)
            ax_gauge.set_ylim(0, 3)
            ax_gauge.axis('off')
            plt.tight_layout()
            st.pyplot(fig_gauge, use_container_width=True)

    with right_result_placeholder:
        with st.container(border=True):
            st.subheader("Where You Sit in the Dataset")
            fig_hist, ax_hist = plt.subplots(figsize=(6, 2.6))
            fig_hist.patch.set_facecolor(BOX_BG)
            ax_hist.set_facecolor(BOX_BG)
            ax_hist.hist(all_train_probas, bins=30, color=BLUE, alpha=0.85)
            ax_hist.axvline(likelihood, color=RED, linewidth=2.5, linestyle='--')
            ax_hist.text(likelihood, ax_hist.get_ylim()[1]*0.9, ' You', color='white', fontweight='bold')
            ax_hist.tick_params(colors='white')
            ax_hist.set_xlabel('Likelihood Estimate', color='white')
            for spine in ax_hist.spines.values():
                spine.set_color('white')
            plt.tight_layout()
            st.pyplot(fig_hist, use_container_width=True)
            st.caption("Distribution across the full dataset, with your result marked.")

        st.write("")

        explainer = shap.Explainer(model.predict_proba, X_train, feature_names=X_train.columns.tolist())
        shap_values = explainer(person_data)
        person_shap = shap_values[0, :, 1].values
        feature_names = X_train.columns.tolist()
        top5_contributions = sorted(zip(feature_names, person_shap), key=lambda x: abs(x[1]), reverse=True)[:5]

        with st.container(border=True):
            st.subheader("Top 5 Contributing Factors")
            chart_features = [f for f, v in top5_contributions][::-1]
            chart_values = [v for f, v in top5_contributions][::-1]
            chart_colors = [RED if v > 0 else GREEN for v in chart_values]

            fig_bar, ax_bar = plt.subplots(figsize=(6, 2.8))
            fig_bar.patch.set_facecolor(BOX_BG)
            ax_bar.set_facecolor(BOX_BG)
            ax_bar.barh(chart_features, chart_values, color=chart_colors)
            ax_bar.axvline(0, color='white', linewidth=0.8)
            ax_bar.tick_params(colors='white')
            ax_bar.xaxis.label.set_color('white')
            ax_bar.set_xlabel('Effect on likelihood estimate')
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            st.caption("Red = increases likelihood. Green = decreases it.")

        st.write("")

        good_factors = []
        candidate_risks = []
        if smoker == "No": good_factors.append('Smoker')
        else: candidate_risks.append('Smoker')
        if phys_activity == "Yes": good_factors.append('PhysActivity')
        else: candidate_risks.append('PhysActivity')
        if fruits == "Yes": good_factors.append('Fruits')
        else: candidate_risks.append('Fruits')
        if veggies == "Yes": good_factors.append('Veggies')
        else: candidate_risks.append('Veggies')
        if hvy_alcohol == "No": good_factors.append('HvyAlcoholConsump')
        else: candidate_risks.append('HvyAlcoholConsump')
        if high_bp == "No": good_factors.append('HighBP')
        else: candidate_risks.append('HighBP')
        if high_chol == "No": good_factors.append('HighChol')
        else: candidate_risks.append('HighChol')
        if 18.5 <= bmi_calculated <= 24.9: good_factors.append('BMI_healthy')
        elif bmi_calculated > 24.9: candidate_risks.append('BMI_high')

        with st.container(border=True):
            st.subheader("Your Habits at a Glance")
            chart_grid_col1, chart_grid_col2 = st.columns(2)

            with chart_grid_col1:
                fig_donut, ax_donut = plt.subplots(figsize=(3.2, 3.2))
                fig_donut.patch.set_facecolor(BOX_BG)
                n_good = len(good_factors)
                n_risk = len(candidate_risks)
                if n_good + n_risk > 0:
                    ax_donut.pie([n_good, n_risk], colors=[GREEN, RED], startangle=90,
                            wedgeprops=dict(width=0.4, edgecolor=BOX_BG, linewidth=3))
                    ax_donut.text(0, 0, f"{n_good}/{n_good+n_risk}\nHealthy", ha='center', va='center',
                             color='white', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_donut, use_container_width=True)
                st.caption("Habits Breakdown")

            with chart_grid_col2:
                radar_categories = ['Activity', 'Fruits', 'Veggies', 'BP Health', 'Chol Health', 'Non-Smoker']
                radar_values = [
                    1 if phys_activity == "Yes" else 0,
                    1 if fruits == "Yes" else 0,
                    1 if veggies == "Yes" else 0,
                    1 if high_bp == "No" else 0,
                    1 if high_chol == "No" else 0,
                    1 if smoker == "No" else 0,
                ]
                radar_values_closed = radar_values + radar_values[:1]
                radar_angles = np.linspace(0, 2*np.pi, len(radar_categories), endpoint=False).tolist()
                radar_angles += radar_angles[:1]

                fig_radar, ax_radar = plt.subplots(figsize=(3.2, 3.2), subplot_kw={'projection': 'polar'})
                fig_radar.patch.set_facecolor(BOX_BG)
                ax_radar.set_facecolor(BOX_BG)
                ax_radar.plot(radar_angles, radar_values_closed, color=BLUE, linewidth=2)
                ax_radar.fill(radar_angles, radar_values_closed, color=BLUE, alpha=0.35)
                ax_radar.set_xticks(radar_angles[:-1])
                ax_radar.set_xticklabels(radar_categories, color='white', size=7)
                ax_radar.set_yticks([])
                ax_radar.spines['polar'].set_color('white')
                plt.tight_layout()
                st.pyplot(fig_radar, use_container_width=True)
                st.caption("Health Snapshot")

        st.write("")

        with st.container(border=True):
            st.subheader("How Likelihood Changes With Age")
            age_labels = ["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"]
            age_trend_values = []
            for a in range(1, 14):
                trend_row = person_row.copy()
                trend_row['Age'] = a
                trend_data = pd.DataFrame([trend_row])[X_train.columns]
                age_trend_values.append(model.predict_proba(trend_data)[0][1])

            fig_trend, ax_trend = plt.subplots(figsize=(7, 2.8))
            fig_trend.patch.set_facecolor(BOX_BG)
            ax_trend.set_facecolor(BOX_BG)
            ax_trend.plot(range(1, 14), age_trend_values, color=BLUE, linewidth=2, marker='o', markersize=4)
            ax_trend.axvline(person_row['Age'], color=AMBER, linewidth=2, linestyle='--')
            ax_trend.text(person_row['Age'], max(age_trend_values)*0.9, ' Your age', color='white', fontweight='bold')
            ax_trend.set_xticks(range(1, 14))
            ax_trend.set_xticklabels(age_labels, rotation=45, ha='right', color='white', fontsize=7)
            ax_trend.tick_params(colors='white')
            ax_trend.set_ylabel('Likelihood Estimate', color='white')
            for spine in ax_trend.spines.values():
                spine.set_color('white')
            plt.tight_layout()
            st.pyplot(fig_trend, use_container_width=True)
            st.caption("Illustrative: how the estimate would change across age brackets, everything else held the same.")

        st.write("")

        with st.container(border=True):
            st.subheader("Compared to Your Age Group")
            same_age_group = X_train[X_train['Age'] == person_row['Age']]
            if len(same_age_group) >= 20:
                age_sample = same_age_group.sample(n=min(100, len(same_age_group)), random_state=42)
                age_shap_values = explainer(age_sample)
                mean_abs_shap = np.abs(age_shap_values[:, :, 1].values).mean(axis=0)
                age_top3 = pd.Series(mean_abs_shap, index=X_train.columns).sort_values(ascending=False).head(3)
                st.write("Factors that matter most for your age group, on average:")
                for factor in age_top3.index:
                    st.write(f"- {factor}")
            else:
                st.write("Not enough people in this age group for a reliable comparison.")

    RECOMMENDATION_LIBRARY = {
        'PhysActivity': "Your activity level is a contributor to your result. The NZ Ministry of Health recommends at least 150 minutes of moderate-intensity activity per week for heart health.",
        'Smoker': "Smoking status is a contributor to your result. Quitline NZ (0800 778 778) provides free, confidential support. Blood pressure begins to fall within 20 minutes of quitting.",
        'HvyAlcoholConsump': "Your reported alcohol intake is a contributor to your result. Try to stay under 14 drinks/week (men) or 7 drinks/week (women), the threshold used in this model. Source: NZ Health Promotion Agency.",
        'Fruits': "Fruit intake is a contributor to your result. The NZ '5+ A Day' guideline recommends at least 2 servings of fruit daily.",
        'Veggies': "Vegetable intake is a contributor to your result. The NZ '5+ A Day' guideline recommends at least 5 servings of vegetables daily.",
        'HighBP': "Blood pressure is a contributor to your result. Two things help most: eat less salt (under 5g/day, about 1 teaspoon), and stay active (150+ min/week). Target: 120/80 mmHg or below. Source: Heart Foundation NZ.",
        'HighChol': "Cholesterol is a contributor to your result. Swap butter and fatty meat for nuts, olive oil, and oats. Target: under 5.5 mmol/L. Source: Heart Foundation NZ.",
        'BMI_high': "Your BMI is above the healthy range. Heart Foundation NZ notes a healthy BMI range is 18.5-24.9, best discussed with your GP alongside other factors.",
    }

    POSITIVE_MESSAGES = {
        'PhysActivity': "You're meeting the recommended activity level (150+ min/week). Keep it up.",
        'Smoker': "You don't smoke - one of the most protective things you can do for your heart.",
        'HvyAlcoholConsump': "Your alcohol intake is within a healthy range for your sex.",
        'Fruits': "You're meeting the NZ '5+ A Day' fruit guideline (2+ servings daily).",
        'Veggies': "You're meeting the NZ '5+ A Day' vegetable guideline (5+ servings daily).",
        'HighBP': "Your blood pressure is working in your favour. Target is 120/80 mmHg or below.",
        'HighChol': "Your cholesterol is working in your favour. Target is below 5.5 mmol/L.",
        'BMI_healthy': "Your BMI is within the healthy range (18.5-24.9).",
    }

    SUPPORTIVE_MESSAGES = {
        'GenHlth': "Your self-rated general health is one of the factors linked to this result. If you haven't discussed your overall wellbeing with your GP recently, it may be worth raising.",
        'MentHlth': "The number of recent days your mental health hasn't been great is a factor here. If this has been ongoing, it's worth mentioning to your GP.",
        'PhysHlth': "The number of recent days your physical health hasn't been great is a factor here. If this has been ongoing, it's worth mentioning to your GP.",
        'Stroke': "A history of stroke is a factor in this result. Ongoing management with your healthcare provider is the appropriate path here, rather than a lifestyle change.",
        'Diabetes': "Diabetes status is a factor in this result. Ongoing management with your healthcare provider is the appropriate path here, rather than a lifestyle change.",
        'DiffWalk': "Difficulty walking or climbing stairs is a factor in this result. Since this can have many different causes, it's best discussed directly with your GP.",
        'CholCheck': "You haven't had a cholesterol check in the past 5 years. Heart Foundation NZ recommends getting this checked, especially from age 45 onward.",
        'NoDocbcCost': "It looks like cost may have been a barrier to seeing a doctor recently. Community health services and budget GP options may be available in your area - Healthline NZ (0800 611 116) can help you find local options.",
    }

    supportive_notes = []
    if chol_check == "No": supportive_notes.append(SUPPORTIVE_MESSAGES['CholCheck'])
    if no_doc_cost == "Yes": supportive_notes.append(SUPPORTIVE_MESSAGES['NoDocbcCost'])
    if stroke == "Yes": supportive_notes.append(SUPPORTIVE_MESSAGES['Stroke'])
    if diabetes != "No": supportive_notes.append(SUPPORTIVE_MESSAGES['Diabetes'])
    if diff_walk == "Yes": supportive_notes.append(SUPPORTIVE_MESSAGES['DiffWalk'])
    if gen_health_value >= 4: supportive_notes.append(SUPPORTIVE_MESSAGES['GenHlth'])
    if ment_hlth_days >= 14: supportive_notes.append(SUPPORTIVE_MESSAGES['MentHlth'])
    if phys_hlth_days >= 14: supportive_notes.append(SUPPORTIVE_MESSAGES['PhysHlth'])

    candidate_with_impact = []
    for feature in candidate_risks:
        if feature == 'BMI_high':
            candidate_with_impact.append((feature, None, 999))
            continue
        changed_row = person_row.copy()
        changed_row[feature] = 1.0 if feature in ['PhysActivity', 'Fruits', 'Veggies'] else 0.0
        changed_data = pd.DataFrame([changed_row])[X_train.columns]
        changed_proba = model.predict_proba(changed_data)[0][1]
        impact = likelihood - changed_proba
        candidate_with_impact.append((feature, changed_proba, impact))

    candidate_with_impact.sort(key=lambda x: x[2], reverse=True)

    risk_factors = []
    if candidate_with_impact:
        risk_factors.append((candidate_with_impact[0][0], candidate_with_impact[0][1]))
        for feature, changed_proba, impact in candidate_with_impact[1:]:
            if impact >= 0.01 and len(risk_factors) < 2:
                risk_factors.append((feature, changed_proba))

    temp_dir = tempfile.mkdtemp()
    gauge_path = os.path.join(temp_dir, 'gauge.png')
    hist_path = os.path.join(temp_dir, 'hist.png')
    bar_path = os.path.join(temp_dir, 'bar.png')
    donut_path = os.path.join(temp_dir, 'donut.png')
    radar_path = os.path.join(temp_dir, 'radar.png')
    trend_path = os.path.join(temp_dir, 'trend.png')

    fig_gauge.savefig(gauge_path, dpi=150, bbox_inches='tight', facecolor='white')
    fig_hist.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)
    fig_bar.savefig(bar_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)
    fig_donut.savefig(donut_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)
    fig_radar.savefig(radar_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)
    fig_trend.savefig(trend_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)

    st.session_state['risk_factors'] = risk_factors
    st.session_state['good_factors'] = good_factors
    st.session_state['supportive_notes'] = supportive_notes
    st.session_state['likelihood'] = likelihood
    st.session_state['percentile'] = percentile
    st.session_state['RECOMMENDATION_LIBRARY'] = RECOMMENDATION_LIBRARY
    st.session_state['POSITIVE_MESSAGES'] = POSITIVE_MESSAGES
    st.session_state['chart_paths'] = {
        'gauge': gauge_path, 'hist': hist_path, 'bar': bar_path,
        'donut': donut_path, 'radar': radar_path, 'trend': trend_path
    }
    st.session_state['show_results'] = True

if st.session_state.get('show_results'):
    st.write("---")
    risk_factors = st.session_state['risk_factors']
    good_factors = st.session_state['good_factors']
    supportive_notes = st.session_state['supportive_notes']
    likelihood = st.session_state['likelihood']
    percentile = st.session_state['percentile']
    RECOMMENDATION_LIBRARY = st.session_state['RECOMMENDATION_LIBRARY']
    POSITIVE_MESSAGES = st.session_state['POSITIVE_MESSAGES']
    chart_paths = st.session_state['chart_paths']

    def show_recommendations():
        if risk_factors:
            st.subheader(f"{len(risk_factors)} Recommendation(s) For You")
            for feature, changed_proba in risk_factors:
                st.markdown(f"**{feature}**")
                st.write(RECOMMENDATION_LIBRARY[feature])
                if changed_proba is not None:
                    st.write(f"*Illustrative model scenario: if {feature} changed, likelihood would move from {likelihood:.1%} to {changed_proba:.1%}*")
                st.write("")

    def show_good_factors():
        if good_factors:
            st.subheader(f"👍 What's Already Working In Your Favour ({len(good_factors)})")
            for feature in good_factors:
                st.write(f"- {POSITIVE_MESSAGES[feature]}")

    def show_supportive_notes():
        if supportive_notes:
            st.subheader("📋 Worth Knowing")
            for note in supportive_notes:
                st.write(f"- {note}")

    if likelihood < 0.10:
        st.success("✅ Your result looks positive!")
        show_good_factors()
        show_recommendations()
        if not risk_factors:
            st.write("Keep up your current habits, and continue regular check-ins with your GP.")
    else:
        show_recommendations()
        show_good_factors()

    show_supportive_notes()

    st.write("---")
    st.subheader("Download Your Summary")
    st.write("Includes your charts, recommendations, and GP discussion points.")

    def clean_text(text):
        return text.encode('ascii', 'ignore').decode('ascii')

    pdf = FPDF()
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, "Cardiovascular Health-Awareness Summary")
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, "Empowering Patient-Led Heart Health in Aotearoa")
    pdf.ln(4)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, f"Likelihood estimate: {likelihood:.1%}")
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, clean_text(f"Compared to others in this dataset: higher than {percentile:.0f}% of people"))
    pdf.ln(2)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, clean_text("This is a dataset-based likelihood estimate, not a validated clinical risk score. Please discuss this result with your GP alongside your full medical history."))
    pdf.ln(4)

    pdf.image(chart_paths['gauge'], x=70, w=70)
    pdf.ln(3)
    pdf.image(chart_paths['hist'], x=25, w=160)
    pdf.ln(3)
    pdf.image(chart_paths['bar'], x=25, w=160)
    pdf.ln(3)

    pdf.add_page()
    pdf.set_x(15)
    pdf.image(chart_paths['donut'], x=15, w=85)
    pdf.image(chart_paths['radar'], x=105, w=85)
    pdf.ln(3)
    pdf.image(chart_paths['trend'], x=25, w=160)
    pdf.ln(5)

    if risk_factors:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, "Recommendations to Discuss with Your GP")
        for feature, changed_proba in risk_factors:
            pdf.set_x(15)
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, clean_text(feature))
            pdf.set_x(15)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, clean_text(RECOMMENDATION_LIBRARY[feature]))
            pdf.ln(2)

    if good_factors:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, "What's Working in Your Favour")
        pdf.set_font("Helvetica", "", 10)
        for feature in good_factors:
            pdf.set_x(15)
            pdf.multi_cell(0, 5, clean_text(f"- {POSITIVE_MESSAGES[feature]}"))

    pdf.ln(2)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, clean_text("This is not a diagnosis. Generated for informational purposes to support a conversation with your GP."))

    pdf_bytes = bytes(pdf.output())
    st.download_button(
        label="Download PDF Summary (with charts)",
        data=pdf_bytes,
        file_name="heart_health_summary.pdf",
        mime="application/pdf"
    )
