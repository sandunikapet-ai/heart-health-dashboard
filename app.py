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

GREEN = '#27AE60'
RED = '#E74C3C'
AMBER = '#F39C12'
BLUE = '#5DADE2'
BOX_BG = '#1B3A5C'

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
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

model = joblib.load('model_ebm.pkl')
X_train = pd.read_csv('X_train.csv')

if 'view' not in st.session_state:
    st.session_state['view'] = 'form'

# ============================================================
# FORM VIEW
# ============================================================
if st.session_state['view'] == 'form':

        logo_col, attr_col = st.columns([5, 1.3])
    with logo_col:
        inner1, inner2, inner3 = st.columns([2, 1, 2])
        with inner2:
            st.image("logo.png", width=160)
    with attr_col:
        st.markdown("""
        <div style='background-color:#1B3A5C; padding:10px; border-radius:8px; text-align:center; font-size:10px; color:#5DADE2; margin-top:20px; border:1px solid #4A9EFF;'>
            Developed by<br>Petronilda Biyanwila<br>Yoobee College of<br>Creative Innovation<br>MBI908 Capstone
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    get_result_top = st.button("Get My Result", use_container_width=True, key="btn_top")
    st.caption("Fill in the sections below, then click here (or the button at the bottom) when ready.")
    st.write("---")

    step1, step2, step3 = st.columns(3)

    with step1:
        with st.container(border=True):
            st.subheader("Step 1: Required Info")
            st.caption("Yes/No answers, since guessing could mislead the result.")
            high_bp = st.radio("High blood pressure diagnosed by a doctor?", ["No", "Yes"])
            high_chol = st.radio("High cholesterol diagnosed by a doctor?", ["No", "Yes"])
            st.caption("Asks if a doctor EVER told you this, even years ago - may not reflect current level if not rechecked recently.")
            chol_check = st.radio("Cholesterol checked in the past 5 years?", ["No", "Yes"])
            st.caption("Asks if you've had a test at all, regardless of result.")
            stroke = st.radio("Ever told you had a stroke?", ["No", "Yes"])
            diabetes = st.selectbox("Do you have diabetes?", ["No", "Pre-diabetes/borderline", "Yes"])

    with step2:
        with st.container(border=True):
            st.subheader("Step 2: About You")
            st.caption("If unsure about income/education, select 'Unsure'.")
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

    with step3:
        with st.container(border=True):
            st.subheader("Step 3: Lifestyle")
            st.caption("These directly shape your recommendations.")
            smoker = st.radio(
                "Smoked 100+ cigarettes in your lifetime (~5 packs)? Includes past smoking, even if quit.",
                ["No", "Yes"]
            )
            st.markdown("**Activity**")
            st.caption("🔴 Low: 0 min/wk. 🟡 Moderate: ~10-25 min/day avg. 🟢 High: 30+ min/day most days.")
            activity_level = st.selectbox("Your activity level?", ["Low", "Moderate", "High"])
            phys_activity_live = activity_level != "Low"

            fruits_per_day = st.number_input("Fruit servings/day", min_value=0, max_value=10, value=0)
            fruits = "Yes" if fruits_per_day >= 1 else "No"
            veggies_per_day = st.number_input("Vegetable servings/day", min_value=0, max_value=10, value=0)
            veggies = "Yes" if veggies_per_day >= 1 else "No"
            st.caption("🟢 Targets: 2+/day fruit, 5+/day veg (NZ '5+ A Day').")

            st.markdown("**Heavy Alcohol**")
            if sex == "Male":
                st.caption("🔴 Heavy: more than 14 drinks/week (men).")
            else:
                st.caption("🔴 Heavy: more than 7 drinks/week (women).")
            st.caption("1 drink ≈ 330ml beer(4%), 100ml wine(12.5%), or 30ml spirits(42%).")
            hvy_alcohol = st.radio("Is your drinking heavy?", ["No", "Yes"])

    step4, step5 = st.columns(2)

    with step4:
        with st.container(border=True):
            st.subheader("Step 4: Body Measurements")
            height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            height_inches = height_cm / 2.54
            st.caption(f"≈ {height_inches:.1f} in")
            weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
            weight_lb = weight_kg * 2.20462
            st.caption(f"≈ {weight_lb:.1f} lb")
            bmi_calculated = weight_kg / ((height_cm / 100) ** 2)
            st.write(f"BMI: **{bmi_calculated:.1f}**")

    with step5:
        with st.container(border=True):
            st.subheader("Step 5: Additional Health Info")
            gen_health = st.selectbox("Rate your general health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
            with st.expander("What do these mean?"):
                st.caption("🟢 **Excellent**: No ongoing problems; full activity ability.")
                st.caption("🟢 **Very good**: Minor, infrequent issues, no limits.")
                st.caption("🟡 **Good**: A manageable condition may exist; most activities unaffected.")
                st.caption("🟠 **Fair**: A condition limits some activities sometimes.")
                st.caption("🔴 **Poor**: A condition limits activities most of the time.")
            ment_hlth_days = st.number_input("Days mental health wasn't good (past 30)", min_value=0, max_value=30, value=0)
            phys_hlth_days = st.number_input("Days physical health wasn't good (past 30)", min_value=0, max_value=30, value=0)
            diff_walk = st.radio("Serious difficulty walking/climbing stairs?", ["No", "Yes"])
            st.caption("Includes cardiovascular causes (breathlessness from heart failure) and others (stroke weakness, arthritis, COPD).")
            any_healthcare = st.radio("Have health care coverage?", ["Yes", "No"])
            no_doc_cost = st.radio("Skipped a doctor visit due to cost (past year)?", ["No", "Yes"])

    get_result_bottom = st.button("Get My Result", use_container_width=True, key="btn_bottom")

    with st.expander("What do these terms mean?"):
        st.write("**Likelihood estimate**: How closely your profile matches patterns linked to heart disease in this dataset. This is not a diagnosis or a guaranteed future outcome.")
        st.write("**Contributing factor**: Shows which of your answers had the biggest effect on your result, and whether each one pushed it up or down.")
        st.write("**Illustrative model scenario**: A 'what-if' example showing how the estimate would change if one factor changed. It reflects patterns in the data, not a promise about your real health.")
        st.write("**Percentile**: Shows where your result sits compared to everyone else in this dataset, not a clinical category.")

    get_result = get_result_top or get_result_bottom

    if get_result:
        phys_activity = "No" if activity_level == "Low" else "Yes"
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
            tier_label, tier_color = "🔴 HIGHER LIKELIHOOD", "red"
        elif likelihood >= 0.10:
            tier_label, tier_color = "🟡 MODERATE LIKELIHOOD", "orange"
        else:
            tier_label, tier_color = "🟢 LOWER LIKELIHOOD", "green"

        all_train_probas = model.predict_proba(X_train)[:, 1]
        percentile = (all_train_probas < likelihood).mean() * 100

        hvy_alcohol_live = hvy_alcohol == "Yes"
        snapshot_labels = ["High BP", "High Chol", "Stroke", "Smoker", "Active", "Fruits", "Veggies", "Alcohol"]
        snapshot_values = [
            1 if high_bp == "Yes" else 0, 1 if high_chol == "Yes" else 0,
            1 if stroke == "Yes" else 0, 1 if smoker == "Yes" else 0,
            1 if phys_activity_live else 0, 1 if fruits == "Yes" else 0,
            1 if veggies == "Yes" else 0, 1 if hvy_alcohol_live else 0,
        ]
        protective_when_yes = {"Active", "Fruits", "Veggies"}
        snapshot_colors = []
        for label, val in zip(snapshot_labels, snapshot_values):
            if label in protective_when_yes:
                snapshot_colors.append(GREEN if val == 1 else RED)
            else:
                snapshot_colors.append(RED if val == 1 else GREEN)
        fig_snap, ax_snap = plt.subplots(figsize=(6, 3))
        fig_snap.patch.set_facecolor(BOX_BG)
        ax_snap.set_facecolor(BOX_BG)
        ax_snap.barh(snapshot_labels[::-1], [1]*8, color=snapshot_colors[::-1])
        ax_snap.set_xlim(0, 1)
        ax_snap.set_xticks([])
        ax_snap.tick_params(colors='white')
        for spine in ax_snap.spines.values():
            spine.set_visible(False)
        plt.tight_layout()

        fig_gauge, ax_gauge = plt.subplots(figsize=(3.6, 2.2), subplot_kw={'projection': 'polar'})
        fig_gauge.patch.set_alpha(0)
        gauge_colors = [GREEN, AMBER, RED]
        bounds = [0, 0.10, 0.30, 1.0]
        for i in range(3):
            theta1 = np.pi * (1 - bounds[i])
            theta2 = np.pi * (1 - bounds[i+1])
            ax_gauge.bar(x=(theta1+theta2)/2, height=1, width=abs(theta1-theta2), bottom=2, color=gauge_colors[i], edgecolor=BOX_BG, linewidth=2)
        needle_angle = np.pi * (1 - likelihood)
        ax_gauge.plot([needle_angle, needle_angle], [0, 2.3], color='white', linewidth=3)
        ax_gauge.set_theta_zero_location('W')
        ax_gauge.set_theta_direction(1)
        ax_gauge.set_thetamin(0)
        ax_gauge.set_thetamax(180)
        ax_gauge.set_ylim(0, 3)
        ax_gauge.axis('off')
        plt.tight_layout()

        explainer = shap.Explainer(model.predict_proba, X_train, feature_names=X_train.columns.tolist())
        shap_values = explainer(person_data)
        person_shap = shap_values[0, :, 1].values
        feature_names = X_train.columns.tolist()
        top5_contributions = sorted(zip(feature_names, person_shap), key=lambda x: abs(x[1]), reverse=True)[:5]
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

        good_factors, candidate_risks = [], []
        good_factor_labels = {'Smoker': 'Not smoking', 'PhysActivity': 'Physical activity', 'Fruits': 'Fruit intake',
                               'Veggies': 'Vegetable intake', 'HvyAlcoholConsump': 'Alcohol intake',
                               'HighBP': 'Blood pressure', 'HighChol': 'Cholesterol', 'BMI_healthy': 'BMI'}
        risk_factor_labels = {'Smoker': 'Smoking', 'PhysActivity': 'Low physical activity', 'Fruits': 'Low fruit intake',
                               'Veggies': 'Low vegetable intake', 'HvyAlcoholConsump': 'Heavy alcohol consumption',
                               'HighBP': 'Blood pressure', 'HighChol': 'Cholesterol', 'BMI_high': 'BMI'}
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

        fig_donut, ax_donut = plt.subplots(figsize=(3.2, 3.2))
        fig_donut.patch.set_facecolor(BOX_BG)
        n_good, n_risk = len(good_factors), len(candidate_risks)
        if n_good + n_risk > 0:
            ax_donut.pie([n_good, n_risk], colors=[GREEN, RED], startangle=90, wedgeprops=dict(width=0.4, edgecolor=BOX_BG, linewidth=3))
            ax_donut.text(0, 0, f"{n_good}/{n_good+n_risk}\nHealthy", ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        plt.tight_layout()

        radar_categories = ['Activity', 'Fruits', 'Veggies', 'BP Health', 'Chol Health', 'Non-Smoker']
        radar_values = [1 if phys_activity == "Yes" else 0, 1 if fruits == "Yes" else 0, 1 if veggies == "Yes" else 0,
                         1 if high_bp == "No" else 0, 1 if high_chol == "No" else 0, 1 if smoker == "No" else 0]
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
        bar_path = os.path.join(temp_dir, 'bar.png')
        donut_path = os.path.join(temp_dir, 'donut.png')
        radar_path = os.path.join(temp_dir, 'radar.png')
        trend_path = os.path.join(temp_dir, 'trend.png')
        snap_path = os.path.join(temp_dir, 'snap.png')
        fig_gauge.savefig(gauge_path, dpi=150, bbox_inches='tight', facecolor='white')
        fig_bar.savefig(bar_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)
        fig_donut.savefig(donut_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)
        fig_radar.savefig(radar_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)
        fig_trend.savefig(trend_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)
        fig_snap.savefig(snap_path, dpi=150, bbox_inches='tight', facecolor=BOX_BG)

        st.session_state['risk_factors'] = risk_factors
        st.session_state['good_factors'] = good_factors
        st.session_state['candidate_risks'] = candidate_risks
        st.session_state['supportive_notes'] = supportive_notes
        st.session_state['likelihood'] = likelihood
        st.session_state['percentile'] = percentile
        st.session_state['tier_label'] = tier_label
        st.session_state['tier_color'] = tier_color
        st.session_state['bmi_calculated'] = bmi_calculated
        st.session_state['bmi_status'] = "Healthy range" if 18.5 <= bmi_calculated <= 24.9 else "Outside healthy range"
        st.session_state['RECOMMENDATION_LIBRARY'] = RECOMMENDATION_LIBRARY
        st.session_state['POSITIVE_MESSAGES'] = POSITIVE_MESSAGES
        st.session_state['good_factor_labels'] = good_factor_labels
        st.session_state['risk_factor_labels'] = risk_factor_labels
        st.session_state['chart_paths'] = {'gauge': gauge_path, 'bar': bar_path, 'donut': donut_path, 'radar': radar_path, 'trend': trend_path, 'snapshot': snap_path}
        st.session_state['view'] = 'results'
        st.rerun()

# ============================================================
# RESULTS VIEW
# ============================================================
elif st.session_state['view'] == 'results':
    if st.button("← Back to Form"):
        st.session_state['view'] = 'form'
        st.rerun()

    risk_factors = st.session_state['risk_factors']
    good_factors = st.session_state['good_factors']
    candidate_risks = st.session_state['candidate_risks']
    supportive_notes = st.session_state['supportive_notes']
    likelihood = st.session_state['likelihood']
    percentile = st.session_state['percentile']
    tier_label = st.session_state['tier_label']
    tier_color = st.session_state['tier_color']
    bmi_calculated = st.session_state['bmi_calculated']
    bmi_status = st.session_state['bmi_status']
    RECOMMENDATION_LIBRARY = st.session_state['RECOMMENDATION_LIBRARY']
    POSITIVE_MESSAGES = st.session_state['POSITIVE_MESSAGES']
    good_factor_labels = st.session_state['good_factor_labels']
    risk_factor_labels = st.session_state['risk_factor_labels']
    chart_paths = st.session_state['chart_paths']

    st.header("Your Result")
    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        st.metric("Likelihood Estimate", f"{likelihood:.1%}")
        st.markdown(f":{tier_color}[**{tier_label}**]")
        st.caption(f"Higher than {percentile:.0f}% of people in this dataset. Dataset-based estimate, not a clinical risk score.")
        st.write(f"BMI: **{bmi_calculated:.1f}** ({bmi_status})")
    with res_col2:
        st.image(chart_paths['gauge'])

    st.write("---")

    if likelihood < 0.10:
        st.success("✅ Your result looks positive!")
        if good_factors:
            st.subheader(f"👍 What's Already Working In Your Favour ({len(good_factors)})")
            for f in good_factors:
                st.write(f"- {POSITIVE_MESSAGES[f]}")
        if risk_factors:
            st.subheader(f"{len(risk_factors)} Recommendation(s) For You")
            for feature, changed_proba in risk_factors:
                st.markdown(f"**{feature}**")
                st.write(RECOMMENDATION_LIBRARY[feature])
                if changed_proba is not None:
                    st.write(f"*Illustrative: if {feature} changed, likelihood would move from {likelihood:.1%} to {changed_proba:.1%}*")
        if not risk_factors:
            st.write("Keep up your current habits, and continue regular check-ins with your GP.")
    else:
        if risk_factors:
            st.subheader(f"{len(risk_factors)} Recommendation(s) For You")
            for feature, changed_proba in risk_factors:
                st.markdown(f"**{feature}**")
                st.write(RECOMMENDATION_LIBRARY[feature])
                if changed_proba is not None:
                    st.write(f"*Illustrative: if {feature} changed, likelihood would move from {likelihood:.1%} to {changed_proba:.1%}*")
        if good_factors:
            st.subheader(f"👍 What's Already Working In Your Favour ({len(good_factors)})")
            for f in good_factors:
                st.write(f"- {POSITIVE_MESSAGES[f]}")

    if supportive_notes:
        st.subheader("📋 Worth Knowing")
        for note in supportive_notes:
            st.write(f"- {note}")

    st.write("---")
    st.subheader("Charts")

    ch1, ch2 = st.columns(2)
    with ch1:
        st.image(chart_paths['snapshot'], caption="Live Snapshot of Your Answers")
    with ch2:
        st.image(chart_paths['bar'], caption="Top 5 Contributing Factors")

    ch3, ch4 = st.columns(2)
    with ch3:
        st.image(chart_paths['donut'], caption="Habits Breakdown")
        list_col1, list_col2 = st.columns(2)
        with list_col1:
            st.markdown(f":green[**✅ Good ({len(good_factors)})**]")
            for f in good_factors:
                st.write(f"- {good_factor_labels[f]}")
        with list_col2:
            st.markdown(f":red[**⚠️ Needs Improvement ({len(candidate_risks)})**]")
            if candidate_risks:
                for f in candidate_risks:
                    st.write(f"- {risk_factor_labels[f]}")
            else:
                st.write("- None")
    with ch4:
        st.image(chart_paths['radar'], caption="Health Snapshot")
        st.image(chart_paths['trend'], caption="How Likelihood Changes With Age")

    st.write("---")
    st.subheader("Download Your Summary")

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
    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, f"Likelihood estimate: {likelihood:.1%}")
    pdf.set_x(15)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, clean_text(f"Compared to others in this dataset: higher than {percentile:.0f}% of people"))
    pdf.ln(2)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, clean_text("This is a dataset-based likelihood estimate, not a validated clinical risk score. Please discuss this result with your GP alongside your full medical history."))
    pdf.ln(4)

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
    pdf.ln(4)

    pdf.add_page()
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, "Charts")
    pdf.ln(2)
    pdf.image(chart_paths['gauge'], x=70, w=70)
    pdf.ln(3)
    pdf.image(chart_paths['snapshot'], x=25, w=160)
    pdf.ln(3)
    pdf.image(chart_paths['bar'], x=25, w=160)
    pdf.ln(3)
    pdf.image(chart_paths['donut'], x=15, w=85)
    pdf.image(chart_paths['radar'], x=105, w=85)
    pdf.ln(3)
    pdf.image(chart_paths['trend'], x=25, w=160)
    pdf.ln(5)

    pdf.set_x(15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, clean_text("This is not a diagnosis. Generated for informational purposes to support a conversation with your GP."))

    pdf_bytes = bytes(pdf.output())
    st.download_button(label="Download PDF Summary", data=pdf_bytes, file_name="heart_health_summary.pdf", mime="application/pdf")
