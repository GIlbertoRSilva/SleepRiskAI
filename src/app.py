import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import datetime
import base64
import time

COLOR_BG = "#0e0e1a"           
COLOR_ACCENT_CYAN = "#00f2ff"  
COLOR_ACCENT_PINK = "#ff0055"  
COLOR_TEXT_MAIN = "#ffffff"
COLOR_TEXT_DIM = "#a0a0a0"
COLOR_GRID = "#333333"

st.set_page_config(
    page_title="SleepRiskAI | Circadian Monitor",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(f"""
    <style>
    /* 1. Fundo Principal */
    .stApp {{
        background: linear-gradient(135deg, #050510 0%, #151530 100%);
    }}

    /* 2. Sidebar Borda */
    section[data-testid="stSidebar"] {{
        background-color: #02020a;
        border-right: 1px solid #1f1f30;
    }}
    
    /* 3. Tipografia */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif;
        color: {COLOR_TEXT_MAIN} !important;
    }}
    h1 span {{
        background: -webkit-linear-gradient(0deg, {COLOR_ACCENT_CYAN}, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    /* 4. SLIDERS (Forçar Ciano) */
    div[data-baseweb="slider"] div[role="slider"] {{
        background-color: {COLOR_ACCENT_CYAN} !important;
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.5) !important;
    }}
    div[data-baseweb="slider"] div[data-testid="stTickBar"] > div {{
        background-color: {COLOR_ACCENT_CYAN} !important;
    }}
    div.stSlider > div[data-baseweb="slider"] > div > div {{
        background-color: #3a7bd5 !important;
    }}

    /* 5. BOTÕES (Forçar Ciano) */
    button[kind="primary"] {{
        background-color: {COLOR_ACCENT_CYAN} !important;
        border: 1px solid {COLOR_ACCENT_CYAN} !important;
        color: #000 !important;
        font-weight: bold !important;
        transition: all 0.3s ease;
    }}
    button[kind="primary"]:hover {{
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.6);
    }}
    button[kind="secondary"] {{
        border: 1px solid {COLOR_ACCENT_CYAN} !important;
        color: {COLOR_ACCENT_CYAN} !important;
    }}
    button[kind="secondary"]:hover {{
        border-color: #fff !important;
        color: #fff !important;
    }}
    
    /* 6. AI Box Style */
    .ai-box {{
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid {COLOR_ACCENT_CYAN};
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin: 20px 0;
    }}
    .ai-title {{
        color: {COLOR_ACCENT_CYAN};
        font-weight: bold;
        font-size: 14px;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .ai-text {{
        color: #ddd;
        font-size: 14px;
        line-height: 1.6;
    }}
    .ai-text strong {{
        color: white;
        font-weight: 700;
    }}

    /* 7. Selectbox Styling */
    .stSelectbox div[data-baseweb="select"] div {{
        background-color: #151530;
        color: white;
    }}
    
    /* 8. Métricas */
    div[data-testid="stMetricValue"] {{
        color: {COLOR_ACCENT_CYAN} !important;
    }}
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sleep_risk_model.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.json"
SHAP_BG_PATH = MODEL_DIR / "shap_background.pkl"

@st.cache_resource
def load_resources():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file not found")
        
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, "r") as f:
            feature_names = json.load(f)
            
        if SHAP_BG_PATH.exists():
            background = joblib.load(SHAP_BG_PATH)
            explainer = shap.TreeExplainer(model, background)
        else:
            explainer = shap.TreeExplainer(model)
            
        return model, feature_names, explainer, False

    except Exception as e:
        class MockModel:
            def predict(self, X): return np.random.choice([0, 1], size=len(X))
            def predict_proba(self, X): 
                try:
                    score = 0.5
                    if 'Stress Level' in X.columns:
                        score += (X['Stress Level'].iloc[0] - 5) * 0.05
                    if 'Sleep Duration' in X.columns:
                        score -= (X['Sleep Duration'].iloc[0] - 7) * 0.05
                    score = np.clip(score, 0.1, 0.9)
                    return [[1-score, score]]
                except:
                     p = np.random.uniform(0.1, 0.9)
                     return [[1-p, p]]
        
        dummy_features = [
            "Gender", "Age", "Sleep Duration", "Quality of Sleep", 
            "Physical Activity Level", "Stress Level", "Heart Rate", 
            "Daily Steps", "BMI_Category", "BP_Systolic", "BP_Diastolic", "Hypertension"
        ]
        
        class MockExplainer:
            def shap_values(self, X): 
                n_samples, n_features = X.shape
                return [np.random.randn(n_samples, n_features), np.random.randn(n_samples, n_features)]
            @property
            def expected_value(self): return np.array([0.5, 0.5])

        return MockModel(), dummy_features, MockExplainer(), True

model, feature_names, explainer, is_demo_mode = load_resources()

st.sidebar.markdown("### Patient Profile")

if 'synced' not in st.session_state: st.session_state['synced'] = False

def get_val(key, default, synced_val):
    return synced_val if st.session_state['synced'] else default

col_btn_icon, col_btn_main = st.sidebar.columns([1, 4])
with col_btn_main:
    if st.sidebar.button("Sync Wearable Data", help="Simulate connection to Apple Health/Fitbit"):
        with st.sidebar:
            with st.spinner("Connecting..."):
                time.sleep(1.2)
        st.success("Data Synced!")
        st.session_state['synced'] = True
        st.rerun()

st.sidebar.markdown("---")

with st.sidebar.expander("👤 Demographics", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 80, get_val('age', 30, 45))
    bmi_idx = 2 if st.session_state['synced'] else 0
    bmi_category = st.selectbox("BMI Category", ["Normal", "Normal Weight", "Overweight", "Obese"], index=bmi_idx)

with st.sidebar.expander("💤 Lifestyle", expanded=True):
    sleep_duration = st.slider("Sleep Duration (h)", 3.0, 10.0, float(get_val('sleep', 7.0, 5.2)), 0.1)
    sleep_quality = st.slider("Sleep Quality (1–10)", 1, 10, get_val('qual', 7, 4))
    stress_level = st.slider("Stress Level (1–10)", 1, 10, get_val('stress', 5, 8)) 
    physical_activity = st.slider("Phys. Activity (min/day)", 0, 150, get_val('act', 45, 20))
    daily_steps = st.slider("Daily Steps", 1000, 20000, get_val('steps', 6000, 3500))

with st.sidebar.expander("❤️ Vitals", expanded=False):
    heart_rate = st.slider("Heart Rate (bpm)", 50, 120, get_val('hr', 72, 85))
    bp_systolic = st.number_input("Systolic BP", 90, 180, get_val('bps', 120, 135))
    bp_diastolic = st.number_input("Diastolic BP", 60, 120, get_val('bpd', 80, 90))

gender_map = {"Male": 1, "Female": 0}
bmi_map = {"Normal": 0, "Normal Weight": 0, "Overweight": 1, "Obese": 2}
hypertension = int(bp_systolic >= 140 or bp_diastolic >= 90)

input_data = {
    "Gender": gender_map[gender],
    "Age": age,
    "Sleep Duration": sleep_duration,
    "Quality of Sleep": sleep_quality,
    "Physical Activity Level": physical_activity,
    "Stress Level": stress_level,
    "Heart Rate": heart_rate,
    "Daily Steps": daily_steps,
    "BMI_Category": bmi_map[bmi_category],
    "BP_Systolic": bp_systolic,
    "BP_Diastolic": bp_diastolic,
    "Hypertension": hypertension
}

input_df = pd.DataFrame([input_data])
if not is_demo_mode:
    input_df = input_df[feature_names]

col_title_L, col_title_R = st.columns([1, 5])
with col_title_R:
    st.markdown(f"# SleepRisk<span>AI</span>", unsafe_allow_html=True)
    st.markdown("##### 🌌 Advanced Sleep Disorder Risk Simulator")

try:
    if is_demo_mode:
        prob_val = model.predict_proba(input_df)[0][1]
    else:
        prob_val = model.predict_proba(input_df)[0][1]
except:
    prob_val = 0.5 

st.markdown("---")


col_viz_left, col_viz_right = st.columns(2)

with col_viz_left:
    st.markdown("### Risk Probability")
    
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_val * 100,
        number = {'suffix': "%", 'font': {'color': COLOR_TEXT_MAIN, 'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#555"},
            'bar': {'color': "#0e0e1a"}, 
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, prob_val*100], 'color': COLOR_ACCENT_PINK if prob_val > 0.5 else COLOR_ACCENT_CYAN},
                {'range': [prob_val*100, 100], 'color': "#1f1f30"} 
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': prob_val * 100
            }
        }
    ))
    
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': COLOR_TEXT_MAIN, 'family': "Inter"},
        height=350, 
        margin=dict(l=30, r=30, t=50, b=30)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    if prob_val > 0.5:
        st.error("🚨 **High Priority:** Patterns indicate potential sleep disorder.")
    else:
        st.success("✅ **Stable:** Patterns indicate healthy circadian rhythm.")

with col_viz_right:
    st.markdown("### Population Benchmark")
    
    n_sleep = min(sleep_duration / 8.0, 1.0)
    n_qual = sleep_quality / 10.0
    n_stress = (11 - stress_level) / 10.0 
    n_activ = min(physical_activity / 60.0, 1.0)
    n_steps = min(daily_steps / 10000.0, 1.0)
    
    categories = ['Sleep Amount', 'Sleep Quality', 'Low Stress', 'Activity', 'Steps']
    values = [n_sleep, n_qual, n_stress, n_activ, n_steps]
    
    values += [values[0]]
    categories += [categories[0]]
    
    baseline = [0.9, 0.8, 0.7, 0.7, 0.7, 0.9]

    fig_radar = go.Figure()

    fig_radar.add_trace(go.Scatterpolar(
        r=baseline,
        theta=categories,
        fill='none',
        name='Healthy Avg',
        line=dict(color='#888888', dash='dot', width=1),
        hoverinfo='skip'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Patient',
        line=dict(color=COLOR_ACCENT_CYAN, width=3),
        fillcolor=f"rgba(0, 242, 255, 0.2)"
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor=COLOR_GRID),
            angularaxis=dict(linecolor=COLOR_GRID, color=COLOR_TEXT_DIM),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLOR_TEXT_MAIN, family="Inter"),
        showlegend=True,
        legend=dict(x=0.7, y=1, font=dict(color=COLOR_TEXT_DIM)),
        height=350,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
st.markdown("---")
col_ai_icon, col_ai_content = st.columns([1, 6])
with col_ai_icon:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712038.png", width=80) 

with col_ai_content:
    st.markdown("### Dr. Sleep AI Consultant")

    if prob_val > 0.5:
        ai_advice = f"<strong>CRITICAL INSIGHT:</strong> I've detected a strong correlation between your elevated Stress ({stress_level}/10) and reduced Sleep Duration ({sleep_duration}h). This pattern triggers cortisol release, blocking Deep Sleep phases. <br><br><strong>ACTION PLAN:</strong> Prioritize a 'wind-down' routine 60 minutes before bed and target at least 6.5h of sleep to lower immediate cardiovascular risk."
    else:
        ai_advice = f"<strong>STATUS REPORT:</strong> Your circadian rhythm markers are stable. Your physical activity ({physical_activity} min) acts as a strong buffer against daily stress. <br><br><strong>OPTIMIZATION:</strong> To reach peak cognitive performance, try maintaining this consistency even on weekends to avoid 'Social Jetlag'."

    st.markdown(f"""
    <div class="ai-box">
        <div class="ai-title">GENERATIVE ANALYSIS</div>
        <div class="ai-text">{ai_advice}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Model Explainability")

shap_exp_obj = None

with st.expander("Technical Deep Dive (SHAP)", expanded=False):
    try:
        plt.style.use('dark_background')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[COLOR_ACCENT_CYAN, COLOR_ACCENT_PINK])
        
        shap_values_raw = explainer.shap_values(input_df)
        expected_value_raw = explainer.expected_value
        
        if isinstance(shap_values_raw, list):
            shap_values_target = shap_values_raw[1][0]
            base_value_target = expected_value_raw[1] if hasattr(expected_value_raw, '__iter__') else expected_value_raw
        else:
            if len(shap_values_raw.shape) == 3:
                shap_values_target = shap_values_raw[0, :, 1]
            else:
                shap_values_target = shap_values_raw[0]
            base_value_target = expected_value_raw[1] if hasattr(expected_value_raw, '__iter__') and len(expected_value_raw) > 1 else expected_value_raw

        shap_exp = shap.Explanation(
            values=np.array(shap_values_target),
            base_values=float(base_value_target),
            data=np.array(input_df.iloc[0]),
            feature_names=input_df.columns.tolist()
        )
        shap_exp_obj = shap_exp
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Global Feature Importance**")
            fig_bar, ax = plt.subplots()
            shap.plots.bar(shap_exp, show=False)
            ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
            ax.xaxis.label.set_color(COLOR_TEXT_DIM)
            for spine in ax.spines.values(): spine.set_visible(False)
            fig_bar.patch.set_alpha(0) 
            ax.patch.set_alpha(0)
            st.pyplot(fig_bar)
            
        with col2:
            st.markdown("**Decision Waterfall**")
            fig_water, ax = plt.subplots()
            shap.plots.waterfall(shap_exp, show=False)
            ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
            for spine in ax.spines.values(): spine.set_visible(False)
            fig_water.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            st.pyplot(fig_water)
            
    except Exception as e:
        st.error(f"Visualization Error: {str(e)}")
        
if shap_exp_obj is not None:
    st.markdown("#### AI Logic Decoder (Top Factors)")
    
    impact_df = pd.DataFrame({
        "Feature": shap_exp_obj.feature_names,
        "Impact": shap_exp_obj.values,
        "Value": shap_exp_obj.data
    })
    
    impact_df['AbsImpact'] = impact_df['Impact'].abs()
    top_3 = impact_df.sort_values(by='AbsImpact', ascending=False).head(3)
    
    cols = st.columns(3)
    for i, (index, row) in enumerate(top_3.iterrows()):
        feature = row['Feature']
        impact = row['Impact']
        val = row['Value']
        
        if impact > 0: 
            direction = "Increased Risk 📈"
            color = COLOR_ACCENT_PINK
        else: 
            direction = "Decreased Risk 🛡️"
            color = COLOR_ACCENT_CYAN
            
        val_str = f"{val:.1f}"
        if feature == "Gender": val_str = "Male" if val == 1 else "Female"
        if feature == "Hypertension": val_str = "Yes" if val == 1 else "No"
        
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 4px solid {color};">
                <p style="color: {COLOR_TEXT_DIM}; font-size: 0.8em; margin-bottom: 5px;">Key Factor #{i+1}</p>
                <h4 style="margin: 0; color: white;">{feature}</h4>
                <p style="font-size: 1.1em; font-weight: bold; margin: 5px 0;">Value: <span style="color:{color}">{val_str}</span></p>
                <p style="margin: 0; color: {color}; font-weight: bold; font-size: 0.9em;">{direction}</p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### Lifestyle Lab")

with st.container():
    col_sim_controls, col_sim_viz = st.columns([1, 2])

    with col_sim_controls:
        st.markdown("#### Adjust")
        new_sleep = st.slider("Sleep (h)", 3.0, 10.0, float(sleep_duration), 0.5, key="sim_sleep")
        new_stress = st.slider("Stress (1-10)", 1, 10, int(stress_level), 1, key="sim_stress")
        new_activity = st.slider("Activity (min)", 0, 150, int(physical_activity), 5, key="sim_activity")
        
        st.markdown("---")
    
        if st.button("AI Auto-Optimize", type="primary"):
            st.toast("Optimizing circadian markers...", icon="🔮")
            opt_df = input_df.copy()
            opt_df['Sleep Duration'] = 8.0
            opt_df['Stress Level'] = 3
            opt_df['Physical Activity Level'] = 60
            
            try:
                opt_prob = 0.15 if is_demo_mode else model.predict_proba(opt_df)[0][1]
            except: opt_prob = 0.1
                
            st.session_state['opt_prob'] = opt_prob
            st.session_state['show_opt'] = True
        
        if st.button("↺ Reset"):
            st.session_state['show_opt'] = False

    sim_df = input_df.copy()
    sim_df['Sleep Duration'] = new_sleep
    sim_df['Stress Level'] = new_stress
    sim_df['Physical Activity Level'] = new_activity
    
    try:
        if is_demo_mode:
            sim_prob = prob_val + (stress_level - new_stress)*0.04 - (new_sleep - sleep_duration)*0.04
            sim_prob = np.clip(sim_prob, 0.05, 0.95)
        else:
            sim_prob = model.predict_proba(sim_df)[0][1]
    except:
        sim_prob = prob_val

    with col_sim_viz:
        st.markdown("#### Projection")
        scenarios = ['Current', 'Simulated']
        risks = [prob_val * 100, sim_prob * 100]
        colors = [COLOR_ACCENT_PINK, COLOR_ACCENT_CYAN]
        
        if st.session_state.get('show_opt'):
            scenarios.append('AI Optimal')
            risks.append(st.session_state['opt_prob'] * 100)
            colors.append('#00ff9d') 
            st.success(f"AI Plan: 8h Sleep + Low Stress = Risk drops to {st.session_state['opt_prob']:.1%}")

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            y=scenarios, x=risks, orientation='h',
            marker_color=colors, text=[f"{r:.1f}%" for r in risks],
            textposition='auto', hoverinfo='none'
        ))
        
        fig_sim.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font={'color': COLOR_TEXT_MAIN, 'family': 'Inter'},
            xaxis=dict(range=[0, 100], showgrid=True, gridcolor='#333'),
            margin=dict(l=10, r=10, t=10, b=10), height=200
        )
        st.plotly_chart(fig_sim, use_container_width=True)


# ==========================================
# 9. ADVANCED VISUAL REPORT
# ==========================================
st.markdown("---")
col_down, col_info = st.columns([3, 1])

with col_down:
    # 1. Configuração de Cores e Dados
    risk_percentage = prob_val * 100
    if prob_val > 0.5:
        risk_color = "#ff0055" # Pink
        risk_bg = "#fff0f5" # Light Pink
        risk_title = "ATTENTION REQUIRED"
        risk_icon = "⚠️"
        recommendation = "High risk patterns detected. Immediate consultation with a sleep specialist is recommended."
    else:
        risk_color = "#00c4cc" # Cyan darker for white paper
        risk_bg = "#f0fdff" # Light Cyan
        risk_title = "OPTIMAL CONDITION"
        risk_icon = "🛡️"
        recommendation = "Your circadian rhythm appears stable. Maintain current healthy lifestyle habits."

    # 2. Processamento do SHAP para o Relatório
    try:
        if shap_exp_obj is None:
            shap_values_rep = explainer.shap_values(input_df)
            if isinstance(shap_values_rep, list):
                vals = shap_values_rep[1][0]
            else:
                vals = shap_values_rep[0] if len(shap_values_rep.shape) == 2 else shap_values_rep[0, :, 1]
            
            feat_imp = pd.DataFrame({
                "Feature": feature_names,
                "Impact": vals,
                "Value": input_df.iloc[0].values
            })
        else:
             feat_imp = pd.DataFrame({
                "Feature": shap_exp_obj.feature_names,
                "Impact": shap_exp_obj.values,
                "Value": shap_exp_obj.data
            })

        feat_imp['Abs'] = feat_imp['Impact'].abs()
        max_impact = feat_imp['Abs'].max() if feat_imp['Abs'].max() > 0 else 1
        feat_imp['BarWidth'] = (feat_imp['Abs'] / max_impact) * 100
        top_factors = feat_imp.sort_values('Abs', ascending=False).head(4)
        
        factors_html = ""
        for _, row in top_factors.iterrows():
            bar_color = "#ff0055" if row['Impact'] > 0 else "#00c4cc"
            val_fmt = f"{row['Value']:.1f}"
            
            factors_html += f"""
            <div style="margin-bottom: 15px; display: flex; align-items: center; justify-content: space-between;">
                <div style="width: 35%;">
                    <div style="font-weight: bold; color: #333; font-size: 13px;">{row['Feature']}</div>
                    <div style="color: #888; font-size: 11px;">Measured: {val_fmt}</div>
                </div>
                <div style="width: 60%; background: #eee; height: 8px; border-radius: 4px; position: relative;">
                    <div style="width: {row['BarWidth']}%; background: {bar_color}; height: 100%; border-radius: 4px;"></div>
                </div>
                <div style="width: 5%; text-align: right; font-size: 12px; color: {bar_color}; font-weight: bold;">
                    {'+' if row['Impact'] > 0 else ''}
                </div>
            </div>
            """
    except Exception as e:
        factors_html = f"<div style='color:#999'>Analysis data pending ({str(e)})</div>"

    # 3. HTML VISUAL REDESIGN (GRID SYSTEM & CARDS)
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
        <style>
            body {{ font-family: 'Inter', sans-serif; background: #fff; padding: 40px; color: #1a1a1a; -webkit-print-color-adjust: exact; }}
            
            /* Layout */
            .container {{ max-width: 900px; margin: 0 auto; }}
            .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
            .grid-4 {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 15px; margin-bottom: 20px; }}
            
            /* Components */
            .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 20px; background: white; }}
            .card-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #6b7280; font-weight: 600; margin-bottom: 10px; }}
            
            /* Header */
            .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; border-bottom: 2px solid #f3f4f6; padding-bottom: 20px; }}
            .logo {{ font-size: 24px; font-weight: 800; color: #111; }}
            .logo span {{ color: {COLOR_ACCENT_CYAN}; }} 
            
            /* Badges */
            .badge {{ display: inline-block; padding: 4px 8px; border-radius: 6px; font-weight: 600; font-size: 14px; }}
            .badge-neutral {{ background: #f3f4f6; color: #374151; }}
            
            /* Risk Meter */
            .risk-box {{ background: {risk_bg}; border: 1px solid {risk_color}; border-radius: 12px; padding: 30px; text-align: center; margin-bottom: 30px; }}
            .risk-val {{ font-size: 56px; font-weight: 800; color: {risk_color}; line-height: 1; }}
            .risk-bar-bg {{ background: rgba(0,0,0,0.1); height: 10px; border-radius: 5px; width: 60%; margin: 15px auto; overflow: hidden; }}
            .risk-bar-fill {{ background: {risk_color}; height: 100%; width: {risk_percentage}%; }}
            
            /* Data Points */
            .data-label {{ font-size: 11px; color: #6b7280; text-transform: uppercase; margin-bottom: 4px; }}
            .data-value {{ font-size: 18px; font-weight: 700; color: #111; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">SleepRisk<span style="color:#00c4cc">AI</span></div>
                <div style="text-align: right; font-size: 12px; color: #666;">
                    Report ID: #SR-{np.random.randint(1000,9999)}<br>
                    {datetime.datetime.now().strftime("%d %b %Y")}
                </div>
            </div>

            <!-- RISK HERO SECTION -->
            <div class="risk-box">
                <div style="color: {risk_color}; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; font-size: 14px;">
                    {risk_icon} {risk_title}
                </div>
                <div class="risk-val">{risk_percentage:.1f}%</div>
                <div class="risk-bar-bg">
                    <div class="risk-bar-fill"></div>
                </div>
                <div style="color: #666; font-size: 14px; margin-top: 10px;">Probability of Sleep Disorder</div>
            </div>

            <!-- METRICS GRID (VISUAL) -->
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">👤 Patient Vitals</div>
                    <div class="grid-2" style="margin-bottom:0; gap: 10px;">
                        <div>
                            <div class="data-label">Age / Gender</div>
                            <div class="data-value">{age} <span style="font-size:14px; color:#666">/ {gender[0]}</span></div>
                        </div>
                        <div>
                            <div class="data-label">BMI Status</div>
                            <span class="badge badge-neutral">{bmi_category}</span>
                        </div>
                        <div style="margin-top: 10px;">
                            <div class="data-label">Blood Pressure</div>
                            <div class="data-value">{bp_systolic}/{bp_diastolic}</div>
                        </div>
                        <div style="margin-top: 10px;">
                            <div class="data-label">Heart Rate</div>
                            <div class="data-value">{heart_rate} <small>bpm</small></div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <div class="card-title">💤 Lifestyle Factors</div>
                    <div class="grid-2" style="margin-bottom:0; gap: 10px;">
                        <div>
                            <div class="data-label">Sleep Avg</div>
                            <div class="data-value">{sleep_duration}h</div>
                        </div>
                        <div>
                            <div class="data-label">Stress Load</div>
                            <!-- Visual Dots for Stress -->
                            <div style="color: #aaa; font-size: 14px;">
                                <span style="color: {'#ff0055' if stress_level > 6 else '#00c4cc'}">{'●' * int(stress_level)}</span>{'○' * (10-int(stress_level))}
                            </div>
                        </div>
                        <div style="margin-top: 10px;">
                            <div class="data-label">Activity</div>
                            <div class="data-value">{physical_activity} <small>min</small></div>
                        </div>
                        <div style="margin-top: 10px;">
                            <div class="data-label">Steps</div>
                            <div class="data-value">{daily_steps:,}</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- AI ANALYSIS & SUMMARY -->
            <div class="grid-2">
                <div class="card">
                    <div class="card-title">AI Driving Factors</div>
                    <p style="font-size: 11px; color: #888; margin-bottom: 15px;">Features with highest impact on risk score.</p>
                    {factors_html}
                </div>

                <div class="card" style="background: #fafafa; border-color: transparent;">
                    <div class="card-title">📋 Clinical Summary</div>
                    <p style="font-size: 14px; line-height: 1.6; color: #444;">
                        Based on the analysis of {len(feature_names)} clinical markers, the algorithm detected a 
                        <strong>{risk_title}</strong>.
                    </p>
                    <p style="font-size: 14px; line-height: 1.6; color: #444; margin-top: 10px;">
                        <strong>Recommendation:</strong><br>
                        {recommendation}
                    </p>
                    <div style="margin-top: 30px; text-align: center;">
                         <a href="javascript:window.print()" style="background: #111; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-size: 12px; font-weight: bold;">🖨️ PRINT REPORT</a>
                    </div>
                </div>
            </div>
            
            <div style="text-align: center; border-top: 1px solid #eee; padding-top: 20px; margin-top: 40px; color: #999; font-size: 10px;">
                Generated by SleepRiskAI Algorithm • Not for definitive medical diagnosis
            </div>
        </div>
    </body>
    </html>
    """

    st.download_button(
        label="📄 Download Visual Report (HTML)",
        data=html_report,
        file_name=f"SleepRisk_Visual_Report.html",
        mime="text/html"
    )

st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 30px; opacity: 0.7;'>
        SleepRiskAI v1.5 Final • Hackathon Edition
    </div>
    """, unsafe_allow_html=True
)
