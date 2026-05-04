"""
Student Performance Prediction - Streamlit Application
Using Support Vector Machine (SVM) Model

Correctly processes student data matching the ML.ipynb pipeline
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════
#  CUSTOM CSS — Dark Academia / Neon Accent
# ═══════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

:root {
    --bg-deep:    #0a0e1a;
    --bg-card:    #111827;
    --bg-hover:   #1a2236;
    --accent:     #00f5d4;
    --accent2:    #7b61ff;
    --danger:     #ff4d6d;
    --success:    #00f5d4;
    --text-main:  #e8eaf6;
    --text-muted: #8892b0;
    --border:     rgba(0,245,212,0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-deep);
    color: var(--text-main);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1280px; }

[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,245,212,.12);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-size: .82rem;
    color: var(--text-muted);
    margin-top: .4rem;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.hero {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a0a2e 50%, #0a1a1f 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '🎓';
    position: absolute;
    right: 3rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: .12;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 .5rem;
}
.hero p { color: var(--text-muted); margin: 0; font-size: 1rem; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: .1em;
    border-left: 3px solid var(--accent);
    padding-left: .8rem;
    margin: 1.8rem 0 1rem;
}

.result-pass {
    background: linear-gradient(135deg, rgba(0,245,212,.08), rgba(0,245,212,.02));
    border: 1px solid rgba(0,245,212,.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-fail {
    background: linear-gradient(135deg, rgba(255,77,109,.08), rgba(255,77,109,.02));
    border: 1px solid rgba(255,77,109,.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.result-emoji { font-size: 3.5rem; line-height: 1; }
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin: .5rem 0 .3rem;
}
.result-sub { color: var(--text-muted); font-size: .9rem; }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div input,
textarea {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #0a0e1a;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: .7rem 2rem;
    font-size: 1rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: .04em;
    transition: opacity .2s, transform .15s;
    width: 100%;
}
.stButton > button:hover { opacity: .88; transform: scale(1.02); }

.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: 12px;
    padding: .3rem;
    gap: .3rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #0a0e1a !important;
}

hr { border-color: var(--border); }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  LOAD MODELS AND ENCODERS (with error handling)
# ═══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load all trained models and encoders"""
    try:
        pass_model = joblib.load("svm_pass_model.joblib")
        grade_model = joblib.load("svm_final_grade_model.joblib")
        bin_encoders = joblib.load("bin_encoders.joblib")
        ord_encoders = joblib.load("ord_encoders.joblib")
        le_pass = joblib.load("le_pass.joblib")
        model_columns = joblib.load("model_columns.joblib")
        
        return {
            'pass_model': pass_model,
            'grade_model': grade_model,
            'bin_encoders': bin_encoders,
            'ord_encoders': ord_encoders,
            'le_pass': le_pass,
            'model_columns': model_columns,
            'status': 'success'
        }
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'message': f"Missing file: {str(e)}"
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Error loading artifacts: {str(e)}"
        }


# Load artifacts
artifacts = load_artifacts()

if artifacts['status'] != 'success':
    st.error(f"⚠️ {artifacts['message']}")
    st.info("Make sure all `.joblib` files are in the working directory")
    st.stop()

pass_model = artifacts['pass_model']
grade_model = artifacts['grade_model']
bin_encoders = artifacts['bin_encoders']
ord_encoders = artifacts['ord_encoders']
le_pass = artifacts['le_pass']
model_columns = artifacts['model_columns']


# ═══════════════════════════════════════════════
#  DATA PREPROCESSING FUNCTIONS
# ═══════════════════════════════════════════════
def preprocess_input(input_df):
    """
    Preprocess input data to match training pipeline
    Matches: ML.ipynb preprocessing steps
    """
    df = input_df.copy()
    
    try:
        # ────── Binary Encoding ──────
        binary_cols = ["gender", "internet_access", "school_type"]
        for col in binary_cols:
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return None
            df[col] = bin_encoders[col].transform(df[[col]])
        
        # ────── Ordinal Encoding ──────
        ordinal_config = {
            "family_income": ["Low", "Medium", "High"],
            "parent_education": ["High School", "Bachelor", "Master", "PhD"],
        }
        for col in ordinal_config.keys():
            if col not in df.columns:
                st.error(f"Missing column: {col}")
                return None
            df[[col]] = ord_encoders[col].transform(df[[col]])
        
        # ────── One-Hot Encoding ──────
        nominal_cols = ["device_type"]
        df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)
        
        # ────── Align columns ──────
        df = df.reindex(columns=model_columns, fill_value=0)
        
        return df
    
    except KeyError as e:
        st.error(f"Encoding error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None


# ═══════════════════════════════════════════════
#  BUILD INPUT DATAFRAME FROM USER INPUT
# ═══════════════════════════════════════════════
def build_input_dataframe(input_dict):
    """Convert user inputs to DataFrame"""
    return pd.DataFrame([input_dict])


# ═══════════════════════════════════════════════
#  HERO SECTION
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>Student Performance AI</h1>
  <p>Predict academic outcomes using Support Vector Machine — enter student data to get instant insights.</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  SIDEBAR — STUDENT INFO INPUT
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("##  Student Profile")
    st.markdown("---")

    # ── Demographics ──
    st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
    age = st.slider("Age", 15, 25, 20)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    # ── Academic Background ──
    st.markdown('<div class="section-title">Academic Background</div>', unsafe_allow_html=True)
    study_hours = st.slider("Study Hours per Week", 0, 50, 10)
    attendance = st.slider("Attendance Rate (%)", 0, 100, 80)
    sleep_hours = st.slider("Sleep Hours per Day", 0, 12, 7)
    previous_grade = st.slider("Previous Grade (0-100)", 0, 100, 70)
    assignments_completed = st.slider("Assignments Completed (%)", 0, 100, 80)
    practice_tests_taken = st.slider("Practice Tests Taken", 0, 50, 10)
    
    # ── Learning & Health ──
    st.markdown('<div class="section-title">Learning & Health</div>', unsafe_allow_html=True)
    group_study_hours = st.slider("Group Study Hours", 0, 20, 2)
    notes_quality_score = st.slider("Notes Quality (1-10)", 1, 10, 7)
    time_management_score = st.slider("Time Management (1-10)", 1, 10, 6)
    motivation_level = st.slider("Motivation Level (1-10)", 1, 10, 7)
    mental_health_score = st.slider("Mental Health (1-10)", 1, 10, 7)
    
    # ── Lifestyle ──
    st.markdown('<div class="section-title">Lifestyle & Habits</div>', unsafe_allow_html=True)
    screen_time = st.slider("Daily Screen Time (hours)", 0, 12, 4)
    social_media_hours = st.slider("Social Media Time (hours/day)", 0, 8, 2)
    
    # ── Family & Education ──
    st.markdown('<div class="section-title">Family & Education</div>', unsafe_allow_html=True)
    family_income = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
    parent_education = st.selectbox("Parent Education Level", 
                                   ["High School", "Bachelor", "Master", "PhD"])
    internet_access = st.selectbox("Internet Access at Home?", ["Yes", "No"])
    school_type = st.selectbox("School Type", ["Public", "Private"])
    device_type = st.selectbox("Device Type Used for Studies", 
                              ["Laptop", "Desktop", "Tablet", "Smartphone"])
    
    
    predict_btn = st.button("   Predict Now")


# ═══════════════════════════════════════════════
#  BUILD INPUT AND MAKE PREDICTION
# ═══════════════════════════════════════════════
tabs_placeholder = st.container()

if predict_btn:
    with st.spinner(" Analyzing student data..."):
        time.sleep(0.5)  # UX cosmetic delay
        
        # Build input dictionary
        input_data = {
            "age": age,
            "gender": "Male" if gender == "Male" else "Female",
            "study_hours": study_hours,
            "attendance": attendance,
            "sleep_hours": sleep_hours,
            "previous_grade": previous_grade,
            "assignments_completed": assignments_completed,
            "practice_tests_taken": practice_tests_taken,
            "group_study_hours": group_study_hours,
            "notes_quality_score": notes_quality_score,
            "time_management_score": time_management_score,
            "motivation_level": motivation_level,
            "mental_health_score": mental_health_score,
            "screen_time": screen_time,
            "social_media_hours": social_media_hours,
            "family_income": family_income,
            "parent_education": parent_education,
            "internet_access": "Yes" if internet_access == "Yes" else "No",
            "school_type": "Public" if school_type == "Public" else "Private",
            "device_type": device_type,
            
        }
        
        # Build DataFrame
        df_raw = build_input_dataframe(input_data)
        
        # Preprocess
        df_proc = preprocess_input(df_raw)
        
        if df_proc is None:
            st.error(" Failed to preprocess input data")
            st.stop()
        
        # Make predictions
        try:
            pass_pred = pass_model.predict(df_proc)[0]
            grade_pred = grade_model.predict(df_proc)[0]
            grade_pred = float(np.clip(grade_pred, 0, 100))
            
            # Get probability
            try:
                pass_proba = pass_model.predict_proba(df_proc)[0]
                confidence = max(pass_proba) * 100
                has_proba = True
            except:
                has_proba = False
                confidence = 0
            
            # ════════════════════════════════════════════════════
            #  TABS
            # ════════════════════════════════════════════════════
            tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])
            
            # ════════════════════════════════════════════════════
            #  TAB 1 — PREDICTION RESULTS
            # ════════════════════════════════════════════════════
            with tab1:
                st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
                
                # Determine pass/fail
                passed = pass_pred == 1
                
                col_res, col_grade = st.columns([1.2, 1])
                
                with col_res:
                    if passed:
                        st.markdown(f"""
                        <div class="result-pass">
                          <div class="result-emoji">🏆</div>
                          <div class="result-label" style="color:#00f5d4;">PASS</div>
                          <div class="result-sub">Student is predicted to pass the course</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-fail">
                          <div class="result-emoji">⚠️</div>
                          <div class="result-label" style="color:#ff4d6d;">FAIL</div>
                          <div class="result-sub">Student is at risk — intervention recommended</div>
                        </div>""", unsafe_allow_html=True)
                
                with col_grade:
                    st.markdown(f"""
                    <div class="metric-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
                      <div class="metric-value">{grade_pred:.1f}<span style="font-size:1rem;color:#8892b0">/100</span></div>
                      <div class="metric-label">Predicted Final Grade</div>
                      <div style="margin-top:.8rem;background:#1a2236;border-radius:8px;height:8px;overflow:hidden;">
                        <div style="width:{min(grade_pred/100*100, 100):.0f}%;height:100%;background:linear-gradient(90deg,#00f5d4,#7b61ff);border-radius:8px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
                
                # Confidence gauge
                if has_proba:
                    st.markdown('<div class="section-title">Confidence Score</div>', unsafe_allow_html=True)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        number={"suffix": "%", "font": {"color": "#00f5d4", "family": "Space Mono", "size": 36}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
                            "bar": {"color": "#00f5d4"},
                            "bgcolor": "#111827",
                            "bordercolor": "rgba(0,245,212,.2)",
                            "steps": [
                                {"range": [0, 50], "color": "#1a0a2e"},
                                {"range": [50, 75], "color": "#0d1b2a"},
                                {"range": [75, 100], "color": "#0a1a1f"},
                            ],
                            "threshold": {"line": {"color": "#ff4d6d", "width": 2}, "value": 50},
                        },
                        title={"text": "Model Confidence", "font": {"color": "#8892b0", "size": 14}},
                    ))
                    fig_gauge.update_layout(
                        height=260, paper_bgcolor="rgba(0,0,0,0)", font_color="#e8eaf6",
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Key factors radar chart
                st.markdown('<div class="section-title">Student Profile Analysis</div>', unsafe_allow_html=True)
                factors = {
                    "Study Hours": min(study_hours, 50),
                    "Attendance": attendance,
                    "Sleep Quality": sleep_hours * 10,
                    "Motivation": motivation_level * 10,
                    "Mental Health": mental_health_score * 10,
                    "Assignment Rate": assignments_completed,
                }
                
                fig_radar = go.Figure(go.Scatterpolar(
                    r=list(factors.values()),
                    theta=list(factors.keys()),
                    fill='toself',
                    fillcolor='rgba(0,245,212,0.08)',
                    line=dict(color='#00f5d4', width=2),
                    marker=dict(color='#00f5d4', size=6),
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(0,0,0,0)',
                        angularaxis=dict(color='#8892b0', gridcolor='rgba(255,255,255,.07)'),
                        radialaxis=dict(visible=True, range=[0, 100], color='#8892b0',
                                        gridcolor='rgba(255,255,255,.07)'),
                    ),
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#e8eaf6',
                    height=380, margin=dict(l=40, r=40, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # ════════════════════════════════════════════════════
            #  TAB 2 — ANALYTICS
            # ════════════════════════════════════════════════════
            with tab2:
                st.markdown('<div class="section-title">Feature Importance (Domain Knowledge)</div>', unsafe_allow_html=True)
                
                features = ["Study Hours", "Attendance", "Sleep", "Motivation", "Mental Health", 
                           "Assignment Completion", "Group Study", "Time Management", "Notes Quality", "Practice Tests"]
                importance = [0.20, 0.18, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.05]
                
                fig_bar = go.Figure(go.Bar(
                    x=importance, y=features,
                    orientation='h',
                    marker=dict(
                        color=importance,
                        colorscale=[[0,"#1a0a2e"],[0.5,"#7b61ff"],[1,"#00f5d4"]],
                        showscale=False,
                    ),
                    text=[f"{v:.0%}" for v in importance],
                    textposition='outside', textfont=dict(color='#8892b0', size=12),
                ))
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(gridcolor='rgba(255,255,255,.06)', color='#e8eaf6', tickfont_size=12),
                    margin=dict(l=10, r=60, t=20, b=10), height=360,
                    font_color='#e8eaf6',
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pass/Fail distribution
                st.markdown('<div class="section-title">Expected Distribution</div>', unsafe_allow_html=True)
                fig_pie = go.Figure(go.Pie(
                    labels=['Pass', 'Fail'],
                    values=[70, 30],
                    hole=.55,
                    marker=dict(colors=['#00f5d4', '#ff4d6d'],
                                line=dict(color='#0a0e1a', width=3)),
                    textfont=dict(color='#e8eaf6', size=14),
                    textinfo='label+percent',
                ))
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False, height=300,
                    margin=dict(l=20,r=20,t=20,b=20),
                    annotations=[dict(text='Students', x=.5, y=.5, font_size=14,
                                      font_color='#8892b0', showarrow=False)],
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # ════════════════════════════════════════════════════
            #  TAB 3 — ABOUT
            # ════════════════════════════════════════════════════
            with tab3:
                st.markdown("""
                <div class="metric-card" style="margin-bottom:1.2rem;">
                  <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00f5d4;margin-bottom:.8rem;">ℹ️ Project Overview</div>
                  <p style="color:#8892b0;line-height:1.7;margin:0;">
                    This application uses <strong style="color:#e8eaf6;">Support Vector Machine (SVM)</strong> models trained on 
                    comprehensive student data to predict two critical outcomes:
                  </p>
                  <ul style="color:#8892b0;margin-top:.8rem;line-height:1.8;">
                    <li><strong style="color:#00f5d4;">Pass / Fail</strong> — Binary classification task</li>
                    <li><strong style="color:#7b61ff;">Final Grade (0-100)</strong> — Regression task</li>
                  </ul>
                </div>

                <div class="metric-card" style="margin-bottom:1.2rem;">
                  <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00f5d4;margin-bottom:.8rem;">📊 Model Details</div>
                  <ul style="color:#8892b0;margin:0;line-height:1.8;">
                    <li><strong>Algorithm:</strong> Support Vector Machine (SVM)</li>
                    <li><strong>Features Used:</strong> 20+ student attributes</li>
                    <li><strong>Data Preprocessing:</strong> Label Encoding, Ordinal Encoding, One-Hot Encoding</li>
                    <li><strong>Imbalance Handling:</strong> SMOTE for classification</li>
                  </ul>
                </div>

                <div class="metric-card">
                  <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00f5d4;margin-bottom:.8rem;">⚙️ Key Features</div>
                  <ul style="color:#8892b0;margin:0;line-height:1.8;">
                    <li>Real-time predictions with confidence scores</li>
                    <li>Feature importance visualization</li>
                    <li>Student profile analysis with radar charts</li>
                    <li>Consistent, reproducible results (random_state=42)</li>
                  </ul>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f" Prediction failed: {str(e)}")
            st.info("Check that the input data matches the training data format")

else:
    # No prediction yet - show placeholder
    with tabs_placeholder:
        tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])
        
        with tab1:
            st.markdown("""
            <div style="text-align:center;padding:4rem 2rem;color:#8892b0;">
              <div style="font-size:4rem;margin-bottom:1rem;">🎓</div>
              <div style="font-size:1.1rem;font-family:'Space Mono',monospace;">Fill in the student profile</div>
              <div style="font-size:.9rem;margin-top:.5rem;">then click <strong style="color:#00f5d4;"> Predict Now</strong> in the sidebar</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title">Model Summary</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            cards = [
                ("SVM", "Algorithm"),
                ("Pass / Fail", "Classification"),
                ("0-100", "Grade Range"),
                ("20+", "Features"),
            ]
            for col, (val, lbl) in zip([c1, c2, c3, c4], cards):
                col.markdown(f"""
                <div class="metric-card">
                  <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                  <div class="metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)
        
        with tab2:
            st.info("📈 Make a prediction to see analytics")
        
        with tab3:
            st.markdown("""
            <div class="metric-card">
              <div style="font-family:'Space Mono',monospace;font-size:1.1rem;color:#00f5d4;margin-bottom:.8rem;">About This App</div>
              <p style="color:#8892b0;line-height:1.7;">
                This Streamlit application provides real-time predictions for student academic performance 
                using a trained SVM model. The model analyzes 20+ features including study habits, mental health, 
                family background, and lifestyle factors to predict both pass/fail outcomes and final grades.
              </p>
            </div>
            """, unsafe_allow_html=True)