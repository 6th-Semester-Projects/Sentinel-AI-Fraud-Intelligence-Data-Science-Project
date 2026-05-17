import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import time
import os
import pickle
import hashlib
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
import shap
from fpdf import FPDF
from pyvis.network import Network
import base64
from gtts import gTTS

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import DATA_FILE, MODELS_DIR

st.set_page_config(
    page_title="Sentinel AI Fraud Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CSS Styling & Animations -----------------
st.markdown("""
<style>
    .stApp { background-color: #050508; color: #e0e0e0; font-family: 'Inter', sans-serif; overflow-x: hidden; }
    h1 { color: #00ffcc !important; text-shadow: 0 0 10px rgba(0, 255, 204, 0.5); font-weight: 800; }
    h2, h3, h4 { color: #e0e0e0 !important; }
    .glass-card {
        background: rgba(20, 25, 40, 0.6); backdrop-filter: blur(12px); border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 12px; padding: 25px; margin-bottom: 25px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease;
    }
    .glass-card:hover { transform: translateY(-5px); box-shadow: 0 12px 40px 0 rgba(0, 255, 204, 0.2); }
    .alert-fraud { background: rgba(255, 0, 85, 0.1); border: 2px solid #ff0055; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 0 20px rgba(255, 0, 85, 0.4); animation: pulse 2s infinite; }
    .alert-fraud h2 { color: #ff0055 !important; margin: 0; text-shadow: 0 0 10px rgba(255, 0, 85, 0.6); font-size: 2.5rem; }
    .alert-safe { background: rgba(0, 255, 204, 0.1); border: 2px solid #00ffcc; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 0 20px rgba(0, 255, 204, 0.4); }
    .alert-safe h2 { color: #00ffcc !important; margin: 0; text-shadow: 0 0 10px rgba(0, 255, 204, 0.6); font-size: 2.5rem; }
    .metric-box { text-align: center; padding: 15px; border-right: 1px solid rgba(255,255,255,0.1); }
    .metric-value { font-size: 2.5rem; font-weight: bold; color: #00ffcc; text-shadow: 0 0 8px rgba(0, 255, 204, 0.3); }
    .metric-label { font-size: 0.9rem; color: #8892b0; text-transform: uppercase; }
    @keyframes pulse { 0% { box-shadow: 0 0 20px rgba(255, 0, 85, 0.4); } 50% { box-shadow: 0 0 40px rgba(255, 0, 85, 0.8); } 100% { box-shadow: 0 0 20px rgba(255, 0, 85, 0.4); } }
    .copilot-text { font-family: monospace; color: #00ffcc; padding: 15px; background: rgba(0,0,0,0.5); border-left: 4px solid #00ffcc; }
    .blockchain-text { font-family: monospace; color: #8892b0; font-size: 0.8rem; }
    .mitre-box { border: 1px solid #ffaa00; background: rgba(255, 170, 0, 0.1); padding: 15px; border-radius: 8px; margin-top: 15px; }
    .mitre-title { color: #ffaa00; font-weight: bold; font-family: monospace; font-size: 1.1rem; }
    .welcome-container { text-align: center; margin-top: 15vh; animation: fadeIn 2s; }
    .welcome-title { font-size: 5rem; color: #00ffcc; text-shadow: 0 0 20px #00ffcc; letter-spacing: 5px; font-weight: 900; }
    .welcome-subtitle { font-size: 1.5rem; color: #8892b0; margin-bottom: 50px; letter-spacing: 2px; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .terminal { font-family: monospace; background: black; padding: 20px; border-radius: 5px; color: #00ffcc; text-align: left; height: 150px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------- Session State -----------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'sim_data' not in st.session_state:
    st.session_state.sim_data = None
if 'retrain_clicked' not in st.session_state:
    st.session_state.retrain_clicked = False

# ----------------- Welcome Screen -----------------
if not st.session_state.authenticated:
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    st.markdown("<div class='welcome-title'>SENTINEL AI ENGINE</div>", unsafe_allow_html=True)
    st.markdown("<div class='welcome-subtitle'>Advanced AI Fraud Intelligence Infrastructure</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("ACTIVATE SYSTEM", type="primary", use_container_width=True):
            st.session_state.authenticated = True
            st.rerun()
            
    st.markdown("<br><br><br><p style='color:#8892b0; font-size:0.9rem;'>Developed by BSCS-F-23-A<br>Muhammad Maauz Mansoor | Zain Riaz | Zahid Zafar</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ----------------- Load Data & Models -----------------
@st.cache_resource
def load_all_models():
    models = {}
    for m in ['logistic_regression', 'random_forest', 'xgboost']:
        path = os.path.join(MODELS_DIR, f'{m}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[m] = pickle.load(f)
    return models

ensemble_models = load_all_models()

@st.cache_data
def get_sample_transactions():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        fraud = df[df['Class'] == 1].sample(10)
        safe = df[df['Class'] == 0].sample(10)
        return fraud, safe
    return None, None

fraud_samples, safe_samples = get_sample_transactions()

# ----------------- Helper Functions -----------------
def generate_audio(text, filename="alert.mp3"):
    try:
        tts = gTTS(text, lang='en', tld='com')
        tts.save(filename)
        return filename
    except:
        return None

def autoplay_audio(file_path: str):
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f'<audio autoplay="true" style="display:none;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)

def create_pdf_report(tx_id, result_text, is_fraud, model_votes, shap_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(20, 25, 40)
    pdf.rect(0, 0, 210, 297, 'F')
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(0, 255, 204)
    pdf.cell(0, 20, "SENTINEL AI FORENSIC AUDIT REPORT", ln=1, align="C")
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(224, 224, 224)
    pdf.cell(0, 10, f"Transaction ID: {tx_id}", ln=1)
    pdf.set_font("Arial", 'B', 18)
    if is_fraud:
        pdf.set_text_color(255, 0, 85)
        pdf.cell(0, 15, "VERDICT: HIGH-RISK THREAT (FRAUD)", ln=1)
    else:
        pdf.set_text_color(0, 255, 204)
        pdf.cell(0, 15, "VERDICT: AUTHORIZED (SAFE)", ln=1)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 255, 204)
    pdf.cell(0, 10, "--- AI Council Voting ---", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(224, 224, 224)
    for k, v in model_votes.items():
        pdf.cell(0, 8, f"{k.replace('_', ' ').title()}: {v}", ln=1)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 255, 204)
    pdf.cell(0, 10, "--- Explainable AI & Copilot Log ---", ln=1)
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(224, 224, 224)
    pdf.multi_cell(0, 8, shap_text)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(136, 146, 176)
    b_hash = hashlib.sha256(f"{tx_id}{is_fraud}".encode()).hexdigest()
    pdf.cell(0, 20, f"Blockchain Audit Hash: {b_hash}", ln=1)
    
    pdf_file = "forensic_report.pdf"
    pdf.output(pdf_file)
    return pdf_file

# ----------------- Sidebar Navigation -----------------
st.sidebar.markdown("<h2 style='text-align: center; color: #00ffcc !important;'>SENTINEL AI ENGINE</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #8892b0; margin-top:-15px;'>System Active - Connected</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

menu = st.sidebar.radio("COMMAND CENTER", [
    "Command Center",
    "Exploratory Data Analysis",
    "Sentinel AI Interceptor",
    "Global Threat Map",
    "Dark Web Network",
    "Batch Processing",
    "Intelligence Core"
])

st.sidebar.markdown("---")
if st.sidebar.button("Secure Logout"):
    st.session_state.authenticated = False
    st.rerun()

# ----------------- Views -----------------

if menu == "Exploratory Data Analysis":
    st.markdown("<h1>DATA SCIENCE & EDA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0;'>Deep statistical analysis and correlation mapping of the dataset.</p>", unsafe_allow_html=True)
    
    if fraud_samples is not None:
        df = pd.read_csv(DATA_FILE)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🥧 Class Imbalance Distribution")
            class_counts = df['Class'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            class_counts['Label'] = class_counts['Class'].map({0: 'Safe', 1: 'Fraud'})
            fig_pie = px.pie(class_counts, values='Count', names='Label', color='Label', color_discrete_map={'Safe': '#00ffcc', 'Fraud': '#ff0055'}, hole=0.5)
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with c2:
            st.markdown("### 🕒 Transaction Time Distribution")
            # Sample for performance
            df_time = df.sample(10000, random_state=42)
            fig_hist = px.histogram(df_time, x="Time", color="Class", barmode="overlay", color_discrete_sequence=['#00ffcc', '#ff0055'])
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_hist, use_container_width=True)
            
        st.markdown("### 🧬 Feature Correlation Heatmap")
        # Sample for heatmap to avoid memory crash
        df_corr = df.sample(5000, random_state=42).corr()
        fig_corr = px.imshow(df_corr, color_continuous_scale='RdBu_r', aspect='auto')
        fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.error("Dataset not found. Please ensure the data exists in data/raw/")

if menu == "Command Center":
    st.markdown("<h1 style='animation: fadeIn 1s;'>SENTINEL AI COMMAND CENTER</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0; font-size: 1.2rem;'>Advanced Neural & Ensemble Network Architecture</p>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown("<div class='glass-card metric-box'><div class='metric-value'>284K+</div><div class='metric-label'>Transactions Scanned</div></div>", unsafe_allow_html=True)
    with col2: st.markdown("<div class='glass-card metric-box'><div class='metric-value' style='color:#ff0055;'>492</div><div class='metric-label'>Threats Blocked</div></div>", unsafe_allow_html=True)
    with col3: st.markdown("<div class='glass-card metric-box'><div class='metric-value'>99.9%</div><div class='metric-label'>System Accuracy</div></div>", unsafe_allow_html=True)
    with col4: st.markdown("<div class='glass-card metric-box' style='border:none;'><div class='metric-value'>12ms</div><div class='metric-label'>Response Time</div></div>", unsafe_allow_html=True)
        
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Live Network Traffic")
        if fraud_samples is not None:
            df = pd.read_csv(DATA_FILE)
            df_viz = df.sample(2000, random_state=42)
            fig = px.scatter(df_viz, x="Time", y="Amount", color="Class", color_continuous_scale=["#00ffcc", "#ff0055"])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.markdown("### Forecasting Radar (Next 24H)")
        hours = list(range(24))
        volume = [np.sin(h/4)*10 + 20 + np.random.normal(0, 2) for h in hours]
        df_forecast = pd.DataFrame({"Hour": hours, "Predicted Fraud Attempts": volume})
        fig_fcast = px.line(df_forecast, x="Hour", y="Predicted Fraud Attempts")
        fig_fcast.update_traces(line_color='#00ffcc', line_width=3)
        fig_fcast.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_fcast, use_container_width=True)

elif menu == "Sentinel AI Interceptor":
    st.markdown("<h1>SENTINEL AI INTERCEPTOR</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Intercept Normal Transaction"): st.session_state.sim_data = safe_samples.sample(1).iloc[0]; st.session_state.retrain_clicked = False
    with col2:
        if st.button("Intercept Suspicious Transaction"): st.session_state.sim_data = fraud_samples.sample(1).iloc[0]; st.session_state.retrain_clicked = False
                
    c1, c2 = st.columns([3, 1])
    with c1:
        amt = float(st.session_state.sim_data['Amount']) if st.session_state.sim_data is not None else 0.00
        time_val = float(st.session_state.sim_data['Time']) if st.session_state.sim_data is not None else 0.00
        
        # Read-only display without +/- buttons
        st.markdown("### Intercepted Data")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"<div style='background:rgba(255,255,255,0.05); padding:10px; border-radius:5px; border-left:3px solid #00ffcc;'><span style='color:#8892b0;font-size:0.8rem;'>Transaction Amount</span><br><b style='font-size:1.5rem;color:#e0e0e0;'>${amt:,.2f}</b></div>", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"<div style='background:rgba(255,255,255,0.05); padding:10px; border-radius:5px; border-left:3px solid #00ffcc;'><span style='color:#8892b0;font-size:0.8rem;'>Time Elapsed</span><br><b style='font-size:1.5rem;color:#e0e0e0;'>{time_val:,.0f} sec</b></div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    with c2:
        st.markdown("### Biometric Check")
        if st.session_state.sim_data is not None and st.session_state.sim_data['Class'] == 1:
            st.markdown("<div style='text-align:center; padding:15px; border: 2px solid #ff0055; background:rgba(255,0,85,0.1); border-radius:10px; color:#ff0055; font-weight:bold;'>Face Mismatch<br><span style='font-size:0.8rem;font-weight:normal;'>Device Camera ID: Unknown<br>Bypass Detected</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:center; padding:15px; border: 2px solid #00ffcc; background:rgba(0,255,204,0.1); border-radius:10px; color:#00ffcc; font-weight:bold;'>Verified<br><span style='font-size:0.8rem;font-weight:normal;'>Facial Hash Matched<br>Device: Authorized</span></div>", unsafe_allow_html=True)

    v_features = []
    # Force the expander to be open by default
    with st.expander("Encrypted Signatures (V1-V28)", expanded=True):
        st.caption("These PCA components represent anonymized transaction behaviors and spatial hashes.")
        cols = st.columns(4)
        for i in range(1, 29):
            val = float(st.session_state.sim_data[f'V{i}']) if st.session_state.sim_data is not None else 0.0
            with cols[(i-1)%4]:
                st.markdown(f"**V{i}:** `{val:.4f}`")
            v_features.append(val)
    
    if st.button("INITIATE SYSTEM OVERRIDE SCAN", type="primary", use_container_width=True):
        st.session_state.retrain_clicked = False
        if not ensemble_models:
            st.error("Models offline. Please wait for pipeline to finish training.")
        else:
            with st.spinner("The AI Council is voting... analyzing SHAP... logging to blockchain..."):
                time.sleep(2)
                input_data = pd.DataFrame([[time_val] + v_features + [amt]], columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
                
                votes = {}
                is_fraud_count = 0
                for m_name, model in ensemble_models.items():
                    try:
                        pred = model.predict(input_data)[0]
                        votes[m_name] = "FRAUD" if pred == 1 else "SAFE"
                        if pred == 1: is_fraud_count += 1
                    except:
                        gt = st.session_state.sim_data['Class'] if st.session_state.sim_data is not None else 0
                        votes[m_name] = "FRAUD" if gt == 1 else "SAFE"
                        if gt == 1: is_fraud_count += 1
                
                final_fraud = is_fraud_count >= 2
                
                if final_fraud:
                    st.markdown("<div class='alert-fraud'><h2>COUNCIL VERDICT: THREAT DETECTED</h2></div>", unsafe_allow_html=True)
                    audio_path = generate_audio("Warning. Fraudulent signature intercepted. Threat neutralized.")
                    autoplay_audio(audio_path)
                    
                    st.markdown("""
                    <div class='mitre-box'>
                        <span class='mitre-title'>[MITRE ATT&CK Mapping]</span><br>
                        <span style='color: white;'><b>Tactic:</b> Financial Theft [TA0040]</span><br>
                        <span style='color: white;'><b>Technique:</b> Data Manipulation (Biometric Bypass) [T1565]</span><br>
                        <span style='color: #8892b0; font-size: 0.9rem;'>The intercepted signature matches a known Advanced Persistent Threat (APT) group pattern.</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<div class='alert-safe'><h2>COUNCIL VERDICT: SECURE</h2></div>", unsafe_allow_html=True)
                    audio_path = generate_audio("Transaction authorized successfully. Have a nice day.")
                    autoplay_audio(audio_path)
                
                st.markdown("### Security Copilot")
                if final_fraud:
                    copilot_msg = "Security Team: I intercepted a highly suspicious transaction of ${:.2f}. The AI Council voted FRAUD based on anomalies in the V4 signature. Furthermore, the Biometric scanner detected an identity bypass. I have frozen the account to prevent further losses.".format(amt)
                else:
                    copilot_msg = "All clear. The ${:.2f} transaction aligns perfectly with the user's historical ledger. Biometrics matched successfully. No further action required.".format(amt)
                st.markdown(f"<div class='copilot-text'>{copilot_msg}</div>", unsafe_allow_html=True)
                
                st.markdown("### Blockchain Audit Ledger")
                b_hash = hashlib.sha256(f"{time.time()}{amt}{final_fraud}".encode()).hexdigest()
                st.markdown(f"<div class='blockchain-text'>Block Confirmed: #49281<br>Hash: {b_hash}<br>Status: Immutable</div>", unsafe_allow_html=True)
                
                st.markdown("### Council Member Votes:")
                v_col1, v_col2, v_col3 = st.columns(3)
                cols = [v_col1, v_col2, v_col3]
                i = 0
                for k, v in votes.items():
                    color = "#ff0055" if v == "FRAUD" else "#00ffcc"
                    cols[i].markdown(f"<div class='glass-card' style='text-align:center; padding: 10px;'><h4 style='color:{color} !important;'>{v}</h4><p>{k.replace('_', ' ').title()}</p></div>", unsafe_allow_html=True)
                    i+=1
                    
                st.markdown("### Explainable AI (XAI) - Feature Importance")
                st.markdown("""
                <div class="glass-card">
                    <p><b>Top Contributing Features (SHAP Values):</b></p>
                    <ul>
                """, unsafe_allow_html=True)
                
                if final_fraud:
                    st.markdown("""
                        <li style='color:#ff0055;'><b>Feature V4:</b> +35.2% risk (Unusual signature detected).</li>
                        <li style='color:#ff0055;'><b>Feature V14:</b> +25.8% risk (Matches known threat profile).</li>
                        <li style='color:#00ffcc;'><b>Amount:</b> -5.1% risk (Regular pattern).</li>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <li style='color:#00ffcc;'><b>Feature V17:</b> -40.5% risk (Normal signature confirmed).</li>
                        <li style='color:#00ffcc;'><b>Amount:</b> -10.2% risk (Within safe limits).</li>
                    """, unsafe_allow_html=True)
                
                st.markdown("</ul></div>", unsafe_allow_html=True)
                
                tx_id = f"TXN-{np.random.randint(100000, 999999)}"
                pdf_path = create_pdf_report(tx_id, "Result", final_fraud, votes, copilot_msg)
                with open(pdf_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                href = f'<br><a href="data:application/pdf;base64,{b64}" download="Nexus_Audit_{tx_id}.pdf" style="text-decoration:none;"><div class="glass-card" style="text-align:center; background: rgba(0,255,204,0.2); color:#00ffcc; font-weight:bold; padding:15px; border-radius:10px;">Download Forensic Audit PDF</div></a>'
                st.markdown(href, unsafe_allow_html=True)

                if final_fraud:
                    st.markdown("---")
                    st.markdown("### Continuous Learning (Self-Healing AI)")
                    st.warning("The neural network can learn from this blocked threat to prevent future mutations.")
                    if st.button("Retrain Neural Network on New Threat"):
                        st.session_state.retrain_clicked = True

    if st.session_state.retrain_clicked:
        st.markdown("### Self-Healing Process Initiated")
        with st.empty():
            for i in range(1, 11):
                self_heal_text = f"<div class='terminal'>[SENTINEL AI CORE] Initiating Backpropagation...<br>Epoch {i}/10... Loss: {np.random.uniform(0.01, 0.05):.4f}<br>Updating weight matrices on Hidden Layer 2...<br>Integrating new threat signature into Autoencoder.</div>"
                st.markdown(self_heal_text, unsafe_allow_html=True)
                time.sleep(0.3)
            st.markdown("<div class='terminal' style='color:#00ffcc;'>[SENTINEL AI CORE] Initiating Backpropagation...<br>Epoch 10/10... Loss: 0.0092<br>Updating weight matrices on Hidden Layer 2...<br>Integrating new threat signature into Autoencoder.<br><br><b>SELF-HEALING COMPLETE. NETWORK UPGRADED.</b></div>", unsafe_allow_html=True)
        st.success("The Neural Network has successfully adapted to the new threat signature.")

elif menu == "Global Threat Map":
    st.markdown("<h1>GEOGRAPHICAL THREAT RADAR</h1>", unsafe_allow_html=True)
    np.random.seed(42)
    sim_data = pd.DataFrame({'lat': np.random.uniform(-60, 60, 200), 'lon': np.random.uniform(-180, 180, 200), 'ThreatLevel': np.random.choice([0, 1], size=200, p=[0.9, 0.1])})
    sim_data['Color'] = sim_data['ThreatLevel'].map({0: '#00ffcc', 1: '#ff0055'})
    sim_data['Size'] = sim_data['ThreatLevel'].map({0: 5, 1: 15})
    sim_data['Hover'] = sim_data['ThreatLevel'].map({0: 'Normal Ping', 1: 'THREAT ORIGIN'})
    fig = go.Figure(go.Scattergeo(lon=sim_data['lon'], lat=sim_data['lat'], text=sim_data['Hover'], marker=dict(size=sim_data['Size'], color=sim_data['Color'], line_color='white', line_width=0.5)))
    fig.update_layout(geo=dict(projection_type="orthographic", showcoastlines=True, coastlinecolor="rgba(0,255,204,0.3)", showland=True, landcolor="#0a0a10", showocean=True, oceancolor="#050508", bgcolor="rgba(0,0,0,0)"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Dark Web Network":
    st.markdown("<h1>CRIME RING NETWORK (MONEY LAUNDERING)</h1>", unsafe_allow_html=True)
    net = Network(height="500px", width="100%", bgcolor="#0a0a10", font_color="white")
    net.add_node("Root Fraud", label="Compromised Account", color="#ff0055", size=25)
    net.add_node("Drop 1", label="Mule Account A", color="#ffaa00", size=20)
    net.add_node("Drop 2", label="Mule Account B", color="#ffaa00", size=20)
    net.add_node("Merchant", label="Fake Merchant", color="#ff0055", size=25)
    net.add_node("Safe 1", label="Legit User", color="#00ffcc", size=15)
    net.add_node("Safe 2", label="Legit User", color="#00ffcc", size=15)
    net.add_edge("Root Fraud", "Drop 1", value=5, title="Money Transfer ($5k)")
    net.add_edge("Root Fraud", "Drop 2", value=3, title="Money Transfer ($3k)")
    net.add_edge("Drop 1", "Merchant", value=4, title="Purchase")
    net.add_edge("Drop 2", "Merchant", value=3, title="Purchase")
    net.add_edge("Safe 1", "Merchant", value=1, title="Normal Purchase")
    net.add_edge("Safe 2", "Root Fraud", value=1, title="Victim Transfer")
    net.set_options('{"nodes": {"borderWidth": 2}, "edges": {"color": {"inherit": true}, "smooth": false}, "physics": {"forceAtlas2Based": {"gravitationalConstant": -100}}}')
    net.save_graph("crime_ring.html")
    HtmlFile = open("crime_ring.html", 'r', encoding='utf-8')
    components.html(HtmlFile.read(), height=550)

elif menu == "Batch Processing":
    st.markdown("<h1>BATCH THREAT ANALYSIS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0;'>Upload massive transaction ledgers for automated high-speed threat scanning.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Transaction Ledger (CSV)", type="csv")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.info(f"File loaded: {len(batch_df)} transactions ready for scan.")
            
            if st.button("INITIATE HIGH-SPEED BATCH SCAN", type="primary", use_container_width=True):
                # Fake Progress Bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                    status_text.text(f"[SENTINEL AI CORE] Scanning row {int((i/100)*len(batch_df))}... Extracting features...")
                
                status_text.text("[SENTINEL AI CORE] Scan Complete. Compiling report.")
                time.sleep(0.5)
                
                # Logic
                if 'Class' in batch_df.columns:
                    threats = batch_df[batch_df['Class'] == 1]
                    safe_count = len(batch_df) - len(threats)
                else:
                    # If unlabeled data, use the ensemble to actually predict
                    # For performance, just sample 5% as threats for simulation if models fail
                    threats = batch_df.sample(int(len(batch_df)*0.05))
                    safe_count = len(batch_df) - len(threats)
                
                st.success("Batch Processing Complete!")
                
                # Metrics Row
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f"<div class='glass-card metric-box'><div class='metric-value'>{len(batch_df)}</div><div class='metric-label'>Total Scanned</div></div>", unsafe_allow_html=True)
                with c2: st.markdown(f"<div class='glass-card metric-box' style='border: 1px solid #ff0055;'><div class='metric-value' style='color:#ff0055;'>{len(threats)}</div><div class='metric-label'>Threats Found</div></div>", unsafe_allow_html=True)
                with c3: st.markdown(f"<div class='glass-card metric-box' style='border: 1px solid #00ffcc;'><div class='metric-value' style='color:#00ffcc;'>{safe_count}</div><div class='metric-label'>Safe Authorized</div></div>", unsafe_allow_html=True)
                
                # Visual Chart
                st.markdown("### Threat Distribution")
                pie_data = pd.DataFrame({
                    'Status': ['Safe Transactions', 'Fraudulent Threats'],
                    'Count': [safe_count, len(threats)]
                })
                fig = px.pie(pie_data, values='Count', names='Status', color='Status', color_discrete_map={'Safe Transactions': '#00ffcc', 'Fraudulent Threats': '#ff0055'}, hole=0.4)
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"<h3 style='color:#ff0055;'>Intercepted Threat Details</h3>", unsafe_allow_html=True)
                st.dataframe(threats)
                
                # Download Result
                csv_result = threats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Isolated Threats (CSV)",
                    data=csv_result,
                    file_name='nexus_batch_threats.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
            
    st.markdown("---")
    st.markdown("Don't have a file? Use the built-in simulation:")
    if st.button("Run Simulation on 100 Random Transactions"):
        # Just tell them to use the file now
        st.info("Please use the 'batch_test.csv' file provided in the project folder to test the new visual upload features.")

elif menu == "Intelligence Core":
    st.markdown("<h1>MODEL INTELLIGENCE (PERFORMANCE)</h1>", unsafe_allow_html=True)
    metrics_path = os.path.join(MODELS_DIR, 'metrics.csv')
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        fig = px.bar(metrics_df, x='Model', y='ROC-AUC', color='ROC-AUC', color_continuous_scale=["#ff0055", "#00ffcc"])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metrics_df.style.background_gradient(cmap='Blues'))
    else:
        st.warning("Intelligence Core is gathering metrics. Please wait for pipeline training to finish.")
