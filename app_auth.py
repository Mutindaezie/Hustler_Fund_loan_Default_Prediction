import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from auth_manager import AuthManager
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Hustler Fund | Credit Risk",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0f1e;
    color: #e8eaf6;
}

section[data-testid="stSidebar"] {
    background: #0d1526;
    border-right: 1px solid #1e2d4a;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #00e5a0 !important;
}

h4, h5, h6 {
    color: #b0bec5 !important;
}

[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
}

[data-testid="metric-container"] label {
    color: #64b5f6 !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00e5a0 !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
}

.stButton > button {
    background: linear-gradient(135deg, #00e5a0, #00b4d8);
    color: #0a0f1e;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-size: 0.9rem;
    letter-spacing: 1px;
    transition: all 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 229, 160, 0.3);
}

.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: #90a4ae;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    border-radius: 8px;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #00e5a0 !important;
}

.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #90a4ae;
    font-size: 0.82rem;
    font-family: 'Space Mono', monospace;
}

.risk-high {
    background: linear-gradient(135deg, #ff1744, #d50000);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 2px;
}

.risk-low {
    background: linear-gradient(135deg, #00e5a0, #00b4d8);
    color: #0a0f1e;
    padding: 16px 24px;
    border-radius: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 2px;
}

.info-box {
    background: #111827;
    border-left: 3px solid #00e5a0;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #cfd8dc;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data & Model (cached)
# ─────────────────────────────────────────────
@st.cache_data
def generate_data(n=10000):
    np.random.seed(42)
    ages = np.random.randint(18, 65, n)
    gender = np.random.choice(['Male', 'Female'], n)
    county = np.random.choice(
        ['Nairobi', 'Mombasa', 'Kisumu', 'Machakos', 'Kiambu', 'Nakuru', 'Meru'], n)
    income = np.random.normal(25000, 12000, n).clip(5000, 150000)
    loan_amount = np.random.randint(500, 50000, n)
    loan_purpose = np.random.choice(
        ['Business', 'School Fees', 'Emergency', 'Personal', 'Farm Inputs'], n)
    mpesa_transactions = np.random.randint(5, 200, n)
    mpesa_volume = np.random.normal(15000, 7000, n).clip(1000, 100000)
    previous_loans = np.random.poisson(3, n)
    previous_defaults = np.random.binomial(previous_loans, 0.15)
    repayment_score = np.random.normal(0.68, 0.15, n).clip(0, 1)
    credit_score = (0.5 * repayment_score +
                    0.3 * (1 - previous_defaults / (previous_loans + 1)) +
                    0.2 * (income / 150000)).clip(0, 1)
    loan_repaid = np.random.binomial(
        1,
        (0.55 * repayment_score +
         0.25 * (income / 150000) +
         0.20 * (1 - previous_defaults / (previous_loans + 1))).clip(0, 1)
    )
    df = pd.DataFrame({
        'Age': ages, 'Gender': gender, 'County': county,
        'Monthly_Income': income.astype(int), 'Loan_Amount': loan_amount,
        'Loan_Purpose': loan_purpose,
        'Mpesa_Transactions': mpesa_transactions,
        'Mpesa_Volume': mpesa_volume.astype(int),
        'Repayment_Score': repayment_score.round(3),
        'Credit_Score': credit_score.round(3),
        'Loan_Repaid': loan_repaid
    })
    return df


@st.cache_resource
def train_model(df):
    X = df.drop('Loan_Repaid', axis=1)
    y = df['Loan_Repaid']
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_proc, y_train)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_split=5, max_features='sqrt', random_state=42)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'confusion': confusion_matrix(y_test, y_pred),
        'fpr': roc_curve(y_test, y_prob)[0],
        'tpr': roc_curve(y_test, y_prob)[1],
        'y_test': y_test,
        'y_prob': y_prob,
    }

    ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = num_features + list(ohe_features)
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    return model, preprocessor, metrics, importance, X_test, y_test


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def load_user_data():
    if os.path.exists('users.json'):
        with open('users.json', 'r') as f:
            return json.load(f)
    return {}


def save_user_data(data):
    with open('users.json', 'w') as f:
        json.dump(data, f, indent=4)


def load_predictions():
    if os.path.exists('predictions.json'):
        with open('predictions.json', 'r') as f:
            return json.load(f)
    return {}


def save_predictions(data):
    with open('predictions.json', 'w') as f:
        json.dump(data, f, indent=4)


# ─────────────────────────────────────────────
# Initialize Session State
# ─────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = load_user_data()
if 'predictions' not in st.session_state:
    st.session_state.predictions = load_predictions()


# ─────────────────────────────────────────────
# Login & Sign-up Page
# ─────────────────────────────────────────────
def login_page():
    st.markdown("""
    <div style='text-align:center; padding: 2rem 0 1rem 0;'>
      <h1 style='font-size:2.4rem; letter-spacing:3px;'>💰 HUSTLER FUND</h1>
      <p style='color:#64b5f6; font-family: Space Mono, monospace; font-size:0.85rem; letter-spacing:2px;'>
        CREDIT DEFAULT RISK PREDICTION SYSTEM
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔐 Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True, key="login_btn"):
            if login_username and login_password:
                if login_username in st.session_state.user_data:
                    user = st.session_state.user_data[login_username]
                    if AuthManager.validate_password(login_password, user['password']):
                        if user.get('blocked', 0) == 1:
                            st.error("❌ Your account has been blocked by an administrator.")
                        else:
                            st.session_state.logged_in = True
                            st.session_state.username = login_username
                            st.session_state.is_admin = user.get('is_admin', False)
                            st.success("✅ Logged in successfully!")
                            st.rerun()
                    else:
                        st.error("❌ Invalid password")
                else:
                    st.error("❌ User not found")
            else:
                st.warning("⚠️ Please enter username and password")

    with col2:
        st.subheader("📝 Sign Up")
        signup_username = st.text_input("Username", key="signup_username")
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")

        if st.button("Sign Up", use_container_width=True, key="signup_btn"):
            if not all([signup_username, signup_email, signup_password, signup_password_confirm]):
                st.warning("⚠️ Please fill all fields")
            elif signup_username in st.session_state.user_data:
                st.error("❌ Username already exists")
            elif signup_password != signup_password_confirm:
                st.error("❌ Passwords do not match")
            elif len(signup_password) < 8:
                st.error("❌ Password must be at least 8 characters")
            else:
                hashed_password = AuthManager.hash_password(signup_password)
                st.session_state.user_data[signup_username] = {
                    'password': hashed_password,
                    'email': signup_email,
                    'is_admin': False,
                    'blocked': 0,
                    'created_at': datetime.now().isoformat()
                }
                save_user_data(st.session_state.user_data)
                st.success("✅ Account created successfully! Please login.")


# ─────────────────────────────────────────────
# User Dashboard
# ─────────────────────────────────────────────
def user_dashboard(df, model, preprocessor, metrics, importance):
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
      <h1 style='font-size:2.0rem; letter-spacing:2px;'>💰 HUSTLER FUND</h1>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Predict Risk", "📊 Model Performance", "📈 Data Insights", "📋 Dataset", "📜 My History"])

    with tab1:
        st.markdown("### Applicant Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Personal Info**")
            age = st.slider("Age", 18, 65, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            county = st.selectbox("County", ['Nairobi', 'Mombasa', 'Kisumu', 'Machakos', 'Kiambu', 'Nakuru', 'Meru'])
            monthly_income = st.number_input("Monthly Income (KES)", 5000, 150000, 25000, step=1000)

        with col2:
            st.markdown("**💳 Loan Details**")
            loan_amount = st.number_input("Loan Amount (KES)", 500, 50000, 5000, step=500)
            loan_purpose = st.selectbox("Loan Purpose", ['Business', 'School Fees', 'Emergency', 'Personal', 'Farm Inputs'])

        with col3:
            st.markdown("**📱 M-Pesa Activity & Scores**")
            mpesa_transactions = st.slider("M-Pesa Transactions (monthly)", 5, 200, 50)
            mpesa_volume = st.number_input("M-Pesa Volume (KES)", 1000, 100000, 15000, step=500)
            repayment_score = st.slider("Repayment Score", 0.0, 1.0, 0.68, step=0.01)
            credit_score = st.slider("Credit Score", 0.0, 1.0, 0.65, step=0.01)

        if st.button("⚡ ASSESS CREDIT RISK", use_container_width=True):
            input_df = pd.DataFrame([{
                'Age': age, 'Gender': gender, 'County': county,
                'Monthly_Income': monthly_income, 'Loan_Amount': loan_amount,
                'Loan_Purpose': loan_purpose,
                'Mpesa_Transactions': mpesa_transactions,
                'Mpesa_Volume': mpesa_volume,
                'Repayment_Score': repayment_score,
                'Credit_Score': credit_score
            }])

            input_proc = preprocessor.transform(input_df)
            prob = model.predict_proba(input_proc)[0][1]
            default_prob = 1 - prob

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Repayment Probability", f"{prob*100:.1f}%")
            with r2:
                st.metric("Default Probability", f"{default_prob*100:.1f}%")
            with r3:
                risk_level = "HIGH RISK" if default_prob > 0.4 else ("MEDIUM RISK" if default_prob > 0.25 else "LOW RISK")
                st.metric("Risk Level", risk_level)

            if default_prob > 0.4:
                st.markdown(f'<div class="risk-high">⚠ HIGH DEFAULT RISK — Recommend: DECLINE or REDUCE LOAN</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✓ LOW DEFAULT RISK — Recommend: APPROVE LOAN</div>', unsafe_allow_html=True)

            # Save prediction
            if st.session_state.username not in st.session_state.predictions:
                st.session_state.predictions[st.session_state.username] = []
            
            st.session_state.predictions[st.session_state.username].append({
                'timestamp': datetime.now().isoformat(),
                'default_probability': float(default_prob),
                'risk_level': risk_level
            })
            save_predictions(st.session_state.predictions)
            st.success("✅ Prediction saved to your history!")

    with tab2:
        st.markdown("### Model Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        report = metrics['report']
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        m3.metric("Precision (Class 1)", f"{report['1']['precision']:.4f}")
        m4.metric("Recall (Class 1)", f"{report['1']['recall']:.4f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            sns.heatmap(metrics['confusion'], annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=0.5, annot_kws={'color': 'white', 'size': 13})
            ax.set_xlabel('Predicted', color='#90a4ae')
            ax.set_ylabel('Actual', color='#90a4ae')
            ax.tick_params(colors='#90a4ae')
            st.pyplot(fig)
            plt.close()

        with c2:
            st.markdown("#### ROC Curve")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            ax.plot(metrics['fpr'], metrics['tpr'], color='#00e5a0', lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
            ax.plot([0, 1], [0, 1], '--', color='#455a64', lw=1)
            ax.set_xlabel('False Positive Rate', color='#90a4ae')
            ax.set_ylabel('True Positive Rate', color='#90a4ae')
            ax.tick_params(colors='#90a4ae')
            ax.legend(facecolor='#1e3a5f', labelcolor='white')
            ax.set_title('ROC Curve', color='#00e5a0', pad=12)
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.markdown("### Dataset Insights")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Loan Purpose vs Repayment")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            purpose_rate = df.groupby('Loan_Purpose')['Loan_Repaid'].mean().sort_values()
            ax.barh(purpose_rate.index, purpose_rate.values, color=['#ff1744' if v < 0.5 else '#00e5a0' for v in purpose_rate.values])
            ax.set_xlabel('Repayment Rate', color='#90a4ae')
            ax.tick_params(colors='#90a4ae')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### County-wise Repayment Rate")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            county_rate = df.groupby('County')['Loan_Repaid'].mean().sort_values()
            ax.barh(county_rate.index, county_rate.values, color='#00b4d8')
            ax.set_xlabel('Repayment Rate', color='#90a4ae')
            ax.tick_params(colors='#90a4ae')
            st.pyplot(fig)
            plt.close()

    with tab4:
        st.markdown("### Synthetic Dataset Preview")
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Records", f"{len(df):,}")
        d2.metric("Features", str(df.shape[1] - 1))
        repaid_pct = df['Loan_Repaid'].mean() * 100
        d3.metric("Repayment Rate", f"{repaid_pct:.1f}%")

        st.markdown("#### Sample Data (first 100 rows)")
        st.dataframe(df.head(100), use_container_width=True)

    with tab5:
        st.markdown("### Your Prediction History")
        if st.session_state.username in st.session_state.predictions:
            history = st.session_state.predictions[st.session_state.username]
            if history:
                df_history = pd.DataFrame(history)
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(df_history, use_container_width=True)
                st.metric("Total Predictions Made", len(history))
            else:
                st.info("📊 No predictions yet. Make your first prediction!")
        else:
            st.info("📊 No predictions yet. Make your first prediction!")


# ─────────────────────────────────────────────
# Admin Dashboard
# ─────────────────────────────────────────────
def admin_dashboard():
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
      <h2 style='font-size:1.8rem; letter-spacing:2px;'>👨‍💼 ADMIN DASHBOARD</h2>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["👥 User Management", "📊 System Stats", "📜 Activity Log"])

    with tab1:
        st.subheader("Manage Users")
        users_list = list(st.session_state.user_data.keys())
        
        if users_list:
            for username in users_list:
                with st.expander(f"👤 {username}"):
                    user = st.session_state.user_data[username]
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Email:** {user['email']}")
                        st.write(f"**Admin:** {'Yes ✅' if user.get('is_admin') else 'No'}")
                        st.write(f"**Status:** {'🔒 Blocked' if user.get('blocked') else '✅ Active'}")

                    with col2:
                        if user.get('blocked'):
                            if st.button(f"🔓 Unblock {username}", key=f"unblock_{username}"):
                                st.session_state.user_data[username]['blocked'] = 0
                                save_user_data(st.session_state.user_data)
                                st.success(f"✅ {username} unblocked!")
                                st.rerun()
                        else:
                            if st.button(f"🔒 Block {username}", key=f"block_{username}"):
                                st.session_state.user_data[username]['blocked'] = 1
                                save_user_data(st.session_state.user_data)
                                st.warning(f"⚠️ {username} blocked!")
                                st.rerun()
        else:
            st.info("No users yet.")

    with tab2:
        st.subheader("System Statistics")
        total_users = len(st.session_state.user_data)
        blocked_users = sum(1 for u in st.session_state.user_data.values() if u.get('blocked'))
        admin_users = sum(1 for u in st.session_state.user_data.values() if u.get('is_admin'))
        total_predictions = sum(len(v) for v in st.session_state.predictions.values())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👥 Total Users", total_users)
        m2.metric("✅ Active Users", total_users - blocked_users)
        m3.metric("👨‍💼 Admin Users", admin_users)
        m4.metric("📊 Total Predictions", total_predictions)

    with tab3:
        st.subheader("Prediction Activity")
        if st.session_state.predictions:
            for username, preds in st.session_state.predictions.items():
                if preds:
                    st.write(f"**{username}**: {len(preds)} predictions made")
        else:
            st.info("No prediction activity yet.")


# ─────────────────────────────────────────────
# Main Application Logic
# ─────────────────────────────────────────────

if not st.session_state.logged_in:
    login_page()
else:
    # Load model and data
    df = generate_data()
    with st.spinner("🔄 Training model on synthetic dataset..."):
        model, preprocessor, metrics, importance, X_test, y_test = train_model(df)

    # Sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**👤 Logged in as:** `{st.session_state.username}`")
        
        if st.session_state.is_admin:
            st.markdown("**🔑 Role:** Admin")
        else:
            st.markdown("**🔑 Role:** User")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.is_admin = False
            st.rerun()

    if st.session_state.is_admin:
        admin_dashboard()
    else:
        user_dashboard(df, model, preprocessor, metrics, importance)
