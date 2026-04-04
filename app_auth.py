import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
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
# Custom CSS with Background
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, rgba(10, 15, 30, 0.95), rgba(13, 21, 38, 0.95)), 
                url('https://images.unsplash.com/photo-1557821552-17105176677c?w=1200') center/cover,
                linear-gradient(to right, #0a0f1e, #1a2332);
    background-blend-mode: overlay, screen, multiply;
    color: #e8eaf6;
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: rgba(13, 21, 38, 0.98);
    border-right: 1px solid #1e2d4a;
}

h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #00e5a0 !important;
    text-shadow: 0 2px 10px rgba(0, 229, 160, 0.3);
}

h4, h5, h6 {
    color: #b0bec5 !important;
}

[data-testid="metric-container"] {
    background: rgba(17, 24, 39, 0.9);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(10px);
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
    background: rgba(17, 24, 39, 0.8);
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
    box-shadow: 0 8px 25px rgba(255, 23, 68, 0.3);
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
    box-shadow: 0 8px 25px rgba(0, 229, 160, 0.3);
}

.risk-medium {
    background: linear-gradient(135deg, #ffab00, #ff6f00);
    color: white;
    padding: 16px 24px;
    border-radius: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 2px;
    box-shadow: 0 8px 25px rgba(255, 171, 0, 0.3);
}

.info-box {
    background: rgba(17, 24, 39, 0.9);
    border-left: 3px solid #00e5a0;
    padding: 14px 18px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #cfd8dc;
    backdrop-filter: blur(10px);
}

.loan-analysis {
    background: rgba(30, 58, 95, 0.8);
    border-left: 4px solid #00e5a0;
    padding: 16px;
    border-radius: 10px;
    margin: 10px 0;
    font-family: 'Space Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Database Functions
# ─────────────────────────────────────────────
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('hustler_fund.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        blocked INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        county TEXT,
        monthly_income INTEGER,
        loan_amount INTEGER,
        loan_purpose TEXT,
        mpesa_transactions INTEGER,
        mpesa_volume INTEGER,
        repayment_score REAL,
        credit_score REAL,
        default_probability REAL,
        risk_level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES users(username)
    )''')
    
    conn.commit()
    conn.close()


def register_user(full_name, username, password):
    """Register a new user in the database"""
    try:
        conn = sqlite3.connect('hustler_fund.db')
        c = conn.cursor()
        
        hashed_password = AuthManager.hash_password(password)
        
        c.execute('INSERT INTO users (full_name, username, password) VALUES (?, ?, ?)',
                  (full_name, username, hashed_password))
        
        conn.commit()
        conn.close()
        return True, "✅ Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "❌ Username already exists"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"


def login_user(username, password):
    """Verify user credentials"""
    try:
        conn = sqlite3.connect('hustler_fund.db')
        c = conn.cursor()
        
        c.execute('SELECT full_name, password, is_admin, blocked FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        
        if user:
            full_name, hashed_password, is_admin, blocked = user
            
            if blocked:
                return False, None, False, "❌ Your account has been blocked"
            
            if AuthManager.validate_password(password, hashed_password):
                return True, full_name, bool(is_admin), "✅ Login successful"
            else:
                return False, None, False, "❌ Invalid password"
        else:
            return False, None, False, "❌ Username not found"
    except Exception as e:
        return False, None, False, f"❌ Error: {str(e)}"


def save_prediction(username, age, gender, county, monthly_income, loan_amount, 
                   loan_purpose, mpesa_transactions, mpesa_volume, repayment_score, 
                   credit_score, default_probability, risk_level):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect('hustler_fund.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO predictions 
                    (username, age, gender, county, monthly_income, loan_amount, loan_purpose,
                     mpesa_transactions, mpesa_volume, repayment_score, credit_score,
                     default_probability, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (username, age, gender, county, monthly_income, loan_amount, loan_purpose,
                   mpesa_transactions, mpesa_volume, repayment_score, credit_score,
                   default_probability, risk_level))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error saving prediction: {str(e)}")
        return False


def get_user_predictions(username):
    """Get all predictions for a user"""
    try:
        conn = sqlite3.connect('hustler_fund.db')
        c = conn.cursor()
        
        c.execute('''SELECT age, gender, county, monthly_income, loan_amount, loan_purpose,
                           mpesa_transactions, mpesa_volume, repayment_score, credit_score,
                           default_probability, risk_level, created_at
                    FROM predictions WHERE username = ? ORDER BY created_at DESC''', (username,))
        
        predictions = c.fetchall()
        conn.close()
        
        return predictions
    except Exception as e:
        st.error(f"Error retrieving predictions: {str(e)}")
        return []


def get_all_users():
    """Get all users from database"""
    try:
        conn = sqlite3.connect('hustler_fund.db')
        c = conn.cursor()
        
        c.execute('SELECT full_name, username, is_admin, blocked, created_at FROM users ORDER BY created_at DESC')
        users = c.fetchall()
        conn.close()
        
        return users
    except Exception as e:
        st.error(f"Error retrieving users: {str(e)}")
        return []


def block_unblock_user(username, block):
    """Block or unblock a user"""
    try:
        conn = sqlite3.connect('hustler_fund.db')
        c = conn.cursor()
        
        c.execute('UPDATE users SET blocked = ? WHERE username = ?', (1 if block else 0, username))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False


# ─────────────────────────────────────────────
# Enhanced Prediction Logic
# ─────────────────────────────────────────────
def calculate_default_risk(age, monthly_income, loan_amount, mpesa_transactions, mpesa_volume):
    """
    Enhanced prediction logic considering key factors:
    - Age (younger = higher risk)
    - Loan to Income Ratio (higher ratio = higher risk)
    - M-Pesa activity (lower activity = higher risk)
    """
    risk_score = 0
    
    # Age risk: Younger = Higher Risk
    if age < 25:
        risk_score += 0.25
    elif age < 35:
        risk_score += 0.15
    elif age > 55:
        risk_score += 0.10
    
    # Loan to Income Ratio Risk
    loan_to_income = loan_amount / monthly_income if monthly_income > 0 else 10
    
    if loan_to_income > 2:  # Loan is more than 2x monthly income
        risk_score += 0.30
    elif loan_to_income > 1.5:  # Loan is 1.5-2x monthly income
        risk_score += 0.20
    elif loan_to_income > 1:  # Loan is 1-1.5x monthly income
        risk_score += 0.10
    
    # M-Pesa Activity Risk: Lower activity = Higher Risk
    if mpesa_transactions < 10:
        risk_score += 0.25
    elif mpesa_transactions < 30:
        risk_score += 0.15
    elif mpesa_transactions < 50:
        risk_score += 0.05
    
    # M-Pesa Volume Risk
    if mpesa_volume < 5000:
        risk_score += 0.15
    elif mpesa_volume < 10000:
        risk_score += 0.08
    
    # Cap risk score at 1.0
    default_probability = min(risk_score, 1.0)
    
    return default_probability


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
# Initialize Session State
# ─────────────────────────────────────────────
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'full_name' not in st.session_state:
    st.session_state.full_name = None
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

init_database()


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
      <p style='color:#90a4ae; font-size:0.75rem; margin-top: 1rem;'>Empowering Hustlers 🇰🇪</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔐 Login")
        st.markdown('<div class="info-box">Enter your credentials to login to your account</div>', unsafe_allow_html=True)
        
        login_username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        login_password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")

        if st.button("🔓 Login", use_container_width=True, key="login_btn"):
            if login_username and login_password:
                success, full_name, is_admin, message = login_user(login_username, login_password)
                
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = login_username
                    st.session_state.full_name = full_name
                    st.session_state.is_admin = is_admin
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.warning("⚠️ Please enter username and password")

    with col2:
        st.subheader("📝 Create Account")
        st.markdown('<div class="info-box">Sign up to create a new account</div>', unsafe_allow_html=True)
        
        signup_full_name = st.text_input("Full Name", key="signup_full_name", placeholder="Enter your full name")
        signup_username = st.text_input("Username", key="signup_username", placeholder="Choose a username")
        signup_password = st.text_input("New Password", type="password", key="signup_password", placeholder="Create a strong password (min 8 chars)")
        signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm", placeholder="Confirm your password")

        if st.button("✅ Sign Up", use_container_width=True, key="signup_btn"):
            if not all([signup_full_name, signup_username, signup_password, signup_password_confirm]):
                st.warning("⚠️ Please fill all fields")
            elif len(signup_password) < 8:
                st.error("❌ Password must be at least 8 characters long")
            elif signup_password != signup_password_confirm:
                st.error("❌ Passwords do not match")
            elif not signup_username.replace('_', '').replace('-', '').isalnum():
                st.error("❌ Username can only contain letters, numbers, hyphens, and underscores")
            else:
                success, message = register_user(signup_full_name, signup_username, signup_password)
                
                if success:
                    st.success(message)
                    st.info("🔐 Your account has been created! Please login with your credentials.")
                else:
                    st.error(message)


# ─────────────────────────────────────────────
# User Dashboard - Simple Prediction Only
# ─────────────────────────────────────────────
def user_dashboard():
    st.markdown(f"""
    <div style='text-align:center; padding: 1rem 0;'>
      <h1 style='font-size:2.0rem; letter-spacing:2px;'>💰 HUSTLER FUND</h1>
      <p style='color:#64b5f6; font-size:0.9rem;'>Welcome, {st.session_state.full_name}! 👋</p>
      <p style='color:#90a4ae; font-size:0.8rem;'>Check your loan eligibility instantly</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔍 Assess Loan", "📜 My History"])

    with tab1:
        st.markdown("### 📋 Loan Application")
        st.markdown('<div class="info-box">Fill in your details to get an instant loan eligibility assessment</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**👤 Personal Information**")
            age = st.slider("Your Age", 18, 65, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            county = st.selectbox("County", ['Nairobi', 'Mombasa', 'Kisumu', 'Machakos', 'Kiambu', 'Nakuru', 'Meru'])
            monthly_income = st.number_input("Monthly Income (KES)", 5000, 150000, 25000, step=1000)

        with col2:
            st.markdown("**💳 Loan Request**")
            loan_amount = st.number_input("Loan Amount (KES)", 500, 50000, 5000, step=500)
            loan_purpose = st.selectbox("Loan Purpose", ['Business', 'School Fees', 'Emergency', 'Personal', 'Farm Inputs'])
            mpesa_transactions = st.slider("M-Pesa Transactions (monthly)", 5, 200, 50)
            mpesa_volume = st.number_input("M-Pesa Volume (KES)", 1000, 100000, 15000, step=500)

        if st.button("⚡ GET ASSESSMENT", use_container_width=True):
            # Calculate using enhanced logic
            default_prob = calculate_default_risk(age, monthly_income, loan_amount, mpesa_transactions, mpesa_volume)
            repayment_prob = 1 - default_prob
            
            if default_prob > 0.4:
                risk_level = "HIGH RISK"
            elif default_prob > 0.25:
                risk_level = "MEDIUM RISK"
            else:
                risk_level = "LOW RISK"

            # Display Results
            st.markdown("---")
            st.markdown("### 📊 Assessment Results")
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Approval Probability", f"{repayment_prob*100:.1f}%")
            with r2:
                st.metric("Default Risk", f"{default_prob*100:.1f}%")
            with r3:
                st.metric("Status", risk_level)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show Risk Classification
            if default_prob > 0.4:
                st.markdown(f'<div class="risk-high">⚠ HIGH RISK — Recommendation: FURTHER REVIEW REQUIRED</div>', unsafe_allow_html=True)
            elif default_prob > 0.25:
                st.markdown(f'<div class="risk-medium">⚡ MEDIUM RISK — Recommendation: CONDITIONAL APPROVAL</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✓ LOW RISK — Recommendation: APPROVED</div>', unsafe_allow_html=True)

            # Risk Analysis
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Risk Analysis")
            
            loan_to_income = loan_amount / monthly_income
            
            analysis = f"""
            <div class="loan-analysis">
            <b>Loan Analysis Factors:</b><br><br>
            • <b>Age Factor:</b> {age} years old - {'⚠ Higher risk for younger applicants' if age < 25 else '✓ Good age for lending'}<br>
            • <b>Loan-to-Income Ratio:</b> {loan_to_income:.2f}x - {'⚠ Loan is too high relative to income' if loan_to_income > 2 else '✓ Manageable loan amount'}<br>
            • <b>Monthly Income:</b> KES {monthly_income:,} - {'⚠ Low income level' if monthly_income < 15000 else '✓ Sufficient income'}<br>
            • <b>M-Pesa Activity:</b> {mpesa_transactions} transactions - {'⚠ Low activity' if mpesa_transactions < 30 else '✓ Good activity level'}<br>
            • <b>M-Pesa Volume:</b> KES {mpesa_volume:,} - {'⚠ Low transaction volume' if mpesa_volume < 10000 else '✓ Healthy transaction volume'}
            </div>
            """
            st.markdown(analysis, unsafe_allow_html=True)

            # Save prediction
            if save_prediction(st.session_state.username, age, gender, county, monthly_income, 
                             loan_amount, loan_purpose, mpesa_transactions, mpesa_volume, 
                             0.5, 0.5, float(default_prob), risk_level):
                st.success("✅ Assessment saved to your history!")

    with tab2:
        st.markdown("### Your Assessment History")
        predictions = get_user_predictions(st.session_state.username)
        
        if predictions:
            history_data = []
            for pred in predictions:
                history_data.append({
                    'Date': pd.to_datetime(pred[12]).strftime('%Y-%m-%d %H:%M'),
                    'Loan Amount': f"KES {pred[4]:,}",
                    'Income': f"KES {pred[3]:,}",
                    'Ratio': f"{pred[4]/pred[3]:.2f}x",
                    'Risk Level': pred[11],
                    'Default Risk': f"{pred[10]*100:.1f}%"
                })
            
            df_history = pd.DataFrame(history_data)
            st.dataframe(df_history, use_container_width=True, hide_index=True)
            st.metric("Total Assessments", len(predictions))
        else:
            st.info("📊 No assessments yet. Make your first assessment above!")


# ─────────────────────────────────────────────
# Admin Dashboard - Full Analytics
# ─────────────────────────────────────────────
def admin_dashboard(df, model, preprocessor, metrics, importance):
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
      <h2 style='font-size:1.8rem; letter-spacing:2px;'>👨‍💼 ADMIN DASHBOARD</h2>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["👥 Users", "📊 Model Performance", "📈 Data Insights", "📋 Dataset", "⚙ System"])

    with tab1:
        st.subheader("User Management")
        users = get_all_users()
        
        if users:
            for full_name, username, is_admin, blocked, created_at in users:
                with st.expander(f"👤 {full_name} (@{username})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Full Name:** {full_name}")
                        st.write(f"**Username:** @{username}")
                        st.write(f"**Admin:** {'Yes ✅' if is_admin else 'No'}")
                        st.write(f"**Status:** {'🔒 Blocked' if blocked else '✅ Active'}")
                        st.write(f"**Joined:** {created_at}")

                    with col2:
                        if blocked:
                            if st.button(f"🔓 Unblock {username}", key=f"unblock_{username}"):
                                block_unblock_user(username, False)
                                st.success(f"✅ {username} unblocked!")
                                st.rerun()
                        else:
                            if st.button(f"🔒 Block {username}", key=f"block_{username}"):
                                block_unblock_user(username, True)
                                st.warning(f"⚠️ {username} blocked!")
                                st.rerun()
        else:
            st.info("No users yet.")

    with tab2:
        st.markdown("### Model Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        report = metrics['report']
        m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        m2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        m3.metric("Precision", f"{report['1']['precision']:.4f}")
        m4.metric("Recall", f"{report['1']['recall']:.4f}")

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
        st.markdown("### Dataset Preview")
        d1, d2, d3 = st.columns(3)
        d1.metric("Total Records", f"{len(df):,}")
        d2.metric("Features", str(df.shape[1] - 1))
        repaid_pct = df['Loan_Repaid'].mean() * 100
        d3.metric("Repayment Rate", f"{repaid_pct:.1f}%")

        st.dataframe(df.head(100), use_container_width=True)

    with tab5:
        st.subheader("System Statistics")
        users = get_all_users()
        
        total_users = len(users)
        blocked_users = sum(1 for u in users if u[3])
        admin_users = sum(1 for u in users if u[2])
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👥 Total Users", total_users)
        m2.metric("✅ Active Users", total_users - blocked_users)
        m3.metric("👨‍💼 Admin Users", admin_users)
        m4.metric("📊 System Status", "🟢 Healthy")


# ─────────────────────────────────────────────
# Main Application Logic
# ─────────────────────────────────────────────

if not st.session_state.logged_in:
    login_page()
else:
    df = generate_data()
    with st.spinner("🔄 Loading system..."):
        model, preprocessor, metrics, importance, X_test, y_test = train_model(df)

    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**👤** {st.session_state.full_name}")
        st.markdown(f"**@** {st.session_state.username}")
        
        if st.session_state.is_admin:
            st.markdown("**🔑 Role:** Admin")
        else:
            st.markdown("**🔑 Role:** User")
        
        st.markdown("---")
        st.markdown("**Powered by:** 🇰🇪 Hustler Fund")
        st.markdown("**President:** William Samoei Ruto")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.full_name = None
            st.session_state.is_admin = False
            st.rerun()

    if st.session_state.is_admin:
        admin_dashboard(df, model, preprocessor, metrics, importance)
    else:
        user_dashboard()
