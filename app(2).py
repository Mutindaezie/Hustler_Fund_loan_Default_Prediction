import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

/* Dark background */
.stApp {
    background: #0a0f1e;
    color: #e8eaf6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1526;
    border-right: 1px solid #1e2d4a;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    color: #00e5a0 !important;
}

h4, h5, h6 {
    color: #b0bec5 !important;
}

/* Metric cards */
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

/* Buttons */
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

/* Tabs */
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

/* Selectbox, sliders */
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #90a4ae;
    font-size: 0.82rem;
    font-family: 'Space Mono', monospace;
}

/* Risk badge */
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

    # Feature importance
    ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
    feature_names = num_features + list(ohe_features)
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    return model, preprocessor, metrics, importance, X_test, y_test


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────
df = generate_data()

with st.spinner("Training model on synthetic dataset..."):
    model, preprocessor, metrics, importance, X_test, y_test = train_model(df)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem 0;'>
  <h1 style='font-size:2.4rem; letter-spacing:3px;'>💰 HUSTLER FUND</h1>
  <p style='color:#64b5f6; font-family: Space Mono, monospace; font-size:0.85rem; letter-spacing:2px;'>
    CREDIT DEFAULT RISK PREDICTION SYSTEM
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict Risk", "📊 Model Performance", "📈 Data Insights", "📋 Dataset"])

# ══════════════════════════════════════════════
# TAB 1 – PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### Applicant Details")
    st.markdown('<div class="info-box">Fill in the applicant\'s information below to get an instant default risk assessment.</div>', unsafe_allow_html=True)

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

    st.markdown("---")
    predict_btn = st.button("⚡ ASSESS CREDIT RISK", use_container_width=True)

    if predict_btn:
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
        prediction = model.predict(input_proc)[0]
        default_prob = 1 - prob

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Repayment Probability", f"{prob*100:.1f}%")
        with r2:
            st.metric("Default Probability", f"{default_prob*100:.1f}%")
        with r3:
            risk_level = "HIGH RISK" if default_prob > 0.4 else ("MEDIUM RISK" if default_prob > 0.25 else "LOW RISK")
            st.metric("Risk Level", risk_level)

        st.markdown("<br>", unsafe_allow_html=True)
        if default_prob > 0.4:
            st.markdown(f'<div class="risk-high">⚠ HIGH DEFAULT RISK — Recommend: DECLINE or REDUCE LOAN</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✓ LOW DEFAULT RISK — Recommend: APPROVE LOAN</div>', unsafe_allow_html=True)

        # Gauge bar
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 1.2))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        ax.barh(0, 1, color='#1e3a5f', height=0.5)
        color = '#ff1744' if default_prob > 0.4 else ('#ffab00' if default_prob > 0.25 else '#00e5a0')
        ax.barh(0, default_prob, color=color, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='#90a4ae')
        ax.set_title("Default Risk Meter", color='#00e5a0', fontsize=11, pad=8)
        ax.spines[:].set_visible(False)
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
# TAB 2 – MODEL PERFORMANCE
# ══════════════════════════════════════════════
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
        sns.heatmap(metrics['confusion'], annot=True, fmt='d', cmap='Blues',
                    ax=ax, linewidths=0.5,
                    annot_kws={'color': 'white', 'size': 13})
        ax.set_xlabel('Predicted', color='#90a4ae')
        ax.set_ylabel('Actual', color='#90a4ae')
        ax.tick_params(colors='#90a4ae')
        ax.set_title('Confusion Matrix', color='#00e5a0', pad=12)
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("#### ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        ax.plot(metrics['fpr'], metrics['tpr'], color='#00e5a0', lw=2,
                label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], '--', color='#455a64', lw=1)
        ax.set_xlabel('False Positive Rate', color='#90a4ae')
        ax.set_ylabel('True Positive Rate', color='#90a4ae')
        ax.tick_params(colors='#90a4ae')
        ax.legend(facecolor='#1e3a5f', labelcolor='white')
        ax.set_title('ROC Curve', color='#00e5a0', pad=12)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a5f')
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Top 10 Feature Importances")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    top10 = importance.head(10)
    colors = ['#00e5a0' if i == 0 else '#00b4d8' if i < 3 else '#1e88e5' for i in range(len(top10))]
    ax.barh(top10.index[::-1], top10.values[::-1], color=colors[::-1])
    ax.set_xlabel('Importance', color='#90a4ae')
    ax.tick_params(colors='#90a4ae', labelsize=9)
    ax.set_title('Feature Importance (Random Forest)', color='#00e5a0', pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
# TAB 3 – DATA INSIGHTS
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### Dataset Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Loan Purpose vs Repayment")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        purpose_rate = df.groupby('Loan_Purpose')['Loan_Repaid'].mean().sort_values()
        bars = ax.barh(purpose_rate.index, purpose_rate.values,
                       color=['#ff1744' if v < 0.5 else '#00e5a0' for v in purpose_rate.values])
        ax.set_xlabel('Repayment Rate', color='#90a4ae')
        ax.tick_params(colors='#90a4ae')
        ax.set_title('Repayment Rate by Loan Purpose', color='#00e5a0', pad=10)
        for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
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
        ax.set_title('Repayment Rate by County', color='#00e5a0', pad=10)
        for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
        st.pyplot(fig)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Income Distribution by Repayment")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        for label, color in [(0, '#ff1744'), (1, '#00e5a0')]:
            subset = df[df['Loan_Repaid'] == label]['Monthly_Income']
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label='Defaulted' if label == 0 else 'Repaid')
        ax.set_xlabel('Monthly Income (KES)', color='#90a4ae')
        ax.set_ylabel('Count', color='#90a4ae')
        ax.tick_params(colors='#90a4ae')
        ax.legend(facecolor='#1e3a5f', labelcolor='white')
        ax.set_title('Income Distribution', color='#00e5a0', pad=10)
        for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("#### Credit Score Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        for label, color in [(0, '#ff1744'), (1, '#00e5a0')]:
            subset = df[df['Loan_Repaid'] == label]['Credit_Score']
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label='Defaulted' if label == 0 else 'Repaid')
        ax.set_xlabel('Credit Score', color='#90a4ae')
        ax.set_ylabel('Count', color='#90a4ae')
        ax.tick_params(colors='#90a4ae')
        ax.legend(facecolor='#1e3a5f', labelcolor='white')
        ax.set_title('Credit Score Distribution', color='#00e5a0', pad=10)
        for spine in ax.spines.values(): spine.set_edgecolor('#1e3a5f')
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, linewidths=0.3, annot_kws={'size': 8})
    ax.tick_params(colors='#90a4ae', labelsize=8)
    ax.set_title('Feature Correlation Matrix', color='#00e5a0', pad=12)
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
# TAB 4 – DATASET
# ══════════════════════════════════════════════
with tab4:
    st.markdown("### Synthetic Dataset Preview")
    d1, d2, d3 = st.columns(3)
    d1.metric("Total Records", f"{len(df):,}")
    d2.metric("Features", str(df.shape[1] - 1))
    repaid_pct = df['Loan_Repaid'].mean() * 100
    d3.metric("Repayment Rate", f"{repaid_pct:.1f}%")

    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("#### Sample Data (first 100 rows)")
    st.dataframe(df.head(100), use_container_width=True)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Full Dataset (CSV)", csv,
                       "hustler_fund_dataset.csv", "text/csv")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0;'>
      <h3 style='font-size:1rem; letter-spacing:2px;'>ℹ ABOUT</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    This app predicts the probability of a Hustler Fund loan being repaid using a <b>Random Forest</b> model trained on 10,000 synthetic applicant records.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Model Details**")
    st.markdown(f"""
    <div class="info-box">
    • Algorithm: Random Forest<br>
    • Balancing: SMOTE<br>
    • Accuracy: <b>{metrics['accuracy']:.3f}</b><br>
    • ROC-AUC: <b>{metrics['roc_auc']:.3f}</b><br>
    • Train/Test: 80/20
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Features Used**")
    features = ['Age', 'Gender', 'County', 'Monthly Income', 'Loan Amount',
                'Loan Purpose', 'M-Pesa Transactions', 'M-Pesa Volume',
                'Repayment Score', 'Credit Score']
    for f in features:
        st.markdown(f"<small style='color:#64b5f6'>▸ {f}</small>", unsafe_allow_html=True)
