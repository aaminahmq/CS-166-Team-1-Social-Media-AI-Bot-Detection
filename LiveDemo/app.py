import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bot Detection — CS166 Team 1",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F8FA;
    color: #1a1d23;
}

.header-bar {
    background: #1E3A5F;
    color: #ffffff;
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem -1rem;
    border-bottom: 3px solid #2563EB;
}
.header-bar h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    font-weight: 400;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.01em;
    color: #ffffff;
}
.header-bar p {
    font-size: 0.85rem;
    color: #93C5FD;
    margin: 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

.result-card {
    border-radius: 6px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    text-align: center;
}
.result-card.bot {
    background: #FEF2F2;
    border: 1.5px solid #FECACA;
    border-left: 5px solid #DC2626;
}
.result-card.human {
    background: #F0FDF4;
    border: 1.5px solid #BBF7D0;
    border-left: 5px solid #16A34A;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 1.1rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-label.bot   { color: #DC2626; }
.result-label.human { color: #16A34A; }
.result-confidence {
    font-size: 2.6rem;
    font-family: 'DM Serif Display', serif;
    line-height: 1;
    margin: 0.5rem 0;
}
.result-confidence.bot   { color: #DC2626; }
.result-confidence.human { color: #16A34A; }
.result-sub {
    font-size: 0.82rem;
    color: #6B7280;
    margin-top: 0.3rem;
}

.signal-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #E5E7EB;
    font-size: 0.88rem;
}
.signal-row:last-child { border-bottom: none; }
.signal-name  { color: #374151; }
.signal-val   { font-family: 'DM Mono', monospace; font-size: 0.82rem; }
.signal-flag  { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
                padding: 0.15rem 0.5rem; border-radius: 3px; }
.flag-warn    { background: #FEF3C7; color: #92400E; }
.flag-ok      { background: #D1FAE5; color: #065F46; }

.sample-card {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    font-size: 0.85rem;
    line-height: 1.7;
}
.sample-card .sample-title {
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.sample-card.bot-sample   .sample-title { color: #DC2626; }
.sample-card.human-sample .sample-title { color: #16A34A; }
.sample-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #374151;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stCheckbox"] label {
    font-size: 0.85rem !important;
    color: #374151 !important;
    font-weight: 500 !important;
}
.stButton > button {
    background: #1E3A5F !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.55rem 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load & train ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    url = "https://raw.githubusercontent.com/aaminahmq/Datafiles/refs/heads/main/CS%20166%20group%20project/twitter_human_bots_dataset.csv"
    df = pd.read_csv(url)
    df["Bot Label"] = (df["account_type"] == "bot").astype(int)
    df["description"] = df["description"].fillna("")
    for col in ["default_profile", "default_profile_image", "geo_enabled", "verified"]:
        df[col] = df[col].astype(int)
    df["description_length"]      = df["description"].str.len()
    df["followers_friends_ratio"] = df["followers_count"] / (df["friends_count"] + 1)
    features = [
        "followers_count", "friends_count", "favourites_count", "statuses_count",
        "verified", "default_profile", "default_profile_image", "geo_enabled",
        "average_tweets_per_day", "account_age_days",
        "description_length", "followers_friends_ratio"
    ]
    X = df[features]
    y = df["Bot Label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train_s, y_train)
    return model, scaler, features


with st.spinner("Loading model..."):
    model, scaler, features = load_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>Social Media Bot Detector</h1>
  <p>CS166 Information Security &nbsp;·&nbsp; Team 1 &nbsp;·&nbsp; Spring 2026</p>
</div>
""", unsafe_allow_html=True)


# ── Sample accounts ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Sample Accounts for Demo</div>', unsafe_allow_html=True)

col_s1, col_s2 = st.columns(2)

with col_s1:
    st.markdown("""
    <div class="sample-card bot-sample">
      <div class="sample-title">Expected: Bot</div>
      <div>Followers: <span class="sample-val">8</span></div>
      <div>Following: <span class="sample-val">4500</span></div>
      <div>Total Likes: <span class="sample-val">0</span></div>
      <div>Total Tweets: <span class="sample-val">48000</span></div>
      <div>Avg Tweets / Day: <span class="sample-val">180.0</span></div>
      <div>Account Age (days): <span class="sample-val">6</span></div>
      <div>Bio Length: <span class="sample-val">0</span></div>
      <div>Verified: <span class="sample-val">No</span></div>
      <div>Default Profile: <span class="sample-val">Yes</span></div>
      <div>Default Image: <span class="sample-val">Yes</span></div>
      <div>Geo Enabled: <span class="sample-val">No</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_s2:
    st.markdown("""
    <div class="sample-card human-sample">
      <div class="sample-title">Expected: Human</div>
      <div>Followers: <span class="sample-val">843</span></div>
      <div>Following: <span class="sample-val">290</span></div>
      <div>Total Likes: <span class="sample-val">4200</span></div>
      <div>Total Tweets: <span class="sample-val">1650</span></div>
      <div>Avg Tweets / Day: <span class="sample-val">2.5</span></div>
      <div>Account Age (days): <span class="sample-val">1380</span></div>
      <div>Bio Length: <span class="sample-val">92</span></div>
      <div>Verified: <span class="sample-val">No</span></div>
      <div>Default Profile: <span class="sample-val">No</span></div>
      <div>Default Image: <span class="sample-val">No</span></div>
      <div>Geo Enabled: <span class="sample-val">Yes</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#E5E7EB; margin: 1.5rem 0;'>", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Enter Account Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    followers      = st.number_input("Followers",              min_value=0,   value=150,  step=1)
    friends        = st.number_input("Following",              min_value=0,   value=300,  step=1)
    favourites     = st.number_input("Total Likes Given",      min_value=0,   value=500,  step=1)
    statuses       = st.number_input("Total Tweets Posted",    min_value=0,   value=800,  step=1)
    avg_tweets_day = st.number_input("Avg Tweets / Day",       min_value=0.0, value=2.5,  step=0.1, format="%.1f")
    account_age    = st.number_input("Account Age (days)",     min_value=0,   value=600,  step=1)

with col2:
    desc_length     = st.number_input("Bio Length (characters)", min_value=0, value=80, step=1)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    verified        = st.checkbox("Verified account",          value=False)
    default_profile = st.checkbox("Default profile settings", value=False)
    default_img     = st.checkbox("Default profile image",    value=False)
    geo_enabled     = st.checkbox("Geo location enabled",     value=True)

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
run = st.button("Run Detection")


# ── Prediction output ─────────────────────────────────────────────────────────
if run:
    ff_ratio = followers / (friends + 1)
    input_df = pd.DataFrame([[
        followers, friends, favourites, statuses,
        int(verified), int(default_profile), int(default_img), int(geo_enabled),
        avg_tweets_day, account_age, desc_length, ff_ratio
    ]], columns=features)

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    prob_bot     = model.predict_proba(input_scaled)[0][1]
    prob_human   = 1 - prob_bot
    is_bot       = prediction == 1

    if is_bot:
        st.markdown(f"""
        <div class="result-card bot">
          <div class="result-label bot">Bot Account Detected</div>
          <div class="result-confidence bot">{prob_bot:.1%}</div>
          <div class="result-sub">bot probability &nbsp;·&nbsp; Random Forest classifier</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card human">
          <div class="result-label human">Human Account</div>
          <div class="result-confidence human">{prob_human:.1%}</div>
          <div class="result-sub">human probability &nbsp;·&nbsp; Random Forest classifier</div>
        </div>
        """, unsafe_allow_html=True)

    # Signal breakdown
    st.markdown('<div class="section-label" style="margin-top:1rem;">Signal Breakdown</div>', unsafe_allow_html=True)

    signals = [
        ("Followers / Following ratio",
         f"{ff_ratio:.3f}",
         "Suspicious — very few followers relative to following" if ff_ratio < 0.1 else "Normal",
         ff_ratio < 0.1),
        ("Account age",
         f"{account_age} days",
         "Suspicious — account is very new" if account_age < 30 else "Normal",
         account_age < 30),
        ("Avg tweets per day",
         f"{avg_tweets_day}",
         "Suspicious — unusually high posting rate" if avg_tweets_day > 50 else "Normal",
         avg_tweets_day > 50),
        ("Default profile image",
         "Yes" if default_img else "No",
         "Suspicious — real users typically set a profile photo" if default_img else "Normal",
         default_img),
        ("Verified",
         "Yes" if verified else "No",
         "Normal — verified accounts are almost always human" if verified else "Neutral",
         False),
        ("Bio length",
         f"{desc_length} characters",
         "Suspicious — no bio is a common bot pattern" if desc_length == 0 else "Normal",
         desc_length == 0),
    ]

    rows_html = ""
    for name, val, note, warn in signals:
        flag_class = "flag-warn" if warn else "flag-ok"
        flag_label = "Suspicious" if warn else "Normal"
        rows_html += f"""
        <div class="signal-row">
          <span class="signal-name">{name}</span>
          <span style="display:flex;align-items:center;gap:0.75rem;">
            <span class="signal-val">{val}</span>
            <span class="signal-flag {flag_class}">{flag_label}</span>
          </span>
        </div>"""

    st.markdown(
        f"<div style='background:#fff;border:1px solid #E5E7EB;border-radius:6px;padding:0.5rem 1rem;'>{rows_html}</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
    <p style="font-size:0.78rem; color:#9CA3AF; margin-top:1.2rem;">
      Model trained on 37,438 labeled Twitter accounts &nbsp;·&nbsp;
      Random Forest (500 trees) &nbsp;·&nbsp;
      Accuracy 88.22% &nbsp;·&nbsp; AUC-ROC 0.9348
    </p>
    """, unsafe_allow_html=True)