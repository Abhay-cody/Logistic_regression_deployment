import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

#Page Configuration
st.set_page_config(
    page_title="Insurance Purchase Prediction",
    page_icon="💵",
    layout="centered"
)

st.title("💵Insurance Purchase Prediction")
st.write("Predict Insurance Purchase using Logistic Regression")

# 1. Load Data
# Note: Ensure "insurance_data.csv" is in the same directory as this script!
try:
    df = pd.read_csv("insurance_data.csv")
    st.write("### Dataset Preview", df.head())
except FileNotFoundError:
    st.error("Please place 'insurance_data.csv' in the same folder as this script.")
    st.stop()

# 3. Train Test Split (Locked with random_state for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    df[['age']], 
    df.bought_insurance, 
    train_size=0.8, 
    random_state=42
)

# 4. Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Model Evaluation
st.write("### Model Performance")
st.write(f"**Model Accuracy Score:** {model.score(X_test, y_test):.2f}")

# Extracting dynamic coefficients so manual math always matches the model
m = model.coef_[0][0]
b = model.intercept_[0]

st.write("### Model Parameters")
st.write(f"**Coefficient (m):** {m:.4f}")
st.write(f"**Intercept (b):** {b:.4f}")

# 6. Manual Math & Sigmoid Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(age):
    # y = mx + b
    z = m * age + b 
    y = sigmoid(z)
    return y

# 7. Interactive Prediction UI
st.write("### Try Your Own Prediction")
user_age = st.number_input("Enter Age:", min_value=1, max_value=100, value=35)
probability = prediction_function(user_age)

st.write(f"Calculated Probability: **{probability:.3f}**")
if probability >= 0.5:
    st.success(f"With a probability of {probability:.3f}, a person aged {user_age} **will** buy insurance.")
else:
    st.warning(f"With a probability of {probability:.3f}, a person aged {user_age} **will not** buy insurance.")

# ==============================
# Sidebar Branding
# ==============================

st.markdown("""
<style>

.profile-container{
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
    padding:40px;
    border-radius:25px;
    box-shadow:0px 15px 40px rgba(0,0,0,0.35);
    color:white;
    margin-top:30px;
    animation: fadeIn 0.8s ease-in-out;
}

.profile-title{
    text-align:center;
    font-size:34px;
    font-weight:700;
    margin-bottom:5px;
}

.profile-subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:30px;
}

.profile-description{
    text-align:center;
    font-size:16px;
    color:#f8fafc;
    line-height:1.8;
    max-width:900px;
    margin:auto;
    margin-bottom:35px;
}

.stats{
    display:flex;
    justify-content:center;
    gap:25px;
    flex-wrap:wrap;
    margin-bottom:35px;
}

.stat-card{
    background:rgba(255,255,255,0.08);
    padding:20px;
    width:170px;
    border-radius:15px;
    text-align:center;
}

.stat-number{
    font-size:30px;
    font-weight:bold;
    color:#38bdf8;
}

.stat-text{
    color:#e2e8f0;
}

.buttons{
    text-align:center;
}

.btn{
    display:inline-block;
    margin:10px;
    padding:14px 28px;
    border-radius:12px;
    text-decoration:none;
    font-size:16px;
    font-weight:bold;
    transition:0.3s;
}

.github{
    background:#181717;
    color:white !important;
}

.linkedin{
    background:#0A66C2;
    color:white !important;
}

.btn:hover{
    transform:translateY(-4px);
    box-shadow:0 10px 25px rgba(0,0,0,0.3);
}

@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(20px);
    }
    to{
        opacity:1;
        transform:translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="profile-container">

<div class="profile-title">
👨‍💻 Abhay Kumar Gupta
</div>

<div class="profile-subtitle">
Machine Learning Engineer • Deep Learning Enthusiast • Python Developer
</div>

<div class="profile-description">
Thank you for exploring this project! I enjoy building intelligent machine learning
solutions that transform real-world data into actionable insights. My interests include
Machine Learning, Deep Learning, Computer Vision, Data Analytics, and AI-powered Web Applications.
</div>

<div class="stats">

<div class="stat-card">
<div class="stat-number">10+</div>
<div class="stat-text">ML Projects</div>
</div>

<div class="stat-card">
<div class="stat-number">Python</div>
<div class="stat-text">Primary Language</div>
</div>

<div class="stat-card">
<div class="stat-number">AI</div>
<div class="stat-text">Machine Learning & Deep Learning</div>
</div>

</div>

<div class="buttons">

<a class="btn github"
href="https://github.com/Abhay-cody"
target="_blank">
🐙 View GitHub
</a>

<a class="btn linkedin"
href="https://www.linkedin.com/in/abhay-kumar-gupta-104a18397"
target="_blank">
💼 Connect on LinkedIn
</a>

</div>

</div>
""", unsafe_allow_html=True)
