import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# BACKGROUND & STYLING
# =====================================================

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#dbeafe,#eff6ff);
}

.title{
text-align:center;
font-size:45px;
font-weight:bold;
color:#1e3a8a;
}

.subtitle{
text-align:center;
font-size:18px;
color:#374151;
margin-bottom:25px;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 4px 10px rgba(0,0,0,0.1);
}

.result-pass{
background:#16a34a;
padding:25px;
border-radius:15px;
color:white;
text-align:center;
font-size:35px;
font-weight:bold;
}

.result-fail{
background:#dc2626;
padding:25px;
border-radius:15px;
color:white;
text-align:center;
font-size:35px;
font-weight:bold;
}

.stButton button{
width:100%;
background:#2563eb;
color:white;
font-size:20px;
border-radius:10px;
height:55px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown(
    "<div class='title'>🎓 Student Performance Prediction Dashboard</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Machine Learning Based Pass/Fail Prediction System</div>",
    unsafe_allow_html=True
)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("student_performance.csv")

# Create Target Variable

df["Result"] = np.where(df["G3"] >= 10, 1, 0)

# Remove grade columns

df = df.drop(["G1", "G2", "G3"], axis=1)

# Encode categorical columns

encoders = {}

for col in df.columns:

    if df[col].dtype == "object":

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

# =====================================================
# MODEL TRAINING
# =====================================================

X = df.drop("Result", axis=1)

y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = accuracy_score(
    y_test,
    model.predict(X_test)
)

# =====================================================
# SIDEBAR INPUTS
# =====================================================

st.sidebar.title("📚 Student Details")

age = st.sidebar.slider(
    "Age",
    15,
    22,
    17
)

studytime = st.sidebar.slider(
    "Study Time",
    1,
    4,
    2
)

failures = st.sidebar.slider(
    "Past Failures",
    0,
    4,
    0
)

absences = st.sidebar.slider(
    "Absences",
    0,
    50,
    5
)

Medu = st.sidebar.slider(
    "Mother Education",
    0,
    4,
    2
)

Fedu = st.sidebar.slider(
    "Father Education",
    0,
    4,
    2
)

health = st.sidebar.slider(
    "Health Status",
    1,
    5,
    3
)

freetime = st.sidebar.slider(
    "Free Time",
    1,
    5,
    3
)

goout = st.sidebar.slider(
    "Going Out Frequency",
    1,
    5,
    3
)

predict = st.sidebar.button("🎯 Predict Result")

# =====================================================
# DASHBOARD INFO CARDS
# =====================================================

c1,c2,c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class='card'>
    <h3>📖 Study Time</h3>
    More study time generally improves academic performance.
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class='card'>
    <h3>📅 Attendance</h3>
    Lower absences often lead to better results.
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class='card'>
    <h3>🏫 Academic History</h3>
    Previous failures can significantly affect future outcomes.
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================
# PREDICTION
# =====================================================

if predict:

    sample = np.zeros(len(X.columns))

    feature_names = list(X.columns)

    def fill(name, value):
        if name in feature_names:
            sample[feature_names.index(name)] = value

    fill("age", age)
    fill("studytime", studytime)
    fill("failures", failures)
    fill("absences", absences)
    fill("Medu", Medu)
    fill("Fedu", Fedu)
    fill("health", health)
    fill("freetime", freetime)
    fill("goout", goout)

    prediction = model.predict([sample])[0]

    if prediction == 1:

        st.markdown(
        """
        <div class='result-pass'>
        ✅ PASS
        </div>
        """,
        unsafe_allow_html=True
        )

    else:

        st.markdown(
        """
        <div class='result-fail'>
        ❌ FAIL
        </div>
        """,
        unsafe_allow_html=True
        )

    # Student Profile Chart

    st.subheader("📊 Student Profile")

    profile = pd.DataFrame({

        "Feature":[
            "Study Time",
            "Absences",
            "Health",
            "Free Time",
            "Go Out"
        ],

        "Value":[
            studytime,
            absences,
            health,
            freetime,
            goout
        ]
    })

    st.bar_chart(
        profile.set_index("Feature")
    )

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.subheader("📈 Factors Affecting Student Performance")

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.set_title(
    "Feature Importance"
)

st.pyplot(fig)

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.subheader("🎯 Model Accuracy")

st.metric(
    "Accuracy",
    f"{accuracy*100:.2f}%"
)

# =====================================================
# DATA PREVIEW
# =====================================================

st.subheader("📄 Dataset Preview")

st.dataframe(df.head())