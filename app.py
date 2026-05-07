# ======================================================
# DADV MINI PROJECT
# Analyzing Behavioural Patterns and Their Impact on Students
# ======================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
# ======================================================
# PAGE CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="Student Behaviour Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS (DARK THEME)
# ======================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# SIDEBAR NAVIGATION
# ======================================================
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "Dashboard",
        "Behaviour Analytics",
        "Machine Learning Prediction"
    ]
)

# ======================================================
# LOAD DATASET
# ======================================================
df = pd.read_csv("student_lifestyle_performance_dataset.csv")

# ======================================================
# MODERN HERO SECTION
# ======================================================
hero_html = """
<div style='padding:35px;border-radius:24px;background:linear-gradient(135deg,#111827,#1f2937,#312e81);margin-bottom:25px;box-shadow:0 8px 24px rgba(0,0,0,0.35);'>
<h1 style='color:white;font-size:42px;font-weight:700;'>📊 Analyzing Behavioural Patterns and Their Impact on Students</h1>
<p style='color:#d1d5db;font-size:18px;line-height:1.7;'>Interactive analytics dashboard for understanding how behavioural and lifestyle attributes influence student academic performance. Built using all 12 dataset attributes with modern frontend design and visual storytelling.</p>
</div>
"""

st.markdown(hero_html, unsafe_allow_html=True)

# ======================================================
# DASHBOARD PAGE
# ======================================================
if page == "Dashboard":

    st.title("📊 Student Behaviour Analysis Dashboard")
    st.markdown("### Analyzing Behavioural Patterns and Their Impact on Students")

    # --------------------------------------------------
    # ADVANCED FILTERS
    # --------------------------------------------------
    st.sidebar.header("Filters")

    branch_filter = st.sidebar.multiselect(
        "Select Branch",
        df['Branch'].unique(),
        default=df['Branch'].unique()
    )

    residence_filter = st.sidebar.multiselect(
        "Select Residence",
        df['Residence'].unique(),
        default=df['Residence'].unique()
    )

    diet_filter = st.sidebar.multiselect(
        "Select Diet Type",
        df['Diet_Type'].unique(),
        default=df['Diet_Type'].unique()
    )

    age_filter = st.sidebar.slider(
        "Select Age Range",
        int(df['Age'].min()),
        int(df['Age'].max()),
        (int(df['Age'].min()), int(df['Age'].max()))
    )

    filtered_df = df[
        (df['Branch'].isin(branch_filter)) &
        (df['Residence'].isin(residence_filter)) &
        (df['Diet_Type'].isin(diet_filter)) &
        (df['Age'] >= age_filter[0]) &
        (df['Age'] <= age_filter[1])
    ]

    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------
    avg_cgpa = round(filtered_df['CGPA'].mean(), 2)
    avg_study = round(filtered_df['Study_Hours_per_Day'].mean(), 2)
    avg_sleep = round(filtered_df['Sleep_Hours'].mean(), 2)
    avg_attendance = round(filtered_df['Attendance_Percentage'].mean(), 2)
    avg_stress = round(filtered_df['Stress_Level_1_to_10'].mean(), 2)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("CGPA", avg_cgpa)
    col2.metric("Study Hours", avg_study)
    col3.metric("Sleep Hours", avg_sleep)
    col4.metric("Attendance", avg_attendance)
    col5.metric("Stress", avg_stress)

    st.markdown("---")

    # --------------------------------------------------
    # CHARTS
    # --------------------------------------------------

    # Study Hours vs CGPA
    fig1 = px.scatter(
        filtered_df,
        x='Study_Hours_per_Day',
        y='CGPA',
        color='Branch',
        size='Attendance_Percentage',
        title='Study Hours vs CGPA'
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Screen Time vs CGPA
    fig2 = px.scatter(
        filtered_df,
        x='Screen_Time_Hours',
        y='CGPA',
        color='Stress_Level_1_to_10',
        title='Screen Time vs CGPA'
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Branch-wise CGPA
    branch_avg = filtered_df.groupby('Branch')['CGPA'].mean().reset_index()

    fig3 = px.bar(
        branch_avg,
        x='Branch',
        y='CGPA',
        text_auto=True,
        title='Branch-wise Average CGPA'
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Attendance vs Internal Marks
    fig_att = px.scatter(
        filtered_df,
        x='Attendance_Percentage',
        y='Internal_Marks',
        color='Branch',
        title='Attendance vs Internal Marks'
    )

    st.plotly_chart(fig_att, use_container_width=True)

    # Sleep Hours vs Stress Level
    fig_sleep = px.scatter(
        filtered_df,
        x='Sleep_Hours',
        y='Stress_Level_1_to_10',
        color='Residence',
        title='Sleep Hours vs Stress Level'
    )

    st.plotly_chart(fig_sleep, use_container_width=True)

    # Gym Hours Analysis
    fig_gym = px.box(
        filtered_df,
        x='Branch',
        y='Gym_Hours_per_Week',
        color='Residence',
        title='Gym Hours Per Week Analysis'
    )

    st.plotly_chart(fig_gym, use_container_width=True)

    # Age Distribution
    fig_age = px.histogram(
        filtered_df,
        x='Age',
        nbins=15,
        title='Age Distribution of Students'
    )

    st.plotly_chart(fig_age, use_container_width=True)

    # Internal Marks Distribution
    fig_internal = px.histogram(
        filtered_df,
        x='Internal_Marks',
        nbins=20,
        color='Branch',
        title='Internal Marks Distribution'
    )

    st.plotly_chart(fig_internal, use_container_width=True)

    # Residence Analysis
    residence_chart = px.sunburst(
        filtered_df,
        path=['Residence', 'Diet_Type', 'Branch'],
        values='CGPA',
        title='Residence and Diet Analysis'
    )

    st.plotly_chart(residence_chart, use_container_width=True)

    # Multi Attribute Bubble Chart
    bubble_chart = px.scatter(
        filtered_df,
        x='Study_Hours_per_Day',
        y='Internal_Marks',
        size='CGPA',
        color='Stress_Level_1_to_10',
        hover_name='Branch',
        title='Multi-Attribute Student Performance Analysis'
    )

    st.plotly_chart(bubble_chart, use_container_width=True)

    # Pie Chart
    fig4 = px.pie(
        filtered_df,
        names='Diet_Type',
        title='Diet Type Distribution'
    )

    st.plotly_chart(fig4, use_container_width=True)

    # Histogram
    fig5 = px.histogram(
        filtered_df,
        x='Stress_Level_1_to_10',
        nbins=10,
        title='Stress Level Distribution'
    )

    st.plotly_chart(fig5, use_container_width=True)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

    st.pyplot(fig)

# ======================================================
# BEHAVIOUR ANALYTICS PAGE
# ======================================================
elif page == "Behaviour Analytics":

    st.title("📈 Behaviour Analytics & Dataset Insights")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Data Types")
    st.write(df.dtypes)

# ======================================================
# MACHINE LEARNING PAGE
# ======================================================
elif page == "Machine Learning Prediction":

    st.title("🤖 CGPA Prediction Model")

    features = [
        'Age',
        'Study_Hours_per_Day',
        'Sleep_Hours',
        'Screen_Time_Hours',
        'Gym_Hours_per_Week',
        'Attendance_Percentage',
        'Stress_Level_1_to_10',
        'Internal_Marks'
    ]

    X = df[features]
    y = df['CGPA']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = r2_score(y_test, predictions)

    st.success(f"Model Accuracy: {round(accuracy * 100, 2)}%")

    st.subheader("Enter Student Details")

    age = st.slider("Age", 15, 30, 20)
    study = st.slider("Study Hours", 1, 15, 5)
    sleep = st.slider("Sleep Hours", 1, 12, 7)
    screen = st.slider("Screen Time", 1, 15, 5)
    gym = st.slider("Gym Hours Per Week", 0, 20, 3)
    attendance = st.slider("Attendance %", 50, 100, 80)
    stress = st.slider("Stress Level", 1, 10, 5)
    internal = st.slider("Internal Marks", 0, 100, 70)

    if st.button("Predict CGPA"):

        input_data = pd.DataFrame([
            [age, study, sleep, screen, gym, attendance, stress, internal]
        ], columns=features)

        predicted_cgpa = model.predict(input_data)[0]

        st.success(f"Predicted CGPA: {round(predicted_cgpa, 2)}")



# ======================================================
# FOOTER
# ======================================================
st.markdown("---")
st.markdown("### Student Behaviour Analysis Dashboard")
