import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Student Behaviour Dashboard", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Student Behaviour Analysis Dashboard")
st.markdown("### Analyzing Behavioural Patterns and Their Impact on Students")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("student_lifestyle_performance_dataset.csv")

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")

branch_filter = st.sidebar.multiselect(
    "Select Branch",
    options=df['Branch'].unique(),
    default=df['Branch'].unique()
)

residence_filter = st.sidebar.multiselect(
    "Select Residence",
    options=df['Residence'].unique(),
    default=df['Residence'].unique()
)

# Apply filters
filtered_df = df[
    (df['Branch'].isin(branch_filter)) &
    (df['Residence'].isin(residence_filter))
]

# -----------------------------
# KPI CARDS
# -----------------------------
avg_cgpa = round(filtered_df['CGPA'].mean(), 2)
avg_study = round(filtered_df['Study_Hours_per_Day'].mean(), 2)
avg_sleep = round(filtered_df['Sleep_Hours'].mean(), 2)
avg_attendance = round(filtered_df['Attendance_Percentage'].mean(), 2)
avg_stress = round(filtered_df['Stress_Level_1_to_10'].mean(), 2)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Average CGPA", avg_cgpa)
col2.metric("Study Hours", avg_study)
col3.metric("Sleep Hours", avg_sleep)
col4.metric("Attendance %", avg_attendance)
col5.metric("Stress Level", avg_stress)

st.markdown("---")

# -----------------------------
# CHART 1 - STUDY HOURS VS CGPA
# -----------------------------
st.subheader("Study Hours vs CGPA")

fig1 = px.scatter(
    filtered_df,
    x='Study_Hours_per_Day',
    y='CGPA',
    color='Branch',
    size='Attendance_Percentage',
    hover_data=['Residence'],
    title='Relationship Between Study Hours and CGPA'
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# CHART 2 - SCREEN TIME VS CGPA
# -----------------------------
st.subheader("Screen Time vs CGPA")

fig2 = px.scatter(
    filtered_df,
    x='Screen_Time_Hours',
    y='CGPA',
    color='Stress_Level_1_to_10',
    title='Impact of Screen Time on CGPA'
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# CHART 3 - BRANCH WISE CGPA
# -----------------------------
st.subheader("Branch-wise Average CGPA")

branch_avg = filtered_df.groupby('Branch')['CGPA'].mean().reset_index()

fig3 = px.bar(
    branch_avg,
    x='Branch',
    y='CGPA',
    text_auto=True,
    title='Average CGPA by Branch'
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# CHART 4 - ATTENDANCE VS INTERNAL MARKS
# -----------------------------
st.subheader("Attendance vs Internal Marks")

fig4 = px.scatter(
    filtered_df,
    x='Attendance_Percentage',
    y='Internal_Marks',
    color='Residence',
    title='Attendance Impact on Internal Marks'
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# CHART 5 - DIET TYPE DISTRIBUTION
# -----------------------------
st.subheader("Diet Type Distribution")

fig5 = px.pie(
    filtered_df,
    names='Diet_Type',
    title='Veg vs Non-Veg Students'
)

st.plotly_chart(fig5, use_container_width=True)

# -----------------------------
# CHART 6 - STRESS LEVEL DISTRIBUTION
# -----------------------------
st.subheader("Stress Level Distribution")

fig6 = px.histogram(
    filtered_df,
    x='Stress_Level_1_to_10',
    nbins=10,
    title='Stress Level Distribution'
)

st.plotly_chart(fig6, use_container_width=True)

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
st.subheader("Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])

corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

st.pyplot(fig)

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(filtered_df)

# -----------------------------
# KEY INSIGHTS
# -----------------------------
st.subheader("Key Insights")

st.markdown("""
- Students with higher study hours generally achieve better CGPA.
- High attendance positively impacts internal marks.
- Excessive screen time may reduce academic performance.
- Proper sleep helps in reducing stress levels.
- Balanced lifestyle habits improve student performance.
""")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("### DADV Mini Project")
st.markdown("Created using Python, Streamlit, Plotly, Pandas and Seaborn")


# -----------------------------
# HOW TO RUN THE PROJECT
# -----------------------------
'''
1. Save this file as app.py
2. Keep dataset file in the same folder
3. Open terminal
4. Install required libraries:

pip install streamlit pandas plotly seaborn matplotlib

5. Run the dashboard:

streamlit run app.py
'''
