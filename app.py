import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Deep Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/multipleChoiceResponses.csv",
        low_memory=False
    )

    freeform = pd.read_csv(
        "data/freeformResponses.csv",
        low_memory=False
    )

    schema = pd.read_csv(
        "data/schema.csv",
        encoding="latin1"
    )

    return df, freeform, schema


df, freeform, schema = load_data()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("📊 Deep Analytics Survey Dashboard")

st.markdown("""
### Interactive Analytics using Streamlit + Plotly

This dashboard provides:
- KPI Analytics
- Country Filtering
- Employment Insights
- Age Analysis
- Programming Language Trends
- WordCloud Insights
- Raw Dataset Explorer
""")

st.divider()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("🎯 Filters")

if "Country" in df.columns:
    country_list = sorted(df["Country"].dropna().unique())

    selected_country = st.sidebar.selectbox(
        "Select Country",
        ["All"] + list(country_list)
    )

    if selected_country == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["Country"] == selected_country]

else:
    filtered_df = df.copy()

# ---------------------------------------------------
# KPI SECTION
# ---------------------------------------------------
st.subheader("📌 Dashboard Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Responses",
        f"{len(filtered_df):,}"
    )

with col2:
    if "Country" in filtered_df.columns:
        st.metric(
            "Countries",
            filtered_df["Country"].nunique()
        )

with col3:
    st.metric(
        "Columns",
        len(filtered_df.columns)
    )

with col4:
    missing_values = filtered_df.isnull().sum().sum()

    st.metric(
        "Missing Values",
        f"{missing_values:,}"
    )

st.divider()

# ---------------------------------------------------
# EMPLOYMENT STATUS ANALYSIS
# ---------------------------------------------------
if "EmploymentStatus" in filtered_df.columns:

    st.subheader("💼 Employment Status Analysis")

    employment_data = (
        filtered_df["EmploymentStatus"]
        .value_counts()
        .reset_index()
    )

    employment_data.columns = ["Employment Status", "Count"]

    fig = px.bar(
        employment_data,
        x="Employment Status",
        y="Count",
        title="Employment Distribution",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# AGE DISTRIBUTION
# ---------------------------------------------------
if "Age" in filtered_df.columns:

    st.subheader("🎂 Age Distribution")

    age_df = filtered_df["Age"].dropna()

    fig2 = px.histogram(
        age_df,
        nbins=25,
        title="Age Distribution Analysis"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ---------------------------------------------------
# GENDER ANALYSIS
# ---------------------------------------------------
if "GenderSelect" in filtered_df.columns:

    st.subheader("🧑 Gender Analysis")

    gender_df = (
        filtered_df["GenderSelect"]
        .value_counts()
        .reset_index()
    )

    gender_df.columns = ["Gender", "Count"]

    fig3 = px.pie(
        gender_df,
        names="Gender",
        values="Count",
        title="Gender Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# ---------------------------------------------------
# PROGRAMMING LANGUAGE ANALYSIS
# ---------------------------------------------------
language_columns = [
    col for col in filtered_df.columns
    if "Language" in col
][:10]

if language_columns:

    st.subheader("💻 Programming Language Trends")

    language_data = []

    for col in language_columns:

        values = (
            filtered_df[col]
            .dropna()
            .value_counts()
        )

        for language, count in values.items():
            language_data.append(
                [language, count]
            )

    language_df = pd.DataFrame(
        language_data,
        columns=["Language", "Count"]
    )

    language_df = (
        language_df
        .groupby("Language")
        .sum()
        .reset_index()
        .sort_values(
            by="Count",
            ascending=False
        )
        .head(10)
    )

    fig4 = px.bar(
        language_df,
        x="Language",
        y="Count",
        title="Top Programming Languages",
        text_auto=True
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# ---------------------------------------------------
# SALARY ANALYSIS
# ---------------------------------------------------
salary_columns = [
    col for col in filtered_df.columns
    if "Salary" in col
]

if salary_columns:

    st.subheader("💰 Salary Related Columns")

    st.write(salary_columns)

# ---------------------------------------------------
# WORD CLOUD
# ---------------------------------------------------
st.subheader("☁️ Freeform Insights WordCloud")

try:

    text_data = " ".join(
        freeform
        .astype(str)
        .fillna("")
        .values
        .flatten()
        .tolist()
    )

    wordcloud = WordCloud(
        width=1400,
        height=600,
        background_color="white"
    ).generate(text_data)

    fig5, ax = plt.subplots(figsize=(15, 6))

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    st.pyplot(fig5)

except Exception as e:
    st.error(f"WordCloud Error: {e}")

# ---------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------
st.subheader("📈 Correlation Heatmap")

numeric_df = filtered_df.select_dtypes(
    include=["number"]
)

if not numeric_df.empty:

    correlation = numeric_df.corr()

    fig6 = px.imshow(
        correlation,
        text_auto=False,
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

# ---------------------------------------------------
# MISSING VALUES ANALYSIS
# ---------------------------------------------------
st.subheader("🧹 Missing Values Analysis")

missing_df = (
    filtered_df
    .isnull()
    .sum()
    .reset_index()
)

missing_df.columns = ["Column", "Missing Values"]

missing_df = missing_df.sort_values(
    by="Missing Values",
    ascending=False
).head(15)

fig7 = px.bar(
    missing_df,
    x="Column",
    y="Missing Values",
    title="Top Missing Value Columns"
)

st.plotly_chart(
    fig7,
    use_container_width=True
)

# ---------------------------------------------------
# KEY INSIGHTS
# ---------------------------------------------------
st.subheader("🔍 Deep Insights")

insights = []

insights.append(
    f"Dataset contains {len(filtered_df):,} responses."
)

if "Country" in filtered_df.columns:
    insights.append(
        f"Data collected from {filtered_df['Country'].nunique()} countries."
    )

insights.append(
    "Programming language adoption trends are visible."
)

insights.append(
    "Employment distribution helps identify workforce patterns."
)

insights.append(
    "Age analytics provides demographic understanding."
)

for insight in insights:
    st.success(insight)

# ---------------------------------------------------
# RAW DATA EXPLORER
# ---------------------------------------------------
with st.expander("📂 View Raw Dataset"):

    st.dataframe(
        filtered_df.head(100),
        use_container_width=True
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.divider()

st.markdown("""
### 🚀 Built With
- Streamlit
- Plotly
- Pandas
- WordCloud
- Matplotlib

Made for Deep Analytics & Interactive Data Exploration
""")
