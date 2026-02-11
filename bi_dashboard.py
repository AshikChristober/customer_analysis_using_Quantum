import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Behaviour BI Dashboard", layout="wide")

# Load data
df = pd.read_csv("../data/cleaned_data.csv")

# Tabs = BI Dashboards
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview",
    "👥 Customer Segmentation",
    "📉 Churn Behaviour",
    "💰 Revenue & Loyalty",
    "🚨 Risk & Action"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.header("Executive Overview Dashboard")

    total_customers = len(df)
    churned = df["Churn"].sum()
    churn_rate = churned / total_customers * 100
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Customers", total_customers)
    col2.metric("Churned Customers", churned)
    col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")
    col4.metric("Avg Monthly Charges", f"{avg_monthly:.2f}")
    col5.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")

    st.info("This dashboard provides a high-level summary of customer behaviour and churn.")

# ---------------- TAB 2 ----------------
with tab2:
    st.header("Customer Segmentation Dashboard")

    df["TenureSegment"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 100],
        labels=["New", "Mid-term", "Long-term", "Loyal"]
    )

    fig, ax = plt.subplots()
    sns.countplot(x="TenureSegment", data=df, ax=ax)
    ax.set_title("Customer Distribution by Loyalty Segment")
    st.pyplot(fig)

    st.markdown("""
    **Insight:**
    - Majority of churn originates from **new and mid-term customers**
    - Loyal customers form a stable base
    """)

# ---------------- TAB 3 ----------------
with tab3:
    st.header("Churn Behaviour Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(x="Churn", y="tenure", data=df, ax=ax)
        ax.set_title("Tenure vs Churn")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
        ax.set_title("Monthly Charges vs Churn")
        st.pyplot(fig)

    st.markdown("""
    **Insights:**
    - Low tenure customers churn more frequently
    - High monthly charges increase churn probability
    """)

# ---------------- TAB 4 ----------------
with tab4:
    st.header("Revenue & Loyalty Dashboard")

    revenue_loyalty = df.groupby("TenureSegment")[["TotalCharges", "MonthlyCharges"]].mean()

    fig, ax = plt.subplots()
    revenue_loyalty["TotalCharges"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Average Total Charges")
    ax.set_title("Lifetime Revenue by Loyalty Segment")
    st.pyplot(fig)

    st.markdown("""
    **Insight:**
    - Loyal customers generate significantly higher lifetime value
    - Retention directly impacts long-term revenue
    """)

# ---------------- TAB 5 ----------------
with tab5:
    st.header("High-Risk Customer Action Dashboard")

    high_risk = df[
        (df["tenure"] < 12) &
        (df["MonthlyCharges"] > df["MonthlyCharges"].quantile(0.75))
    ]

    st.metric("High-Risk Customers", len(high_risk))
    st.dataframe(high_risk.head(20))

    st.markdown("""
    **Actionable Insights:**
    - These customers should be targeted with:
      - Discounts
      - Contract upgrades
      - Personalized offers
    """)

