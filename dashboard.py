import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/cleaned_data.csv")

st.title("Customer Behaviour Analysis Dashboard")

# ---- KPI SECTION ----
st.subheader("Customer Overview")

total_customers = len(df)
churned = df["Churn"].sum()
churn_rate = (churned / total_customers) * 100
avg_monthly = df["MonthlyCharges"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned)
col3.metric("Churn Rate (%)", f"{churn_rate:.2f}")
col4.metric("Avg Monthly Charges", f"{avg_monthly:.2f}")

# ---- TENURE VS CHURN ----
st.subheader("Tenure vs Churn Behaviour")
fig, ax = plt.subplots()
sns.boxplot(x="Churn", y="tenure", data=df, ax=ax)
st.pyplot(fig)

# ---- MONTHLY CHARGES VS CHURN ----
st.subheader("Monthly Charges vs Churn")
fig, ax = plt.subplots()
sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax)
st.pyplot(fig)

# ---- CONTRACT TYPE ----
st.subheader("Contract Type vs Churn")
contract_churn = df.groupby("Contract")["Churn"].mean()

fig, ax = plt.subplots()
contract_churn.plot(kind="bar", ax=ax)
ax.set_ylabel("Churn Rate")
st.pyplot(fig)

st.markdown("""
### Behavioural Insights
- Customers with **low tenure** show higher churn probability.
- Customers with **higher monthly charges** are more price-sensitive.
- **Month-to-month contracts** show higher churn compared to long-term contracts.
""")

