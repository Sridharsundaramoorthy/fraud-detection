
import streamlit as st
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import get_db_manager

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

# Check authentication
if not st.session_state.get('authenticated'):
    st.error("⛔ Please login first")
    st.stop()

# Check role
if st.session_state.user_role not in ['manager', 'admin']:
    st.error("⛔ Access denied. This page is only for managers and admins.")
    st.stop()

st.title("📊 Fraud Detection Analytics Dashboard")

# Initialize DB
db = get_db_manager()

# Time period selector
time_period = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=2, format_func=lambda x: f"Last {x} days")

# Get statistics
stats = db.get_fraud_statistics(days=time_period)

# KPI Metrics
st.markdown("### 📈 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Returns", stats['total_returns'])
with col2:
    st.metric("Approved", stats['approved'], delta=f"{stats['approved'] / max(stats['total_returns'], 1) * 100:.1f}%")
with col3:
    st.metric("Rejected (Fraud)", stats['rejected'], delta=f"{stats['fraud_detection_rate']:.1f}%", delta_color="inverse")
with col4:
    st.metric("Manual Review", stats['manual_review'])
with col5:
    st.metric("Avg Fraud Score", f"{stats['avg_fraud_score']:.1f}%")

st.markdown("---")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Decision Distribution")
    decision_data = pd.DataFrame({
        'Decision': ['Approved', 'Rejected', 'Manual Review'],
        'Count': [stats['approved'], stats['rejected'], stats['manual_review']]
    })
    fig = px.pie(decision_data, values='Count', names='Decision', 
                 color='Decision',
                 color_discrete_map={'Approved':'green', 'Rejected':'red', 'Manual Review':'orange'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📈 Fraud Detection Trend")
    trend_data = db.get_daily_fraud_trend(days=time_period)
    if trend_data:
        df_trend = pd.DataFrame(trend_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_trend['_id'], y=df_trend['total_returns'], 
                                name='Total Returns', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=df_trend['_id'], y=df_trend['fraud_cases'], 
                                name='Fraud Cases', mode='lines+markers', 
                                line=dict(color='red')))
        fig.update_layout(xaxis_title="Date", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

# Top returned products
st.markdown("### 🔝 Most Returned Products")
top_products = db.get_top_returned_products(limit=10)
if top_products:
    df_products = pd.DataFrame(top_products)
    df_products.columns = ['Product ID', 'Return Count', 'Avg Fraud Score']
    df_products['Avg Fraud Score'] = df_products['Avg Fraud Score'].round(2)
    st.dataframe(df_products, use_container_width=True)

# High-risk customers
st.markdown("### ⚠️ High-Risk Customers")
high_risk = db.get_high_risk_customers(limit=10)
if high_risk:
    df_risk = pd.DataFrame([{
        'Customer ID': c['customer_id'],
        'Risk Score': f"{c['risk_score'] * 100:.1f}%",
        'Total Returns': c.get('total_returns', 0),
        'Fraud Cases': c.get('fraud_cases', 0),
        'Status': c['status'].replace('_', ' ').title()
    } for c in high_risk])
    st.dataframe(df_risk, use_container_width=True)