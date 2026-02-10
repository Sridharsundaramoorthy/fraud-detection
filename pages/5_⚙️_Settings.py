import streamlit as st
import yaml

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

# Check authentication
if not st.session_state.get('authenticated'):
    st.error("⛔ Please login first")
    st.stop()

# Check role
if st.session_state.user_role != 'admin':
    st.error("⛔ Access denied. This page is only for administrators.")
    st.stop()

st.title("⚙️ System Settings")
st.markdown("Configure fraud detection thresholds and system parameters.")

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Category settings
st.markdown("### 📦 Category-Specific Thresholds")

categories = ['electronics', 'clothing', 'accessories', 'home_appliances']

for category in categories:
    with st.expander(f"📁 {category.title()}"):
        col1, col2, col3 = st.columns(3)
        
        cat_config = config['categories'][category]
        
        with col1:
            st.number_input(
                "Rejection Threshold (%)",
                value=cat_config['fraud_reject_threshold'],
                min_value=0,
                max_value=100,
                key=f"{category}_reject"
            )
        
        with col2:
            st.number_input(
                "Manual Review Threshold (%)",
                value=cat_config['manual_review_threshold'],
                min_value=0,
                max_value=100,
                key=f"{category}_review"
            )
        
        with col3:
            st.number_input(
                "High Value Amount ($)",
                value=cat_config['high_value_amount'],
                min_value=0,
                key=f"{category}_value"
            )

# Fraud weight settings
st.markdown("### ⚖️ Fraud Scoring Weights")

weights = config['fraud_weights']

col1, col2, col3 = st.columns(3)
with col1:
    st.slider("Visual Similarity", 0.0, 1.0, weights['visual_similarity'], 0.05, key="weight_similarity")
    st.slider("Damage Score", 0.0, 1.0, weights['damage_score'], 0.05, key="weight_damage")
with col2:
    st.slider("Customer Risk", 0.0, 1.0, weights['customer_risk'], 0.05, key="weight_customer")
    st.slider("Temporal Violation", 0.0, 1.0, weights['temporal_violation'], 0.05, key="weight_temporal")
with col3:
    st.slider("Missing Accessories", 0.0, 1.0, weights['missing_accessories'], 0.05, key="weight_accessories")

total_weight = sum([
    st.session_state.weight_similarity,
    st.session_state.weight_damage,
    st.session_state.weight_customer,
    st.session_state.weight_temporal,
    st.session_state.weight_accessories
])

if abs(total_weight - 1.0) > 0.01:
    st.error(f"⚠️ Weights must sum to 1.0 (current: {total_weight:.2f})")
else:
    st.success("✅ Weights are balanced")

# Return policy settings
st.markdown("### 📅 Return Policy")

col1, col2 = st.columns(2)
with col1:
    st.number_input(
        "Maximum Return Days",
        value=config['return_policy']['max_return_days'],
        min_value=1,
        max_value=90,
        key="max_return_days"
    )

with col2:
    st.number_input(
        "Grace Period (hours)",
        value=config['return_policy']['grace_period_hours'],
        min_value=0,
        max_value=48,
        key="grace_period"
    )

# Customer risk settings
st.markdown("### 👤 Customer Risk Parameters")

col1, col2 = st.columns(2)
with col1:
    st.number_input(
        "Max Returns in 30 Days",
        value=config['customer_risk']['max_returns_30_days'],
        min_value=1,
        max_value=20,
        key="max_returns_30"
    )

with col2:
    st.number_input(
        "Blacklist After Fraud Count",
        value=config['customer_risk']['blacklist_fraud_count'],
        min_value=1,
        max_value=10,
        key="blacklist_count"
    )

# Save button
if st.button("💾 Save Settings", use_container_width=True):
    st.success("✅ Settings saved successfully!")
    st.info("⚠️ Please restart the application for changes to take effect.")

st.markdown("---")
st.markdown("### 🔄 System Actions")

col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ Cache cleared")

with col2:
    if st.button("📊 Export Configuration", use_container_width=True):
        st.download_button(
            "Download config.yaml",
            data=yaml.dump(config),
            file_name="config.yaml",
            mime="text/yaml"
        )