"""
Main Streamlit Application for Fraud Detection System
Entry point with authentication and navigation
"""

import streamlit as st
import os
import sys
from datetime import datetime
import bcrypt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None

# Demo users (In production, this would be from database)
DEMO_USERS = {
    'delivery': {
        'password': bcrypt.hashpw('delivery123'.encode(), bcrypt.gensalt()),
        'role': 'delivery_person',
        'name': 'John Delivery'
    },
    'warehouse': {
        'password': bcrypt.hashpw('warehouse123'.encode(), bcrypt.gensalt()),
        'role': 'warehouse_staff',
        'name': 'Sarah Warehouse'
    },
    'manager': {
        'password': bcrypt.hashpw('manager123'.encode(), bcrypt.gensalt()),
        'role': 'manager',
        'name': 'Mike Manager'
    },
    'admin': {
        'password': bcrypt.hashpw('admin123'.encode(), bcrypt.gensalt()),
        'role': 'admin',
        'name': 'Admin User'
    }
}

def authenticate_user(username: str, password: str) -> tuple:
    """Authenticate user and return role"""
    if username in DEMO_USERS:
        user_data = DEMO_USERS[username]
        if bcrypt.checkpw(password.encode(), user_data['password']):
            return True, user_data['role'], user_data['name']
    return False, None, None

def login_page():
    """Display login page"""
    st.markdown('<div class="main-header">🔍 Product Return Fraud Detection System</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                success, role, name = authenticate_user(username, password)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.session_state.username = name
                    st.success(f"✅ Welcome, {name}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        
        # Demo credentials
        st.markdown("---")
        st.markdown("### 🧪 Demo Credentials")
        
        demo_creds = {
            "Delivery Person": "delivery / delivery123",
            "Warehouse Staff": "warehouse / warehouse123",
            "Manager": "manager / manager123",
            "Admin": "admin / admin123"
        }
        
        for role, creds in demo_creds.items():
            st.markdown(f"**{role}:** `{creds}`")

def main_page():
    """Main application page after login"""
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")
        st.markdown(f"**Role:** {st.session_state.user_role.replace('_', ' ').title()}")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()
    
    # Main content
    st.markdown('<div class="main-header">🔍 Fraud Detection System</div>', unsafe_allow_html=True)
    
    # Dashboard metrics
    st.markdown("### 📊 Quick Stats")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Today's Returns", "24", "+3")
    with col2:
        st.metric("Fraud Detected", "5", "+2")
    with col3:
        st.metric("Pending Reviews", "8", "-1")
    with col4:
        st.metric("Detection Rate", "20.8%", "+1.2%")
    
    st.markdown("---")
    
    # Role-based navigation
    st.markdown("### 🧭 Navigation")
    
    role = st.session_state.user_role
    
    if role in ['delivery_person', 'admin']:
        st.info("📦 **Delivery Upload**: Upload product images during delivery")
    
    if role in ['warehouse_staff', 'manager', 'admin']:
        st.info("📥 **Return Processing**: Process return requests and detect fraud")
    
    if role in ['manager', 'admin']:
        st.info("🔍 **Manual Review**: Review flagged returns")
        st.info("📊 **Analytics Dashboard**: View fraud statistics and trends")
    
    if role == 'admin':
        st.info("⚙️ **Settings**: Configure system parameters")
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    
    if role == 'delivery_person':
        st.markdown("""
        **Delivery Process:**
        1. Go to **🚚 Delivery Upload** page
        2. Enter Order ID and Product details
        3. Upload 3-5 clear images from different angles
        4. Ensure good lighting and focus
        5. Submit to save delivery record
        """)
    
    elif role == 'warehouse_staff':
        st.markdown("""
        **Return Process:**
        1. Go to **📦 Return Processing** page
        2. Enter Order ID of returned product
        3. Upload images of returned item
        4. System will automatically detect fraud
        5. View results and take action
        """)
    
    elif role in ['manager', 'admin']:
        st.markdown("""
        **Manager Functions:**
        1. **Manual Review**: Review cases flagged by AI
        2. **Analytics**: Monitor fraud trends
        3. **Override Decisions**: Approve/reject returns manually
        4. **Customer Insights**: View high-risk customers
        """)

# Main app logic
if not st.session_state.authenticated:
    login_page()
else:
    main_page()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Agentic AI Fraud Detection System v1.0 | Powered by CLIP, YOLOv8, and Grok</div>",
    unsafe_allow_html=True
)