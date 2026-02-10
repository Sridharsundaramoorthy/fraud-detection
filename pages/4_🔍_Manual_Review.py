import streamlit as st
import sys
import os
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import get_db_manager

st.set_page_config(page_title="Manual Review", page_icon="🔍", layout="wide")

# Check authentication
if not st.session_state.get('authenticated'):
    st.error("⛔ Please login first")
    st.stop()

# Check role
if st.session_state.user_role not in ['manager', 'admin']:
    st.error("⛔ Access denied. This page is only for managers and admins.")
    st.stop()

st.title("🔍 Manual Review Queue")
st.markdown("Review returns flagged for manual inspection.")

# Initialize DB
db = get_db_manager()

# Get pending reviews
pending = db.get_pending_reviews(limit=50)

if not pending:
    st.success("✅ No pending reviews! All returns have been processed.")
else:
    st.info(f"📋 {len(pending)} returns awaiting manual review")
    
    # Select return to review
    order_ids = [p['order_id'] for p in pending]
    selected_order = st.selectbox("Select Order to Review", order_ids)
    
    if selected_order:
        # Get return data
        return_data = db.get_return_data(selected_order)
        delivery_data = db.get_delivery_data(selected_order)
        
        # Display fraud analysis
        st.markdown("---")
        st.markdown("### 📊 Fraud Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Score", f"{return_data['fraud_score']:.1f}%")
        with col2:
            st.metric("Visual Similarity", f"{return_data['similarity_score'] * 100:.1f}%")
        with col3:
            st.metric("Customer Risk", f"{return_data['customer_risk_score'] * 100:.1f}%")
        
        # Explanation
        st.markdown("### 💡 AI Explanation")
        st.info(return_data['explanation'])
        
        # Images
        st.markdown("### 🖼️ Image Comparison")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Delivery Images**")
            for img_path in delivery_data['image_paths'][:3]:
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
        
        with col2:
            st.markdown("**Return Images**")
            for img_path in return_data['return_image_paths'][:3]:
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
        
        # Review form
        st.markdown("---")
        st.markdown("### ⚖️ Manager Decision")
        
        with st.form("review_form"):
            final_decision = st.radio(
                "Final Decision",
                ["approved", "rejected"],
                format_func=lambda x: "✅ Approve Return" if x == "approved" else "❌ Reject Return"
            )
            
            notes = st.text_area("Review Notes (Optional)", placeholder="Enter any additional comments or observations...")
            
            submit = st.form_submit_button("Submit Decision", use_container_width=True)
            
            if submit:
                success = db.update_return_decision(
                    order_id=selected_order,
                    new_decision=final_decision,
                    reviewed_by=st.session_state.username,
                    notes=notes
                )
                
                if success:
                    db.resolve_alert(selected_order, st.session_state.username)
                    st.success(f"✅ Decision saved: {final_decision.upper()}")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Failed to update decision")