
import streamlit as st
import os
import sys
from datetime import datetime
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import get_db_manager
from agents.all_agents import create_agents

st.set_page_config(page_title="Return Processing", page_icon="📦", layout="wide")

# Check authentication
if not st.session_state.get('authenticated'):
    st.error("⛔ Please login first")
    st.stop()

# Check role
if st.session_state.user_role not in ['warehouse_staff', 'manager', 'admin']:
    st.error("⛔ Access denied. This page is only for warehouse staff and managers.")
    st.stop()

st.title("📦 Return Request Processing")
st.markdown("Upload images of returned products for automated fraud detection.")

# Initialize agents
if 'agents' not in st.session_state:
    with st.spinner("Loading AI agents..."):
        grok_key = os.getenv('GROK_API_KEY', 'demo_key')  # Fallback for demo
        st.session_state.agents = create_agents(grok_api_key=grok_key)

agents = st.session_state.agents

# Initialize DB
db = get_db_manager()

# Form
with st.form("return_form"):
    order_id = st.text_input("Order ID *", placeholder="e.g., ORD-2025-001")
    
    st.markdown("---")
    st.markdown("### 📸 Upload Return Images")
    st.markdown("Upload 3-5 clear images of the returned product.")
    
    uploaded_files = st.file_uploader(
        "Choose return images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True
    )
    
    submit = st.form_submit_button("🔍 Analyze Return", use_container_width=True)

# Process submission
if submit:
    if not order_id:
        st.error("❌ Please enter Order ID")
    elif not uploaded_files:
        st.error("❌ Please upload images of returned product")
    else:
        # Check if delivery exists
        delivery_data = db.get_delivery_data(order_id)
        
        if not delivery_data:
            st.error(f"❌ No delivery record found for Order ID: {order_id}")
            st.info("Please verify the Order ID or ensure delivery images were uploaded first.")
        else:
            # Check if return already processed
            existing_return = db.get_return_data(order_id)
            if existing_return:
                st.warning(f"⚠️ Return for {order_id} was already processed on {existing_return['return_timestamp']}")
                st.info("Decision: " + existing_return['decision'].upper())
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create return image directory
                return_dir = f"./data/images/returns/{order_id}"
                os.makedirs(return_dir, exist_ok=True)
                
                # Save return images
                return_image_paths = []
                return_embeddings = []
                return_serials = []
                
                status_text.text("Processing return images...")
                for idx, uploaded_file in enumerate(uploaded_files):
                    image_path = os.path.join(return_dir, f"return_{idx + 1}.jpg")
                    image = Image.open(uploaded_file)
                    image.save(image_path)
                    return_image_paths.append(image_path)
                    
                    # Generate embedding
                    embedding = agents['embedding'].generate_embedding(image_path)
                    return_embeddings.append(embedding)
                    
                    # Extract serial numbers
                    serials = agents['ocr'].extract_serial_numbers(image_path)
                    return_serials.extend(serials)
                
                return_serials = list(set(return_serials))
                progress_bar.progress(0.2)
                
                # Run fraud detection agents
                st.markdown("---")
                st.markdown("### 🤖 AI Agent Analysis")
                
                # Agent 1: Visual Similarity
                status_text.text("🔍 Analyzing visual similarity...")
                
                # CRITICAL FIX: Convert embeddings properly from MongoDB
                delivery_embeddings = []
                for emb in delivery_data['embeddings']:
                    if isinstance(emb, list):
                        delivery_embeddings.append(np.array(emb))
                    else:
                        delivery_embeddings.append(emb)
                
                # Debug: Check embeddings
                st.write(f"**Debug Info:**")
                st.write(f"- Delivery images: {len(delivery_data['image_paths'])}")
                st.write(f"- Delivery embeddings: {len(delivery_embeddings)}")
                st.write(f"- Return images: {len(return_image_paths)}")
                st.write(f"- Return embeddings: {len(return_embeddings)}")
                
                if len(delivery_embeddings) > 0:
                    st.write(f"- Delivery embedding shape: {delivery_embeddings[0].shape}")
                    st.write(f"- Delivery embedding norm: {np.linalg.norm(delivery_embeddings[0]):.4f}")
                
                if len(return_embeddings) > 0:
                    st.write(f"- Return embedding shape: {return_embeddings[0].shape}")
                    st.write(f"- Return embedding norm: {np.linalg.norm(return_embeddings[0]):.4f}")
                
                similarity_result = agents['similarity'].compare_image_sets(
                    delivery_embeddings,
                    return_embeddings
                )
                progress_bar.progress(0.3)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Visual Similarity", f"{similarity_result['max_similarity'] * 100:.1f}%")
                with col2:
                    st.metric("Average Match", f"{similarity_result['avg_similarity'] * 100:.1f}%")
                
                # Agent 2: Serial Number Verification
                status_text.text("🔢 Verifying serial numbers...")
                delivery_serials = delivery_data.get('serial_numbers', [])
                serial_match, serial_confidence = agents['ocr'].compare_serial_numbers(
                    delivery_serials,
                    return_serials
                )
                progress_bar.progress(0.4)
                
                if delivery_serials:
                    st.write(f"**Serial Match:** {'✅ Match Found' if serial_match else '❌ No Match'} (confidence: {serial_confidence * 100:.1f}%)")
                
                # Agent 3: Damage Detection
                status_text.text("🔧 Detecting damage...")
                damage_results = []
                for img_path in return_image_paths:
                    damage = agents['damage'].detect_damage(img_path)
                    damage_results.append(damage)
                
                avg_damage = np.mean([d['damage_score'] for d in damage_results])
                progress_bar.progress(0.5)
                
                st.metric("Damage Score", f"{avg_damage * 100:.1f}%")
                
                # Agent 4: Accessory Verification
                status_text.text("📦 Verifying accessories...")
                accessory_check = agents['accessory'].verify_accessories(
                    return_image_paths,
                    delivery_data['category']
                )
                progress_bar.progress(0.6)
                
                # Display accessory check results
                expected_items = accessory_check.get('expected_items', [])
                if expected_items:
                    detected_items = accessory_check.get('detected_items', [])
                    missing_items = accessory_check.get('missing_items', [])
                    
                    st.write(f"**Accessories Present:** {len(detected_items)}/{len(expected_items)}")
                    if missing_items:
                        st.warning(f"**Missing Items:** {', '.join(missing_items)}")
                else:
                    st.info(accessory_check.get('message', 'No accessory verification required'))
                
                # Agent 5: Temporal Analysis
                status_text.text("⏰ Analyzing return timing...")
                temporal_check = agents['temporal'].check_return_window(
                    delivery_data['delivery_timestamp'],
                    datetime.utcnow()
                )
                progress_bar.progress(0.7)
                
                if not temporal_check['is_valid']:
                    st.error(f"❌ {temporal_check['message']}")
                else:
                    st.info(f"✅ {temporal_check['message']}")
                
                # Agent 6: Customer Risk Analysis
                status_text.text("👤 Analyzing customer risk...")
                customer_profile = db.get_customer_profile(delivery_data['customer_id'])
                customer_risk = agents['customer_risk'].calculate_customer_risk(customer_profile)
                progress_bar.progress(0.8)
                
                st.metric("Customer Risk Score", f"{customer_risk['risk_score'] * 100:.1f}%", 
                         delta=customer_risk['risk_level'].upper())
                
                # Agent 7: Fraud Scoring
                status_text.text("🎯 Calculating fraud score...")
                signals = {
                    'similarity_score': similarity_result['max_similarity'],
                    'damage_score': avg_damage,
                    'customer_risk_score': customer_risk['risk_score'],
                    'temporal_risk_score': temporal_check['temporal_risk_score'],
                    'missing_accessories_score': 1.0 - accessory_check['verification_score']
                }
                
                fraud_analysis = agents['fraud_scoring'].calculate_fraud_score(
                    signals,
                    delivery_data['category']
                )
                progress_bar.progress(0.9)
                
                # Agent 8: Decision Making
                status_text.text("⚖️ Making decision...")
                decision = agents['decision'].make_decision(
                    fraud_analysis,
                    delivery_data['category'],
                    delivery_data.get('product_value', 0)
                )
                
                # Agent 9: Generate Explanation
                if agents['explanation']:
                    status_text.text("📝 Generating explanation...")
                    explanation = agents['explanation'].generate_explanation(
                        fraud_analysis,
                        decision
                    )
                else:
                    explanation = decision['reason']
                
                progress_bar.progress(1.0)
                status_text.text("✅ Analysis complete!")
                
                # Display Results
                st.markdown("---")
                st.markdown("### 📊 Fraud Detection Results")
                
                # Decision banner
                if decision['decision'] == 'approved':
                    st.success(f"## ✅ RETURN APPROVED")
                elif decision['decision'] == 'rejected':
                    st.error(f"## ❌ RETURN REJECTED")
                else:
                    st.warning(f"## ⚠️ MANUAL REVIEW REQUIRED")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraud Score", f"{fraud_analysis['fraud_score']:.1f}%")
                with col2:
                    st.metric("Fraud Level", fraud_analysis['fraud_level'].replace('_', ' ').upper())
                with col3:
                    st.metric("Decision", decision['decision'].replace('_', ' ').upper())
                
                # Explanation
                st.markdown("### 💡 Explanation")
                st.info(explanation)
                
                # Detailed breakdown
                with st.expander("📋 Detailed Component Analysis"):
                    st.json(fraud_analysis['component_scores'])
                
                # Images comparison
                st.markdown("### 🖼️ Image Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Delivery Images**")
                    for img_path in delivery_data['image_paths'][:3]:
                        st.image(img_path, use_column_width=True)
                
                with col2:
                    st.markdown("**Return Images**")
                    for img_path in return_image_paths[:3]:
                        st.image(img_path, use_column_width=True)
                
                # Save results
                return_data = {
                    'order_id': order_id,
                    'customer_id': delivery_data['customer_id'],
                    'product_id': delivery_data['product_id'],
                    'category': delivery_data['category'],
                    'return_image_paths': return_image_paths,
                    'similarity_score': similarity_result['max_similarity'],
                    'damage_score': avg_damage,
                    'customer_risk_score': customer_risk['risk_score'],
                    'temporal_valid': temporal_check['is_valid'],
                    'fraud_score': fraud_analysis['fraud_score'],
                    'fraud_level': fraud_analysis['fraud_level'],
                    'decision': decision['decision'],
                    'explanation': explanation,
                    'processed_by': st.session_state.username,
                    'fraud_details': fraud_analysis
                }
                
                db.save_return_data(return_data)
                
                st.success("✅ Return analysis saved to database")
                
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
                import traceback
                st.code(traceback.format_exc())