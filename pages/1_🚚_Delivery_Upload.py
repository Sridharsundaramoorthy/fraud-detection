import streamlit as st
import os
import sys
from datetime import datetime
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import get_db_manager
from agents.all_agents import create_agents

st.set_page_config(page_title="Delivery Upload", page_icon="🚚", layout="wide")

# Check authentication
if not st.session_state.get('authenticated'):
    st.error("⛔ Please login first")
    st.stop()

# Check role
if st.session_state.user_role not in ['delivery_person', 'admin']:
    st.error("⛔ Access denied. This page is only for delivery personnel.")
    st.stop()

st.title("🚚 Delivery Image Upload")
st.markdown("Upload product images at the time of delivery for fraud verification.")

# Initialize agents (only quality check needed for delivery)
if 'agents' not in st.session_state:
    with st.spinner("Loading AI agents..."):
        st.session_state.agents = create_agents(grok_api_key=os.getenv('GROK_API_KEY'))

agents = st.session_state.agents

# Initialize DB
db = get_db_manager()

# Form
with st.form("delivery_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        order_id = st.text_input("Order ID *", placeholder="e.g., ORD-2025-001")
        product_id = st.text_input("Product ID *", placeholder="e.g., PROD-12345")
        product_name = st.text_input("Product Name", placeholder="e.g., iPhone 15 Pro")
        
    with col2:
        customer_id = st.text_input("Customer ID *", placeholder="e.g., CUST001")
        category = st.selectbox("Product Category *", [
            "electronics", "clothing", "accessories", "home_appliances"
        ])
        product_value = st.number_input("Product Value ($)", min_value=0.0, value=0.0, step=10.0)
    
    st.markdown("---")
    st.markdown("### 📸 Upload Product Images")
    st.markdown("Upload 3-5 clear images from different angles. Ensure good lighting and focus.")
    
    uploaded_files = st.file_uploader(
        "Choose images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload 3-5 images showing the product from different angles"
    )
    
    submit = st.form_submit_button("✅ Submit Delivery", use_container_width=True)

# Process submission
if submit:
    # Validation
    if not all([order_id, product_id, customer_id, category]):
        st.error("❌ Please fill all required fields (*)")
    elif not uploaded_files:
        st.error("❌ Please upload at least one image")
    elif len(uploaded_files) < 3:
        st.error("❌ Please upload at least 3 images")
    elif len(uploaded_files) > 5:
        st.error("❌ Maximum 5 images allowed")
    else:
        # Check if order already exists
        existing = db.get_delivery_data(order_id)
        if existing:
            st.error(f"❌ Order {order_id} already exists in the system!")
        else:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create image directory
                image_dir = f"./data/images/delivery/{order_id}"
                os.makedirs(image_dir, exist_ok=True)
                
                image_paths = []
                embeddings = []
                serial_numbers = []
                
                # Process each image
                for idx, uploaded_file in enumerate(uploaded_files):
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing image {idx + 1}/{len(uploaded_files)}...")
                    
                    # Save image
                    image_path = os.path.join(image_dir, f"image_{idx + 1}.jpg")
                    image = Image.open(uploaded_file)
                    image.save(image_path)
                    
                    # Quality check
                    is_valid, message = agents['quality'].validate(image_path)
                    if not is_valid:
                        st.error(f"❌ Image {idx + 1} quality check failed: {message}")
                        st.error("Please retake the image and resubmit.")
                        # Clean up
                        for path in image_paths:
                            if os.path.exists(path):
                                os.remove(path)
                        st.stop()
                    
                    image_paths.append(image_path)
                    
                    # Generate embedding
                    embedding = agents['embedding'].generate_embedding(image_path)
                    embeddings.append(embedding.tolist())
                    
                    # Extract serial numbers
                    serials = agents['ocr'].extract_serial_numbers(image_path)
                    serial_numbers.extend(serials)
                
                # Remove duplicate serial numbers
                serial_numbers = list(set(serial_numbers))
                
                # Save to database
                status_text.text("Saving to database...")
                delivery_data = {
                    'order_id': order_id,
                    'product_id': product_id,
                    'product_name': product_name,
                    'customer_id': customer_id,
                    'category': category,
                    'product_value': product_value,
                    'image_paths': image_paths,
                    'embeddings': embeddings,
                    'serial_numbers': serial_numbers,
                    'uploaded_by': st.session_state.username,
                    'num_images': len(image_paths)
                }
                
                db.save_delivery_data(delivery_data)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Delivery record saved successfully!")
                
                # Success message
                st.success(f"""
                ### ✅ Delivery Record Created Successfully!
                
                **Order ID:** {order_id}  
                **Product:** {product_name}  
                **Images Uploaded:** {len(image_paths)}  
                **Serial Numbers Detected:** {len(serial_numbers)}  
                **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
                
                # Display images
                st.markdown("### 📸 Uploaded Images")
                cols = st.columns(len(image_paths))
                for idx, (col, img_path) in enumerate(zip(cols, image_paths)):
                    with col:
                        st.image(img_path, caption=f"Image {idx + 1}", use_column_width=True)
                
                if serial_numbers:
                    st.markdown("### 🔢 Detected Serial Numbers")
                    for sn in serial_numbers:
                        st.code(sn)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
