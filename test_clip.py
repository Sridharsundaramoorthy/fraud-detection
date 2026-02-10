"""
Test CLIP Similarity - Verify embeddings are working correctly
Run this to debug the 65% similarity issue

Usage: python test_clip_similarity.py
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
import open_clip

def test_clip_embeddings():
    """Test CLIP model with your actual images"""
    
    print("=" * 70)
    print("🔍 CLIP EMBEDDING SIMILARITY TEST")
    print("=" * 70)
    
    # Load CLIP model
    print("\n📦 Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B/32',
        pretrained='openai'
    )
    model = model.to(device)
    model.eval()
    print("✅ CLIP model loaded")
    
    # Get your image paths
    print("\n📁 Enter image paths:")
    delivery_path = input("   Delivery image (iPhone 17 Pro): ").strip()
    return_path = input("   Return image (Samsung Galaxy A35): ").strip()
    
    if not os.path.exists(delivery_path):
        print(f"❌ Delivery image not found: {delivery_path}")
        return
    
    if not os.path.exists(return_path):
        print(f"❌ Return image not found: {return_path}")
        return
    
    # Generate embeddings
    print("\n🔬 Generating embeddings...")
    
    # Delivery image
    delivery_img = Image.open(delivery_path).convert('RGB')
    print(f"   Delivery image size: {delivery_img.size}")
    delivery_tensor = preprocess(delivery_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        delivery_emb = model.encode_image(delivery_tensor)
        delivery_emb = delivery_emb / delivery_emb.norm(dim=-1, keepdim=True)
    
    delivery_emb_np = delivery_emb.cpu().numpy().flatten()
    print(f"   Delivery embedding shape: {delivery_emb_np.shape}")
    print(f"   Delivery embedding norm: {np.linalg.norm(delivery_emb_np):.6f}")
    
    # Return image
    return_img = Image.open(return_path).convert('RGB')
    print(f"   Return image size: {return_img.size}")
    return_tensor = preprocess(return_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        return_emb = model.encode_image(return_tensor)
        return_emb = return_emb / return_emb.norm(dim=-1, keepdim=True)
    
    return_emb_np = return_emb.cpu().numpy().flatten()
    print(f"   Return embedding shape: {return_emb_np.shape}")
    print(f"   Return embedding norm: {np.linalg.norm(return_emb_np):.6f}")
    
    # Calculate similarity
    print("\n📊 Calculating similarity...")
    
    # Method 1: Cosine similarity
    cosine_sim = np.dot(delivery_emb_np, return_emb_np)
    print(f"   Cosine similarity: {cosine_sim:.6f} ({cosine_sim * 100:.2f}%)")
    
    # Method 2: Euclidean distance
    euclidean_dist = np.linalg.norm(delivery_emb_np - return_emb_np)
    print(f"   Euclidean distance: {euclidean_dist:.6f}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("📋 INTERPRETATION:")
    print("=" * 70)
    
    if cosine_sim > 0.8:
        print(f"✅ VERY SIMILAR ({cosine_sim * 100:.1f}%) - Same product expected")
    elif cosine_sim > 0.6:
        print(f"⚠️ MODERATELY SIMILAR ({cosine_sim * 100:.1f}%) - Could be similar products")
    elif cosine_sim > 0.4:
        print(f"⚠️ SOMEWHAT SIMILAR ({cosine_sim * 100:.1f}%) - Different but related")
    else:
        print(f"❌ VERY DIFFERENT ({cosine_sim * 100:.1f}%) - Completely different products")
    
    print("\n🎯 EXPECTED FOR IPHONE vs SAMSUNG:")
    print("   Similarity should be: 20-40% (different products)")
    print(f"   Your actual result: {cosine_sim * 100:.1f}%")
    
    if cosine_sim > 0.5:
        print("\n❌ PROBLEM DETECTED!")
        print("   Similarity is too high for different products.")
        print("   Possible causes:")
        print("   1. Images might be too similar (same background/angle)")
        print("   2. Images might be corrupted or wrong")
        print("   3. CLIP model might not be loaded correctly")
        print("\n   Recommendation:")
        print("   - Check if you uploaded the correct images")
        print("   - Try with very different products (phone vs shoe)")
        print("   - Check image quality and content")
    else:
        print("\n✅ CLIP IS WORKING CORRECTLY!")
        print("   The similarity score is appropriate for different products.")
    
    # Additional debug info
    print("\n" + "=" * 70)
    print("🔧 DEBUG INFO:")
    print("=" * 70)
    print(f"Delivery embedding - First 10 values:")
    print(f"   {delivery_emb_np[:10]}")
    print(f"Return embedding - First 10 values:")
    print(f"   {return_emb_np[:10]}")
    print(f"Difference in embeddings:")
    print(f"   Mean absolute difference: {np.mean(np.abs(delivery_emb_np - return_emb_np)):.6f}")
    print(f"   Max difference: {np.max(np.abs(delivery_emb_np - return_emb_np)):.6f}")


def test_same_image():
    """Test that same image gives 100% similarity"""
    print("\n" + "=" * 70)
    print("🧪 SANITY CHECK: Testing same image twice")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B/32',
        pretrained='openai'
    )
    model = model.to(device)
    model.eval()
    
    image_path = input("\nEnter any image path for sanity check: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb1 = model.encode_image(tensor)
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        
        # Generate again
        emb2 = model.encode_image(tensor)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)
    
    emb1_np = emb1.cpu().numpy().flatten()
    emb2_np = emb2.cpu().numpy().flatten()
    
    similarity = np.dot(emb1_np, emb2_np)
    
    print(f"\n📊 Same image similarity: {similarity:.10f}")
    
    if similarity > 0.9999:
        print("✅ PASS: Same image gives ~100% similarity")
    else:
        print(f"❌ FAIL: Same image should give 100%, got {similarity * 100:.4f}%")
        print("   This indicates a problem with the CLIP model or preprocessing")


if __name__ == "__main__":
    print("\n🔍 CLIP Similarity Debugging Tool")
    print("\nThis will help diagnose why iPhone vs Samsung shows 65% similarity")
    print("\nMake sure you have the actual image file paths ready!")
    
    # Run sanity check first
    print("\n" + "=" * 70)
    choice = input("\n1. Test your actual images (iPhone vs Samsung)\n2. Sanity check (same image twice)\n3. Both\n\nChoice (1/2/3): ").strip()
    
    if choice == "1":
        test_clip_embeddings()
    elif choice == "2":
        test_same_image()
    elif choice == "3":
        test_same_image()
        print("\n\n")
        test_clip_embeddings()
    else:
        print("Invalid choice")