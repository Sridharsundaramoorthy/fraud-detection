"""
Complete AI Agent System for Fraud Detection
Implements all 12 specialized agents with production-ready code
"""

import cv2
import numpy as np
import torch
from PIL import Image
import easyocr
from ultralytics import YOLO
import open_clip
from typing import List, Dict, Tuple, Optional
import os
from datetime import datetime, timedelta
import re
import hashlib
import requests
import yaml


class ImageQualityAgent:
    """Agent 1: Validates image quality before processing"""
    
    def __init__(self, config: Dict):
        self.config = config['image_quality']
    
    def validate(self, image_path: str) -> Tuple[bool, str]:
        """
        Validate image quality
        Returns: (is_valid, reason)
        """
        if not os.path.exists(image_path):
            return False, "Image file not found"
        
        # Check file size
        file_size_kb = os.path.getsize(image_path) / 1024
        if file_size_kb < self.config['min_file_size_kb']:
            return False, f"Image too small ({file_size_kb:.1f}KB). Minimum {self.config['min_file_size_kb']}KB required"
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return False, "Invalid image format or corrupted file"
        
        # Check resolution
        height, width = img.shape[:2]
        min_width, min_height = self.config['min_resolution']
        if width < min_width or height < min_height:
            return False, f"Resolution {width}x{height} too low. Minimum {min_width}x{min_height} required"
        
        # Check blur (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.config['max_blur_threshold']:
            return False, f"Image too blurry (score: {blur_score:.1f}). Please retake with better focus"
        
        # Check brightness
        brightness = np.mean(gray)
        if brightness < self.config['min_brightness']:
            return False, f"Image too dark (brightness: {brightness:.1f}). Please retake in better lighting"
        if brightness > self.config['max_brightness']:
            return False, f"Image too bright (brightness: {brightness:.1f}). Please reduce exposure"
        
        return True, "Image quality acceptable"


class ImagePreprocessingAgent:
    """Agent 2: Preprocesses images for analysis"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess image: resize, denoise, normalize
        Returns: preprocessed image array
        """
        img = cv2.imread(image_path)
        
        # Resize while maintaining aspect ratio
        img = self._resize_with_padding(img, self.target_size)
        
        # Denoise
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # Normalize lighting (CLAHE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return img
    
    def _resize_with_padding(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image with padding to maintain aspect ratio"""
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return padded


class EmbeddingGeneratorAgent:
    """Agent 3: Generates CLIP-based visual embeddings"""
    
    def __init__(self, config: Dict):
        self.config = config['clip']
        self.device = "cuda" if torch.cuda.is_available() and self.config['device'] == "cuda" else "cpu"
        
        # Load CLIP model
        print(f"Loading CLIP model on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config['model_name'],
            pretrained='openai'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        print("✅ CLIP model loaded")
    
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate visual embedding for an image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Debug: Check image
        print(f"🔍 Generating embedding for: {os.path.basename(image_path)}")
        print(f"   Image size: {image.size}")
        
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Debug: Check embedding
        print(f"   Embedding shape: {embedding_np.shape}")
        print(f"   Embedding norm: {np.linalg.norm(embedding_np):.4f}")
        print(f"   Embedding mean: {np.mean(embedding_np):.4f}")
        
        return embedding_np
    
    def generate_batch_embeddings(self, image_paths: List[str]) -> np.ndarray:
        """Generate embeddings for multiple images"""
        embeddings = []
        for path in image_paths:
            emb = self.generate_embedding(path)
            embeddings.append(emb)
        return np.array(embeddings)


class OCRExtractionAgent:
    """Agent 4: Extracts serial numbers and barcodes using OCR"""
    
    def __init__(self, config: Dict):
        self.config = config['ocr']
        if self.config['enabled']:
            print("Loading EasyOCR...")
            self.reader = easyocr.Reader(self.config['languages'], gpu=torch.cuda.is_available())
            print("✅ EasyOCR loaded")
        else:
            self.reader = None
    
    def extract_text(self, image_path: str) -> List[Dict]:
        """Extract all text from image with confidence scores"""
        if not self.config['enabled'] or self.reader is None:
            return []
        
        results = self.reader.readtext(image_path)
        
        extracted = []
        for bbox, text, confidence in results:
            if confidence >= self.config['confidence_threshold']:
                extracted.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return extracted
    
    def extract_serial_numbers(self, image_path: str) -> List[str]:
        """Extract potential serial numbers using regex patterns"""
        texts = self.extract_text(image_path)
        
        serial_numbers = []
        for item in texts:
            text = item['text']
            for pattern in self.config['serial_patterns']:
                matches = re.findall(pattern, text)
                serial_numbers.extend(matches)
        
        return list(set(serial_numbers))  # Remove duplicates
    
    def compare_serial_numbers(self, delivery_serials: List[str], 
                               return_serials: List[str]) -> Tuple[bool, float]:
        """
        Compare serial numbers from delivery and return
        Returns: (match_found, confidence)
        """
        if not delivery_serials or not return_serials:
            return False, 0.0
        
        # Check for exact matches
        delivery_set = set(delivery_serials)
        return_set = set(return_serials)
        
        matches = delivery_set.intersection(return_set)
        
        if matches:
            confidence = len(matches) / max(len(delivery_set), len(return_set))
            return True, confidence
        
        return False, 0.0


class VisualSimilarityAgent:
    """Agent 5: Compares visual similarity using embeddings"""
    
    def calculate_similarity(self, embedding1: np.ndarray, 
                            embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        # Ensure embeddings are normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def compare_image_sets(self, delivery_embeddings: List[np.ndarray],
                          return_embeddings: List[np.ndarray]) -> Dict:
        """
        Compare sets of delivery and return images
        Returns best match and average similarity
        """
        similarities = []
        
        # Debug: Print embedding info
        print(f"🔍 DEBUG - Comparing embeddings:")
        print(f"   Delivery embeddings: {len(delivery_embeddings)} images")
        print(f"   Return embeddings: {len(return_embeddings)} images")
        
        for i, d_emb in enumerate(delivery_embeddings):
            for j, r_emb in enumerate(return_embeddings):
                sim = self.calculate_similarity(d_emb, r_emb)
                similarities.append(sim)
                print(f"   Delivery[{i}] vs Return[{j}]: {sim:.4f}")
        
        if not similarities:
            return {
                'max_similarity': 0.0, 
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'std_similarity': 0.0
            }
        
        max_sim = float(np.max(similarities))
        avg_sim = float(np.mean(similarities))
        min_sim = float(np.min(similarities))
        std_sim = float(np.std(similarities))
        
        print(f"   📊 Results: Max={max_sim:.4f}, Avg={avg_sim:.4f}, Min={min_sim:.4f}")
        
        # CRITICAL: If max similarity is high but products are different,
        # this indicates a problem with embeddings
        if max_sim > 0.6:
            print(f"   ⚠️ WARNING: High similarity detected - verify embeddings are correct!")
        
        return {
            'max_similarity': max_sim,
            'avg_similarity': avg_sim,
            'min_similarity': min_sim,
            'std_similarity': std_sim
        }


class DamageDetectionAgent:
    """Agent 6: Detects physical damage using YOLO and OpenCV"""
    
    def __init__(self, config: Dict):
        self.config = config['damage_detection']
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8 model: {self.config['model']}...")
        self.model = YOLO(self.config['model'] + '.pt')
        print("✅ YOLOv8 loaded")
    
    def detect_damage(self, image_path: str) -> Dict:
        """
        Detect physical damage in image
        Returns damage score and detected defects
        """
        img = cv2.imread(image_path)
        
        # Method 1: Edge detection for scratches/cracks
        edge_score = self._detect_edges(img)
        
        # Method 2: YOLO detection (if trained on damage classes)
        yolo_detections = self._yolo_detect(image_path)
        
        # Method 3: Color analysis for discoloration
        color_score = self._detect_discoloration(img)
        
        # Combine scores
        total_damage_score = (edge_score * 0.4 + 
                            yolo_detections['score'] * 0.4 + 
                            color_score * 0.2)
        
        return {
            'damage_score': min(total_damage_score, 1.0),
            'edge_score': edge_score,
            'yolo_detections': yolo_detections['detections'],
            'color_score': color_score,
            'has_damage': total_damage_score > 0.3
        }
    
    def _detect_edges(self, img: np.ndarray) -> float:
        """Detect edges that might indicate scratches or cracks"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / edges.size
        
        # Normalize to 0-1 range (threshold at 10% edge density)
        return min(edge_density * 10, 1.0)
    
    def _yolo_detect(self, image_path: str) -> Dict:
        """Use YOLO to detect damage (if model trained on damage classes)"""
        results = self.model(image_path, conf=self.config['confidence_threshold'])
        
        detections = []
        damage_score = 0.0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                
                # Check if detected class is in damage classes
                if class_name.lower() in self.config['damage_classes']:
                    detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
                    damage_score = max(damage_score, confidence)
        
        return {
            'detections': detections,
            'score': damage_score
        }
    
    def _detect_discoloration(self, img: np.ndarray) -> float:
        """Detect color inconsistencies that might indicate damage"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate color variance
        h, s, v = cv2.split(hsv)
        
        # High variance in hue might indicate discoloration
        hue_variance = np.var(h) / 255.0
        
        return min(hue_variance, 1.0)


class AccessoryVerificationAgent:
    """Agent 7: Verifies accessories using object detection"""
    
    def __init__(self, config: Dict):
        self.config = config['accessory_detection']
        
        # Load YOLOv8 model for object detection
        print(f"Loading YOLOv8 for accessory detection: {self.config['model']}...")
        self.model = YOLO(self.config['model'] + '.pt')
        print("✅ YOLOv8 for accessories loaded")
    
    def detect_items(self, image_path: str) -> List[str]:
        """Detect all items in image"""
        results = self.model(image_path, conf=self.config['confidence_threshold'])
        
        detected_items = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                detected_items.append(class_name.lower())
        
        return detected_items
    
    def verify_accessories(self, image_paths: List[str], 
                          product_category: str) -> Dict:
        """
        Verify all expected accessories are present
        Returns missing items and verification score
        """
        # Get expected items for this category
        expected_items = self.config['expected_items'].get(
            product_category.lower(), 
            []
        )
        
        # Detect items across all images
        all_detected = []
        for path in image_paths:
            detected = self.detect_items(path)
            all_detected.extend(detected)
        
        # Remove duplicates
        all_detected = list(set(all_detected))
        
        # Find missing items
        missing_items = []
        for expected in expected_items:
            if expected not in all_detected:
                missing_items.append(expected)
        
        # Calculate verification score
        if len(expected_items) == 0:
            score = 1.0
            message = 'No accessory verification required for this category'
        else:
            score = 1.0 - (len(missing_items) / len(expected_items))
            message = f'{len(all_detected)}/{len(expected_items)} expected items detected'
        
        return {
            'expected_items': expected_items,
            'detected_items': all_detected,
            'missing_items': missing_items,
            'verification_score': score,
            'all_present': len(missing_items) == 0,
            'message': message
        }


class TemporalAnalysisAgent:
    """Agent 8: Analyzes temporal patterns"""
    
    def __init__(self, config: Dict):
        self.config = config['return_policy']
    
    def check_return_window(self, delivery_timestamp: datetime, 
                           return_timestamp: datetime) -> Dict:
        """
        Check if return is within allowed window
        Returns: validity and temporal risk score
        """
        time_diff = return_timestamp - delivery_timestamp
        max_days = self.config['max_return_days']
        grace_hours = self.config['grace_period_hours']
        
        max_allowed = timedelta(days=max_days, hours=grace_hours)
        
        is_valid = time_diff <= max_allowed
        
        # Calculate temporal risk score
        # Very quick returns (< 1 day) are suspicious
        # Returns right at deadline are also suspicious
        if time_diff < timedelta(days=1):
            risk_score = 0.3  # Suspicious - too quick
        elif time_diff > max_allowed:
            risk_score = 1.0  # Invalid - outside window
        elif time_diff > timedelta(days=max_days - 1):
            risk_score = 0.2  # Slightly suspicious - right at deadline
        else:
            risk_score = 0.0  # Normal
        
        return {
            'is_valid': is_valid,
            'days_elapsed': time_diff.days,
            'hours_elapsed': time_diff.total_seconds() / 3600,
            'temporal_risk_score': risk_score,
            'message': self._get_message(is_valid, time_diff, max_days)
        }
    
    def _get_message(self, is_valid: bool, time_diff: timedelta, max_days: int) -> str:
        """Generate human-readable message"""
        days = time_diff.days
        if not is_valid:
            return f"Return window expired. Maximum {max_days} days allowed, but {days} days have elapsed."
        elif days < 1:
            return f"Return initiated very quickly ({time_diff.total_seconds() / 3600:.1f} hours after delivery)"
        else:
            return f"Return initiated {days} days after delivery (within {max_days}-day window)"


class CustomerRiskProfilingAgent:
    """Agent 9: Analyzes customer risk based on history"""
    
    def __init__(self, config: Dict):
        self.config = config['customer_risk']
    
    def calculate_customer_risk(self, customer_profile: Optional[Dict]) -> Dict:
        """
        Calculate customer risk score based on history
        Returns: risk score and risk factors
        """
        if not customer_profile:
            return {
                'risk_score': 0.5,  # Neutral for new customers
                'risk_level': 'unknown',
                'factors': ['New customer - no history available']
            }
        
        risk_factors = []
        risk_score = 0.0
        
        # Factor 1: Return frequency
        total_returns = customer_profile.get('total_returns', 0)
        total_orders = customer_profile.get('total_orders', 1)
        return_rate = total_returns / max(total_orders, 1)
        
        if return_rate > 0.5:
            risk_factors.append(f'High return rate: {return_rate * 100:.1f}%')
            risk_score += 0.3
        
        # Factor 2: Fraud history
        fraud_cases = customer_profile.get('fraud_cases', 0)
        if fraud_cases > 0:
            risk_factors.append(f'Previous fraud cases: {fraud_cases}')
            risk_score += fraud_cases * 0.2
        
        # Factor 3: Customer status
        status = customer_profile.get('status', 'normal')
        if status == 'blacklisted':
            risk_factors.append('Customer is blacklisted')
            risk_score = 1.0
        elif status == 'high_risk':
            risk_factors.append('Customer flagged as high risk')
            risk_score += 0.3
        elif status == 'trusted':
            risk_factors.append('Trusted customer')
            risk_score = max(0, risk_score - 0.3)
        
        # Factor 4: Existing risk score
        existing_risk = customer_profile.get('risk_score', 0)
        risk_score = (risk_score + existing_risk) / 2
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'high'
        elif risk_score > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_level': risk_level,
            'factors': risk_factors if risk_factors else ['No significant risk factors'],
            'customer_status': status
        }


class FraudScoringAgent:
    """Agent 10: Combines all signals into fraud score"""
    
    def __init__(self, config: Dict):
        self.weights = config['fraud_weights']
        self.categories = config['categories']
    
    def calculate_fraud_score(self, signals: Dict, category: str = 'default') -> Dict:
        """
        Calculate weighted fraud score
        signals = {
            'similarity_score': 0.85,
            'damage_score': 0.2,
            'customer_risk_score': 0.3,
            'temporal_risk_score': 0.1,
            'missing_accessories_score': 0.0
        }
        """
        # Calculate weighted score
        fraud_score = (
            self.weights['visual_similarity'] * (1 - signals.get('similarity_score', 0)) +
            self.weights['damage_score'] * signals.get('damage_score', 0) +
            self.weights['customer_risk'] * signals.get('customer_risk_score', 0) +
            self.weights['temporal_violation'] * signals.get('temporal_risk_score', 0) +
            self.weights['missing_accessories'] * signals.get('missing_accessories_score', 0)
        )
        
        fraud_score = min(fraud_score * 100, 100)  # Convert to percentage
        
        # Get category-specific thresholds
        category_config = self.categories.get(category, self.categories['default'])
        
        # Determine fraud level
        if fraud_score < 20:
            fraud_level = 'very_low'
        elif fraud_score < 40:
            fraud_level = 'low'
        elif fraud_score < 60:
            fraud_level = 'medium'
        elif fraud_score < 80:
            fraud_level = 'high'
        else:
            fraud_level = 'very_high'
        
        return {
            'fraud_score': round(fraud_score, 2),
            'fraud_level': fraud_level,
            'component_scores': signals,
            'weights_used': self.weights,
            'category': category,
            'thresholds': category_config
        }


class DecisionAgent:
    """Agent 11: Makes final decision based on fraud score"""
    
    def __init__(self, config: Dict):
        self.categories = config['categories']
    
    def make_decision(self, fraud_analysis: Dict, category: str = 'default',
                     product_value: float = 0) -> Dict:
        """
        Make final decision: approve, reject, or manual review
        """
        fraud_score = fraud_analysis['fraud_score']
        category_config = self.categories.get(category, self.categories['default'])
        
        reject_threshold = category_config['fraud_reject_threshold']
        manual_threshold = category_config['manual_review_threshold']
        high_value_amount = category_config['high_value_amount']
        
        # Override for high-value items
        if product_value > high_value_amount:
            decision = 'manual_review'
            reason = f'High-value item (${product_value:.2f}) - requires manual review'
        
        # Decision based on thresholds
        elif fraud_score >= reject_threshold:
            decision = 'rejected'
            reason = f'Fraud score {fraud_score:.1f}% exceeds rejection threshold ({reject_threshold}%)'
        
        elif fraud_score >= manual_threshold:
            decision = 'manual_review'
            reason = f'Fraud score {fraud_score:.1f}% requires manual review (threshold: {manual_threshold}%)'
        
        else:
            decision = 'approved'
            reason = f'Fraud score {fraud_score:.1f}% is below threshold ({manual_threshold}%)'
        
        return {
            'decision': decision,
            'reason': reason,
            'fraud_score': fraud_score,
            'category': category,
            'requires_manual_review': decision == 'manual_review'
        }


class ExplanationAgent:
    """Agent 12: Generates human-readable explanations using Grok API"""
    
    def __init__(self, config: Dict, api_key: str):
        self.config = config['grok_api']
        self.api_key = api_key
    
    def generate_explanation(self, fraud_analysis: Dict, decision: Dict) -> str:
        """
        Generate human-readable explanation using Grok API
        """
        # Prepare context for Grok
        context = self._prepare_context(fraud_analysis, decision)
        
        # Call Grok API
        try:
            explanation = self._call_grok_api(context)
            return explanation
        except Exception as e:
            print(f"⚠️ Grok API error: {e}")
            # Fallback to template-based explanation
            return self._template_explanation(fraud_analysis, decision)
    
    def _prepare_context(self, fraud_analysis: Dict, decision: Dict) -> str:
        """Prepare context for Grok"""
        fraud_score = fraud_analysis['fraud_score']
        decision_text = decision['decision']
        
        components = fraud_analysis['component_scores']
        
        context = f"""
Analyze this product return fraud detection case and provide a clear, professional explanation.

Decision: {decision_text.upper()}
Fraud Score: {fraud_score:.1f}%

Component Analysis:
- Visual Similarity: {components.get('similarity_score', 0) * 100:.1f}%
- Damage Detected: {components.get('damage_score', 0) * 100:.1f}%
- Customer Risk: {components.get('customer_risk_score', 0) * 100:.1f}%
- Temporal Risk: {components.get('temporal_risk_score', 0) * 100:.1f}%
- Missing Accessories: {components.get('missing_accessories_score', 0) * 100:.1f}%

Provide a 2-3 sentence explanation for warehouse staff explaining why this decision was made.
Be professional, clear, and actionable.
"""
        return context
    
    def _call_grok_api(self, context: str) -> str:
        """Call Grok API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.config['model'],
            "messages": [
                {
                    "role": "system",
                    "content": "You are an AI assistant helping warehouse staff understand fraud detection decisions. Be concise and professional."
                },
                {
                    "role": "user",
                    "content": context
                }
            ],
            "max_tokens": self.config['max_tokens'],
            "temperature": self.config['temperature']
        }
        
        response = requests.post(
            self.config['endpoint'],
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API returned status {response.status_code}")
    
    def _template_explanation(self, fraud_analysis: Dict, decision: Dict) -> str:
        """Fallback template-based explanation"""
        fraud_score = fraud_analysis['fraud_score']
        decision_text = decision['decision']
        components = fraud_analysis['component_scores']
        
        similarity = components.get('similarity_score', 0) * 100
        damage = components.get('damage_score', 0) * 100
        customer_risk = components.get('customer_risk_score', 0) * 100
        
        if decision_text == 'approved':
            return f"Return APPROVED (fraud risk: {fraud_score:.1f}%). The returned product shows {similarity:.1f}% visual match with delivery images, minimal damage ({damage:.1f}%), and low customer risk ({customer_risk:.1f}%). All verification checks passed."
        
        elif decision_text == 'rejected':
            reasons = []
            if similarity < 70:
                reasons.append(f"low visual similarity ({similarity:.1f}%)")
            if damage > 50:
                reasons.append(f"significant damage detected ({damage:.1f}%)")
            if customer_risk > 70:
                reasons.append(f"high customer risk ({customer_risk:.1f}%)")
            
            reason_text = ", ".join(reasons) if reasons else "multiple fraud indicators"
            return f"Return REJECTED (fraud risk: {fraud_score:.1f}%). Detection system flagged: {reason_text}. Please verify product authenticity before processing."
        
        else:  # manual_review
            return f"Manual review REQUIRED (fraud risk: {fraud_score:.1f}%). The system detected moderate fraud indicators: visual similarity {similarity:.1f}%, damage score {damage:.1f}%, customer risk {customer_risk:.1f}%. Please inspect the product physically before making final decision."


# Factory function to create all agents
def create_agents(config_path: str = "config.yaml", grok_api_key: str = None) -> Dict:
    """
    Create all agent instances
    Returns dictionary of all agents
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    agents = {
        'quality': ImageQualityAgent(config),
        'preprocessing': ImagePreprocessingAgent(),
        'embedding': EmbeddingGeneratorAgent(config),
        'ocr': OCRExtractionAgent(config),
        'similarity': VisualSimilarityAgent(),
        'damage': DamageDetectionAgent(config),
        'accessory': AccessoryVerificationAgent(config),
        'temporal': TemporalAnalysisAgent(config),
        'customer_risk': CustomerRiskProfilingAgent(config),
        'fraud_scoring': FraudScoringAgent(config),
        'decision': DecisionAgent(config),
        'explanation': ExplanationAgent(config, grok_api_key) if grok_api_key else None
    }
    
    print("✅ All agents initialized successfully")
    return agents