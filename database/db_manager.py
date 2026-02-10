"""
MongoDB Database Manager for Fraud Detection System
Handles all database operations with proper error handling and validation
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, OperationFailure
import yaml
import hashlib
import json


class DatabaseManager:
    """Centralized database manager for all collections"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize database connection"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        db_config = self.config['database']
        
        # Connect to MongoDB
        try:
            self.client = MongoClient(
                host=db_config['host'],
                port=db_config['port'],
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.client.admin.command('ping')
            print("✅ MongoDB connection successful")
        except ConnectionFailure:
            raise Exception("❌ Failed to connect to MongoDB. Is MongoDB running?")
        
        self.db = self.client[db_config['name']]
        self._initialize_collections()
        self._create_indexes()
    
    def _initialize_collections(self):
        """Initialize all collections"""
        self.products = self.db['products']
        self.returns = self.db['returns']
        self.customers = self.db['customers']
        self.fraud_patterns = self.db['fraud_patterns']
        self.audit_logs = self.db['audit_logs']
        self.alerts = self.db['alerts']
        self.users = self.db['users']
    
    def _create_indexes(self):
        """Create indexes for better query performance"""
        # Products indexes
        self.products.create_index([("order_id", ASCENDING)], unique=True)
        self.products.create_index([("product_id", ASCENDING)])
        self.products.create_index([("delivery_timestamp", DESCENDING)])
        
        # Returns indexes
        self.returns.create_index([("order_id", ASCENDING)])
        self.returns.create_index([("return_timestamp", DESCENDING)])
        self.returns.create_index([("fraud_score", DESCENDING)])
        self.returns.create_index([("decision", ASCENDING)])
        
        # Customers indexes
        self.customers.create_index([("customer_id", ASCENDING)], unique=True)
        self.customers.create_index([("risk_score", DESCENDING)])
        
        # Alerts indexes
        self.alerts.create_index([("status", ASCENDING)])
        self.alerts.create_index([("created_at", DESCENDING)])
    
    # ==================== PRODUCT OPERATIONS ====================
    
    def save_delivery_data(self, delivery_data: Dict) -> str:
        """Save delivery images and embeddings"""
        delivery_data['delivery_timestamp'] = datetime.utcnow()
        delivery_data['status'] = 'delivered'
        
        # Generate image hash for tamper detection
        if self.config['security']['enable_image_hashing']:
            delivery_data['image_hashes'] = self._generate_image_hashes(
                delivery_data['image_paths']
            )
        
        result = self.products.insert_one(delivery_data)
        
        # Log audit trail
        self._log_audit(
            action="delivery_upload",
            order_id=delivery_data['order_id'],
            user=delivery_data.get('uploaded_by', 'unknown'),
            details={"num_images": len(delivery_data['image_paths'])}
        )
        
        return str(result.inserted_id)
    
    def get_delivery_data(self, order_id: str) -> Optional[Dict]:
        """Retrieve delivery data by order ID"""
        return self.products.find_one({"order_id": order_id})
    
    def verify_image_integrity(self, order_id: str, image_paths: List[str]) -> bool:
        """Verify images haven't been tampered with"""
        delivery_data = self.get_delivery_data(order_id)
        if not delivery_data or 'image_hashes' not in delivery_data:
            return True  # Skip if hashing not enabled
        
        current_hashes = self._generate_image_hashes(image_paths)
        original_hashes = delivery_data['image_hashes']
        
        return current_hashes == original_hashes
    
    # ==================== RETURN OPERATIONS ====================
    
    def save_return_data(self, return_data: Dict) -> str:
        """Save return request with fraud analysis"""
        return_data['return_timestamp'] = datetime.utcnow()
        return_data['processed'] = True
        
        result = self.returns.insert_one(return_data)
        
        # Update customer risk profile
        self._update_customer_risk(
            customer_id=return_data.get('customer_id'),
            fraud_score=return_data['fraud_score'],
            decision=return_data['decision']
        )
        
        # Create alert if needed
        if return_data['decision'] == 'manual_review':
            self._create_alert(return_data)
        
        # Log audit trail
        self._log_audit(
            action="return_processed",
            order_id=return_data['order_id'],
            user=return_data.get('processed_by', 'system'),
            details={
                "fraud_score": return_data['fraud_score'],
                "decision": return_data['decision']
            }
        )
        
        return str(result.inserted_id)
    
    def get_return_data(self, order_id: str) -> Optional[Dict]:
        """Retrieve return data by order ID"""
        return self.returns.find_one({"order_id": order_id})
    
    def get_pending_reviews(self, limit: int = 50) -> List[Dict]:
        """Get returns flagged for manual review"""
        return list(self.returns.find(
            {"decision": "manual_review", "reviewed": {"$ne": True}}
        ).sort("return_timestamp", DESCENDING).limit(limit))
    
    def update_return_decision(self, order_id: str, new_decision: str, 
                               reviewed_by: str, notes: str = "") -> bool:
        """Update decision after manual review"""
        result = self.returns.update_one(
            {"order_id": order_id},
            {
                "$set": {
                    "decision": new_decision,
                    "reviewed": True,
                    "reviewed_by": reviewed_by,
                    "review_notes": notes,
                    "review_timestamp": datetime.utcnow()
                }
            }
        )
        
        if result.modified_count > 0:
            self._log_audit(
                action="manual_review",
                order_id=order_id,
                user=reviewed_by,
                details={"new_decision": new_decision, "notes": notes}
            )
            return True
        return False
    
    # ==================== CUSTOMER OPERATIONS ====================
    
    def _update_customer_risk(self, customer_id: str, fraud_score: float, 
                              decision: str):
        """Update customer risk profile based on return"""
        if not customer_id:
            return
        
        # Get or create customer record
        customer = self.customers.find_one({"customer_id": customer_id})
        
        if not customer:
            customer = {
                "customer_id": customer_id,
                "total_orders": 0,
                "total_returns": 0,
                "fraud_cases": 0,
                "risk_score": 0.0,
                "status": "normal",
                "created_at": datetime.utcnow()
            }
        
        # Update return history
        customer['total_returns'] = customer.get('total_returns', 0) + 1
        
        if decision == 'rejected':
            customer['fraud_cases'] = customer.get('fraud_cases', 0) + 1
        
        # Calculate returns in last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_returns = self.returns.count_documents({
            "customer_id": customer_id,
            "return_timestamp": {"$gte": thirty_days_ago}
        })
        
        # Update risk score
        risk_config = self.config['customer_risk']
        risk_factors = []
        
        # Factor 1: Recent return frequency
        if recent_returns > risk_config['max_returns_30_days']:
            risk_factors.append(0.3)
        
        # Factor 2: Fraud ratio
        if customer['total_returns'] > 0:
            fraud_ratio = customer['fraud_cases'] / customer['total_returns']
            risk_factors.append(fraud_ratio * 0.4)
        
        # Factor 3: Current fraud score
        risk_factors.append(fraud_score / 100 * 0.3)
        
        customer['risk_score'] = min(sum(risk_factors), 1.0)
        
        # Update status
        if customer['fraud_cases'] >= risk_config['blacklist_fraud_count']:
            customer['status'] = 'blacklisted'
        elif customer['risk_score'] > risk_config['high_risk_score_threshold']:
            customer['status'] = 'high_risk'
        elif customer['total_orders'] >= risk_config['whitelist_min_orders']:
            customer['status'] = 'trusted'
        else:
            customer['status'] = 'normal'
        
        customer['last_updated'] = datetime.utcnow()
        
        # Upsert customer record
        self.customers.update_one(
            {"customer_id": customer_id},
            {"$set": customer},
            upsert=True
        )
    
    def get_customer_profile(self, customer_id: str) -> Optional[Dict]:
        """Get customer risk profile"""
        return self.customers.find_one({"customer_id": customer_id})
    
    def get_high_risk_customers(self, limit: int = 20) -> List[Dict]:
        """Get customers with high fraud risk"""
        return list(self.customers.find(
            {"status": {"$in": ["high_risk", "blacklisted"]}}
        ).sort("risk_score", DESCENDING).limit(limit))
    
    # ==================== FRAUD PATTERN OPERATIONS ====================
    
    def save_fraud_pattern(self, pattern_data: Dict):
        """Save detected fraud pattern for ML training"""
        pattern_data['detected_at'] = datetime.utcnow()
        self.fraud_patterns.insert_one(pattern_data)
    
    def get_similar_fraud_cases(self, product_id: str, limit: int = 5) -> List[Dict]:
        """Find similar fraud cases for pattern detection"""
        return list(self.fraud_patterns.find(
            {"product_id": product_id}
        ).sort("detected_at", DESCENDING).limit(limit))
    
    # ==================== ALERT OPERATIONS ====================
    
    def _create_alert(self, return_data: Dict):
        """Create alert for manual review"""
        alert = {
            "order_id": return_data['order_id'],
            "customer_id": return_data.get('customer_id'),
            "fraud_score": return_data['fraud_score'],
            "priority": "high" if return_data['fraud_score'] > 80 else "medium",
            "status": "pending",
            "created_at": datetime.utcnow(),
            "details": return_data.get('fraud_details', {})
        }
        self.alerts.insert_one(alert)
    
    def get_pending_alerts(self, limit: int = 50) -> List[Dict]:
        """Get all pending alerts"""
        return list(self.alerts.find(
            {"status": "pending"}
        ).sort([("priority", DESCENDING), ("created_at", DESCENDING)]).limit(limit))
    
    def resolve_alert(self, order_id: str, resolved_by: str):
        """Mark alert as resolved"""
        self.alerts.update_one(
            {"order_id": order_id},
            {
                "$set": {
                    "status": "resolved",
                    "resolved_by": resolved_by,
                    "resolved_at": datetime.utcnow()
                }
            }
        )
    
    # ==================== AUDIT LOG OPERATIONS ====================
    
    def _log_audit(self, action: str, order_id: str, user: str, details: Dict):
        """Log all system actions for audit trail"""
        log_entry = {
            "action": action,
            "order_id": order_id,
            "user": user,
            "details": details,
            "timestamp": datetime.utcnow(),
            "ip_address": None  # Can be added from request context
        }
        self.audit_logs.insert_one(log_entry)
    
    def get_audit_trail(self, order_id: str) -> List[Dict]:
        """Get complete audit trail for an order"""
        return list(self.audit_logs.find(
            {"order_id": order_id}
        ).sort("timestamp", ASCENDING))
    
    # ==================== ANALYTICS OPERATIONS ====================
    
    def get_fraud_statistics(self, days: int = 30) -> Dict:
        """Get fraud detection statistics"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        total_returns = self.returns.count_documents({
            "return_timestamp": {"$gte": start_date}
        })
        
        approved = self.returns.count_documents({
            "return_timestamp": {"$gte": start_date},
            "decision": "approved"
        })
        
        rejected = self.returns.count_documents({
            "return_timestamp": {"$gte": start_date},
            "decision": "rejected"
        })
        
        manual_review = self.returns.count_documents({
            "return_timestamp": {"$gte": start_date},
            "decision": "manual_review"
        })
        
        # Average fraud score
        pipeline = [
            {"$match": {"return_timestamp": {"$gte": start_date}}},
            {"$group": {"_id": None, "avg_fraud_score": {"$avg": "$fraud_score"}}}
        ]
        avg_result = list(self.returns.aggregate(pipeline))
        avg_fraud_score = avg_result[0]['avg_fraud_score'] if avg_result else 0
        
        return {
            "total_returns": total_returns,
            "approved": approved,
            "rejected": rejected,
            "manual_review": manual_review,
            "avg_fraud_score": round(avg_fraud_score, 2),
            "fraud_detection_rate": round((rejected / total_returns * 100) if total_returns > 0 else 0, 2)
        }
    
    def get_top_returned_products(self, limit: int = 10) -> List[Dict]:
        """Get most frequently returned products"""
        pipeline = [
            {"$group": {
                "_id": "$product_id",
                "return_count": {"$sum": 1},
                "avg_fraud_score": {"$avg": "$fraud_score"}
            }},
            {"$sort": {"return_count": -1}},
            {"$limit": limit}
        ]
        return list(self.returns.aggregate(pipeline))
    
    def get_daily_fraud_trend(self, days: int = 30) -> List[Dict]:
        """Get daily fraud detection trends"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        pipeline = [
            {"$match": {"return_timestamp": {"$gte": start_date}}},
            {"$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$return_timestamp"
                    }
                },
                "total_returns": {"$sum": 1},
                "fraud_cases": {
                    "$sum": {"$cond": [{"$eq": ["$decision", "rejected"]}, 1, 0]}
                },
                "avg_fraud_score": {"$avg": "$fraud_score"}
            }},
            {"$sort": {"_id": 1}}
        ]
        return list(self.returns.aggregate(pipeline))
    
    # ==================== UTILITY METHODS ====================
    
    def _generate_image_hashes(self, image_paths: List[str]) -> List[str]:
        """Generate SHA256 hashes for images"""
        hashes = []
        for path in image_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    hashes.append(file_hash)
            else:
                hashes.append(None)
        return hashes
    
    def populate_sample_data(self):
        """Populate database with sample test data"""
        print("📦 Populating sample data...")
        
        # Sample customers
        sample_customers = [
            {
                "customer_id": "CUST001",
                "total_orders": 15,
                "total_returns": 1,
                "fraud_cases": 0,
                "risk_score": 0.1,
                "status": "trusted"
            },
            {
                "customer_id": "CUST002",
                "total_orders": 5,
                "total_returns": 4,
                "fraud_cases": 3,
                "risk_score": 0.85,
                "status": "high_risk"
            },
            {
                "customer_id": "CUST003",
                "total_orders": 8,
                "total_returns": 2,
                "fraud_cases": 0,
                "risk_score": 0.25,
                "status": "normal"
            }
        ]
        
        for customer in sample_customers:
            self.customers.update_one(
                {"customer_id": customer['customer_id']},
                {"$set": customer},
                upsert=True
            )
        
        print("✅ Sample data populated successfully!")
    
    def close(self):
        """Close database connection"""
        self.client.close()
        print("🔒 Database connection closed")


# Singleton instance
_db_instance = None

def get_db_manager(config_path: str = "config.yaml") -> DatabaseManager:
    """Get singleton database manager instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager(config_path)
    return _db_instance