import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from typing import List, Dict, Tuple
from app.config import settings
import os

class RecommendationEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = os.path.join(settings.MODELS_PATH, 'recommendation_model.pkl')
        self.scaler_path = os.path.join(settings.MODELS_PATH, 'scaler.pkl')
        
    def prepare_features(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for training"""
        # Merge products with ratings
        data = ratings_df.merge(products_df, left_on='product_id', right_on='id')
        
        feature_data = []
        
        for _, row in data.iterrows():
            features = {}
            
            # Basic categorical features
            features['gender_male'] = 1 if row['gender'] == 'Men' else 0
            features['gender_female'] = 1 if row['gender'] == 'Women' else 0
            
            # Master category features
            features['apparel'] = 1 if row['masterCategory'] == 'Apparel' else 0
            features['accessories'] = 1 if row['masterCategory'] == 'Accessories' else 0
            features['footwear'] = 1 if row['masterCategory'] == 'Footwear' else 0
            
            # Season features
            features['summer'] = 1 if row['season'] == 'Summer' else 0
            features['winter'] = 1 if row['season'] == 'Winter' else 0
            features['monsoon'] = 1 if row['season'] == 'Monsoon' else 0
            features['fall'] = 1 if row['season'] == 'Fall' else 0
            
            # Usage features
            usage = str(row['usage']).lower()
            features['casual'] = 1 if 'casual' in usage else 0
            features['formal'] = 1 if 'formal' in usage else 0
            features['sports'] = 1 if 'sports' in usage else 0
            
            # Color features (extract from color_features JSON if available)
            if pd.notna(row.get('color_features')):
                try:
                    color_data = json.loads(row['color_features'])
                    color_percentages = color_data.get('color_percentages', {})
                    for color, percentage in color_percentages.items():
                        features[f'color_{color}'] = percentage
                except:
                    pass
            
            # Style features (extract from style_features JSON if available)
            if pd.notna(row.get('style_features')):
                try:
                    style_data = json.loads(row['style_features'])
                    for feature_name, value in style_data.items():
                        features[f'style_{feature_name}'] = value
                except:
                    pass
            
            # Add target variable
            features['rating'] = row['rating']
            features['user_id'] = row['user_id']
            features['product_id'] = row['product_id']
            
            feature_data.append(features)
        
        return pd.DataFrame(feature_data)
    
    def train_model(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Dict:
        """Train the recommendation model"""
        # Prepare features
        feature_df = self.prepare_features(products_df, ratings_df)
        
        if len(feature_df) < settings.MIN_RATINGS_FOR_TRAINING:
            return {"error": "Not enough ratings for training"}
        
        # Prepare X and y
        exclude_cols = ['rating', 'user_id', 'product_id']
        X = feature_df.drop(columns=exclude_cols)
        y = (feature_df['rating'] > 0.5).astype(int)  # Convert to binary classification
        
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (using RandomForest for better interpretability)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model and scaler
        os.makedirs(settings.MODELS_PATH, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return {
            "accuracy": accuracy,
            "feature_count": len(self.feature_columns),
            "training_samples": len(X_train)
        }
    
    def load_model(self):
        """Load trained model and scaler"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False
    
    def predict_preference(self, user_features: Dict, product_features: Dict) -> Tuple[float, Dict]:
        """Predict user preference for a product"""
        if not self.model:
            if not self.load_model():
                raise ValueError("No trained model available")
        
        # Combine features
        combined_features = {**user_features, **product_features}
        
        # Create feature vector
        feature_vector = []
        feature_importance = {}
        
        for col in self.feature_columns:
            value = combined_features.get(col, 0)
            feature_vector.append(value)
        
        # Scale and predict
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get probability and feature importance
        probability = self.model.predict_proba(feature_vector_scaled)[0][1]
        
        # Get feature importance for explanation
        importances = self.model.feature_importances_
        for i, col in enumerate(self.feature_columns):
            if feature_vector[0][i] > 0:  # Only include active features
                feature_importance[col] = importances[i] * feature_vector[0][i]
        
        return probability, feature_importance
    
    def get_recommendations(self, user_id: int, products_df: pd.DataFrame, 
                          user_ratings_df: pd.DataFrame, count: int = 20) -> List[Dict]:
        """Get personalized recommendations for a user"""
        if not self.model:
            if not self.load_model():
                raise ValueError("No trained model available")

        if products_df is None or products_df.empty or 'id' not in products_df.columns:
            return []

        rated_product_ids = set(user_ratings_df[user_ratings_df['user_id'] == user_id]['product_id'])
        user_profile = self._build_user_profile(user_id, user_ratings_df, products_df)

        recommendations = []
        for _, product in products_df.iterrows():
            if product['id'] in rated_product_ids:
                continue
            product_features = self._extract_product_features(product)
            score, importance = self.predict_preference(user_profile, product_features)
            explanation = self._generate_explanation(importance, product)
            recommendations.append({
                'product_id': product['id'],
                'score': float(score),
                'explanation': explanation,
                'product_name': product.get('productDisplayName'),
                'image_path': f"{product['id']}.jpg"
            })

        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:count]
    
    # inside RecommendationEngine._build_user_profile
    def _build_user_profile(self, user_id: int, ratings_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict:
        """Build user preference profile from ratings"""
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        if len(user_ratings) == 0:
            return {}

        if products_df is None or products_df.empty or 'id' not in products_df.columns:
            return {}

        liked_products = user_ratings[user_ratings['rating'] > 0.5]['product_id']
        liked_product_data = products_df[products_df['id'].isin(liked_products)]

        profile = {}
        if len(liked_product_data) > 0:
            gender_counts = liked_product_data['gender'].value_counts(dropna=True)
            if len(gender_counts) > 0:
                profile['preferred_gender'] = gender_counts.index[0]

            if 'baseColour' in liked_product_data.columns:
                color_counts = liked_product_data['baseColour'].value_counts(dropna=True)
                profile['preferred_colors'] = color_counts.head(3).index.tolist()

            if 'masterCategory' in liked_product_data.columns:
                category_counts = liked_product_data['masterCategory'].value_counts(dropna=True)
                profile['preferred_categories'] = category_counts.head(2).index.tolist()

            if 'season' in liked_product_data.columns:
                season_counts = liked_product_data['season'].value_counts(dropna=True)
                profile['preferred_seasons'] = season_counts.head(2).index.tolist()

        return profile
    
    def _extract_product_features(self, product: pd.Series) -> Dict:
        """Extract features from product data"""
        features = {}
        
        # Basic features
        features['gender_male'] = 1 if product['gender'] == 'Men' else 0
        features['gender_female'] = 1 if product['gender'] == 'Women' else 0
        
        # Category features
        features['apparel'] = 1 if product['masterCategory'] == 'Apparel' else 0
        features['accessories'] = 1 if product['masterCategory'] == 'Accessories' else 0
        features['footwear'] = 1 if product['masterCategory'] == 'Footwear' else 0
        
        # Season features
        features['summer'] = 1 if product['season'] == 'Summer' else 0
        features['winter'] = 1 if product['season'] == 'Winter' else 0
        features['monsoon'] = 1 if product['season'] == 'Monsoon' else 0
        features['fall'] = 1 if product['season'] == 'Fall' else 0
        
        # Usage features
        usage = str(product['usage']).lower()
        features['casual'] = 1 if 'casual' in usage else 0
        features['formal'] = 1 if 'formal' in usage else 0
        features['sports'] = 1 if 'sports' in usage else 0
        
        # Add color and style features if available
        if pd.notna(product.get('color_features')):
            try:
                color_data = json.loads(product['color_features'])
                color_percentages = color_data.get('color_percentages', {})
                for color, percentage in color_percentages.items():
                    features[f'color_{color}'] = percentage
            except:
                pass
        
        return features
    
    def _generate_explanation(self, importance: Dict, product: pd.Series) -> str:
        """Generate human-readable explanation for recommendation"""
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        explanations = []
        
        for feature, weight in sorted_features[:3]:  # Top 3 features
            if weight < 0.01:  # Skip very low importance features
                continue
                
            if 'color_' in feature:
                color = feature.replace('color_', '')
                explanations.append(f"strong {color} color preference")
            elif feature == 'casual':
                explanations.append("preference for casual wear")
            elif feature == 'formal':
                explanations.append("preference for formal wear")
            elif feature == 'gender_male':
                explanations.append("preference for men's fashion")
            elif feature == 'gender_female':
                explanations.append("preference for women's fashion")
            elif feature in ['summer', 'winter', 'monsoon', 'fall']:
                explanations.append(f"preference for {feature} season items")
        
        if not explanations:
            return f"Recommended based on similar {product['masterCategory'].lower()} preferences"
        
        return f"Recommended due to your {', '.join(explanations[:2])}"