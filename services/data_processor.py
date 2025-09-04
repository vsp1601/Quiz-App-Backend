# services/data_processor.py
from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd

from app.config import settings
from services.feature_extractor import FeatureExtractor

REQUIRED_COLS = [
    'id', 'gender', 'masterCategory', 'subCategory', 'articleType',
    'baseColour', 'season', 'year', 'usage', 'productDisplayName'
]


class DataProcessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()

    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load and validate the fashion dataset (CSV)."""
        try:
            df = pd.read_csv(csv_path)

            # Validate columns
            missing = [c for c in REQUIRED_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            return df
        except Exception as e:
            raise ValueError(f"Error loading dataset: {e}")

    def process_and_store_features(self, products_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for all products and return a new DF with JSON strings."""
        if products_df.empty:
            return products_df.assign(color_features=None, style_features=None)

        processed = products_df.copy()
        if 'color_features' not in processed.columns:
            processed['color_features'] = None
        if 'style_features' not in processed.columns:
            processed['style_features'] = None

        for idx, product in processed.iterrows():
            try:
                feats = self.feature_extractor.process_product_features(
                    int(product['id']),
                    product.to_dict()
                )
                processed.at[idx, 'color_features'] = json.dumps(feats.get('color_features', {}))
                processed.at[idx, 'style_features'] = json.dumps(feats.get('style_features', {}))
            except Exception as e:
                # Keep going even if one product fails
                processed.at[idx, 'color_features'] = json.dumps({})
                processed.at[idx, 'style_features'] = json.dumps({})

        return processed

    def create_sample_ratings(
        self, products_df: pd.DataFrame, num_users: int = 50, ratings_per_user: int = 30
    ) -> pd.DataFrame:
        """Create synthetic ratings for smoke testing (remove in prod)."""
        import random

        if products_df.empty:
            return pd.DataFrame(columns=['user_id', 'product_id', 'rating'])

        products = products_df['id'].tolist()
        ratings_data: List[Dict] = []

        for user_id in range(1, num_users + 1):
            sampled = random.sample(products, k=min(ratings_per_user, len(products)))
            for pid in sampled:
                base_rating = random.uniform(0.3, 0.8)

                # Simulate gender tilt
                row = products_df.loc[products_df['id'] == pid].head(1)
                gender = (row['gender'].iloc[0] if not row.empty else '') or ''
                if user_id % 2 == 0 and gender == 'Women':
                    base_rating += 0.2
                elif user_id % 2 == 1 and gender == 'Men':
                    base_rating += 0.2

                rating = max(0.0, min(1.0, base_rating))
                ratings_data.append({'user_id': user_id, 'product_id': int(pid), 'rating': float(rating)})

        return pd.DataFrame(ratings_data, columns=['user_id', 'product_id', 'rating'])

    def validate_images(self, products_df: pd.DataFrame) -> Dict:
        """Validate that images exist for products (assumes {id}.{ext})."""
        if products_df.empty:
            return {
                'total_products': 0,
                'existing_images': 0,
                'missing_images': 0,
                'missing_image_ids': [],
            }

        missing, existing = [], []
        for pid in products_df['id']:
            found = False
            for ext in ('jpg', 'jpeg', 'png'):
                path = os.path.join(settings.IMAGES_PATH, f"{pid}.{ext}")
                if os.path.exists(path):
                    found = True
                    break
            if found:
                existing.append(pid)
            else:
                missing.append(pid)

        return {
            'total_products': int(len(products_df)),
            'existing_images': int(len(existing)),
            'missing_images': int(len(missing)),
            'missing_image_ids': [int(x) for x in missing[:10]],  # first 10 for debugging
        }
