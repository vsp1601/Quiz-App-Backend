# services/feature_extractor.py
import os
import glob
import json
from typing import Dict
from datetime import datetime

import cv2
import numpy as np

from app.config import settings


class FeatureExtractor:
    def __init__(self):
        # HSV ranges
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'orange': ([11, 50, 50], [25, 255, 255]),
            'yellow': ([26, 50, 50], [35, 255, 255]),
            'green': ([36, 50, 50], [85, 255, 255]),
            'blue': ([86, 50, 50], [125, 255, 255]),
            'purple': ([126, 50, 50], [145, 255, 255]),
            'pink': ([146, 50, 50], [170, 255, 255]),
            'black': ([0, 0, 0], [180, 255, 30]),
            'white': ([0, 0, 200], [180, 30, 255]),
            'grey': ([0, 0, 50], [180, 30, 200]),
        }

    def _find_image_path(self, product_id: int) -> str | None:
        """Try jpg/jpeg/png in IMAGES_PATH; return first that exists."""
        base = os.path.join(settings.IMAGES_PATH, str(product_id))
        for ext in ("jpg", "jpeg", "png"):
            cand = f"{base}.{ext}"
            if os.path.exists(cand):
                return cand
        # also try any file that starts with the id (e.g., id_something.jpg)
        pattern = os.path.join(settings.IMAGES_PATH, f"{product_id}.*")
        matches = sorted(glob.glob(pattern))
        return matches[0] if matches else None

    def extract_color_features(self, image_path: str) -> Dict:
        """Extract color-based features from image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {}

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            color_percentages = {}
            total_pixels = image.shape[0] * image.shape[1]
            if total_pixels == 0:
                return {}

            for color_name, (lower, upper) in self.color_ranges.items():
                lower_arr = np.array(lower, dtype=np.uint8)
                upper_arr = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_arr, upper_arr)
                percentage = float(np.count_nonzero(mask)) / float(total_pixels)
                color_percentages[color_name] = round(percentage, 3)

            dominant = sorted(color_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
            return {
                'color_percentages': color_percentages,
                'dominant_colors': [c[0] for c in dominant],
                'dominant_color_percentages': [c[1] for c in dominant],
            }
        except Exception:
            return {}

    def extract_style_features(self, product_data: Dict) -> Dict:
        """Extract style features from product metadata"""
        features = {}

        usage = (product_data.get('usage') or '').lower()
        features['is_formal'] = 1 if 'formal' in usage else 0
        features['is_casual'] = 1 if 'casual' in usage else 0
        features['is_sports'] = 1 if any(x in usage for x in ('sports', 'gym', 'running')) else 0

        season = (product_data.get('season') or '').lower()
        features['is_summer'] = 1 if season == 'summer' else 0
        features['is_winter'] = 1 if season == 'winter' else 0
        features['is_monsoon'] = 1 if season == 'monsoon' else 0

        master = (product_data.get('masterCategory') or '').lower()
        features['is_apparel'] = 1 if master == 'apparel' else 0
        features['is_accessories'] = 1 if master == 'accessories' else 0
        features['is_footwear'] = 1 if master == 'footwear' else 0

        return features

    def process_product_features(self, product_id: int, product_data: Dict) -> Dict:
        """Process and combine all features for a product"""
        image_path = self._find_image_path(product_id)

        color_features = {}
        if image_path and os.path.exists(image_path):
            color_features = self.extract_color_features(image_path)

        style_features = self.extract_style_features(product_data)

        return {
            'color_features': color_features,
            'style_features': style_features,
            'processed_at': datetime.utcnow().isoformat(),
        }
