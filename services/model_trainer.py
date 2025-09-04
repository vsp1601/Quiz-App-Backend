# services/model_trainer.py
import logging
from typing import Dict, Any

import pandas as pd
from sqlalchemy.orm import Session

from models.database_models import Product, UserRating
from models.recommendation_engine import RecommendationEngine
from services.data_processor import DataProcessor

logger = logging.getLogger(__name__)

# Sensible default if not provided in settings
MIN_TRAIN_RATINGS = 40  # total labeled samples required to train (adjust as you like)

REQUIRED_PRODUCT_COLS = [
    'id', 'gender', 'masterCategory', 'subCategory', 'articleType',
    'baseColour', 'season', 'year', 'usage', 'productDisplayName',
    'color_features', 'style_features'
]


class ModelTrainer:
    def __init__(self):
        self.rec_engine = RecommendationEngine()
        self.data_processor = DataProcessor()

    def _normalize_products_df(self, prods) -> pd.DataFrame:
        rows = []
        for p in prods:
            rows.append({
                'id': p.id,
                'gender': p.gender,
                'masterCategory': p.master_category,
                'subCategory': p.sub_category,
                'articleType': p.article_type,
                'baseColour': p.base_colour,
                'season': p.season,
                'year': p.year,
                'usage': p.usage,
                'productDisplayName': p.product_display_name,
                'color_features': p.color_features,
                'style_features': p.style_features,
            })
        df = pd.DataFrame(rows, columns=REQUIRED_PRODUCT_COLS)
        return df

    def _normalize_ratings_df(self, ratings) -> pd.DataFrame:
        rows = [{'user_id': r.user_id, 'product_id': r.product_id, 'rating': r.rating} for r in ratings]
        df = pd.DataFrame(rows, columns=['user_id', 'product_id', 'rating'])
        return df

    def retrain_model(self, db: Session) -> Dict[str, Any]:
        """Retrain the recommendation model with latest data"""
        logger.info("Starting model retraining...")

        try:
            # ---- Load from DB
            products = db.query(Product).all()
            ratings = db.query(UserRating).all()

            logger.info("Loaded %d products, %d ratings from DB", len(products), len(ratings))

            # ---- Normalize into DataFrames with fixed schema
            products_df = self._normalize_products_df(products)
            ratings_df = self._normalize_ratings_df(ratings)

            # ---- Quick sanity logs
            if not products_df.empty:
                logger.info("Products DF columns: %s", list(products_df.columns))
                logger.info("Products DF sample:\n%s", products_df.head(3).to_string(index=False))
            else:
                logger.warning("Products DF is EMPTY")

            if not ratings_df.empty:
                logger.info("Ratings DF sample:\n%s", ratings_df.head(5).to_string(index=False))
            else:
                logger.warning("Ratings DF is EMPTY")

            # ---- Optional: validate images (helps you catch missing files early)
            try:
                img_report = self.data_processor.validate_images(products_df)
                logger.info(
                    "Images: total=%d, existing=%d, missing=%d, first_missing=%s",
                    img_report['total_products'],
                    img_report['existing_images'],
                    img_report['missing_images'],
                    img_report['missing_image_ids'],
                )
            except Exception as e:
                logger.warning("Image validation skipped due to error: %s", e)

            # ---- Minimum data checks
            if ratings_df.empty or len(ratings_df) < MIN_TRAIN_RATINGS:
                msg = f"Not enough labeled samples to train (have {len(ratings_df)}, need >= {MIN_TRAIN_RATINGS})"
                logger.warning(msg)
                return {"error": msg, "training_samples": int(len(ratings_df))}

            if products_df.empty:
                msg = "No products available to train"
                logger.warning(msg)
                return {"error": msg, "training_samples": int(len(ratings_df))}

            # ---- Train model
            training_result = self.rec_engine.train_model(products_df, ratings_df)

            if 'error' in training_result:
                logger.error("Model training failed: %s", training_result['error'])
            else:
                logger.info(
                    "Model trained successfully. accuracy=%.3f features=%d train_samples=%d",
                    training_result.get('accuracy', -1),
                    training_result.get('feature_count', -1),
                    training_result.get('training_samples', -1),
                )

            return training_result

        except Exception as e:
            logger.exception("Error during model retraining")
            return {"error": str(e)}
