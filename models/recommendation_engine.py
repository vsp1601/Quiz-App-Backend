import os
import json
from typing import List, Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import settings


class RecommendationEngine:
    """
    Supervised like/dislike recommender with user personalization.

    Labels:
      - Binary target from ratings (> 0.5 => like)

    Features:
      - Product one-hots: gender, category, season, usage, optional color/style JSON
      - User preference one-hots (derived from user's liked items)
      - User–item match indicators: match_gender/category/season/color

    Training:
      - RandomForestClassifier
      - StandardScaler (on numeric features)
      - Persist model, scaler, and feature column order
    """

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: List[str] = []

        self.model_path = os.path.join(settings.MODELS_PATH, "recommendation_model.pkl")
        self.scaler_path = os.path.join(settings.MODELS_PATH, "scaler.pkl")
        self.columns_path = os.path.join(settings.MODELS_PATH, "feature_columns.json")

    # ---------------------------------------------------------------------
    # User profile
    # ---------------------------------------------------------------------
    @staticmethod
    def build_user_profile(user_id: int, ratings_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict:
        """
        Build a compact profile from user's liked products (> 0.5).
        Returns:
          {
            'preferred_gender': 'Men'|'Women'|None,
            'preferred_categories': [ ... ],
            'preferred_seasons': [ ... ],
            'preferred_colors': [ ... ],
          }
        """
        if ratings_df is None or ratings_df.empty:
            return {}

        liked_ids = ratings_df[(ratings_df["user_id"] == user_id) & (ratings_df["rating"] > 0.5)]["product_id"]
        if liked_ids.empty:
            return {}

        liked = products_df[products_df["id"].isin(liked_ids)]
        profile: Dict[str, object] = {}

        if not liked.empty:
            if "gender" in liked.columns and not liked["gender"].dropna().empty:
                profile["preferred_gender"] = liked["gender"].value_counts().idxmax()

            if "masterCategory" in liked.columns and not liked["masterCategory"].dropna().empty:
                profile["preferred_categories"] = liked["masterCategory"].value_counts().head(2).index.tolist()

            if "season" in liked.columns and not liked["season"].dropna().empty:
                profile["preferred_seasons"] = liked["season"].value_counts().head(2).index.tolist()

            if "baseColour" in liked.columns and not liked["baseColour"].dropna().empty:
                profile["preferred_colors"] = liked["baseColour"].value_counts().head(3).index.tolist()

        return profile

    # Backward-compat shim for older code that calls _build_user_profile
    def _build_user_profile(self, user_id: int, ratings_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict:
        return self.build_user_profile(user_id, ratings_df, products_df)

    # ---------------------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------------------
    @staticmethod
    def _product_onehots(row: pd.Series) -> Dict[str, float]:
        f: Dict[str, float] = {}
        # Gender
        gender = str(row.get("gender") or "")
        f["gender_male"] = 1.0 if gender == "Men" else 0.0
        f["gender_female"] = 1.0 if gender == "Women" else 0.0
        # Category
        mc = str(row.get("masterCategory") or "")
        f["apparel"] = 1.0 if mc == "Apparel" else 0.0
        f["accessories"] = 1.0 if mc == "Accessories" else 0.0
        f["footwear"] = 1.0 if mc == "Footwear" else 0.0
        # Season
        season = str(row.get("season") or "")
        f["summer"] = 1.0 if season == "Summer" else 0.0
        f["winter"] = 1.0 if season == "Winter" else 0.0
        f["monsoon"] = 1.0 if season == "Monsoon" else 0.0
        f["fall"] = 1.0 if season == "Fall" else 0.0
        # Usage (bag of words)
        usage = str(row.get("usage") or "").lower()
        f["casual"] = 1.0 if "casual" in usage else 0.0
        f["formal"] = 1.0 if "formal" in usage else 0.0
        f["sports"] = 1.0 if "sports" in usage else 0.0
        # Optional color/style JSON
        cf = row.get("color_features")
        if pd.notna(cf):
            try:
                color_data = json.loads(cf)
                for c, pct in (color_data.get("color_percentages") or {}).items():
                    f[f"color_{c}"] = float(pct)
            except Exception:
                pass
        sf = row.get("style_features")
        if pd.notna(sf):
            try:
                style_data = json.loads(sf)
                for k, v in (style_data or {}).items():
                    try:
                        f[f"style_{k}"] = float(v)
                    except Exception:
                        continue
            except Exception:
                pass
        return f

    @staticmethod
    def _user_onehots(profile: Dict) -> Dict[str, float]:
        """Compact user preference one-hots derived from profile."""
        f: Dict[str, float] = {}
        pg = profile.get("preferred_gender")
        f["upref_gender_men"] = 1.0 if pg == "Men" else 0.0
        f["upref_gender_women"] = 1.0 if pg == "Women" else 0.0

        cats = set(profile.get("preferred_categories", []) or [])
        f["upref_apparel"] = 1.0 if "Apparel" in cats else 0.0
        f["upref_accessories"] = 1.0 if "Accessories" in cats else 0.0
        f["upref_footwear"] = 1.0 if "Footwear" in cats else 0.0

        seasons = set(profile.get("preferred_seasons", []) or [])
        f["upref_summer"] = 1.0 if "Summer" in seasons else 0.0
        f["upref_winter"] = 1.0 if "Winter" in seasons else 0.0
        f["upref_monsoon"] = 1.0 if "Monsoon" in seasons else 0.0
        f["upref_fall"] = 1.0 if "Fall" in seasons else 0.0

        return f

    @staticmethod
    def _match_features(profile: Dict, product_row: pd.Series) -> Dict[str, float]:
        """User–item match indicators."""
        f: Dict[str, float] = {}
        # gender match
        pg = profile.get("preferred_gender")
        f["match_gender"] = 1.0 if pg and pg == product_row.get("gender") else 0.0
        # category match (any of top categories)
        cats = set(profile.get("preferred_categories", []) or [])
        f["match_category"] = 1.0 if product_row.get("masterCategory") in cats and len(cats) > 0 else 0.0
        # season match (any of top seasons)
        seasons = set(profile.get("preferred_seasons", []) or [])
        f["match_season"] = 1.0 if product_row.get("season") in seasons and len(seasons) > 0 else 0.0
        # color match (any of top colors)
        colors = set(profile.get("preferred_colors", []) or [])
        f["match_color"] = 1.0 if product_row.get("baseColour") in colors and len(colors) > 0 else 0.0
        return f

    # ---------------------------------------------------------------------
    # Training set construction
    # ---------------------------------------------------------------------
    def prepare_features(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature rows per rating with:
          product one-hots + user one-hots + match features + (rating, user_id, product_id)
        """
        if products_df is None or products_df.empty or ratings_df is None or ratings_df.empty:
            return pd.DataFrame()

        # Merge ratings with product attributes
        data = ratings_df.merge(products_df, left_on="product_id", right_on="id", how="inner")
        if data.empty:
            return pd.DataFrame()

        # Precompute user profiles (cache)
        user_ids = data["user_id"].unique().tolist()
        profile_cache: Dict[int, Dict] = {
            uid: self.build_user_profile(uid, ratings_df, products_df) for uid in user_ids
        }

        rows: List[Dict] = []
        for _, row in data.iterrows():
            uid = int(row["user_id"])
            profile = profile_cache.get(uid, {}) or {}
            prod_feats = self._product_onehots(row)
            user_feats = self._user_onehots(profile)
            match_feats = self._match_features(profile, row)

            feats = {}
            feats.update(prod_feats)
            feats.update(user_feats)
            feats.update(match_feats)

            feats["rating"] = float(row.get("rating", 0.0))
            feats["user_id"] = uid
            feats["product_id"] = int(row.get("product_id"))
            rows.append(feats)

        return pd.DataFrame(rows)

    # ---------------------------------------------------------------------
    # Train / persist
    # ---------------------------------------------------------------------
    def train_model(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Dict:
        feat_df = self.prepare_features(products_df, ratings_df)
        if feat_df.empty or len(feat_df) < getattr(settings, "MIN_RATINGS_FOR_TRAINING", 10):
            return {"error": "Not enough ratings for training"}

        exclude = ["rating", "user_id", "product_id"]
        X: pd.DataFrame = feat_df.drop(columns=exclude, errors="ignore")
        y = (feat_df["rating"] > 0.5).astype(int)

        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        acc = float(accuracy_score(y_test, y_pred))

        os.makedirs(settings.MODELS_PATH, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        try:
            with open(self.columns_path, "w", encoding="utf-8") as f:
                json.dump(self.feature_columns, f)
        except Exception:
            pass

        return {
            "accuracy": acc,
            "feature_count": len(self.feature_columns),
            "training_samples": int(len(X_train)),
        }

    def load_model(self) -> bool:
        if not (os.path.exists(self.model_path) and os.path.exists(self.scaler_path)):
            return False
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        # restore columns
        if os.path.exists(self.columns_path):
            try:
                with open(self.columns_path, "r", encoding="utf-8") as f:
                    self.feature_columns = json.load(f)
            except Exception:
                pass
        # as fallback, try scaler names
        if not self.feature_columns and hasattr(self.scaler, "feature_names_in_"):
            self.feature_columns = list(self.scaler.feature_names_in_)
        return True

    # ---------------------------------------------------------------------
    # Vectorization for inference
    # ---------------------------------------------------------------------
    def _vectorize(self, feats: Dict) -> pd.DataFrame:
        if not self.feature_columns:
            raise ValueError("feature_columns is empty; train_model first.")
        row = {col: feats.get(col, 0) for col in self.feature_columns}
        return pd.DataFrame([row], columns=self.feature_columns)

    def _features_for_user_and_product(self, profile: Dict, product_row: pd.Series) -> Dict[str, float]:
        """Build the SAME feature set used during training for (user, product)."""
        feats = {}
        feats.update(self._product_onehots(product_row))
        feats.update(self._user_onehots(profile))
        feats.update(self._match_features(profile, product_row))
        return feats

    # ---------------------------------------------------------------------
    # Predict single pair
    # ---------------------------------------------------------------------
    def predict_preference(self, user_profile: Dict, product_row: pd.Series) -> Tuple[float, Dict]:
        if not self.model and not self.load_model():
            raise ValueError("No trained model available")

        feats = self._features_for_user_and_product(user_profile, product_row)
        X_row = self._vectorize(feats)
        X_row_scaled = self.scaler.transform(X_row)
        proba = float(self.model.predict_proba(X_row_scaled)[0][1])

        # crude importance: RF importances weighted by active features
        importance: Dict[str, float] = {}
        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
            for i, col in enumerate(self.feature_columns):
                val = float(X_row.iloc[0, i])
                if val != 0.0:
                    importance[col] = float(imps[i]) * val
        return proba, importance

    # ---------------------------------------------------------------------
    # Recommend (per-user)
    # ---------------------------------------------------------------------
    def get_recommendations(
        self,
        user_id: int,
        products_df: pd.DataFrame,
        user_ratings_df: pd.DataFrame,
        count: int = 20,
    ) -> List[Dict]:
        if not self.model and not self.load_model():
            raise ValueError("No trained model available")
        if products_df is None or products_df.empty or "id" not in products_df.columns:
            return []

        rated = set(user_ratings_df[user_ratings_df["user_id"] == user_id]["product_id"])
        profile = self.build_user_profile(user_id, user_ratings_df, products_df)

        recs: List[Dict] = []
        for _, prod in products_df.iterrows():
            pid = int(prod["id"])
            if pid in rated:
                continue

            score, imp = self.predict_preference(profile, prod)
            recs.append(
                {
                    "product_id": pid,
                    "score": float(score),
                    "explanation": self._explain_from_importance(imp, prod, profile),
                    "product_name": prod.get("productDisplayName"),
                    "image_path": f"{pid}.jpg",
                }
            )

        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[: max(1, int(count))]

    # ---------------------------------------------------------------------
    # Explanations (human-readable)
    # ---------------------------------------------------------------------
    def _explain_from_importance(self, imp: Dict[str, float], product: pd.Series, profile: Dict) -> str:
        # If any match features are active, prefer those for explanation
        phrases: List[str] = []
        if profile:
            mf = self._match_features(profile, product)
            if mf.get("match_category"):
                phrases.append("matches your preferred category")
            if mf.get("match_color"):
                phrases.append("uses one of your favorite colors")
            if mf.get("match_season"):
                phrases.append("fits your usual season")
            if mf.get("match_gender"):
                phrases.append("in your preferred gender")

        if phrases:
            return "Recommended because it " + ", ".join(phrases[:2])

        # Fallback to generic feature-based phrasing
        if not imp:
            mc = str(product.get("masterCategory") or "").lower()
            return f"Recommended based on similar {mc} preferences" if mc else "Recommended for you"

        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        generic: List[str] = []
        for feat, w in top[:4]:
            if w < 0.01:
                continue
            if feat.startswith("color_"):
                generic.append(f"strong {feat.replace('color_', '')} color presence")
            elif feat == "casual":
                generic.append("casual style")
            elif feat == "formal":
                generic.append("formal style")
            elif feat == "sports":
                generic.append("sports style")
            elif feat == "gender_male":
                generic.append("men's fashion")
            elif feat == "gender_female":
                generic.append("women's fashion")
            elif feat in {"summer", "winter", "monsoon", "fall"}:
                generic.append(f"{feat} season")
        return "Recommended due to " + ", ".join(generic[:2]) if generic else "Recommended for you"
