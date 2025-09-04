# app/models/recommendation_engine.py
import os
import json
from typing import List, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import settings


class RecommendationEngine:
    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: List[str] = []

        self.model_path = os.path.join(settings.MODELS_PATH, "recommendation_model.pkl")
        self.scaler_path = os.path.join(settings.MODELS_PATH, "scaler.pkl")
        self.columns_path = os.path.join(settings.MODELS_PATH, "feature_columns.json")  # NEW

    # ---------- feature prep (same as before) ----------
    def prepare_features(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame) -> pd.DataFrame:
        if ratings_df is None or ratings_df.empty:
            return pd.DataFrame()
        data = ratings_df.merge(products_df, left_on="product_id", right_on="id", how="inner")
        if data.empty:
            return pd.DataFrame()

        rows: List[Dict] = []
        for _, row in data.iterrows():
            f: Dict[str, float | int | str] = {}

            gender = str(row.get("gender", "")).strip()
            f["gender_male"] = 1 if gender == "Men" else 0
            f["gender_female"] = 1 if gender == "Women" else 0

            mc = str(row.get("masterCategory", "")).strip()
            f["apparel"] = 1 if mc == "Apparel" else 0
            f["accessories"] = 1 if mc == "Accessories" else 0
            f["footwear"] = 1 if mc == "Footwear" else 0

            season = str(row.get("season", "")).strip()
            f["summer"] = 1 if season == "Summer" else 0
            f["winter"] = 1 if season == "Winter" else 0
            f["monsoon"] = 1 if season == "Monsoon" else 0
            f["fall"] = 1 if season == "Fall" else 0

            usage = str(row.get("usage", "")).lower()
            f["casual"] = 1 if "casual" in usage else 0
            f["formal"] = 1 if "formal" in usage else 0
            f["sports"] = 1 if "sports" in usage else 0

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
                        f[f"style_{k}"] = float(v)
                except Exception:
                    pass

            f["rating"] = float(row.get("rating", 0.0))
            f["user_id"] = int(row.get("user_id"))
            f["product_id"] = int(row.get("product_id"))
            rows.append(f)

        return pd.DataFrame(rows)

    # ---------- train / persist ----------
    def train_model(self, products_df: pd.DataFrame, ratings_df: pd.DataFrame) -> Dict:
        feat = self.prepare_features(products_df, ratings_df)
        if feat.empty or len(feat) < settings.MIN_RATINGS_FOR_TRAINING:
            return {"error": "Not enough ratings for training"}

        X = feat.drop(columns=["rating", "user_id", "product_id"], errors="ignore")
        y = (feat["rating"] > 0.5).astype(int)

        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
        )
        self.model.fit(X_train_scaled, y_train)

        acc = float(accuracy_score(y_test, self.model.predict(X_test_scaled)))

        os.makedirs(settings.MODELS_PATH, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        # NEW: persist column order so we can restore on restart
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

        # Try to restore feature columns from sidecar; fallback to scaler names
        cols = None
        if os.path.exists(self.columns_path):
            try:
                with open(self.columns_path, "r", encoding="utf-8") as f:
                    cols = json.load(f)
            except Exception:
                cols = None
        if not cols and hasattr(self.scaler, "feature_names_in_"):
            cols = list(self.scaler.feature_names_in_)
        self.feature_columns = cols or self.feature_columns

        return True

    # ---------- prediction ----------
    def _vectorize_features(self, combined: Dict) -> pd.DataFrame:
        if not self.feature_columns:
            raise ValueError("feature_columns is empty; train_model first.")
        row = {c: combined.get(c, 0) for c in self.feature_columns}
        return pd.DataFrame([row], columns=self.feature_columns)

    def predict_preference(self, user_features: Dict, product_features: Dict) -> Tuple[float, Dict]:
        if not self.model and not self.load_model():
            raise ValueError("No trained model available")

        X_row_df = self._vectorize_features({**user_features, **product_features})
        X_row_scaled = self.scaler.transform(X_row_df)
        proba = float(self.model.predict_proba(X_row_scaled)[0][1])

        importance: Dict[str, float] = {}
        if hasattr(self.model, "feature_importances_"):
            imps = self.model.feature_importances_
            for i, col in enumerate(self.feature_columns):
                val = float(X_row_df.iloc[0, i])
                if val:
                    importance[col] = float(imps[i]) * val
        return proba, importance

    # ---------- public API ----------
    def get_recommendations(
        self, user_id: int, products_df: pd.DataFrame, user_ratings_df: pd.DataFrame, count: int = 20
    ) -> List[Dict]:
        if not self.model and not self.load_model():
            raise ValueError("No trained model available")
        if products_df is None or products_df.empty or "id" not in products_df.columns:
            return []

        rated = set(user_ratings_df[user_ratings_df["user_id"] == user_id]["product_id"])
        profile = self._build_user_profile(user_id, user_ratings_df, products_df)

        recs: List[Dict] = []
        for _, p in products_df.iterrows():
            pid = int(p["id"])
            if pid in rated:
                continue
            score, imp = self.predict_preference(profile, self._extract_product_features(p))
            recs.append(
                {
                    "product_id": pid,
                    "score": float(score),
                    "explanation": self._generate_explanation(imp, p),
                    "product_name": p.get("productDisplayName"),
                    "image_path": f"{pid}.jpg",
                }
            )
        recs.sort(key=lambda x: x["score"], reverse=True)
        return recs[: max(1, int(count))]

    # ---------- helpers (unchanged) ----------
    def _build_user_profile(self, user_id: int, ratings_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict:
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return {}
        liked_ids = user_ratings[user_ratings["rating"] > 0.5]["product_id"]
        liked = products_df[products_df["id"].isin(liked_ids)]

        profile: Dict[str, object] = {}
        if not liked.empty:
            if "gender" in liked.columns and not liked["gender"].dropna().empty:
                profile["preferred_gender"] = liked["gender"].value_counts().idxmax()
            if "baseColour" in liked.columns and not liked["baseColour"].dropna().empty:
                profile["preferred_colors"] = liked["baseColour"].value_counts().head(3).index.tolist()
            if "masterCategory" in liked.columns and not liked["masterCategory"].dropna().empty:
                profile["preferred_categories"] = liked["masterCategory"].value_counts().head(2).index.tolist()
            if "season" in liked.columns and not liked["season"].dropna().empty:
                profile["preferred_seasons"] = liked["season"].value_counts().head(2).index.tolist()
        return profile

    def _extract_product_features(self, product: pd.Series) -> Dict:
        f: Dict[str, float | int] = {}
        g = str(product.get("gender", "")).strip()
        f["gender_male"] = 1 if g == "Men" else 0
        f["gender_female"] = 1 if g == "Women" else 0
        m = str(product.get("masterCategory", "")).strip()
        f["apparel"] = 1 if m == "Apparel" else 0
        f["accessories"] = 1 if m == "Accessories" else 0
        f["footwear"] = 1 if m == "Footwear" else 0
        s = str(product.get("season", "")).strip()
        f["summer"] = 1 if s == "Summer" else 0
        f["winter"] = 1 if s == "Winter" else 0
        f["monsoon"] = 1 if s == "Monsoon" else 0
        f["fall"] = 1 if s == "Fall" else 0
        usage = str(product.get("usage", "")).lower()
        f["casual"] = 1 if "casual" in usage else 0
        f["formal"] = 1 if "formal" in usage else 0
        f["sports"] = 1 if "sports" in usage else 0
        cf = product.get("color_features")
        if pd.notna(cf):
            try:
                cd = json.loads(cf)
                for c, pct in (cd.get("color_percentages") or {}).items():
                    f[f"color_{c}"] = float(pct)
            except Exception:
                pass
        sf = product.get("style_features")
        if pd.notna(sf):
            try:
                sd = json.loads(sf)
                for k, v in (sd or {}).items():
                    f[f"style_{k}"] = float(v)
            except Exception:
                pass
        return f

    def _generate_explanation(self, imp: Dict[str, float], product: pd.Series) -> str:
        if not imp:
            mc = str(product.get("masterCategory") or "").lower()
            return f"Recommended based on similar {mc} preferences" if mc else "Recommended for you"
        top = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        phrases: List[str] = []
        for feat, w in top[:3]:
            if w < 0.01:
                continue
            if feat.startswith("color_"):
                phrases.append(f"strong {feat.replace('color_', '')} color preference")
            elif feat == "casual":
                phrases.append("preference for casual wear")
            elif feat == "formal":
                phrases.append("preference for formal wear")
            elif feat == "sports":
                phrases.append("preference for sportswear")
            elif feat == "gender_male":
                phrases.append("preference for men's fashion")
            elif feat == "gender_female":
                phrases.append("preference for women's fashion")
            elif feat in {"summer", "winter", "monsoon", "fall"}:
                phrases.append(f"preference for {feat} season items")
        return f"Recommended due to your {', '.join(phrases[:2])}" if phrases else \
               (f"Recommended based on similar {str(product.get('masterCategory') or '').lower()} preferences" or "Recommended for you")
