# app/api/recommendations.py
from fastapi import APIRouter, Depends, Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List, Dict
import pandas as pd
import time

from app.database import get_db, get_current_user
from models.database_models import User, Product, UserRating, RecommendationLog
from models.recommendation_engine import RecommendationEngine
from ._helpers import normalize_gender, normalize_category, img_url_from_product

router = APIRouter()

# ---- Tunables ----
# When user doesn't pass gender/category, don't try to score the entire catalog.
DEFAULT_CANDIDATE_LIMIT = 500


def _products_df_from_models(products: List[Product]) -> pd.DataFrame:
    """Build products_df with the schema the engine expects."""
    cols = [
        "id",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "year",
        "usage",
        "productDisplayName",
        "color_features",
        "style_features",
        "image_path",
    ]
    rows = [
        {
            "id": p.id,
            "gender": p.gender,
            "masterCategory": p.master_category,
            "subCategory": p.sub_category,
            "articleType": p.article_type,
            "baseColour": p.base_colour,
            "season": p.season,
            "year": p.year,
            "usage": p.usage,
            "productDisplayName": p.product_display_name,
            "color_features": p.color_features,
            "style_features": p.style_features,
            "image_path": p.image_path,
        }
        for p in products
    ]
    return pd.DataFrame(rows, columns=cols)


def _ratings_df_from_models(ratings: List[UserRating]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"user_id": r.user_id, "product_id": r.product_id, "rating": r.rating} for r in ratings],
        columns=["user_id", "product_id", "rating"],
    )


@router.get("/recommendations")
async def get_recommendations(
    request: Request,
    count: int = 20,
    gender: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Personalized recommendations for the current user.
    - Normalizes & applies filters.
    - Caps candidate set if no filters to keep latency predictable.
    - Batch-predicts for speed.
    - Falls back to popular if model unavailable / error / cold user / empty candidates.
    """
    t0 = time.time()
    print(">>> GET /api/v1/recommendations/recommendations START")

    try:
        # ---- Normalize filters ----
        gender_n = normalize_gender(gender)
        category_n = normalize_category(category)

        # ---- Candidate products ----
        pq = db.query(Product)
        if gender_n:
            pq = pq.filter(func.lower(Product.gender) == gender_n.lower())
        if category_n:
            pq = pq.filter(func.lower(Product.master_category) == category_n.lower())

        # Safety cap if no filters (prevents walking entire catalog every time)
        if not gender_n and not category_n:
            pq = pq.limit(DEFAULT_CANDIDATE_LIMIT)

        products: List[Product] = pq.all()
        print(f"   candidates: {len(products)}")

        if not products:
            # Nothing matches -> popular fallback ignoring filters (or keep filters by design)
            return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        # ---- Build dataframes ----
        products_df = _products_df_from_models(products)
        ratings = db.query(UserRating).all()
        ratings_df = _ratings_df_from_models(ratings)

        # ---- Cold user -> popular fallback ----
        user_rating_count = int((ratings_df["user_id"] == current_user.id).sum()) if not ratings_df.empty else 0
        print(f"   user_rating_count: {user_rating_count}")
        if user_rating_count < 5:
            print("   cold user -> popular fallback")
            return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        # ---- No candidates after filters -> popular fallback ----
        if products_df.empty:
            print("   empty products_df -> popular fallback")
            return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        # ---- Engine: load or train-once (optional) ----
        engine = RecommendationEngine()
        if not engine.load_model():
            # If a model isn't present yet but we have enough ratings, attempt a quick train once.
            from app.config import settings
            if len(ratings_df) >= getattr(settings, "MIN_RATINGS_FOR_TRAINING", 10):
                print("   no model -> training once (warm start)")
                engine.train_model(products_df, ratings_df)
            else:
                print("   no model and not enough ratings -> popular fallback")
                return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        # ---- Batch prediction for all candidates (fast) ----
        # Build product-side feature dicts -> DataFrame aligned to engine.feature_columns
        feat_rows: List[Dict] = []
        id_index: List[int] = []
        # Build minimal user profile once
        user_profile = engine._build_user_profile(current_user.id, ratings_df, products_df)

        # Use engine's private extractor to keep parity with training features
        for _, prod in products_df.iterrows():
            pid = int(prod["id"])
            id_index.append(pid)
            prod_feats = engine._extract_product_features(prod)  # dict
            # Merge with user_profile so we can vectorize against engine.feature_columns
            merged = {**user_profile, **prod_feats}
            # vectorize just collects values in correct order; do it ourselves in batch:
            row = {col: merged.get(col, 0) for col in engine.feature_columns}
            feat_rows.append(row)

        if not feat_rows:
            print("   no feature rows -> popular fallback")
            return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        X_df = pd.DataFrame(feat_rows, columns=engine.feature_columns)
        # Scale & predict in one shot
        X_scaled = engine.scaler.transform(X_df)
        proba = engine.model.predict_proba(X_scaled)[:, 1]  # numpy array

        # ---- Build result list sorted by score ----
        # Quick lookup map for image_url enrichment
        prod_by_id: Dict[int, Product] = {p.id: p for p in products}
        scored = []
        for pid, score in zip(id_index, proba):
            p = prod_by_id.get(pid)
            scored.append(
                {
                    "product_id": pid,
                    "product_name": p.product_display_name if p else None,
                    "score": float(score),
                    "explanation": "",  # optional: could compute top features per row if needed
                    "image_url": img_url_from_product(request, p) if p else None,
                }
            )

        scored.sort(key=lambda r: r["score"], reverse=True)
        out = {
            "recommendations": scored[: max(1, int(count))],
            "recommendation_type": "personalized",
            "user_rating_count": user_rating_count,
        }

        # ---- Best-effort log persist (top K only to limit write volume) ----
        try:
            for r in out["recommendations"][:50]:
                db.add(
                    RecommendationLog(
                        user_id=current_user.id,
                        product_id=r["product_id"],
                        score=float(r["score"]),
                        reason=r.get("explanation") or "",
                    )
                )
            db.commit()
        except Exception:
            db.rollback()

        dur = (time.time() - t0) * 1000
        print(f"<<< /recommendations OK in {dur:.1f} ms")
        return out

    except Exception as ex:
        # Never hang—return popular fallback on any unexpected error
        print(f"!!! /recommendations error: {ex}")
        return await _popular_recommendations_response(request, count, gender, category, db)


@router.get("/popular")
async def get_popular_recommendations(
    request: Request,
    count: int = 20,
    gender: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Public endpoint to fetch popular items (avg rating, desc).
    """
    return await _popular_recommendations_response(request, count, gender, category, db)


@router.post("/retrain-model")
async def retrain_model(db: Session = Depends(get_db)):
    """
    Train/refresh the recommendation model from all products + user ratings.
    Returns training metrics.
    """
    products = db.query(Product).all()
    products_df = _products_df_from_models(products)
    ratings = db.query(UserRating).all()
    ratings_df = _ratings_df_from_models(ratings)

    engine = RecommendationEngine()
    report = engine.train_model(products_df, ratings_df)
    if "error" in report:
        raise HTTPException(status_code=400, detail=report["error"])
    return report


# ----------------------- Internal helpers -----------------------

async def _popular_recommendations_response(
    request: Request,
    count: int,
    gender: Optional[str],
    category: Optional[str],
    db: Session,
):
    """
    Popular items by average rating (case-insensitive filters).
    """
    from sqlalchemy.orm import aliased

    gender_n = normalize_gender(gender)
    category_n = normalize_category(category)

    p = aliased(Product)
    ur = aliased(UserRating)

    q = (
        db.query(
            p.id.label("id"),
            p.product_display_name.label("name"),
            func.coalesce(func.avg(ur.rating), 0.0).label("avg_rating"),
            func.count(ur.rating).label("rating_count"),
        )
        .outerjoin(ur, ur.product_id == p.id)
    )
    if gender_n:
        q = q.filter(func.lower(p.gender) == gender_n.lower())
    if category_n:
        q = q.filter(func.lower(p.master_category) == category_n.lower())

    q = (
        q.group_by(p.id, p.product_display_name)
        .order_by(func.coalesce(func.avg(ur.rating), 0.0).desc(), func.count(ur.rating).desc())
        .limit(count)
    )

    rows = q.all()
    product_map: Dict[int, Product] = {
        prod.id: prod for prod in db.query(Product).filter(Product.id.in_([r.id for r in rows])).all()
    }

    recs = []
    for row in rows:
        prod = product_map.get(row.id)
        recs.append(
            {
                "product_id": row.id,
                "product_name": row.name,
                "score": float(row.avg_rating or 0.0),
                "explanation": f"Popular item with {int(row.rating_count)} ratings",
                "image_url": img_url_from_product(request, prod) if prod else None,
            }
        )

    return {
        "recommendations": recs,
        "recommendation_type": "popular",
    }
