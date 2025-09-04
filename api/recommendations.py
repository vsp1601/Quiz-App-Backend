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

# Cap candidates when no filters to keep latency predictable
DEFAULT_CANDIDATE_LIMIT = 500


def _products_df_from_models(products: List[Product]) -> pd.DataFrame:
    cols = [
        "id", "gender", "masterCategory", "subCategory", "articleType",
        "baseColour", "season", "year", "usage",
        "productDisplayName", "color_features", "style_features", "image_path",
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
    Personalized recommendations:
    - Applies filters (case-insensitive).
    - Caps candidates if no filters (DEFAULT_CANDIDATE_LIMIT).
    - Uses engine.build_user_profile + engine._features_for_user_and_product for batch prediction.
    - Falls back to /popular when model missing, cold user, or any error.
    """
    t0 = time.time()
    print(">>> GET /api/v1/recommendations/recommendations START")

    try:
        # Normalize filters
        gender_n = normalize_gender(gender)
        category_n = normalize_category(category)

        # Candidate query
        pq = db.query(Product)
        if gender_n:
            pq = pq.filter(func.lower(Product.gender) == gender_n.lower())
        if category_n:
            pq = pq.filter(func.lower(Product.master_category) == category_n.lower())

        if not gender_n and not category_n:
            pq = pq.limit(DEFAULT_CANDIDATE_LIMIT)

        products: List[Product] = pq.all()
        print(f"   candidates: {len(products)}")

        if not products:
            return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        # Build dataframes
        products_df = _products_df_from_models(products)
        ratings = db.query(UserRating).all()
        ratings_df = _ratings_df_from_models(ratings)

        # Cold-user fallback
        user_rating_count = int((ratings_df["user_id"] == current_user.id).sum()) if not ratings_df.empty else 0
        print(f"   user_rating_count: {user_rating_count}")
        if user_rating_count < 5:
            print("   cold user -> popular fallback")
            return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        if products_df.empty:
            print("   empty products_df -> popular fallback")
            return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        # Load/prepare engine
        engine = RecommendationEngine()
        if not engine.load_model():
            from app.config import settings
            if len(ratings_df) >= getattr(settings, "MIN_RATINGS_FOR_TRAINING", 10):
                print("   no model -> training once (warm start)")
                engine.train_model(products_df, ratings_df)
            else:
                print("   no model and not enough ratings -> popular fallback")
                return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        # Build user profile ONCE
        user_profile = engine.build_user_profile(current_user.id, ratings_df, products_df)

        # Batch feature matrix for all candidates (product + user + match features)
        feat_rows: List[Dict] = []
        id_index: List[int] = []
        for _, prod in products_df.iterrows():
            pid = int(prod["id"])
            id_index.append(pid)
            # Compose SAME features as training:
            feats = engine._features_for_user_and_product(user_profile, prod)
            row = {col: feats.get(col, 0) for col in engine.feature_columns}
            feat_rows.append(row)

        if not feat_rows:
            print("   no feature rows -> popular fallback")
            return await _popular_recommendations_response(request, count, gender_n, category_n, db)

        X_df = pd.DataFrame(feat_rows, columns=engine.feature_columns)
        X_scaled = engine.scaler.transform(X_df)
        proba = engine.model.predict_proba(X_scaled)[:, 1]

        # Build response
        prod_by_id: Dict[int, Product] = {p.id: p for p in products}
        scored = []
        for pid, score in zip(id_index, proba):
            p = prod_by_id.get(pid)
            scored.append(
                {
                    "product_id": pid,
                    "product_name": p.product_display_name if p else None,
                    "score": float(score),
                    "explanation": "",  # keep light; can compute per-topK if needed
                    "image_url": img_url_from_product(request, p) if p else None,
                }
            )

        scored.sort(key=lambda r: r["score"], reverse=True)
        out = {
            "recommendations": scored[: max(1, int(count))],
            "recommendation_type": "personalized",
            "user_rating_count": user_rating_count,
        }

        # Best-effort logging (limit volume)
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
    return await _popular_recommendations_response(request, count, gender, category, db)


@router.post("/retrain-model")
async def retrain_model(db: Session = Depends(get_db)):
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
