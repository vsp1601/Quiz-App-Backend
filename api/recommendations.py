# api/recommendations.py
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional, List, Dict
import pandas as pd

from app.database import get_db, get_current_user
from models.database_models import User, Product, UserRating, RecommendationLog
from models.recommendation_engine import RecommendationEngine
from ._helpers import normalize_gender, normalize_category, img_url_from_product

router = APIRouter()

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
    Get personalized recommendations for current user.
    - Normalizes filters
    - Case-insensitive DB filtering
    - Graceful popular fallback if model missing / too few ratings / no candidates
    """
    # --- Normalize incoming filters to match DB values ---
    gender_n = normalize_gender(gender)
    category_n = normalize_category(category)

    # --- Candidate products (case-insensitive) ---
    pq = db.query(Product)
    if gender_n:
        pq = pq.filter(func.lower(Product.gender) == gender_n.lower())
    if category_n:
        pq = pq.filter(func.lower(Product.master_category) == category_n.lower())
    products: List[Product] = pq.all()

    # --- Build products_df with a fixed schema even if empty ---
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
    products_df = pd.DataFrame(rows, columns=cols)

    # --- Ratings DF (all users; filtered later by user_id) ---
    ratings = db.query(UserRating).all()
    ratings_df = pd.DataFrame(
        [{"user_id": r.user_id, "product_id": r.product_id, "rating": r.rating} for r in ratings],
        columns=["user_id", "product_id", "rating"],
    )

    # --- Cold user -> popular fallback ---
    user_rating_count = int((ratings_df["user_id"] == current_user.id).sum()) if not ratings_df.empty else 0
    if user_rating_count < 5:
        return await _popular_recommendations_response(request, count, gender_n, category_n, db)

    # --- No candidates after filters -> popular fallback ---
    if products_df.empty:
        return await _popular_recommendations_response(request, count, gender_n, category_n, db)

    # --- Personalized via model ---
    rec_engine = RecommendationEngine()
    try:
        recs = rec_engine.get_recommendations(current_user.id, products_df, ratings_df, count)
    except ValueError:
        # model not trained / unavailable
        return await _popular_recommendations_response(request, count, gender_n, category_n, db)

    # --- Enrich with absolute image_url and product_name ---
    # Avoid extra DB hits: use the filtered set first; fallback to db.get if needed
    product_by_id: Dict[int, Product] = {p.id: p for p in products}
    enriched = []
    for r in recs:
        pid = r.get("product_id")
        p = product_by_id.get(pid) or db.get(Product, pid)
        enriched.append(
            {
                **r,
                "product_name": r.get("product_name") or (p.product_display_name if p else None),
                "image_url": img_url_from_product(request, p) if p else None,
            }
        )

    # --- Persist logs (best-effort) ---
    for r in enriched:
        try:
            db.add(
                RecommendationLog(
                    user_id=current_user.id,
                    product_id=r.get("product_id"),
                    score=float(r.get("score") or 0.0),
                    reason=r.get("explanation") or "",
                )
            )
        except Exception:
            # don't block user on log errors
            pass
    db.commit()

    return {
        "recommendations": enriched,
        "recommendation_type": "personalized",
        "user_rating_count": user_rating_count,
    }


@router.get("/popular")
async def get_popular_recommendations(
    request: Request,
    count: int = 20,
    gender: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Public endpoint to fetch popular items (debugging / fallback).
    """
    return await _popular_recommendations_response(request, count, gender, category, db)


# -------------- Internal helpers --------------

async def _popular_recommendations_response(
    request: Request,
    count: int,
    gender: Optional[str],
    category: Optional[str],
    db: Session,
):
    """
    Popular items by average rating (SQLAlchemy only, case-insensitive filters).
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
    # Preload products to compute image_url
    product_map: Dict[int, Product] = {prod.id: prod for prod in db.query(Product).filter(Product.id.in_([r.id for r in rows])).all()}

    recs = []
    for row in rows:
        prod = product_map.get(row.id) or db.get(Product, row.id)
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
