from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.database import get_db, get_current_user
from models.database_models import User, UserRating, Product

router = APIRouter()

class RatingInput(BaseModel):
    product_id: int
    rating: float  # 0.0 to 1.0

class BulkRatingInput(BaseModel):
    ratings: List[RatingInput]

@router.post("/rate-product")
async def rate_product(
    rating_input: RatingInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Rate a single product"""
    # Check if product exists
    product = db.query(Product).filter(Product.id == rating_input.product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Check if user already rated this product
    existing_rating = db.query(UserRating).filter(
        UserRating.user_id == current_user.id,
        UserRating.product_id == rating_input.product_id
    ).first()
    
    if existing_rating:
        # Update existing rating
        existing_rating.rating = rating_input.rating
    else:
        # Create new rating
        new_rating = UserRating(
            user_id=current_user.id,
            product_id=rating_input.product_id,
            rating=rating_input.rating
        )
        db.add(new_rating)
    
    db.commit()
    return {"message": "Rating saved successfully"}

@router.post("/rate-products-bulk")
async def rate_products_bulk(
    ratings_input: BulkRatingInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Rate multiple products at once"""
    for rating in ratings_input.ratings:
        # Check if product exists
        product = db.query(Product).filter(Product.id == rating.product_id).first()
        if not product:
            continue  # Skip invalid products
        
        # Check if user already rated this product
        existing_rating = db.query(UserRating).filter(
            UserRating.user_id == current_user.id,
            UserRating.product_id == rating.product_id
        ).first()
        
        if existing_rating:
            existing_rating.rating = rating.rating
        else:
            new_rating = UserRating(
                user_id=current_user.id,
                product_id=rating.product_id,
                rating=rating.rating
            )
            db.add(new_rating)
    
    db.commit()
    return {"message": f"Processed {len(ratings_input.ratings)} ratings"}

@router.get("/my-ratings")
async def get_user_ratings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all ratings by current user"""
    ratings = db.query(UserRating).filter(
        UserRating.user_id == current_user.id
    ).all()
    
    return [
        {
            "product_id": rating.product_id,
            "rating": rating.rating,
            "created_at": rating.created_at
        }
        for rating in ratings
    ]