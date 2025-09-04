# api/products.py
from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from app.database import get_db
from models.database_models import Product
from ._helpers import normalize_gender, normalize_category, img_url_from_product

router = APIRouter()

@router.get("/products/random", response_model=List[dict])
async def get_random_products(
    request: Request,
    count: int = 20,
    gender: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
):
    gender_n = normalize_gender(gender)
    category_n = normalize_category(category)

    q = db.query(Product)
    if gender_n:
        q = q.filter(func.lower(Product.gender) == gender_n.lower())
    if category_n:
        q = q.filter(func.lower(Product.master_category) == category_n.lower())

    # Postgres/SQLite: random order
    q = q.order_by(func.random()).limit(count)
    products = q.all()

    return [
        {
            "id": p.id,
            "name": p.product_display_name,
            "gender": p.gender,
            "category": p.master_category,
            "subcategory": p.sub_category,
            "color": p.base_colour,
            "season": p.season,
            "usage": p.usage,
            "image_url": img_url_from_product(request, p),
        }
        for p in products
    ]

@router.get("/products/{product_id}")
async def get_product(product_id: int, request: Request, db: Session = Depends(get_db)):
    p = db.query(Product).filter(Product.id == product_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return {
        "id": p.id,
        "name": p.product_display_name,
        "gender": p.gender,
        "category": p.master_category,
        "subcategory": p.sub_category,
        "article_type": p.article_type,
        "color": p.base_colour,
        "season": p.season,
        "year": p.year,
        "usage": p.usage,
        "image_url": img_url_from_product(request, p),
    }

@router.get("/products/")
async def get_products(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    gender: Optional[str] = None,
    category: Optional[str] = None,
    color: Optional[str] = None,
    db: Session = Depends(get_db),
):
    gender_n = normalize_gender(gender)
    category_n = normalize_category(category)

    q = db.query(Product)
    if gender_n:
        q = q.filter(func.lower(Product.gender) == gender_n.lower())
    if category_n:
        q = q.filter(func.lower(Product.master_category) == category_n.lower())
    if color:
        q = q.filter(func.lower(Product.base_colour) == color.strip().lower())

    products = q.offset(skip).limit(limit).all()
    return [
        {
            "id": p.id,
            "name": p.product_display_name,
            "gender": p.gender,
            "category": p.master_category,
            "color": p.base_colour,
            "image_url": img_url_from_product(request, p),
        }
        for p in products
    ]
