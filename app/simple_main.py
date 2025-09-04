from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

# Database setup
DATABASE_URL = "sqlite:///./fashion_db.sqlite"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Product model
class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    gender = Column(String)
    master_category = Column(String)
    sub_category = Column(String)
    article_type = Column(String)
    base_colour = Column(String)
    season = Column(String)
    year = Column(Integer)
    usage = Column(String)
    product_display_name = Column(String)
    image_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Pydantic models
class ProductResponse(BaseModel):
    id: int
    name: str
    gender: str
    category: str
    color: str
    image_url: str

# Create FastAPI app
app = FastAPI(title="Fashion Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount images directory
if os.path.exists("./images"):
    app.mount("/images", StaticFiles(directory="./images"), name="images")

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Routes
@app.get("/")
async def root():
    return {"message": "Fashion Recommendation API", "status": "running"}

@app.get("/api/v1/products/random")
async def get_random_products(count: int = 20):
    """Get random products"""
    db = SessionLocal()
    try:
        products = db.query(Product).limit(count).all()
        
        return [
            {
                "id": product.id,
                "name": product.product_display_name,
                "gender": product.gender,
                "category": product.master_category,
                "color": product.base_colour,
                "season": product.season,
                "usage": product.usage,
                "image_url": f"/images/{product.id}.jpg"
            }
            for product in products
        ]
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.get("/api/v1/products/{product_id}")
async def get_product(product_id: int):
    """Get specific product"""
    db = SessionLocal()
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return {
            "id": product.id,
            "name": product.product_display_name,
            "gender": product.gender,
            "category": product.master_category,
            "subcategory": product.sub_category,
            "article_type": product.article_type,
            "color": product.base_colour,
            "season": product.season,
            "year": product.year,
            "usage": product.usage,
            "image_url": f"/images/{product.id}.jpg"
        }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

@app.get("/api/v1/stats")
async def get_stats():
    """Get database statistics"""
    db = SessionLocal()
    try:
        total_products = db.query(Product).count()
        
        # Count by category
        categories = db.query(Product.master_category).distinct().all()
        category_counts = {}
        for cat in categories:
            if cat[0]:
                count = db.query(Product).filter(Product.master_category == cat[0]).count()
                category_counts[cat[0]] = count
        
        return {
            "total_products": total_products,
            "categories": category_counts,
            "database_file": "fashion_db.sqlite"
        }
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        db.close()

# Initialize database tables on startup
@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)