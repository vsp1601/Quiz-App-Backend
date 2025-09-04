#!/usr/bin/env python3
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database setup
DATABASE_URL = "postgresql://postgres:password@localhost:5432/fashion_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Simple Product model
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

def load_data():
    """Load data from dataset"""
    # Find dataset file
    dataset_paths = ["Dataset.xlsx", "data/raw/Dataset.xlsx", "fashion_dataset.csv"]
    dataset_path = None
    
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("Dataset file not found!")
        return
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset
    try:
        if dataset_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(dataset_path)
        else:
            df = pd.read_csv(dataset_path)
        
        print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables created")
    except Exception as e:
        print(f"Database connection error: {e}")
        print("Make sure PostgreSQL is running and database exists")
        return
    
    # Load data to database
    db = SessionLocal()
    try:
        products_inserted = 0
        
        for _, row in df.iterrows():
            try:
                # Check if image exists
                image_path = f"./images/{row['id']}.jpg"
                if not os.path.exists(image_path):
                    continue  # Skip products without images
                
                # Check if product already exists
                existing = db.query(Product).filter(Product.id == row['id']).first()
                if existing:
                    continue
                
                # Create product
                product = Product(
                    id=int(row['id']),
                    gender=str(row['gender']),
                    master_category=str(row['masterCategory']),
                    sub_category=str(row['subCategory']),
                    article_type=str(row['articleType']),
                    base_colour=str(row['baseColour']),
                    season=str(row['season']),
                    year=int(row['year']) if pd.notna(row['year']) else None,
                    usage=str(row['usage']),
                    product_display_name=str(row['productDisplayName']),
                    image_path=f"{row['id']}.jpg"
                )
                
                db.add(product)
                products_inserted += 1
                
                if products_inserted % 100 == 0:
                    db.commit()
                    print(f"Inserted {products_inserted} products...")
                    
            except Exception as e:
                print(f"Error processing product {row['id']}: {e}")
                continue
        
        db.commit()
        print(f"\n✓ Successfully loaded {products_inserted} products to database")
        
    except Exception as e:
        print(f"Database error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    load_data()
