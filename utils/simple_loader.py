#!/usr/bin/env python3
import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Database setup
DATABASE_URL = "postgresql://postgres:password@localhost:5432/fashion_db"

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
except Exception as e:
    print(f"Database connection error: {e}")
    print("Make sure PostgreSQL is running and database exists")
    exit(1)

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

def main():
    # Find dataset
    dataset_paths = ["Dataset.xlsx", "data/raw/Dataset.xlsx"]
    dataset_path = None
    
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("Dataset not found! Place Dataset.xlsx in root directory")
        return
    
    print(f"Loading dataset: {dataset_path}")
    
    # Load dataset
    try:
        df = pd.read_excel(dataset_path)
        print(f"Loaded {len(df)} products")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Load to database
    db = SessionLocal()
    try:
        inserted = 0
        for _, row in df.iterrows():
            # Check if image exists
            if not os.path.exists(f"./images/{row['id']}.jpg"):
                continue
            
            # Check if already exists
            if db.query(Product).filter(Product.id == row['id']).first():
                continue
            
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
            inserted += 1
            
            if inserted % 100 == 0:
                db.commit()
                print(f"Inserted {inserted} products...")
        
        db.commit()
        print(f"Successfully loaded {inserted} products!")
        
    except Exception as e:
        print(f"Database error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()