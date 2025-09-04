#!/usr/bin/env python3
import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import json  # for storing features as JSON strings

# SQLite database
DATABASE_URL = "sqlite:///./fashion_db.sqlite"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

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
    color_features = Column(Text, nullable=True)  # store as JSON string
    style_features = Column(Text, nullable=True)  # store as JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

def main():
    print("Fashion Data Loader (SQLite)")
    print("=" * 40)
    
    # Target CSV file
    dataset_path = "../data/raw/styles.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"✓ Found dataset: {dataset_path}")
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        print(f"✓ Loaded {len(df)} products")
        print(f"✓ Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check images
    images_dir = "../images"
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"✓ Found {len(image_files)} image files")
    
    # Create database and tables
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")
    
    db = SessionLocal()
    try:
        products_inserted = 0
        products_skipped = 0
        
        for _, row in df.iterrows():
            try:
                image_path = f"../images/{int(row['id'])}.jpg"
                if not os.path.exists(image_path):
                    products_skipped += 1
                    continue
                
                existing = db.query(Product).filter(Product.id == row['id']).first()
                if existing:
                    products_skipped += 1
                    continue
                
                # Convert features to JSON string if they exist
                color_features = json.dumps(row['color_features']) if 'color_features' in row else None
                style_features = json.dumps(row['style_features']) if 'style_features' in row else None
                
                product = Product(
                    id=int(row['id']),
                    gender=str(row['gender']) if pd.notna(row['gender']) else '',
                    master_category=str(row['masterCategory']) if pd.notna(row['masterCategory']) else '',
                    sub_category=str(row['subCategory']) if pd.notna(row['subCategory']) else '',
                    article_type=str(row['articleType']) if pd.notna(row['articleType']) else '',
                    base_colour=str(row['baseColour']) if pd.notna(row['baseColour']) else '',
                    season=str(row['season']) if pd.notna(row['season']) else '',
                    year=int(row['year']) if pd.notna(row['year']) else None,
                    usage=str(row['usage']) if pd.notna(row['usage']) else '',
                    product_display_name=str(row['productDisplayName']) if pd.notna(row['productDisplayName']) else '',
                    image_path=f"{row['id']}.jpg",
                    color_features=color_features,
                    style_features=style_features
                )
                
                db.add(product)
                products_inserted += 1
                
                if products_inserted % 50 == 0:
                    db.commit()
                    print(f"  Processed {products_inserted} products...")
                    
            except Exception as e:
                print(f"  Error with product {row['id']}: {e}")
                continue
        
        db.commit()
        print(f"\n✓ SUCCESS!")
        print(f"  Inserted: {products_inserted} products")
        print(f"  Skipped: {products_skipped} products")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
