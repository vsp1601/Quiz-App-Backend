#!/usr/bin/env python3
"""
Quick start script for fashion recommendation backend
Place this file in the root directory and run: python quick_start.py
"""

import os
import sys

def setup_environment():
    """Setup the basic environment"""
    print("Setting up Fashion Recommendation Backend...")
    
    # Create directories
    directories = [
        'app', 'models', 'services', 'utils', 'api', 'tests',
        'data/raw', 'data/processed', 'data/models', 'images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory in ['app', 'models', 'services', 'utils', 'api', 'tests']:
            init_file = os.path.join(directory, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('# Package initialization\n')
    
    print("✓ Directories and packages created")
    
    # Create .env file
    if not os.path.exists('.env'):
        env_content = '''DATABASE_URL=postgresql://postgres:password@localhost:5432/fashion_db
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-secret-key-here-change-in-production
IMAGES_PATH=./images
DATA_PATH=./data
'''
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✓ Environment file created")
    
    # Check for dataset
    dataset_files = ['Dataset.xlsx', 'data/raw/Dataset.xlsx', 'fashion_dataset.csv']
    dataset_found = None
    
    for file_path in dataset_files:
        if os.path.exists(file_path):
            dataset_found = file_path
            break
    
    if dataset_found:
        print(f"✓ Dataset found: {dataset_found}")
    else:
        print("⚠ Dataset not found. Please place your Dataset.xlsx file in the root directory")
    
    # Check for images
    if os.path.exists('images') and os.listdir('images'):
        image_count = len([f for f in os.listdir('images') if f.endswith('.jpg')])
        print(f"✓ Found {image_count} images")
    else:
        print("⚠ No images found in ./images/ directory")
    
    return dataset_found

def create_simple_loader():
    """Create a simple data loader script"""
    loader_content = '''#!/usr/bin/env python3
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
        print(f"\\n✓ Successfully loaded {products_inserted} products to database")
        
    except Exception as e:
        print(f"Database error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    load_data()
'''
    
    with open('simple_data_loader.py', 'w', encoding="utf-8") as f:
        f.write(loader_content)

    
    print("✓ Created simple_data_loader.py")

def main():
    """Main setup function"""
    dataset_found = setup_environment()
    create_simple_loader()
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Install requirements:")
    print("   pip install fastapi uvicorn sqlalchemy pandas openpyxl psycopg2-binary")
    
    print("\n2. Setup PostgreSQL:")
    print("   createdb fashion_db")
    
    print("\n3. Load your data:")
    print("   python simple_data_loader.py")
    
    print("\n4. Start the API:")
    print("   uvicorn app.main:app --reload")
    
    if not dataset_found:
        print("\n⚠ IMPORTANT: Place your Dataset.xlsx file in the root directory!")
    
    if not os.path.exists('images') or not os.listdir('images'):
        print("\n⚠ IMPORTANT: Place your images in the ./images/ directory!")

if __name__ == "__main__":
    main()