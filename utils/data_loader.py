#!/usr/bin/env python3
"""
Data loader script with proper imports
Run this from the project root directory
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database setup
DATABASE_URL = "postgresql://postgres:password@localhost:5432/fashion_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Import models after path setup
try:
    from models.database_models import Base, Product
    from services.feature_extractor import FeatureExtractor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    print("And that all __init__.py files exist")
    sys.exit(1)

class SimpleDataLoader:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV or Excel"""
        try:
            if file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)
            
            print(f"Loaded dataset with {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            
            # Validate required columns
            required_cols = ['id', 'gender', 'masterCategory', 'subCategory', 
                           'articleType', 'baseColour', 'season', 'year', 
                           'usage', 'productDisplayName']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def validate_images(self, products_df: pd.DataFrame, images_dir: str = "./images") -> dict:
        """Check which products have corresponding images"""
        existing_images = []
        missing_images = []
        
        for product_id in products_df['id']:
            image_path = os.path.join(images_dir, f"{product_id}.jpg")
            if os.path.exists(image_path):
                existing_images.append(product_id)
            else:
                missing_images.append(product_id)
        
        return {
            'total_products': len(products_df),
            'existing_images': len(existing_images),
            'missing_images': len(missing_images),
            'existing_image_ids': existing_images,
            'missing_image_ids': missing_images[:10]  # Show first 10
        }
    
    def load_to_database(self, df: pd.DataFrame, db: Session):
        """Load products into database"""
        try:
            # Create tables
            Base.metadata.create_all(bind=engine)
            
            products_inserted = 0
            products_updated = 0
            
            for _, row in df.iterrows():
                try:
                    # Check if product exists
                    existing_product = db.query(Product).filter(Product.id == row['id']).first()
                    
                    if existing_product:
                        # Update existing product
                        existing_product.gender = str(row['gender'])
                        existing_product.master_category = str(row['masterCategory'])
                        existing_product.sub_category = str(row['subCategory'])
                        existing_product.article_type = str(row['articleType'])
                        existing_product.base_colour = str(row['baseColour'])
                        existing_product.season = str(row['season'])
                        existing_product.year = int(row['year']) if pd.notna(row['year']) else None
                        existing_product.usage = str(row['usage'])
                        existing_product.product_display_name = str(row['productDisplayName'])
                        existing_product.image_path = f"{row['id']}.jpg"
                        products_updated += 1
                    else:
                        # Create new product
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
                    
                    # Commit every 50 products
                    if (products_inserted + products_updated) % 50 == 0:
                        db.commit()
                        print(f"Processed {products_inserted + products_updated} products...")
                        
                except Exception as e:
                    print(f"Error processing product {row['id']}: {e}")
                    continue
            
            # Final commit
            db.commit()
            
            return {
                "products_inserted": products_inserted,
                "products_updated": products_updated,
                "total_processed": products_inserted + products_updated
            }
            
        except Exception as e:
            db.rollback()
            raise Exception(f"Error loading data to database: {e}")

def main():
    """Main function to load data"""
    print("Fashion Recommendation Data Loader")
    print("=" * 40)
    
    # Initialize loader
    loader = SimpleDataLoader()
    
    # Check for dataset file
    possible_paths = [
        "./data/raw/Dataset.xlsx",
        "./data/raw/fashion_dataset.csv", 
        "./Dataset.xlsx",
        "./fashion_dataset.csv"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        print("Dataset not found. Please place your dataset file in one of these locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"Found dataset: {dataset_path}")
    
    # Load dataset
    df = loader.load_dataset(dataset_path)
    if df is None:
        print("Failed to load dataset")
        return
    
    # Validate images
    print("\nValidating images...")
    validation_result = loader.validate_images(df)
    print(f"Products with images: {validation_result['existing_images']}/{validation_result['total_products']}")
    
    if validation_result['existing_images'] == 0:
        print("No images found! Please ensure images are in ./images/ directory")
        print("Image files should be named as {id}.jpg (e.g., 1234.jpg)")
        return
    
    # Filter to only products with images
    products_with_images = df[df['id'].isin(validation_result['existing_image_ids'])]
    print(f"Will process {len(products_with_images)} products that have images")
    
    # Connect to database
    print("\nConnecting to database...")
    try:
        db = SessionLocal()
        
        # Load data to database
        print("Loading data to database...")
        result = loader.load_to_database(products_with_images, db)
        
        print(f"\nSuccess!")
        print(f"Inserted: {result['products_inserted']} products")
        print(f"Updated: {result['products_updated']} products")
        print(f"Total: {result['total_processed']} products")
        
    except Exception as e:
        print(f"Database error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Create database: createdb fashion_db")
        print("3. Update DATABASE_URL in .env file")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()