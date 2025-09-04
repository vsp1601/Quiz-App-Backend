from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship  # Updated import
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    preferences = relationship("UserPreference", back_populates="user")
    ratings = relationship("UserRating", back_populates="user")

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
    
    # Extracted features (will be populated by feature extraction)
    color_features = Column(Text)  # JSON string
    style_features = Column(Text)  # JSON string
    
    # Relationships
    ratings = relationship("UserRating", back_populates="product")

class UserRating(Base):
    __tablename__ = "user_ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    rating = Column(Float)  # 0.0 (dislike) to 1.0 (like)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    product = relationship("Product", back_populates="ratings")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    preference_type = Column(String)  # 'color', 'style', 'category', etc.
    preference_value = Column(String)
    weight = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="preferences")

class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    score = Column(Float)
    reason = Column(Text)
    recommended_at = Column(DateTime, default=datetime.utcnow)