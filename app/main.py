from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine
from models.database_models import Base
from api import auth, products, recommendations, user_preferences
import logging

logging.basicConfig(
    level=logging.INFO,  # Show INFO and above
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.PROJECT_NAME)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images
app.mount("/images", StaticFiles(directory=settings.IMAGES_PATH), name="images")

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(products.router, prefix=f"{settings.API_V1_STR}/products", tags=["products"])
app.include_router(recommendations.router, prefix=f"{settings.API_V1_STR}/recommendations", tags=["recommendations"])
app.include_router(user_preferences.router, prefix=f"{settings.API_V1_STR}/preferences", tags=["preferences"])

@app.get("/")
async def root():
    return {"message": "Fashion Recommendation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}