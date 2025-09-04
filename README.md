# Fashion Recommendation API

A sophisticated machine learning-powered fashion recommendation system built with FastAPI, featuring personalized recommendations, user authentication, and intelligent product matching.

## 🚀 Features

- **Personalized Recommendations**: ML-powered recommendation engine that learns from user preferences
- **User Authentication**: Secure JWT-based authentication system
- **Product Management**: Comprehensive product catalog with filtering and search
- **User Preferences**: Track and store user preferences for better recommendations
- **Image Processing**: Support for product images with automatic feature extraction
- **RESTful API**: Clean, well-documented REST API endpoints
- **Database Support**: SQLite (default) and PostgreSQL support
- **Docker Ready**: Containerized deployment with Docker and Docker Compose
- **Caching**: Redis integration for improved performance

## 🏗️ Architecture

```
Backend/
├── app/                    # Main application
│   ├── main.py            # FastAPI application entry point
│   ├── config.py          # Configuration settings
│   └── database.py        # Database connection and session management
├── api/                   # API endpoints
│   ├── auth.py            # Authentication endpoints
│   ├── products.py        # Product management endpoints
│   ├── recommendations.py # Recommendation engine endpoints
│   └── user_preferences.py # User preference management
├── models/                # Data models and ML engine
│   ├── database_models.py # SQLAlchemy database models
│   └── recommendation_engine.py # ML recommendation engine
├── services/              # Business logic services
│   ├── data_processor.py  # Data processing utilities
│   ├── feature_extractor.py # Feature extraction for ML
│   └── model_trainer.py   # Model training utilities
├── data/                  # Data storage
│   ├── models/           # Trained ML models
│   └── processed/        # Processed datasets
├── images/               # Product images
└── tests/               # Test files
```

## 🛠️ Tech Stack

- **Backend Framework**: FastAPI
- **Database**: SQLite (default)
- **ORM**: SQLAlchemy
- **Authentication**: JWT with PassLib
- **Machine Learning**: scikit-learn, pandas, numpy
- **Image Processing**: OpenCV, Pillow
- **Caching**: Redis
- **Containerization**: Docker, Docker Compose
- **API Documentation**: Swagger UI

## 📋 Prerequisites

- Python 3.11+
- pip (Python package manager)
- Docker & Docker Compose (optional)
- PostgreSQL (if not using SQLite)

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Backend
   ```

2. **Start with Docker Compose**
   ```bash
   docker-compose up --build
   ```

   This will start:
   - FastAPI application on `http://localhost:8000`
   - PostgreSQL database on port 5432
   - Redis cache on port 6379

### Option 2: Local Development

1. **Clone and navigate to the repository**
   ```bash
   git clone <repository-url>
   cd Backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize the database**
   ```bash
   python quick_start.py
   ```

6. **Load sample data (if available)**
   ```bash
   python simple_data_loader.py
   ```

7. **Start the development server**
   ```bash
   uvicorn app.main:app --reload
   ```

## 📊 Data Setup

### Using the Quick Start Script

The `quick_start.py` script will:
- Create necessary directories
- Set up environment configuration
- Initialize the database
- Create a data loader script

### Loading Your Dataset

1. Place your fashion dataset (Excel/CSV) in the root directory
2. Place product images in the `./images/` directory
3. Run the data loader:
   ```bash
   python simple_data_loader.py
   ```

### Expected Dataset Format

Your dataset should include columns:
- `id`: Unique product identifier
- `gender`: Product gender (Men/Women/Boys/Girls)
- `masterCategory`: Main category
- `subCategory`: Sub-category
- `articleType`: Type of article
- `baseColour`: Primary color
- `season`: Season (Summer/Winter/Fall/Spring)
- `year`: Year of production
- `usage`: Usage type
- `productDisplayName`: Product name

## 🔗 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login

### Products
- `GET /api/v1/products/` - List products with filtering
- `GET /api/v1/products/random` - Get random products
- `GET /api/v1/products/{product_id}` - Get specific product

### Recommendations
- `GET /api/v1/recommendations/recommendations` - Personalized recommendations
- `GET /api/v1/recommendations/popular` - Popular products
- `POST /api/v1/recommendations/retrain-model` - Retrain ML model

### User Preferences
- `GET /api/v1/preferences/` - Get user preferences
- `POST /api/v1/preferences/` - Set user preferences

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=sqlite:///./fashion_db.sqlite
# For PostgreSQL: postgresql://user:password@localhost:5432/fashion_db

# Redis
REDIS_URL=redis://localhost:6379

# JWT
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# File Paths
IMAGES_PATH=./images
DATA_PATH=./data
MODELS_PATH=./data/models

# ML Settings
MIN_RATINGS_FOR_TRAINING=10
RECOMMENDATION_COUNT=20
```

## 🤖 Machine Learning Features

### Recommendation Engine

The system uses a sophisticated ML pipeline:

1. **Feature Extraction**: Extracts color and style features from product images
2. **User Profiling**: Builds user profiles based on rating history
3. **Collaborative Filtering**: Uses user-item interactions for recommendations
4. **Content-Based Filtering**: Considers product attributes and features
5. **Hybrid Approach**: Combines multiple recommendation strategies

### Model Training

- Automatic model training when sufficient data is available
- Retrain endpoint for manual model updates
- Fallback to popular items for cold-start users
- Feature scaling and normalization

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app --cov=api --cov=models
```

## 📈 Performance

- **Caching**: Redis integration for improved response times
- **Database Optimization**: Efficient queries with proper indexing
- **Batch Processing**: Batch feature extraction and model predictions
- **Lazy Loading**: On-demand model loading and training

## 🐳 Docker Deployment

### Production Deployment

1. **Build the image**
   ```bash
   docker build -t fashion-recommendation-api .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Development with Docker

```bash
# Start all services
docker-compose up

# Start specific service
docker-compose up app

# View logs
docker-compose logs -f app
```

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔒 Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS configuration
- Input validation with Pydantic
- SQL injection protection with SQLAlchemy


### Scaling Considerations

- Use PostgreSQL for production
- Implement Redis clustering for high availability
- Consider horizontal scaling with load balancers
- Use CDN for image serving

## 🔮 Future Enhancements

- [ ] Real-time recommendations with WebSockets
- [ ] Advanced ML models (Deep Learning)
- [ ] Advanced analytics dashboard


