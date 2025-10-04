"""
SignSpeak Professional API Server
Enterprise-grade ASL recognition API with SQL Server integration
"""

# Suppress TensorFlow and other ML warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Disable oneDNN optimization warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Force CPU usage to avoid GPU warnings

# Suppress other warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import tempfile
import os
from pathlib import Path
import sys

# Try to import ML dependencies with fallback
try:
    import numpy as np
    import cv2
    import mediapipe as mp
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False

# Try to import database and authentication services
try:
    from database.models import initialize_database
    from database.connections import get_database_connection, check_database_health
    from services.auth_service import get_auth_service, AuthenticationService
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False

# Try to import AI models
if ML_AVAILABLE:
    try:
        from feature_extractor import SignSpeakFeatureExtractor
        from asl_model_simple import FeatureBasedASLModel, create_default_model
        from feature_scaler import load_feature_scaler, create_mock_scaler
        MODEL_AVAILABLE = True
    except ImportError as e:
        MODEL_AVAILABLE = False
else:
    MODEL_AVAILABLE = False

# Authentication dependency
security = HTTPBearer()

# Pydantic models for API requests
class RegisterRequest(BaseModel):
    username: str
    email: str  # Using regular str instead of EmailStr
    password: str
    first_name: str
    last_name: str
    profile_data: Optional[Dict[str, Any]] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class ProfileUpdateRequest(BaseModel):
    preferred_language: Optional[str] = None
    high_contrast_mode: Optional[bool] = None
    large_text_mode: Optional[bool] = None
    voice_enabled: Optional[bool] = None
    voice_speed: Optional[float] = None
    voice_gender: Optional[str] = None
    skill_level: Optional[str] = None
    daily_practice_goal: Optional[int] = None
    bio: Optional[str] = None

# Configure logging with reduced verbosity
logging.basicConfig(level=logging.WARNING)  # Reduced from INFO to WARNING
logger = logging.getLogger(__name__)

# Set other loggers to WARNING level to reduce noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events"""
    global feature_extractor, asl_model, feature_scaler, auth_service, db_connection
    global DATABASE_AVAILABLE, MODEL_AVAILABLE
    
    # Startup
    try:
        # Initialize database if available
        if DATABASE_AVAILABLE:
            try:
                db_connection = get_database_connection()
                if initialize_database():
                    pass  # Database initialized successfully
                auth_service = get_auth_service()
            except Exception as e:
                DATABASE_AVAILABLE = False
        
        # Initialize ML models if available
        if MODEL_AVAILABLE:
            try:
                feature_extractor = SignSpeakFeatureExtractor()
                
                # Load trained model if available
                model_path = Path(__file__).parent / "checkpoints" / "sign_language" / "best_model.pth"
                if model_path.exists():
                    try:
                        asl_model = FeatureBasedASLModel.load_model(str(model_path))
                    except Exception as e:
                        asl_model = create_default_model()
                else:
                    asl_model = create_default_model()
                    
                # Load feature scaler if available
                scaler_path = Path(__file__).parent / "models" / "feature_scaler.pkl"
                feature_scaler = load_feature_scaler(str(scaler_path))
                if feature_scaler is None:
                    feature_scaler = create_mock_scaler()
                    
            except Exception as e:
                MODEL_AVAILABLE = False
                
        print("SignSpeak API ready")  # Simple startup message
            
    except Exception as e:
        print(f"Startup error: {e}")
    
    yield
    
    # Shutdown (cleanup if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SignSpeak Professional API",
    description="Enterprise ASL Recognition API with User Management",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and services
feature_extractor = None
asl_model = None
feature_scaler = None
auth_service = None
db_connection = None

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate user authentication token"""
    if not DATABASE_AVAILABLE or not auth_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    try:
        token = credentials.credentials
        is_valid, user_data = auth_service.validate_session(token)
        
        if not is_valid or not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return user_data
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_info = {
        "status": "healthy",
        "version": "3.0.0",
        "mode": "unknown",
        "services": {
            "database": False,
            "authentication": False,
            "feature_extractor": False,
            "asl_model": False,
            "feature_scaler": False
        },
        "capabilities": []
    }
    
    # Check database health
    if DATABASE_AVAILABLE and db_connection:
        try:
            db_health = check_database_health()
            health_info["services"]["database"] = db_health.get("connected", False)
            health_info["database_info"] = db_health
            if health_info["services"]["database"]:
                health_info["capabilities"].append("user_authentication")
        except Exception as e:
            pass  # Database unavailable
    
    # Check authentication service
    if auth_service:
        health_info["services"]["authentication"] = True
        
    # Check ML services
    if MODEL_AVAILABLE:
        health_info["services"]["feature_extractor"] = feature_extractor is not None
        health_info["services"]["asl_model"] = asl_model is not None
        health_info["services"]["feature_scaler"] = feature_scaler is not None
        
        if all([feature_extractor, asl_model, feature_scaler]):
            health_info["capabilities"].append("asl_recognition")
    
    # Determine mode
    if health_info["services"]["database"] and health_info["services"]["asl_model"]:
        health_info["mode"] = "full"
    elif health_info["services"]["database"]:
        health_info["mode"] = "authentication_only"
    else:
        health_info["mode"] = "basic"
    
    return health_info

@app.get("/expressions")
async def get_expressions():
    """Get list of supported ASL expressions"""
    expressions = [
        "AMAZING", "BAD", "BEAUTIFUL", "DEAF", "FINE", "GOOD", "GOODBYE",
        "HEARING", "HELLO", "HELP", "HOW_ARE_YOU", "LEARN", "LOVE", "NAME",
        "NICE_TO_MEET_YOU", "NO", "PLEASE", "SORRY", "THANK_YOU", "UNDERSTAND", "YES"
    ]
    return {"expressions": expressions}

@app.post("/predict/video")
async def predict_from_video(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Predict ASL expression from video file (authenticated)"""
    if not feature_extractor:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")
    
    if not asl_model:
        raise HTTPException(status_code=503, detail="ASL model not loaded")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Extract features from video
        video_features = feature_extractor.extract_from_video(tmp_file_path)
        
        if not video_features.frames:
            raise HTTPException(status_code=400, detail="No features extracted from video")
        
        # Convert to array format
        features_array = video_features.to_array()
        
        # Apply scaling if available
        if feature_scaler:
            try:
                features_array = feature_scaler.transform(features_array.reshape(1, -1))[0]
            except Exception as e:
                pass  # Continue without scaling
        
        # Predict using model
        prediction = asl_model.predict_single(features_array)
        
        # Save recognition history to database
        try:
            with db_connection.get_session() as session:
                from database.models import RecognitionHistory
                
                history_entry = RecognitionHistory(
                    user_id=current_user["user_id"],
                    input_type="video",
                    filename=file.filename,
                    predicted_class=prediction["predicted_class"],
                    confidence_score=float(prediction["confidence"]),
                    processing_time=0.0,  # You can measure this if needed
                    metadata={
                        "num_frames": len(video_features.frames),
                        "probabilities": {k: float(v) for k, v in prediction["probabilities"].items()}
                    }
                )
                
                session.add(history_entry)
                session.commit()
                
        except Exception as e:
            pass  # Failed to save history
        
        return {
            "filename": file.filename,
            "prediction": prediction["predicted_class"],
            "confidence": float(prediction["confidence"]),
            "probabilities": {k: float(v) for k, v in prediction["probabilities"].items()},
            "num_frames": len(video_features.frames),
            "user": current_user["username"]
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.post("/extract/features")
async def extract_features_only(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Extract features from video without prediction (authenticated)"""
    if not feature_extractor:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Extract features from video
        video_features = feature_extractor.extract_from_video(tmp_file_path)
        
        if not video_features.frames:
            raise HTTPException(status_code=400, detail="No features extracted from video")
        
        # Return features as JSON
        return {
            "filename": file.filename,
            "num_frames": len(video_features.frames),
            "user": current_user["username"],
            "features": video_features.to_dict(),
            "feature_shape": video_features.to_array().shape
        }
        
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting features: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/model/info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "asl_model": {
            "loaded": asl_model is not None,
            "architecture": asl_model.model_type if asl_model else None,
            "num_classes": len(asl_model.class_names) if asl_model else None,
            "classes": asl_model.class_names if asl_model else None
        },
        "feature_extractor": {
            "loaded": feature_extractor is not None,
            "feature_dim": 258  # 126 hand + 132 pose
        },
        "scaler": {
            "loaded": feature_scaler is not None
        }
    }

@app.post("/auth/login")
async def login_user(request: LoginRequest):
    """User login endpoint with SQL Server authentication"""
    if not auth_service:
        raise HTTPException(status_code=503, detail="Authentication service not available")
    
    try:
        success, result = auth_service.authenticate_user(
            username=request.username,
            password=request.password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return {
            "message": "Login successful",
            "access_token": result["session_token"],
            "token_type": "bearer",
            "user": {
                "username": result["user"]["username"],
                "name": result["user"].get("full_name", request.username),
                "user_id": result["user"]["user_id"],
                "settings": result["user"].get("settings", {})
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@app.post("/auth/register")
async def register_user(request: RegisterRequest):
    """User registration endpoint with SQL Server backend"""
    if not auth_service:
        raise HTTPException(status_code=503, detail="Authentication service not available")
    
    try:
        success, result = auth_service.register_user(
            username=request.username,
            email=f"{request.username}@signspeak.local",  # Default email format
            password=request.password,
            full_name=request.name or request.username,
            profile_data={}
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result
            )
        
        return {
            "message": "User registered successfully",
            "user_id": result["user_id"]
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.get("/auth/profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """Get current user's profile from SQL Server"""
    try:
        profile = auth_service.get_user_profile(current_user["user_id"])
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        # Return safe profile data
        return {
            "username": profile.get("username"),
            "name": profile.get("full_name", profile.get("username")),
            "email": profile.get("email"),
            "created_at": profile.get("created_at"),
            "settings": profile.get("settings", {}),
            "user_id": profile.get("user_id"),
            "progress": profile.get("progress", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )

@app.get("/user/{username}")
async def get_public_user_profile(username: str):
    """Get public user profile information (legacy endpoint for compatibility)"""
    if not auth_service:
        raise HTTPException(status_code=503, detail="Authentication service not available")
    
    try:
        # Find user by username
        with db_connection.get_session() as session:
            from database.models import User
            user = session.query(User).filter(User.username == username).first()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Return safe public data
            return {
                "username": user.username,
                "name": user.full_name or username,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "public_settings": {},  # Only public settings
                "is_active": user.is_active
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Public profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )

@app.get("/auth/logout")
async def logout_user(current_user: dict = Depends(get_current_user)):
    """Logout user and invalidate session"""
    try:
        success = auth_service.logout_user(current_user.get("session_token"))
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Logout failed"
            )
        
        return {"message": "Successfully logged out"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@app.get("/auth/progress")
async def get_user_progress(current_user: dict = Depends(get_current_user)):
    """Get user's learning progress and statistics"""
    try:
        with db_connection.get_session() as session:
            from database.models import UserProgress, RecognitionHistory
            
            # Get user progress
            progress = session.query(UserProgress).filter(
                UserProgress.user_id == current_user["user_id"]
            ).first()
            
            # Get recognition history count
            history_count = session.query(RecognitionHistory).filter(
                RecognitionHistory.user_id == current_user["user_id"]
            ).count()
            
            # Get recent recognitions
            recent_recognitions = session.query(RecognitionHistory).filter(
                RecognitionHistory.user_id == current_user["user_id"]
            ).order_by(RecognitionHistory.created_at.desc()).limit(10).all()
            
            return {
                "user_id": current_user["user_id"],
                "username": current_user["username"],
                "progress": {
                    "current_level": progress.current_level if progress else 1,
                    "total_score": progress.total_score if progress else 0,
                    "achievements_count": progress.achievements_count if progress else 0,
                    "last_activity": progress.last_activity.isoformat() if progress and progress.last_activity else None
                },
                "statistics": {
                    "total_recognitions": history_count,
                    "recent_recognitions": [
                        {
                            "predicted_class": r.predicted_class,
                            "confidence": r.confidence_score,
                            "timestamp": r.created_at.isoformat(),
                            "input_type": r.input_type
                        } for r in recent_recognitions
                    ]
                }
            }
            
    except Exception as e:
        logger.error(f"Progress retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve progress"
        )

@app.get("/auth/history")
async def get_recognition_history(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get user's recognition history with pagination"""
    try:
        with db_connection.get_session() as session:
            from database.models import RecognitionHistory
            
            # Get paginated history
            history = session.query(RecognitionHistory).filter(
                RecognitionHistory.user_id == current_user["user_id"]
            ).order_by(RecognitionHistory.created_at.desc()).offset(offset).limit(limit).all()
            
            # Get total count
            total_count = session.query(RecognitionHistory).filter(
                RecognitionHistory.user_id == current_user["user_id"]
            ).count()
            
            return {
                "history": [
                    {
                        "id": r.id,
                        "predicted_class": r.predicted_class,
                        "confidence_score": r.confidence_score,
                        "input_type": r.input_type,
                        "filename": r.filename,
                        "processing_time": r.processing_time,
                        "created_at": r.created_at.isoformat(),
                        "metadata": r.metadata
                    } for r in history
                ],
                "pagination": {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": offset + limit < total_count
                }
            }
            
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve history"
        )

if __name__ == "__main__":
    try:
        import uvicorn
        print("🚀 Starting SignSpeak Professional API Server...")
        print("📍 Server will be available at: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("❌ Uvicorn not available. Install with: pip install uvicorn")
        print("🔄 You can still import this app in other ASGI servers")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")