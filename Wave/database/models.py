"""
SQL Server Database Configuration for SignSpeak
Production-ready database setup with user management, progress tracking, and analytics
"""
import os
import sqlalchemy as sa
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Database Configuration
class DatabaseConfig:
    """Database configuration for SQL Server"""
    
    # SQL Server connection settings
    SERVER = os.getenv('SQL_SERVER', 'localhost')
    DATABASE = os.getenv('SQL_DATABASE', 'SignSpeakDB')
    USERNAME = os.getenv('SQL_USERNAME', 'signspeak_user')
    PASSWORD = os.getenv('SQL_PASSWORD', 'SignSpeak2025!')
    DRIVER = 'ODBC Driver 17 for SQL Server'
    
    # Connection string for SQL Server
    CONNECTION_STRING = f'mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}?driver={DRIVER.replace(" ", "+")}'
    
    # Fallback to SQLite for development
    SQLITE_CONNECTION = 'sqlite:///signspeak_dev.db'

# Database Base Model
Base = declarative_base()

# User Management Models
class User(Base):
    """User account information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    progress = relationship("UserProgress", back_populates="user")
    recognition_history = relationship("RecognitionHistory", back_populates="user")
    achievements = relationship("UserAchievement", back_populates="user")

class UserProfile(Base):
    """Extended user profile information"""
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Personal information
    date_of_birth = Column(DateTime)
    phone_number = Column(String(20))
    country = Column(String(50))
    timezone = Column(String(50), default='UTC')
    
    # Accessibility preferences
    preferred_language = Column(String(10), default='en')
    high_contrast_mode = Column(Boolean, default=False)
    large_text_mode = Column(Boolean, default=False)
    voice_enabled = Column(Boolean, default=True)
    voice_speed = Column(Float, default=1.0)
    voice_gender = Column(String(10), default='female')
    
    # Learning preferences
    skill_level = Column(String(20), default='beginner')  # beginner, intermediate, advanced
    learning_goals = Column(Text)
    daily_practice_goal = Column(Integer, default=30)  # minutes
    
    # Profile customization
    avatar_url = Column(String(255))
    bio = Column(Text)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="profile")

class UserProgress(Base):
    """Track user learning progress across different modules"""
    __tablename__ = 'user_progress'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Module information
    module_type = Column(String(20), nullable=False)  # asl_alphabet, mnist, wlasl
    module_item = Column(String(50), nullable=False)  # specific letter, digit, or word
    
    # Progress metrics
    attempts_count = Column(Integer, default=0)
    successful_attempts = Column(Integer, default=0)
    best_accuracy = Column(Float, default=0.0)
    average_accuracy = Column(Float, default=0.0)
    total_practice_time = Column(Integer, default=0)  # seconds
    
    # Streak tracking
    current_streak = Column(Integer, default=0)
    best_streak = Column(Integer, default=0)
    last_practice_date = Column(DateTime)
    
    # Mastery status
    is_mastered = Column(Boolean, default=False)
    mastery_date = Column(DateTime)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="progress")

class RecognitionHistory(Base):
    """Store ASL recognition attempts and results"""
    __tablename__ = 'recognition_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Recognition details
    recognition_type = Column(String(20), nullable=False)  # live_camera, video_upload
    input_type = Column(String(20), nullable=False)  # alphabet, word, sentence
    
    # Results
    predicted_result = Column(String(255), nullable=False)
    confidence_score = Column(Float, nullable=False)
    actual_result = Column(String(255))  # if known (training mode)
    is_correct = Column(Boolean)
    
    # Metadata
    processing_time = Column(Float)  # milliseconds
    num_frames_processed = Column(Integer)
    session_id = Column(String(50))  # group related recognitions
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="recognition_history")

class UserAchievement(Base):
    """Track user achievements and milestones"""
    __tablename__ = 'user_achievements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Achievement details
    achievement_type = Column(String(50), nullable=False)
    achievement_name = Column(String(100), nullable=False)
    achievement_description = Column(Text)
    
    # Progress
    current_progress = Column(Integer, default=0)
    target_progress = Column(Integer, nullable=False)
    is_completed = Column(Boolean, default=False)
    completion_date = Column(DateTime)
    
    # Rewards
    points_awarded = Column(Integer, default=0)
    badge_icon = Column(String(50))
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="achievements")

class UserSession(Base):
    """Track user sessions for analytics"""
    __tablename__ = 'user_sessions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Session information
    session_token = Column(String(255), unique=True, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    
    # Session tracking
    login_time = Column(DateTime, default=func.now())
    last_activity = Column(DateTime, default=func.now())
    logout_time = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Activity metrics
    pages_visited = Column(Integer, default=0)
    actions_performed = Column(Integer, default=0)
    time_spent = Column(Integer, default=0)  # seconds

# Database Manager Class
class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, use_sqlite_fallback=True):
        self.use_sqlite_fallback = use_sqlite_fallback
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            # Try SQL Server first
            self.engine = create_engine(
                DatabaseConfig.CONNECTION_STRING,
                echo=False,  # Set to True for SQL debugging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            
            logger.info("✅ Connected to SQL Server database")
            
        except Exception as sql_error:
            logger.warning(f"SQL Server connection failed: {sql_error}")
            
            if self.use_sqlite_fallback:
                # Fallback to SQLite for development
                self.engine = create_engine(
                    DatabaseConfig.SQLITE_CONNECTION,
                    echo=False,
                    connect_args={"check_same_thread": False}
                )
                logger.info("✅ Using SQLite fallback database")
            else:
                raise sql_error
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created successfully")
        except Exception as e:
            logger.error(f"❌ Error creating tables: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()

# Global database manager instance
db_manager = DatabaseManager()

# Dependency for getting database session
def get_db_session():
    """Dependency for getting database session in FastAPI"""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()

# Initialize database on import
def initialize_database():
    """Initialize database tables"""
    try:
        db_manager.create_tables()
        logger.info("🎉 SignSpeak database initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    # Initialize database when run directly
    initialize_database()