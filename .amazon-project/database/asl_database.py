"""
Comprehensive Database System for ASL Recognition
Handles ASL alphabet data, user profiles, calibration data, and performance metrics
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ASLLetterData:
    """ASL letter data entry"""
    id: str
    letter: str
    image_path: Optional[str]
    landmarks: np.ndarray
    features: Dict[str, float]
    source: str  # 'mnist', 'asl_alphabet', 'user_generated'
    quality_score: float
    created_at: str
    metadata: Dict[str, Any]

@dataclass
class UserSession:
    """User session data"""
    session_id: str
    user_id: str
    start_time: str
    end_time: Optional[str]
    total_duration: float
    letters_detected: int
    words_completed: int
    accuracy_score: float
    performance_metrics: Dict[str, float]
    errors: List[Dict]
    settings_used: Dict[str, Any]

@dataclass
class CalibrationData:
    """User calibration data"""
    user_id: str
    letter: str
    sample_data: List[Dict]
    average_features: np.ndarray
    confidence_threshold: float
    quality_scores: List[float]
    calibration_date: str

class ASLDatabaseManager:
    """
    Comprehensive database manager for ASL recognition system
    Handles all data storage, retrieval, and analysis
    """
    
    def __init__(self, db_path: str = "asl_database.db"):
        """Initialize database manager"""
        self.db_path = Path(db_path)
        self.connection = None
        
        # Initialize database
        self._init_database()
        
        # Create indexes for performance
        self._create_indexes()
    
    def _init_database(self):
        """Initialize all database tables"""
        with self._get_connection() as conn:
            # ASL alphabet data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS asl_letters (
                    id TEXT PRIMARY KEY,
                    letter TEXT NOT NULL,
                    image_path TEXT,
                    landmarks BLOB,
                    features BLOB NOT NULL,
                    source TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT,
                    CHECK (letter IN ('A','B','C','D','E','F','G','H','I','J','K','L','M',
                                     'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'))
                )
            """)
            
            # User profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    full_name TEXT NOT NULL,
                    email TEXT,
                    created_at TEXT NOT NULL,
                    last_login TEXT,
                    total_sessions INTEGER DEFAULT 0,
                    total_letters INTEGER DEFAULT 0,
                    total_words INTEGER DEFAULT 0,
                    average_accuracy REAL DEFAULT 0.0,
                    hand_size_factor REAL DEFAULT 1.0,
                    preferred_settings TEXT,
                    calibration_complete BOOLEAN DEFAULT FALSE,
                    profile_image TEXT,
                    notes TEXT
                )
            """)
            
            # User sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_duration REAL DEFAULT 0.0,
                    letters_detected INTEGER DEFAULT 0,
                    words_completed INTEGER DEFAULT 0,
                    accuracy_score REAL DEFAULT 0.0,
                    performance_metrics TEXT,
                    errors TEXT,
                    settings_used TEXT,
                    session_type TEXT DEFAULT 'practice',
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            # Calibration data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calibration_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    letter TEXT NOT NULL,
                    sample_index INTEGER NOT NULL,
                    sample_data TEXT NOT NULL,
                    features BLOB NOT NULL,
                    quality_score REAL NOT NULL,
                    calibration_date TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id),
                    UNIQUE(user_id, letter, sample_index)
                )
            """)
            
            # Performance metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    metric_type TEXT NOT NULL,  -- 'accuracy', 'speed', 'consistency', etc.
                    measurement_time TEXT NOT NULL,
                    context TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id),
                    FOREIGN KEY (session_id) REFERENCES user_sessions (session_id)
                )
            """)
            
            # Recognition events table (detailed logging)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recognition_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    detected_letter TEXT,
                    actual_letter TEXT,
                    confidence REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    features BLOB,
                    frame_data TEXT,
                    is_correct BOOLEAN,
                    FOREIGN KEY (session_id) REFERENCES user_sessions (session_id),
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            """)
            
            # Model performance table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    test_date TEXT NOT NULL,
                    dataset_size INTEGER NOT NULL,
                    overall_accuracy REAL NOT NULL,
                    per_class_accuracy TEXT NOT NULL,  -- JSON
                    confusion_matrix TEXT NOT NULL,    -- JSON
                    training_params TEXT,              -- JSON
                    validation_metrics TEXT,           -- JSON
                    notes TEXT
                )
            """)
            
            # Dataset statistics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    total_samples INTEGER NOT NULL,
                    samples_per_class TEXT NOT NULL,  -- JSON
                    quality_distribution TEXT,        -- JSON
                    source_distribution TEXT,         -- JSON
                    created_at TEXT NOT NULL,
                    file_path TEXT,
                    checksum TEXT
                )
            """)
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        with self._get_connection() as conn:
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_asl_letters_letter ON asl_letters (letter)",
                "CREATE INDEX IF NOT EXISTS idx_asl_letters_source ON asl_letters (source)",
                "CREATE INDEX IF NOT EXISTS idx_asl_letters_quality ON asl_letters (quality_score)",
                "CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions (user_id)",
                "CREATE INDEX IF NOT EXISTS idx_user_sessions_date ON user_sessions (start_time)",
                "CREATE INDEX IF NOT EXISTS idx_calibration_user_letter ON calibration_data (user_id, letter)",
                "CREATE INDEX IF NOT EXISTS idx_performance_user_metric ON performance_metrics (user_id, metric_name)",
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_session ON recognition_events (session_id)",
                "CREATE INDEX IF NOT EXISTS idx_recognition_events_timestamp ON recognition_events (timestamp)"
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(index_sql)
                except sqlite3.Error as e:
                    logger.warning(f"Index creation failed: {e}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    # ASL Letter Data Management
    def add_asl_letter_data(self, letter_data: ASLLetterData) -> bool:
        """Add ASL letter data to database"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO asl_letters 
                    (id, letter, image_path, landmarks, features, source, 
                     quality_score, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    letter_data.id,
                    letter_data.letter,
                    letter_data.image_path,
                    pickle.dumps(letter_data.landmarks) if letter_data.landmarks is not None else None,
                    pickle.dumps(letter_data.features),
                    letter_data.source,
                    letter_data.quality_score,
                    letter_data.created_at,
                    json.dumps(letter_data.metadata)
                ))
            return True
        except Exception as e:
            logger.error(f"Error adding ASL letter data: {e}")
            return False
    
    def get_asl_letters_by_letter(self, letter: str, limit: Optional[int] = None) -> List[ASLLetterData]:
        """Get all data for specific letter"""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT * FROM asl_letters 
                    WHERE letter = ? 
                    ORDER BY quality_score DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query, (letter,))
                rows = cursor.fetchall()
                
                return [self._row_to_asl_letter_data(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving ASL letters: {e}")
            return []
    
    def get_asl_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        try:
            with self._get_connection() as conn:
                # Total counts by letter
                cursor = conn.execute("""
                    SELECT letter, COUNT(*) as count, AVG(quality_score) as avg_quality
                    FROM asl_letters 
                    GROUP BY letter 
                    ORDER BY letter
                """)
                letter_stats = {row[0]: {'count': row[1], 'avg_quality': row[2]} 
                               for row in cursor.fetchall()}
                
                # Source distribution
                cursor = conn.execute("""
                    SELECT source, COUNT(*) as count
                    FROM asl_letters 
                    GROUP BY source
                """)
                source_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Quality distribution
                cursor = conn.execute("""
                    SELECT 
                        CASE 
                            WHEN quality_score >= 0.9 THEN 'excellent'
                            WHEN quality_score >= 0.8 THEN 'good'
                            WHEN quality_score >= 0.7 THEN 'fair'
                            ELSE 'poor'
                        END as quality_tier,
                        COUNT(*) as count
                    FROM asl_letters 
                    GROUP BY quality_tier
                """)
                quality_stats = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    'letter_distribution': letter_stats,
                    'source_distribution': source_stats,
                    'quality_distribution': quality_stats,
                    'total_samples': sum(source_stats.values()),
                    'generated_at': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting dataset statistics: {e}")
            return {}
    
    # User Management
    def create_user_profile(self, username: str, full_name: str, email: str = None) -> Optional[str]:
        """Create new user profile"""
        try:
            user_id = str(uuid.uuid4())
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO user_profiles 
                    (user_id, username, full_name, email, created_at, last_login)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    username,
                    full_name,
                    email,
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))
            return user_id
        except sqlite3.IntegrityError:
            logger.error(f"Username '{username}' already exists")
            return None
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return None
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile by ID"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM user_profiles WHERE user_id = ?
                """, (user_id,))
                row = cursor.fetchone()
                
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    profile = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    if profile['preferred_settings']:
                        profile['preferred_settings'] = json.loads(profile['preferred_settings'])
                    
                    return profile
                return None
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    def update_user_stats(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Update user statistics after session"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE user_profiles 
                    SET 
                        total_sessions = total_sessions + 1,
                        total_letters = total_letters + ?,
                        total_words = total_words + ?,
                        average_accuracy = (
                            (average_accuracy * total_sessions + ?) / (total_sessions + 1)
                        ),
                        last_login = ?
                    WHERE user_id = ?
                """, (
                    session_data.get('letters_detected', 0),
                    session_data.get('words_completed', 0),
                    session_data.get('accuracy_score', 0.0),
                    datetime.now().isoformat(),
                    user_id
                ))
            return True
        except Exception as e:
            logger.error(f"Error updating user stats: {e}")
            return False
    
    # Session Management
    def start_user_session(self, user_id: str, session_type: str = 'practice') -> str:
        """Start new user session"""
        try:
            session_id = str(uuid.uuid4())
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO user_sessions 
                    (session_id, user_id, start_time, session_type)
                    VALUES (?, ?, ?, ?)
                """, (
                    session_id,
                    user_id,
                    datetime.now().isoformat(),
                    session_type
                ))
            return session_id
        except Exception as e:
            logger.error(f"Error starting user session: {e}")
            return ""
    
    def end_user_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """End user session with final data"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE user_sessions 
                    SET 
                        end_time = ?,
                        total_duration = ?,
                        letters_detected = ?,
                        words_completed = ?,
                        accuracy_score = ?,
                        performance_metrics = ?,
                        errors = ?
                    WHERE session_id = ?
                """, (
                    datetime.now().isoformat(),
                    session_data.get('total_duration', 0.0),
                    session_data.get('letters_detected', 0),
                    session_data.get('words_completed', 0),
                    session_data.get('accuracy_score', 0.0),
                    json.dumps(session_data.get('performance_metrics', {})),
                    json.dumps(session_data.get('errors', [])),
                    session_id
                ))
            return True
        except Exception as e:
            logger.error(f"Error ending user session: {e}")
            return False
    
    def get_user_sessions(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user session history"""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM user_sessions 
                    WHERE user_id = ? 
                    ORDER BY start_time DESC 
                    LIMIT ?
                """, (user_id, limit))
                
                columns = [desc[0] for desc in cursor.description]
                sessions = []
                
                for row in cursor.fetchall():
                    session = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    for field in ['performance_metrics', 'errors', 'settings_used']:
                        if session[field]:
                            session[field] = json.loads(session[field])
                    
                    sessions.append(session)
                
                return sessions
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []
    
    # Calibration Data Management
    def save_calibration_data(self, user_id: str, letter: str, 
                            sample_data: List[Dict], features: np.ndarray) -> bool:
        """Save user calibration data"""
        try:
            with self._get_connection() as conn:
                # Clear existing calibration for this letter
                conn.execute("""
                    UPDATE calibration_data 
                    SET is_active = FALSE 
                    WHERE user_id = ? AND letter = ?
                """, (user_id, letter))
                
                # Add new calibration samples
                for idx, sample in enumerate(sample_data):
                    conn.execute("""
                        INSERT INTO calibration_data 
                        (user_id, letter, sample_index, sample_data, features, 
                         quality_score, calibration_date, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id,
                        letter,
                        idx,
                        json.dumps(sample),
                        pickle.dumps(features),
                        sample.get('quality', 0.8),
                        datetime.now().isoformat(),
                        True
                    ))
            return True
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")
            return False
    
    def get_user_calibration(self, user_id: str, letter: str = None) -> Dict[str, List[Dict]]:
        """Get user calibration data"""
        try:
            with self._get_connection() as conn:
                query = """
                    SELECT * FROM calibration_data 
                    WHERE user_id = ? AND is_active = TRUE
                """
                params = [user_id]
                
                if letter:
                    query += " AND letter = ?"
                    params.append(letter)
                
                query += " ORDER BY letter, sample_index"
                
                cursor = conn.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                
                calibration_data = {}
                for row in cursor.fetchall():
                    data = dict(zip(columns, row))
                    letter_key = data['letter']
                    
                    if letter_key not in calibration_data:
                        calibration_data[letter_key] = []
                    
                    # Parse data
                    data['sample_data'] = json.loads(data['sample_data'])
                    data['features'] = pickle.loads(data['features'])
                    
                    calibration_data[letter_key].append(data)
                
                return calibration_data
        except Exception as e:
            logger.error(f"Error getting calibration data: {e}")
            return {}
    
    # Performance Analytics
    def record_performance_metric(self, user_id: str, session_id: str, 
                                 metric_name: str, metric_value: float, 
                                 metric_type: str, context: str = None) -> bool:
        """Record performance metric"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (user_id, session_id, metric_name, metric_value, 
                     metric_type, measurement_time, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    session_id,
                    metric_name,
                    metric_value,
                    metric_type,
                    datetime.now().isoformat(),
                    context
                ))
            return True
        except Exception as e:
            logger.error(f"Error recording performance metric: {e}")
            return False
    
    def get_performance_analytics(self, user_id: str, 
                                time_window: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        try:
            since_date = (datetime.now() - timedelta(days=time_window)).isoformat()
            
            with self._get_connection() as conn:
                # Accuracy trends
                cursor = conn.execute("""
                    SELECT DATE(measurement_time) as date, AVG(metric_value) as avg_accuracy
                    FROM performance_metrics 
                    WHERE user_id = ? AND metric_type = 'accuracy' 
                    AND measurement_time >= ?
                    GROUP BY DATE(measurement_time)
                    ORDER BY date
                """, (user_id, since_date))
                accuracy_trend = [{'date': row[0], 'accuracy': row[1]} 
                                for row in cursor.fetchall()]
                
                # Speed metrics
                cursor = conn.execute("""
                    SELECT AVG(metric_value) as avg_speed, MAX(metric_value) as max_speed
                    FROM performance_metrics 
                    WHERE user_id = ? AND metric_type = 'speed' 
                    AND measurement_time >= ?
                """, (user_id, since_date))
                speed_stats = cursor.fetchone()
                
                # Letter-specific accuracy
                cursor = conn.execute("""
                    SELECT context as letter, AVG(metric_value) as accuracy
                    FROM performance_metrics 
                    WHERE user_id = ? AND metric_type = 'letter_accuracy' 
                    AND measurement_time >= ?
                    AND context IS NOT NULL
                    GROUP BY context
                    ORDER BY accuracy DESC
                """, (user_id, since_date))
                letter_accuracy = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    'accuracy_trend': accuracy_trend,
                    'speed_stats': {
                        'average': speed_stats[0] if speed_stats[0] else 0,
                        'maximum': speed_stats[1] if speed_stats[1] else 0
                    },
                    'letter_accuracy': letter_accuracy,
                    'time_window_days': time_window,
                    'generated_at': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting performance analytics: {e}")
            return {}
    
    def _row_to_asl_letter_data(self, row: Tuple) -> ASLLetterData:
        """Convert database row to ASLLetterData object"""
        return ASLLetterData(
            id=row[0],
            letter=row[1],
            image_path=row[2],
            landmarks=pickle.loads(row[3]) if row[3] else None,
            features=pickle.loads(row[4]),
            source=row[5],
            quality_score=row[6],
            created_at=row[7],
            metadata=json.loads(row[8]) if row[8] else {}
        )
    
    # Data Export/Import
    def export_user_data(self, user_id: str, export_path: str) -> bool:
        """Export all user data to file"""
        try:
            user_data = {
                'profile': self.get_user_profile(user_id),
                'sessions': self.get_user_sessions(user_id, limit=1000),
                'calibration': self.get_user_calibration(user_id),
                'performance': self.get_performance_analytics(user_id, time_window=365),
                'export_date': datetime.now().isoformat()
            }
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            user_data = convert_numpy(user_data)
            
            with open(export_path, 'w') as f:
                json.dump(user_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error exporting user data: {e}")
            return False
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get comprehensive database summary"""
        try:
            with self._get_connection() as conn:
                summary = {}
                
                # Table sizes
                tables = ['asl_letters', 'user_profiles', 'user_sessions', 
                         'calibration_data', 'performance_metrics', 'recognition_events']
                
                for table in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    summary[f"{table}_count"] = cursor.fetchone()[0]
                
                # Database file size
                summary['database_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
                # Most active users
                cursor = conn.execute("""
                    SELECT username, total_sessions, average_accuracy
                    FROM user_profiles 
                    ORDER BY total_sessions DESC 
                    LIMIT 5
                """)
                summary['top_users'] = [
                    {'username': row[0], 'sessions': row[1], 'accuracy': row[2]}
                    for row in cursor.fetchall()
                ]
                
                summary['generated_at'] = datetime.now().isoformat()
                
                return summary
        except Exception as e:
            logger.error(f"Error getting database summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """Clean up old data to maintain performance"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            with self._get_connection() as conn:
                # Clean old recognition events
                cursor = conn.execute("""
                    DELETE FROM recognition_events 
                    WHERE timestamp < ?
                """, (cutoff_date,))
                
                deleted_events = cursor.rowcount
                
                # Clean old inactive calibration data
                cursor = conn.execute("""
                    DELETE FROM calibration_data 
                    WHERE calibration_date < ? AND is_active = FALSE
                """, (cutoff_date,))
                
                deleted_calibration = cursor.rowcount
                
                logger.info(f"Cleaned {deleted_events} recognition events and {deleted_calibration} calibration records")
                
                # Vacuum database to reclaim space
                conn.execute("VACUUM")
                
                return True
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False


# Utility functions for database management
def initialize_database(db_path: str = "asl_database.db") -> ASLDatabaseManager:
    """Initialize and return database manager"""
    return ASLDatabaseManager(db_path)

def import_dataset_to_db(dataset_path: str, db_manager: ASLDatabaseManager, 
                        source: str = "imported") -> bool:
    """Import dataset from file to database"""
    try:
        # This would implement dataset import logic
        # Based on the specific format of your datasets
        logger.info(f"Importing dataset from {dataset_path}")
        
        # Implementation would depend on dataset format
        # For now, return success placeholder
        return True
    except Exception as e:
        logger.error(f"Error importing dataset: {e}")
        return False


if __name__ == "__main__":
    # Test database functionality
    db = ASLDatabaseManager("test_asl.db")
    
    # Create test user
    user_id = db.create_user_profile("testuser", "Test User", "test@example.com")
    if user_id:
        print(f"Created user: {user_id}")
        
        # Get summary
        summary = db.get_database_summary()
        print("Database summary:", summary)