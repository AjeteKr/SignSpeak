"""
Database Connection Management
Handles SQL Server connections, session management, and database operations
"""
import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import time

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Manages database connections and provides session handling"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._connection_established = False
        self._initialize()
    
    def _get_connection_string(self) -> str:
        """Generate connection string based on environment"""
        # Check for SQL Server configuration
        server = os.getenv('SQL_SERVER', 'localhost')
        database = os.getenv('SQL_DATABASE', 'SignSpeakDB')
        username = os.getenv('SQL_USERNAME', 'signspeak_user')
        password = os.getenv('SQL_PASSWORD', 'SignSpeak2025!')
        
        # Try different SQL Server connection methods
        connection_strings = [
            # Windows Authentication
            f'mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes',
            # SQL Server Authentication
            f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server',
            # Alternative driver
            f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=SQL+Server',
            # SQLite fallback
            'sqlite:///signspeak.db'
        ]
        
        return connection_strings
    
    def _initialize(self):
        """Initialize database connection with fallback options"""
        connection_strings = self._get_connection_string()
        
        for i, conn_str in enumerate(connection_strings):
            try:
                logger.info(f"Attempting database connection {i+1}/{len(connection_strings)}...")
                
                if 'sqlite' in conn_str:
                    # SQLite configuration
                    self.engine = create_engine(
                        conn_str,
                        echo=False,
                        connect_args={"check_same_thread": False}
                    )
                    logger.info("🔄 Using SQLite database (development mode)")
                else:
                    # SQL Server configuration
                    self.engine = create_engine(
                        conn_str,
                        echo=False,
                        pool_size=5,
                        max_overflow=10,
                        pool_pre_ping=True,
                        pool_recycle=3600,
                        connect_args={
                            "timeout": 30,
                            "autocommit": False
                        }
                    )
                
                # Test connection
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
                
                # Create session factory
                self.SessionLocal = sessionmaker(
                    autocommit=False,
                    autoflush=False,
                    bind=self.engine
                )
                
                self._connection_established = True
                
                if 'sqlite' in conn_str:
                    logger.info("✅ SQLite database connected successfully")
                else:
                    logger.info("✅ SQL Server database connected successfully")
                break
                
            except Exception as e:
                logger.warning(f"Connection attempt {i+1} failed: {e}")
                continue
        
        if not self._connection_established:
            raise Exception("❌ All database connection attempts failed")
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self._connection_established:
            raise Exception("Database not connected")
        return self.SessionLocal()
    
    @contextmanager
    def get_session_context(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Test if database connection is working"""
        try:
            with self.get_session_context() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def get_health_status(self) -> dict:
        """Get database health status for monitoring"""
        status = {
            "connected": self._connection_established,
            "engine_info": str(self.engine.url) if self.engine else None,
            "pool_status": None,
            "last_check": time.time()
        }
        
        if self.engine and hasattr(self.engine.pool, 'status'):
            try:
                pool = self.engine.pool
                status["pool_status"] = {
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                }
            except Exception as e:
                logger.warning(f"Could not get pool status: {e}")
        
        return status
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self._connection_established = False
            logger.info("Database connection closed")

# Database Operations Helper
class DatabaseOperations:
    """Helper class for common database operations"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def execute_query(self, query: str, params: dict = None) -> list:
        """Execute raw SQL query"""
        try:
            with self.db.get_session_context() as session:
                result = session.execute(text(query), params or {})
                return result.fetchall()
        except SQLAlchemyError as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_update(self, query: str, params: dict = None) -> int:
        """Execute update/insert/delete query"""
        try:
            with self.db.get_session_context() as session:
                result = session.execute(text(query), params or {})
                return result.rowcount
        except SQLAlchemyError as e:
            logger.error(f"Update execution failed: {e}")
            raise
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        try:
            query = """
            SELECT COUNT(*) as count
            FROM information_schema.tables 
            WHERE table_name = :table_name
            """
            result = self.execute_query(query, {"table_name": table_name})
            return result[0][0] > 0 if result else False
        except Exception:
            # Fallback for SQLite
            try:
                query = """
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=:table_name
                """
                result = self.execute_query(query, {"table_name": table_name})
                return len(result) > 0
            except Exception as e:
                logger.error(f"Table existence check failed: {e}")
                return False

# Global database instance
_db_connection: Optional[DatabaseConnection] = None

def get_database_connection() -> DatabaseConnection:
    """Get global database connection instance"""
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection()
    return _db_connection

def get_db_session() -> Session:
    """Get database session for dependency injection"""
    return get_database_connection().get_session()

@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Get database session with context manager"""
    db_conn = get_database_connection()
    with db_conn.get_session_context() as session:
        yield session

def close_database_connection():
    """Close global database connection"""
    global _db_connection
    if _db_connection:
        _db_connection.close()
        _db_connection = None

# Database health check function
def check_database_health() -> dict:
    """Check database health status"""
    try:
        db_conn = get_database_connection()
        return db_conn.get_health_status()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "connected": False,
            "error": str(e),
            "last_check": time.time()
        }

if __name__ == "__main__":
    # Test database connection
    try:
        db_conn = get_database_connection()
        print("✅ Database connection test successful!")
        print(f"Health status: {db_conn.get_health_status()}")
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")