"""
User Authentication Service with SQL Server Integration
Professional user management system for SignSpeak application
"""
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import logging
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

# Import database models
from database.models import User, UserProfile, UserSession
from database.connections import get_db_context

logger = logging.getLogger(__name__)

class AuthenticationService:
    """Professional authentication service with SQL Server backend"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = "HS256"
        self.token_expire_hours = 24
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt, hash_value = stored_hash.split(":", 1)
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return password_hash == hash_value
        except ValueError:
            return False
    
    def _generate_token(self, user_id: int, username: str) -> str:
        """Generate JWT token for user session"""
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=self.token_expire_hours),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def _verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def register_user(self, 
                     username: str, 
                     email: str, 
                     password: str, 
                     first_name: str, 
                     last_name: str,
                     profile_data: Dict[str, Any] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Register a new user with profile information
        
        Returns:
            (success, message, user_data)
        """
        try:
            with get_db_context() as session:
                # Check if username or email already exists
                existing_user = session.query(User).filter(
                    (User.username == username) | (User.email == email)
                ).first()
                
                if existing_user:
                    if existing_user.username == username:
                        return False, "Username already exists", None
                    else:
                        return False, "Email already registered", None
                
                # Create new user
                password_hash = self._hash_password(password)
                new_user = User(
                    username=username,
                    email=email,
                    password_hash=password_hash,
                    first_name=first_name,
                    last_name=last_name,
                    is_active=True,
                    is_verified=False,
                    created_at=datetime.utcnow()
                )
                
                session.add(new_user)
                session.flush()  # Get the user ID
                
                # Create user profile
                profile_defaults = {
                    "preferred_language": "en",
                    "high_contrast_mode": False,
                    "large_text_mode": False,
                    "voice_enabled": True,
                    "voice_speed": 1.0,
                    "voice_gender": "female",
                    "skill_level": "beginner",
                    "daily_practice_goal": 30
                }
                
                if profile_data:
                    profile_defaults.update(profile_data)
                
                new_profile = UserProfile(
                    user_id=new_user.id,
                    **profile_defaults
                )
                
                session.add(new_profile)
                session.commit()
                
                # Return user data
                user_data = {
                    "id": new_user.id,
                    "username": new_user.username,
                    "email": new_user.email,
                    "first_name": new_user.first_name,
                    "last_name": new_user.last_name,
                    "created_at": new_user.created_at.isoformat(),
                    "profile": profile_defaults
                }
                
                logger.info(f"✅ User registered successfully: {username}")
                return True, "User registered successfully", user_data
                
        except IntegrityError as e:
            logger.error(f"Database integrity error during registration: {e}")
            return False, "Registration failed due to data conflict", None
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False, f"Registration failed: {str(e)}", None
    
    def login_user(self, username: str, password: str, ip_address: str = None, user_agent: str = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Authenticate user and create session
        
        Returns:
            (success, message, session_data)
        """
        try:
            with get_db_context() as session:
                # Find user by username or email
                user = session.query(User).filter(
                    (User.username == username) | (User.email == username)
                ).first()
                
                if not user:
                    return False, "Invalid username or password", None
                
                if not user.is_active:
                    return False, "Account is deactivated", None
                
                # Verify password
                if not self._verify_password(password, user.password_hash):
                    return False, "Invalid username or password", None
                
                # Update last login
                user.last_login = datetime.utcnow()
                
                # Create session token
                token = self._generate_token(user.id, user.username)
                
                # Create user session record
                user_session = UserSession(
                    user_id=user.id,
                    session_token=token,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    login_time=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                    is_active=True
                )
                
                session.add(user_session)
                
                # Get user profile
                profile = session.query(UserProfile).filter(UserProfile.user_id == user.id).first()
                
                session.commit()
                
                # Prepare session data
                session_data = {
                    "token": token,
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "last_login": user.last_login.isoformat(),
                        "profile": {
                            "preferred_language": profile.preferred_language if profile else "en",
                            "high_contrast_mode": profile.high_contrast_mode if profile else False,
                            "large_text_mode": profile.large_text_mode if profile else False,
                            "voice_enabled": profile.voice_enabled if profile else True,
                            "voice_speed": profile.voice_speed if profile else 1.0,
                            "voice_gender": profile.voice_gender if profile else "female",
                            "skill_level": profile.skill_level if profile else "beginner",
                            "daily_practice_goal": profile.daily_practice_goal if profile else 30
                        } if profile else {}
                    }
                }
                
                logger.info(f"✅ User logged in successfully: {username}")
                return True, "Login successful", session_data
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False, f"Login failed: {str(e)}", None
    
    def validate_session(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate user session token
        
        Returns:
            (is_valid, user_data)
        """
        try:
            # Verify JWT token
            payload = self._verify_token(token)
            if not payload:
                return False, None
            
            with get_db_context() as session:
                # Check if session exists and is active
                user_session = session.query(UserSession).filter(
                    UserSession.session_token == token,
                    UserSession.is_active == True
                ).first()
                
                if not user_session:
                    return False, None
                
                # Update last activity
                user_session.last_activity = datetime.utcnow()
                
                # Get user data
                user = session.query(User).filter(User.id == payload["user_id"]).first()
                if not user or not user.is_active:
                    return False, None
                
                # Get user profile
                profile = session.query(UserProfile).filter(UserProfile.user_id == user.id).first()
                
                session.commit()
                
                user_data = {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "profile": {
                        "preferred_language": profile.preferred_language if profile else "en",
                        "high_contrast_mode": profile.high_contrast_mode if profile else False,
                        "large_text_mode": profile.large_text_mode if profile else False,
                        "voice_enabled": profile.voice_enabled if profile else True,
                        "voice_speed": profile.voice_speed if profile else 1.0,
                        "voice_gender": profile.voice_gender if profile else "female",
                        "skill_level": profile.skill_level if profile else "beginner"
                    } if profile else {}
                }
                
                return True, user_data
                
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False, None
    
    def logout_user(self, token: str) -> bool:
        """Logout user and deactivate session"""
        try:
            with get_db_context() as session:
                user_session = session.query(UserSession).filter(
                    UserSession.session_token == token,
                    UserSession.is_active == True
                ).first()
                
                if user_session:
                    user_session.is_active = False
                    user_session.logout_time = datetime.utcnow()
                    session.commit()
                    logger.info("✅ User logged out successfully")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def update_user_profile(self, user_id: int, profile_updates: Dict[str, Any]) -> Tuple[bool, str]:
        """Update user profile information"""
        try:
            with get_db_context() as session:
                profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
                
                if not profile:
                    return False, "User profile not found"
                
                # Update profile fields
                for key, value in profile_updates.items():
                    if hasattr(profile, key):
                        setattr(profile, key, value)
                
                profile.updated_at = datetime.utcnow()
                session.commit()
                
                logger.info(f"✅ Profile updated for user {user_id}")
                return True, "Profile updated successfully"
                
        except Exception as e:
            logger.error(f"Profile update error: {e}")
            return False, f"Profile update failed: {str(e)}"
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information by ID"""
        try:
            with get_db_context() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return None
                
                profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
                
                return {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "profile": {
                        "preferred_language": profile.preferred_language,
                        "high_contrast_mode": profile.high_contrast_mode,
                        "large_text_mode": profile.large_text_mode,
                        "voice_enabled": profile.voice_enabled,
                        "voice_speed": profile.voice_speed,
                        "voice_gender": profile.voice_gender,
                        "skill_level": profile.skill_level,
                        "daily_practice_goal": profile.daily_practice_goal,
                        "bio": profile.bio,
                        "avatar_url": profile.avatar_url
                    } if profile else {}
                }
                
        except Exception as e:
            logger.error(f"Get user error: {e}")
            return None

# Global authentication service instance
auth_service = AuthenticationService()

def get_auth_service() -> AuthenticationService:
    """Get authentication service instance"""
    return auth_service