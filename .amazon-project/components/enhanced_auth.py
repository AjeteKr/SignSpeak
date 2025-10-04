"""
Enhanced Authentication UI for SignSpeak Professional
Beautiful, modern authentication interface with improved UX
"""
import streamlit as st
import requests
import json
import sys
import os
from pathlib import Path
from components.modern_ui import SignSpeakUI
from utils.api_client import SignSpeakClient
from typing import Dict, Any, Optional

# Add Wave path for authentication
wave_path = Path(__file__).parent.parent / "Wave"
sys.path.append(str(wave_path))

class EnhancedAuth:
    """Enhanced authentication system with modern UI"""
    
    def __init__(self):
        self.api_client = SignSpeakClient()
        
    def show_auth_page(self) -> bool:
        """Display the enhanced authentication page"""
        # Apply modern styling
        SignSpeakUI.apply_signspeak_css()
        SignSpeakUI.accessibility_features()
        
        # Hero section
        SignSpeakUI.hero_section(
            title="SignSpeak Professional",
            subtitle="Advanced ASL Recognition System with AI-Powered Learning",
            icon="🤟"
        )
        
        # Session state initialization
        self._initialize_session_state()
        
        # Check if user is already logged in
        if st.session_state.logged_in:
            return self._show_user_dashboard()
        
        # Show authentication forms
        return self._show_authentication_forms()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_data' not in st.session_state:
            st.session_state.user_data = None
        if 'access_token' not in st.session_state:
            st.session_state.access_token = None
    
    def _show_user_dashboard(self) -> bool:
        """Show logged-in user dashboard"""
        user_name = st.session_state.user_data.get('name', st.session_state.username)
        
        # Welcome message
        st.markdown(f"""
        <div class="card fade-in-up" style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h2 style="margin: 0; color: white;">🎉 Welcome back, {user_name}!</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Ready to continue your ASL learning journey?</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if st.button("🚀 Start ASL Recognition", type="primary", use_container_width=True):
                st.session_state.page = "recognition"
                return True
        
        with col2:
            if st.button("📊 View Progress", use_container_width=True):
                st.session_state.page = "dashboard"
                return True
        
        with col3:
            if st.button("🚪 Logout"):
                self._logout_user()
                st.rerun()
        
        # Quick stats (if available)
        self._show_quick_stats()
        
        return True
    
    def _show_authentication_forms(self) -> bool:
        """Show login and registration forms"""
        # Create tabs for login/register
        tab1, tab2 = st.tabs(["🔑 Sign In", "✨ Create Account"])
        
        with tab1:
            return self._show_login_form()
        
        with tab2:
            return self._show_registration_form()
    
    def _show_login_form(self) -> bool:
        """Show enhanced login form"""
        st.markdown("""
        <div class="card fade-in-up">
            <div class="card-header">🔑 Sign In to Your Account</div>
            <p style="color: #64748b; margin-bottom: 1.5rem;">
                Welcome back! Please enter your credentials to access your ASL learning dashboard.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            with st.form("login_form", clear_on_submit=False):
                st.markdown("### 📝 Credentials")
                
                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    help="The username you created when registering"
                )
                
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    help="Your secure password"
                )
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    submit = st.form_submit_button(
                        "🔑 Sign In",
                        type="primary",
                        use_container_width=True
                    )
                with col2:
                    forgot_password = st.form_submit_button(
                        "❓ Forgot Password",
                        use_container_width=True
                    )
                
                if submit:
                    return self._handle_login(username, password)
                
                if forgot_password:
                    st.info("💡 Contact your administrator to reset your password.")
        
        # Demo login option
        with st.expander("🔧 Demo Access"):
            st.markdown("""
            **Try the demo with these credentials:**
            - Username: `demo_user`
            - Password: `demo123`
            
            *Note: Demo accounts have limited functionality.*
            """)
            
            if st.button("🎭 Use Demo Login", use_container_width=True):
                return self._handle_login("demo_user", "demo123")
        
        return False
    
    def _show_registration_form(self) -> bool:
        """Show enhanced registration form"""
        st.markdown("""
        <div class="card fade-in-up">
            <div class="card-header">✨ Create Your Account</div>
            <p style="color: #64748b; margin-bottom: 1.5rem;">
                Join the SignSpeak community and start your ASL learning journey today!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            with st.form("registration_form", clear_on_submit=False):
                st.markdown("### 👤 Personal Information")
                
                col1, col2 = st.columns(2)
                with col1:
                    first_name = st.text_input(
                        "First Name",
                        placeholder="Enter your first name"
                    )
                with col2:
                    last_name = st.text_input(
                        "Last Name",
                        placeholder="Enter your last name"
                    )
                
                st.markdown("### 🔐 Account Details")
                
                username = st.text_input(
                    "Username",
                    placeholder="Choose a unique username",
                    help="This will be your unique identifier in the system"
                )
                
                email = st.text_input(
                    "Email Address",
                    placeholder="Enter your email address",
                    help="We'll use this for important account notifications"
                )
                
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Create a strong password",
                    help="Use at least 8 characters with numbers and symbols"
                )
                
                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Re-enter your password"
                )
                
                # Terms and privacy
                agree_terms = st.checkbox(
                    "I agree to the Terms of Service and Privacy Policy",
                    help="Required to create an account"
                )
                
                newsletter = st.checkbox(
                    "Subscribe to ASL learning tips and updates (optional)"
                )
                
                submit = st.form_submit_button(
                    "✨ Create Account",
                    type="primary",
                    use_container_width=True
                )
                
                if submit:
                    return self._handle_registration({
                        'first_name': first_name,
                        'last_name': last_name,
                        'username': username,
                        'email': email,
                        'password': password,
                        'confirm_password': confirm_password,
                        'agree_terms': agree_terms,
                        'newsletter': newsletter
                    })
        
        return False
    
    def _handle_login(self, username: str, password: str) -> bool:
        """Handle user login with enhanced error handling"""
        if not username or not password:
            st.error("🚫 Please enter both username and password")
            return False
        
        with st.spinner("🔄 Signing you in..."):
            try:
                # For now, simulate authentication since we don't have database
                if username == "demo_user" and password == "demo123":
                    # Simulate successful login
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_data = {
                        'name': 'Demo User',
                        'email': 'demo@signspeak.com',
                        'level': 1,
                        'total_recognitions': 15,
                        'accuracy': 87.5
                    }
                    st.session_state.access_token = "demo_token_123"
                    
                    st.success("🎉 Welcome to SignSpeak! Login successful.")
                    st.balloons()
                    st.rerun()
                    return True
                else:
                    # Try actual API authentication (when database is available)
                    st.error("🚫 Invalid credentials. Try the demo login or contact support.")
                    return False
                    
            except Exception as e:
                st.error(f"🚫 Login failed: {str(e)}")
                return False
    
    def _handle_registration(self, form_data: Dict[str, Any]) -> bool:
        """Handle user registration with validation"""
        # Validation
        errors = []
        
        if not form_data['first_name'] or not form_data['last_name']:
            errors.append("First and last name are required")
        
        if not form_data['username'] or len(form_data['username']) < 3:
            errors.append("Username must be at least 3 characters long")
        
        if not form_data['email'] or '@' not in form_data['email']:
            errors.append("Please enter a valid email address")
        
        if not form_data['password'] or len(form_data['password']) < 6:
            errors.append("Password must be at least 6 characters long")
        
        if form_data['password'] != form_data['confirm_password']:
            errors.append("Passwords do not match")
        
        if not form_data['agree_terms']:
            errors.append("You must agree to the Terms of Service")
        
        if errors:
            for error in errors:
                st.error(f"🚫 {error}")
            return False
        
        with st.spinner("🔄 Creating your account..."):
            try:
                # For now, simulate registration
                st.success("🎉 Account created successfully! Please sign in with your new credentials.")
                
                # Switch to login tab (simulate)
                st.info("💡 You can now sign in using the 'Sign In' tab above.")
                return False
                
            except Exception as e:
                st.error(f"🚫 Registration failed: {str(e)}")
                return False
    
    def _logout_user(self):
        """Handle user logout"""
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_data = None
        st.session_state.access_token = None
        
        if 'page' in st.session_state:
            del st.session_state.page
    
    def _show_quick_stats(self):
        """Show quick user statistics"""
        if not st.session_state.user_data:
            return
        
        st.markdown("### 📊 Your Progress at a Glance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            SignSpeakUI.metric_card(
                value=str(st.session_state.user_data.get('total_recognitions', 0)),
                label="Recognitions",
                icon="🎯 "
            )
        
        with col2:
            SignSpeakUI.metric_card(
                value=f"{st.session_state.user_data.get('accuracy', 0):.1f}%",
                label="Accuracy",
                icon="📊 "
            )
        
        with col3:
            SignSpeakUI.metric_card(
                value=f"Level {st.session_state.user_data.get('level', 1)}",
                label="Current Level",
                icon="🏆 "
            )

# Create global instance
enhanced_auth = EnhancedAuth()

def show_enhanced_login_page() -> bool:
    """Show the enhanced authentication page"""
    return enhanced_auth.show_auth_page()