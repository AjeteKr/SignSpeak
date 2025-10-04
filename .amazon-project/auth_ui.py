import streamlit as st
import requests
import json
import sys
import os

# Add Wave path for authentication
wave_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Wave")
sys.path.append(wave_path)

def show_login_page():
    st.title("🤟 SignSpeak - Sign Language Recognition")
    st.markdown("Please sign in or create an account to access the ASL recognition system")
    
    # Session state initialization
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None
        
    if st.session_state.logged_in:
        st.success(f"🎉 Welcome back, {st.session_state.user_data.get('name', st.session_state.username)}!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Continue to ASL Recognition", type="primary"):
                return True
        with col2:
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.user_data = None
                st.rerun()
        return True
        
    tab1, tab2 = st.tabs(["🔑 Sign In", "✨ Create Account"])
    
    with tab1:
        st.header("Sign In to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("🔑 Sign In", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please enter both username and password")
                    return False
                    
                try:
                    # Try API first, fallback to direct auth if API not available
                    response = requests.post(
                        "http://localhost:8000/login",
                        json={"username": username, "password": password},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Login successful!")
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_data = result.get("user", {})
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                        
                except requests.exceptions.ConnectionError:
                    # Fallback to direct authentication if API is not running
                    try:
                        from user_auth import UserAuthManager
                        auth_manager = UserAuthManager()
                        success, result = auth_manager.login(username, password)
                        
                        if success:
                            st.success("✅ Login successful!")
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_data = result
                            st.rerun()
                        else:
                            st.error(f"❌ {result}")
                    except Exception as e:
                        st.error(f"❌ Authentication error: {str(e)}")
                        
                except Exception as e:
                    st.error(f"❌ Login error: {str(e)}")
    
    with tab2:
        st.header("Create New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username", placeholder="Pick a unique username")
            new_name = st.text_input("Your Name", placeholder="Enter your full name (optional)")
            new_password = st.text_input("Choose Password", type="password", placeholder="Create a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            submit = st.form_submit_button("✨ Create Account", use_container_width=True)
            
            if submit:
                if not new_username or not new_password:
                    st.error("Please enter username and password")
                    return False
                    
                if new_password != confirm_password:
                    st.error("❌ Passwords don't match!")
                    return False
                    
                try:
                    # Try API first, fallback to direct auth if API not available
                    response = requests.post(
                        "http://localhost:8000/register",
                        json={
                            "username": new_username,
                            "password": new_password,
                            "name": new_name or new_username
                        },
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ Registration successful! Please sign in with your new account.")
                    else:
                        error_detail = response.json().get("detail", "Registration failed")
                        st.error(f"❌ {error_detail}")
                        
                except requests.exceptions.ConnectionError:
                    # Fallback to direct authentication if API is not running
                    try:
                        from user_auth import UserAuthManager
                        auth_manager = UserAuthManager()
                        success, message = auth_manager.register_user(new_username, new_password, new_name)
                        
                        if success:
                            st.success("✅ Registration successful! Please sign in with your new account.")
                        else:
                            st.error(f"❌ {message}")
                    except Exception as e:
                        st.error(f"❌ Registration error: {str(e)}")
                        
                except Exception as e:
                    st.error(f"❌ Registration error: {str(e)}")
    
    # Information section
    with st.expander("ℹ️ About SignSpeak"):
        st.markdown("""
        **SignSpeak** is an AI-powered American Sign Language (ASL) recognition system that:
        
        - 🎥 **Real-time Recognition**: Processes video input to recognize ASL gestures
        - 🤖 **AI-Powered**: Uses advanced machine learning for accurate predictions
        - 👤 **Personal Accounts**: Save your progress and customize settings
        - 📊 **Performance Tracking**: Monitor your recognition accuracy over time
        - 🔊 **Text-to-Speech**: Hear the recognized signs spoken aloud
        
        **Supported Signs**: HELLO, THANK_YOU, PLEASE, SORRY, GOOD, BAD, BEAUTIFUL, and many more!
        """)

    return False
