import streamlit as st
import json
import os

st.set_page_config(
    page_title="SignSpeak - Login",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Apply custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        height: 3em;
        margin-top: 20px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    div.css-1v0mbdj.etr89bj1 {
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Simple user database
USERS = {
    "demo": {
        "password": "demo",
        "name": "Demo User"
    }
}

def login_page():
    st.title("🤟 SignSpeak")
    
    # Initialize session state
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None

    # Create two columns for layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("""
            <h3 style='margin-bottom: 20px;'>Welcome to SignSpeak</h3>
            <p style='color: #666; margin-bottom: 30px;'>
            Your AI-powered American Sign Language translator. 
            Log in to start translating signs in real-time.
            </p>
        """, unsafe_allow_html=True)

    # Login form
    with st.container():
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in USERS and USERS[username]["password"] == password:
                # Reset any existing session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Set new session state
                st.session_state.authentication_status = True
                st.session_state.username = username
                st.session_state.name = USERS[username]["name"]
                
                st.success(f'Welcome back, {USERS[username]["name"]}!')
                st.switch_page("Home.py")  # Redirect to main app
            else:
                st.error('Username/password is incorrect')

        if st.session_state.authentication_status == False:
            st.error('Username/password is incorrect')
        elif st.session_state.authentication_status == None:
            st.warning('Please enter your username and password')
        elif st.session_state.authentication_status:
            st.success(f'Welcome back, {st.session_state.name}!')
            st.switch_page("Home.py")

    # Demo account info
    with st.expander("🔑 Demo Account"):
        st.write("Username: demo")
        st.write("Password: demo")

    # Footer
    st.markdown("""
        <div style='position: fixed; bottom: 20px; left: 0; right: 0; text-align: center;'>
            <p style='color: #666;'>© 2025 SignSpeak. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    login_page()
