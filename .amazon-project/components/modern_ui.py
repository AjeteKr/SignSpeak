"""
Modern UI Components for SignSpeak Professional
Beautiful, accessible design system for the ASL recognition application
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional

"""
SignSpeak Professional UI Design System
Beautiful, accessible design with specified color palette and typography
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional

class SignSpeakUI:
    """Professional UI component library for SignSpeak with accessible design"""
    
    # 🎨 Color Palette
    COLORS = {
        # Light Mode
        'primary_blue': '#2563EB',      # Primary Blue - calm, trustworthy
        'secondary_yellow': '#FACC15',   # Secondary Yellow - accent for key actions
        'neutral_white': '#FFFFFF',      # Clean background
        'dark_gray': '#1E293B',         # High-contrast text
        
        # Dark Mode
        'dark_navy': '#0F172A',         # Main background in dark mode
        'light_gray': '#F1F5F9',        # Text/icons in dark mode
        
        # Additional colors for status
        'success': '#10B981',
        'warning': '#F59E0B',
        'error': '#EF4444',
        'info': '#3B82F6'
    }
    
    @staticmethod
    def apply_signspeak_css():
        """Apply the complete SignSpeak design system"""
        st.markdown("""
        <style>
        /* Import Poppins Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* =========================
           GLOBAL STYLES & RESET
        ========================= */
        .stApp {
            font-family: 'Poppins', 'Inter', sans-serif;
            background: #FFFFFF;
            color: #1E293B;
            line-height: 1.6;
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            .stApp {
                background: #0F172A;
                color: #F1F5F9;
            }
        }
        
        /* =========================
           TYPOGRAPHY SYSTEM
        ========================= */
        
        /* Headings - Bold, Blue or White */
        h1, h2, h3, h4, h5, h6, .signspeak-heading {
            font-family: 'Poppins', sans-serif;
            font-weight: 700;
            color: #2563EB;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        
        @media (prefers-color-scheme: dark) {
            h1, h2, h3, h4, h5, h6, .signspeak-heading {
                color: #F1F5F9;
            }
        }
        
        h1, .hero-title {
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        
        h2, .section-title {
            font-size: 2.25rem;
            font-weight: 700;
        }
        
        h3, .subsection-title {
            font-size: 1.875rem;
            font-weight: 600;
        }
        
        /* Body text - 16-18px */
        p, .body-text, .stMarkdown {
            font-size: 18px;
            color: #1E293B;
            line-height: 1.7;
            margin-bottom: 1rem;
        }
        
        @media (prefers-color-scheme: dark) {
            p, .body-text, .stMarkdown {
                color: #F1F5F9;
            }
        }
        
        /* =========================
           HERO SECTION
        ========================= */
        .signspeak-hero {
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
            color: #FFFFFF;
            border-radius: 20px;
            margin-bottom: 3rem;
            box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .hero-subtitle {
            font-size: 1.25rem;
            font-weight: 400;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
        }
        
        /* =========================
           CARD COMPONENTS
        ========================= */
        .signspeak-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(37, 99, 235, 0.1);
            transition: all 0.3s ease;
        }
        
        .signspeak-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.15);
        }
        
        @media (prefers-color-scheme: dark) {
            .signspeak-card {
                background: #1E293B;
                border-color: rgba(241, 245, 249, 0.1);
            }
        }
        
        .card-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2563EB;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        @media (prefers-color-scheme: dark) {
            .card-header {
                color: #F1F5F9;
            }
        }
        
        /* =========================
           BUTTON SYSTEM
        ========================= */
        
        /* Primary Blue Buttons */
        .stButton > button, .btn-primary {
            background: #2563EB !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1rem 2rem !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
            cursor: pointer !important;
        }
        
        .stButton > button:hover, .btn-primary:hover {
            background: #1D4ED8 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4) !important;
        }
        
        .stButton > button:active, .btn-primary:active {
            transform: translateY(0) !important;
        }
        
        /* Secondary Yellow Buttons */
        .btn-secondary {
            background: #FACC15 !important;
            color: #1E293B !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1rem 2rem !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(250, 204, 21, 0.3) !important;
            cursor: pointer !important;
        }
        
        .btn-secondary:hover {
            background: #EAB308 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(250, 204, 21, 0.4) !important;
        }
        
        /* =========================
           STATUS INDICATORS
        ========================= */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 16px;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        
        .status-success {
            background: #10B981;
            color: #FFFFFF;
        }
        
        .status-warning {
            background: #F59E0B;
            color: #FFFFFF;
        }
        
        .status-error {
            background: #EF4444;
            color: #FFFFFF;
        }
        
        .status-info {
            background: #2563EB;
            color: #FFFFFF;
        }
        
        /* =========================
           FORM ELEMENTS
        ========================= */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea > div > div > textarea {
            border: 2px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            font-family: 'Poppins', sans-serif !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            background: #FFFFFF !important;
            color: #1E293B !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
            outline: none !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .stTextInput > div > div > input,
            .stSelectbox > div > div > select,
            .stTextArea > div > div > textarea {
                background: #1E293B !important;
                color: #F1F5F9 !important;
                border-color: #374151 !important;
            }
        }
        
        /* =========================
           METRICS & PROGRESS
        ========================= */
        .metric-card {
            background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%);
            color: #FFFFFF;
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.2);
            transition: all 0.3s ease;
            margin: 1rem 0;
        }
        
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(37, 99, 235, 0.3);
        }
        
        .metric-value {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .metric-label {
            font-size: 1rem;
            font-weight: 500;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Progress Bar */
        .progress-container {
            background: #E2E8F0;
            border-radius: 50px;
            height: 24px;
            margin: 1.5rem 0;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(135deg, #2563EB 0%, #FACC15 100%);
            border-radius: 50px;
            transition: width 0.6s ease;
            box-shadow: 0 2px 10px rgba(37, 99, 235, 0.3);
        }
        
        .progress-text {
            text-align: center;
            font-weight: 600;
            color: #1E293B;
            margin-top: 0.5rem;
        }
        
        @media (prefers-color-scheme: dark) {
            .progress-container {
                background: #374151;
            }
            .progress-text {
                color: #F1F5F9;
            }
        }
        
        /* =========================
           SIDEBAR STYLING
        ========================= */
        .css-1d391kg {
            background: rgba(255, 255, 255, 0.98) !important;
            backdrop-filter: blur(10px) !important;
            border-right: 2px solid rgba(37, 99, 235, 0.1) !important;
        }
        
        @media (prefers-color-scheme: dark) {
            .css-1d391kg {
                background: rgba(30, 41, 59, 0.98) !important;
                border-right-color: rgba(241, 245, 249, 0.1) !important;
            }
        }
        
        /* =========================
           ACCESSIBILITY FEATURES
        ========================= */
        
        /* High contrast focus indicators */
        button:focus,
        input:focus,
        select:focus,
        textarea:focus,
        .stSelectbox:focus-within,
        .stTextInput:focus-within {
            outline: 3px solid #FACC15 !important;
            outline-offset: 2px !important;
        }
        
        /* Screen reader only content */
        .sr-only {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        
        /* =========================
           ANIMATIONS
        ========================= */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .fade-in-up {
            animation: fadeInUp 0.6s ease-out;
        }
        
        .slide-in-left {
            animation: slideInLeft 0.5s ease-out;
        }
        
        /* =========================
           RESPONSIVE DESIGN
        ========================= */
        @media (max-width: 768px) {
            .hero-title {
                font-size: 2.5rem;
            }
            
            .signspeak-card {
                padding: 1.5rem;
                margin: 1rem 0;
            }
            
            .metric-value {
                font-size: 2rem;
            }
        }
        
        /* =========================
           STREAMLIT OVERRIDES
        ========================= */
        
        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Improve spacing */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #F1F5F9;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #2563EB;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #1D4ED8;
        }
        
        /* =========================
           UTILITY CLASSES
        ========================= */
        .text-center { text-align: center; }
        .text-left { text-align: left; }
        .text-right { text-align: right; }
        
        .mb-1 { margin-bottom: 0.5rem; }
        .mb-2 { margin-bottom: 1rem; }
        .mb-3 { margin-bottom: 1.5rem; }
        .mb-4 { margin-bottom: 2rem; }
        
        .mt-1 { margin-top: 0.5rem; }
        .mt-2 { margin-top: 1rem; }
        .mt-3 { margin-top: 1.5rem; }
        .mt-4 { margin-top: 2rem; }
        
        .p-1 { padding: 0.5rem; }
        .p-2 { padding: 1rem; }
        .p-3 { padding: 1.5rem; }
        .p-4 { padding: 2rem; }
        
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def hero_section(title: str, subtitle: str, icon: str = "🤟"):
        """Create a beautiful hero section with SignSpeak branding"""
        st.markdown(f"""
        <div class="signspeak-hero fade-in-up">
            <div class="hero-title">{icon} {title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div class="sr-only">Welcome to SignSpeak Professional ASL Recognition System</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def status_card(title: str, status: str, message: str, icon: str = ""):
        """Create a high-contrast status indicator card"""
        status_class = {
            'success': 'status-success',
            'warning': 'status-warning', 
            'error': 'status-error',
            'info': 'status-info'
        }.get(status, 'status-info')
        
        st.markdown(f"""
        <div class="signspeak-card fade-in-up">
            <div class="card-header" role="heading" aria-level="3">{icon} {title}</div>
            <div class="status-indicator {status_class}" role="status" aria-live="polite">
                {message}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(value: str, label: str, icon: str = ""):
        """Create a beautiful metric display card"""
        st.markdown(f"""
        <div class="metric-card fade-in-up" role="region" aria-label="{label} metric">
            <div class="metric-value" aria-label="{value}">{icon}{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def progress_bar(progress: float, label: str = "", max_value: float = 100.0):
        """Create an accessible animated progress bar"""
        percentage = min(max(progress, 0), max_value)
        
        st.markdown(f"""
        <div class="fade-in-up">
            {f'<div class="mb-2" style="font-weight: 600; color: #2563EB;">{label}</div>' if label else ''}
            <div class="progress-container" role="progressbar" 
                 aria-valuenow="{percentage}" 
                 aria-valuemin="0" 
                 aria-valuemax="{max_value}"
                 aria-label="{label if label else 'Progress'}">
                <div class="progress-bar" style="width: {percentage}%"></div>
            </div>
            <div class="progress-text">{percentage:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def feature_card(title: str, description: str, icon: str, action_text: str = "Try Now"):
        """Create a feature showcase card with accessibility"""
        return st.markdown(f"""
        <div class="signspeak-card fade-in-up" role="article">
            <div class="card-header" role="heading" aria-level="3">{icon} {title}</div>
            <p style="color: #1E293B; margin-bottom: 2rem; line-height: 1.7; font-size: 18px;">{description}</p>
            <div style="text-align: center;">
                <button class="btn-secondary" role="button" aria-label="{action_text} for {title}">
                    {action_text}
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def primary_button(text: str, key: str = None, full_width: bool = False):
        """Create a primary blue button"""
        return st.button(
            text, 
            key=key, 
            use_container_width=full_width,
            help=f"Click to {text.lower()}"
        )
    
    @staticmethod
    def secondary_button(text: str, key: str = None, full_width: bool = False):
        """Create a secondary yellow button"""
        # Since Streamlit doesn't directly support custom button classes,
        # we'll use markdown for now
        button_id = key or text.replace(" ", "_").lower()
        st.markdown(f"""
        <button class="btn-secondary" 
                onclick="document.querySelector('[data-testid=\\"{button_id}\\"]').click()"
                style="width: {'100%' if full_width else 'auto'};"
                aria-label="{text}">
            {text}
        </button>
        """, unsafe_allow_html=True)
        
        # Hidden Streamlit button for functionality
        return st.button(text, key=f"hidden_{button_id}", help=f"Click to {text.lower()}")
    
    @staticmethod
    def accessibility_features():
        """Add comprehensive accessibility enhancements"""
        st.markdown("""
        <div class="sr-only">
            <h1>SignSpeak Professional ASL Recognition System</h1>
            <p>This application provides American Sign Language recognition capabilities with user authentication and progress tracking. Navigate using Tab key, activate with Enter or Space.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add skip link
        st.markdown("""
        <a href="#main-content" class="sr-only" style="position: absolute; top: 0; left: 0; background: #2563EB; color: white; padding: 1rem; text-decoration: none; z-index: 9999;">
            Skip to main content
        </a>
        """, unsafe_allow_html=True)
        

class DashboardComponents:
    """Dashboard-specific UI components"""
    
    @staticmethod
    def recognition_stats(stats: Dict[str, Any]):
        """Display recognition statistics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            SignSpeakUI.metric_card(
                value=str(stats.get('total_recognitions', 0)),
                label="Total Recognitions",
                icon="🎯 "
            )
        
        with col2:
            SignSpeakUI.metric_card(
                value=f"{stats.get('accuracy', 0):.1f}%",
                label="Average Accuracy",
                icon="📊 "
            )
        
        with col3:
            SignSpeakUI.metric_card(
                value=str(stats.get('words_learned', 0)),
                label="Words Learned",
                icon="📚 "
            )
        
        with col4:
            SignSpeakUI.metric_card(
                value=f"Level {stats.get('current_level', 1)}",
                label="Current Level",
                icon="🏆 "
            )
    
    @staticmethod
    def recent_activity(activities: List[Dict[str, Any]]):
        """Display recent activity feed"""
        st.markdown('<div class="card fade-in-up">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">📋 Recent Activity</div>', unsafe_allow_html=True)
        
        if not activities:
            st.markdown('<p style="color: #64748b; text-align: center; padding: 2rem;">No recent activity</p>', unsafe_allow_html=True)
        else:
            for activity in activities[:5]:  # Show last 5 activities
                confidence = activity.get('confidence', 0)
                confidence_color = '#10b981' if confidence > 0.8 else '#f59e0b' if confidence > 0.6 else '#ef4444'
                
                st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    background: #f8fafc;
                    border-radius: 10px;
                    border-left: 4px solid {confidence_color};
                ">
                    <div>
                        <strong>{activity.get('predicted_class', 'Unknown')}</strong>
                        <div style="font-size: 0.9rem; color: #64748b;">
                            {activity.get('timestamp', 'Unknown time')}
                        </div>
                    </div>
                    <div style="
                        background: {confidence_color};
                        color: white;
                        padding: 0.25rem 0.75rem;
                        border-radius: 15px;
                        font-size: 0.8rem;
                        font-weight: 500;
                    ">
                        {confidence:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)