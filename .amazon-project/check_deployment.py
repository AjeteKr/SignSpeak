"""
Deployment Readiness Check
Verifies that SignSpeak Pro is ready for production deployment
"""

from pathlib import Path
import importlib
import sys

def check_deployment_readiness():
    """Check if the application is ready for deployment"""
    
    print("🔍 Checking SignSpeak Pro deployment readiness...")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # Check 1: Configuration system
    print("1️⃣ Checking configuration system...")
    try:
        from config.settings import config
        print(f"   ✅ Configuration loaded")
        print(f"   📍 ASL images path: {config.get_asl_images_path()}")
        print(f"   🏗️ Development mode: {config.is_development}")
    except ImportError as e:
        issues.append("Configuration system not available")
        print(f"   ❌ Configuration system not available: {e}")
    
    # Check 2: ASL reference images
    print("\n2️⃣ Checking ASL reference images...")
    try:
        from data_processing.asl_reference_loader import ASLReferenceDatasetLoader
        loader = ASLReferenceDatasetLoader()
        
        if loader.images_folder.exists():
            images, labels, letter_names = loader.load_reference_images()
            if len(images) > 0:
                print(f"   ✅ Found {len(images)} ASL reference images")
                print(f"   📁 Location: {loader.images_folder}")
                print(f"   🔤 Letters: {', '.join(letter_names[:10])}{'...' if len(letter_names) > 10 else ''}")
            else:
                issues.append("No ASL reference images found")
                print(f"   ❌ No images loaded from {loader.images_folder}")
        else:
            issues.append(f"ASL images folder not found: {loader.images_folder}")
            print(f"   ❌ Images folder not found: {loader.images_folder}")
    except Exception as e:
        issues.append(f"ASL loader error: {e}")
        print(f"   ❌ ASL loader error: {e}")
    
    # Check 3: Core dependencies
    print("\n3️⃣ Checking core dependencies...")
    required_packages = [
        'streamlit',
        'opencv-python',
        'mediapipe',
        'numpy',
        'scikit-learn'
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            issues.append(f"Missing required package: {package}")
            print(f"   ❌ Missing: {package}")
    
    # Check 4: Optional dependencies
    print("\n4️⃣ Checking optional dependencies...")
    optional_packages = {
        'tensorflow': 'Advanced CNN models',
        'matplotlib': 'Training visualizations',
        'seaborn': 'Enhanced plots'
    }
    
    for package, purpose in optional_packages.items():
        try:
            importlib.import_module(package)
            print(f"   ✅ {package} - {purpose}")
        except ImportError:
            warnings.append(f"Optional package missing: {package} ({purpose})")
            print(f"   ⚠️  Missing: {package} - {purpose}")
    
    # Check 5: Directory structure
    print("\n5️⃣ Checking directory structure...")
    required_dirs = [
        'models',
        'data',
        'config',
        'data_processing'
    ]
    
    app_root = Path(__file__).parent
    for dir_name in required_dirs:
        dir_path = app_root / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/")
        else:
            issues.append(f"Missing directory: {dir_name}")
            print(f"   ❌ Missing: {dir_name}/")
    
    # Check 6: Main application
    print("\n6️⃣ Checking main application...")
    try:
        main_app = app_root / "signspeak_pro.py"
        if main_app.exists():
            print(f"   ✅ Main application found")
            
            # Try to import without running
            spec = importlib.util.spec_from_file_location("signspeak_pro", main_app)
            if spec:
                print(f"   ✅ Application structure valid")
            else:
                warnings.append("Could not validate application structure")
                print(f"   ⚠️  Could not validate application structure")
        else:
            issues.append("Main application file not found")
            print(f"   ❌ signspeak_pro.py not found")
    except Exception as e:
        warnings.append(f"Application check error: {e}")
        print(f"   ⚠️  Application check error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DEPLOYMENT READINESS SUMMARY")
    print("=" * 50)
    
    if not issues:
        print("🎉 READY FOR DEPLOYMENT!")
        print("✅ All critical requirements met")
        
        if warnings:
            print(f"\n⚠️  {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"   • {warning}")
            print("\nNote: Warnings won't prevent deployment but may limit functionality.")
        
        print("\n🚀 Deployment commands:")
        print("   Local: streamlit run signspeak_pro.py")
        print("   Docker: python prepare_deployment.py && docker build -t signspeak-pro .")
        
    else:
        print("❌ NOT READY FOR DEPLOYMENT")
        print(f"🔧 {len(issues)} issue(s) must be fixed:")
        for issue in issues:
            print(f"   • {issue}")
        
        if warnings:
            print(f"\n⚠️  {len(warnings)} additional warning(s):")
            for warning in warnings:
                print(f"   • {warning}")
        
        print("\n💡 Run 'python prepare_deployment.py' to fix common issues.")
    
    return len(issues) == 0

if __name__ == "__main__":
    import importlib.util
    success = check_deployment_readiness()
    sys.exit(0 if success else 1)