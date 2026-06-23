"""
Verification script to check if the LLM validation framework is set up correctly.
"""

import os
import sys

def check_env_file():
    """Check if .env file exists and has required variables."""
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("   → Copy .env.example to .env and add your API key")
        return False

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')

    if not api_key or api_key == 'your_api_key_here':
        print("❌ OPENAI_API_KEY not set or using placeholder value")
        print("   → Edit .env and add your actual API key")
        return False

    if not base_url:
        print("❌ OPENAI_BASE_URL not set")
        print("   → Edit .env and add the base URL")
        return False

    print("✓ Environment variables configured")
    return True


def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'openai',
        'pandas',
        'numpy',
        'dotenv',
        'sklearn',
        'tiktoken',
    ]

    missing = []
    for package in required_packages:
        try:
            if package == 'dotenv':
                __import__('dotenv')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"❌ {package} not found")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Install missing packages: pip install {' '.join(missing)}")
        return False

    return True


def check_utils():
    """Check if utility modules can be imported."""
    try:
        from utils.test_backtranslation import test_backtranslation
        from utils.test_separation import test_separation
        from utils.test_validity import test_validity
        from utils.llm_api_calls import llm_completion, llm_embedding
        print("✓ All utility modules can be imported")
        return True
    except ImportError as e:
        print(f"❌ Error importing utilities: {e}")
        return False


def test_api_connection():
    """Test if API connection works."""
    try:
        from utils.llm_api_calls import llm_completion

        print("\nTesting API connection...")
        response = llm_completion(
            model="gpt-3.5-turbo",
            temperature=0,
            user_prompt="Say 'Hello' in one word."
        )

        if response and response.choices:
            print("✓ API connection successful")
            return True
        else:
            print("❌ API returned unexpected response")
            return False

    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False


def main():
    print("=" * 60)
    print("LLM VALIDATION FRAMEWORK - SETUP VERIFICATION")
    print("=" * 60)
    print()

    checks = [
        ("Dependencies", check_dependencies),
        ("Utility Modules", check_utils),
        ("Environment Variables", check_env_file),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 60)
        result = check_func()
        results.append(result)

    # Only test API if all previous checks passed
    if all(results):
        print(f"\n{'API Connection'}:")
        print("-" * 60)
        api_result = test_api_connection()
        results.append(api_result)

    print("\n" + "=" * 60)
    if all(results):
        print("✅ All checks passed! You're ready to use the framework.")
        print("\nRun the example:")
        print("  python example_sentiment_news.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
