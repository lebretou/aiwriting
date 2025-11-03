"""Test script to verify the setup is correct"""
import os
import sys

def test_files():
    """Check if all required files exist"""
    required_files = [
        "app.py",
        "utils.py",
        "create_evidence.py",
        "requirements.txt",
        "contexts/truth_context.txt",
        "contexts/fault_context.txt",
    ]

    print("📁 Checking required files...")
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - NOT FOUND")
            all_good = False

    return all_good

def test_imports():
    """Test if key imports work"""
    print("\n📦 Testing imports...")

    imports = [
        ("gradio", "Gradio"),
        ("spacy", "spaCy"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
    ]

    all_good = True
    for module, name in imports:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            all_good = False

    return all_good

def test_spacy_model():
    """Test if spaCy model is downloaded"""
    print("\n🔤 Testing spaCy model...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("  ✅ en_core_web_sm model loaded")
        return True
    except OSError:
        print("  ❌ en_core_web_sm model not found")
        print("     Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"  ❌ Error loading spaCy: {e}")
        return False

def test_evidence_db():
    """Check if evidence database exists"""
    print("\n🗄️  Checking evidence database...")
    if os.path.exists("evidence/evidence_sentences.json"):
        import json
        with open("evidence/evidence_sentences.json") as f:
            data = json.load(f)
        print(f"  ✅ Evidence database exists ({len(data)} sentences)")
        return True
    else:
        print("  ❌ Evidence database not found")
        print("     Run: python create_evidence.py")
        return False

def main():
    print("🧪 Testing Per-Claim Uncertainty Chat Setup\n")
    print("=" * 50)

    results = []
    results.append(("Files", test_files()))
    results.append(("Imports", test_imports()))
    results.append(("spaCy Model", test_spacy_model()))
    results.append(("Evidence DB", test_evidence_db()))

    print("\n" + "=" * 50)
    print("\n📊 Summary:")

    all_passed = all(result for _, result in results)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    if all_passed:
        print("\n🎉 All tests passed! Ready to run the app.")
        print("\nNext steps:")
        print("  1. (Optional) Set OPENAI_API_KEY environment variable")
        print("  2. Run: python app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nTo install dependencies, run:")
        print("  bash setup.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())
