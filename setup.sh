#!/bin/bash
# Setup script for Per-Claim Uncertainty Chat

echo "🔧 Setting up Per-Claim Uncertainty Chat Prototype..."

# Install requirements
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Download spaCy model
echo "📥 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Generate evidence database
echo "🗄️  Generating evidence sentence database..."
python create_evidence.py

echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  python app.py"
echo ""
echo "Optional: Set OPENAI_API_KEY for real LLM responses:"
echo "  export OPENAI_API_KEY='your-key-here'"
