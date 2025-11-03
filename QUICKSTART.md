# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
cd uncertainty-chat-app
bash setup.sh
```

This will:
- Install all Python packages
- Download the spaCy English model
- Generate the evidence sentence database

### Step 2: (Optional) Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Note:** The app works without an API key using mock responses for testing!

### Step 3: Launch the App
```bash
python app.py
```

Open your browser at: **http://127.0.0.1:7860**

## 🧪 Test Your Setup

Before running the app, verify everything is installed:
```bash
python test_setup.py
```

## 💡 Usage Tips

### Try These Questions:
1. "When did the Kyoto Protocol enter into force?"
2. "What were the emission reduction targets?"
3. "Did the United States ratify the protocol?"
4. "What happened with Canada?"

### Compare Contexts:
1. **Select "truth_context"** → Ask a question → Note the confidence scores
2. **Switch to "fault_context"** → Ask the same question → Compare scores
3. **Observe:** Lower confidence on incorrect facts!

### Adjust α (Alpha) Slider:
- **α = 0.0**: Pure evidence-based confidence
- **α = 0.3** (default): Balanced (30% model, 70% evidence)
- **α = 1.0**: Pure model confidence (ignore evidence)

## 📊 Understanding the Output

Each sentence is color-coded:
- 🟢 **Green (0.75-1.0)**: High confidence - likely correct
- 🟡 **Yellow (0.5-0.75)**: Medium confidence - uncertain
- 🔴 **Red (0.0-0.5)**: Low confidence - likely incorrect or contradicted

Hover over sentences to see exact confidence values!

## 🔧 Troubleshooting

### Import Errors
```bash
pip install -r requirements.txt
```

### spaCy Model Missing
```bash
python -m spacy download en_core_web_sm
```

### Evidence Database Missing
```bash
python create_evidence.py
```

### CUDA/GPU Errors
The app works fine on CPU. If you have GPU issues, the models will automatically fall back to CPU.

## 📁 Project Files

```
uncertainty-chat-app/
├── app.py                      # Main Gradio app (run this!)
├── utils.py                    # Uncertainty computation functions
├── create_evidence.py          # Generate evidence database
├── test_setup.py              # Test your installation
├── setup.sh                   # One-command setup
├── requirements.txt           # Dependencies
├── README.md                  # Full documentation
├── QUICKSTART.md             # This file
├── contexts/
│   ├── truth_context.txt     # Correct facts
│   └── fault_context.txt     # Injected errors
└── evidence/
    └── evidence_sentences.json  # Generated truth database
```

## 🎯 What to Expect

### With Truth Context:
```
Q: When did the Kyoto Protocol enter into force?
A: The Kyoto Protocol entered into force on February 16, 2005. (0.87) ← Green!
```

### With Fault Context:
```
Q: When did the Kyoto Protocol enter into force?
A: The Kyoto Protocol entered into force on January 1, 2003. (0.34) ← Red!
```

The fault-injected context contains wrong information, and the uncertainty system correctly flags it!

## 🧠 How It Works (Simple Version)

For each sentence the LLM generates:

1. **Model Confidence**: How certain is the LLM based on its token probabilities?
2. **Evidence Confidence**: Does this match or contradict the truth documents?
   - Uses DeBERTa NLI model to check entailment vs contradiction
   - High entailment + low contradiction = high confidence
3. **Fused Confidence**: Combines both signals with weighted average

## 📚 Next Steps

1. **Test with your own contexts**: Edit files in `contexts/` folder
2. **Try different questions**: See how confidence varies
3. **Experiment with α values**: Find the best fusion weight
4. **Add more evidence**: Expand `truth_context.txt` and regenerate evidence DB

## 🆘 Need Help?

- Check **README.md** for detailed documentation
- Run `python test_setup.py` to diagnose issues
- Ensure all dependencies are installed via `setup.sh`

Happy testing! 🎉
