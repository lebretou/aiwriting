# Per-Claim Uncertainty Estimation Chat Prototype

A minimal prototype for testing **sentence-level uncertainty quantification** in LLM responses with controlled fault injection.

## Features

- 💬 **Interactive chat interface** built with Gradio
- 🎯 **Per-sentence uncertainty estimation** using:
  - Model-side confidence (token log probabilities)
  - Evidence-side confidence (NLI with DeBERTa-V3-base-MNLI)
  - Fused confidence (weighted combination)
- 🔴🟡🟢 **Inline color-coded rendering** of confidence levels
- 🧪 **Controlled fault injection** via switchable context files
- 📊 **Real NLI models** (microsoft/deberta-v3-base-mnli)

## Installation

1. **Clone or create the project directory**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

4. **Set OpenAI API key** (optional - app works with mock data without it):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

5. **Generate evidence database:**
```bash
python create_evidence.py
```

## Usage

1. **Launch the app:**
```bash
python app.py
```

2. **Open browser** at `http://127.0.0.1:7860`

3. **Select context:**
   - `truth_context`: Factually correct Kyoto Protocol information
   - `fault_context`: Modified version with injected errors

4. **Ask questions** about the Kyoto Protocol

5. **Observe confidence scores:**
   - Green (≥0.75): High confidence
   - Yellow (0.5-0.75): Medium confidence
   - Red (<0.5): Low confidence

6. **Adjust α slider** to change model vs evidence weight

## Project Structure

```
uncertainty-chat-app/
├── app.py                    # Main Gradio interface
├── utils.py                  # Helper functions for uncertainty computation
├── create_evidence.py        # Script to generate evidence database
├── requirements.txt          # Python dependencies
├── contexts/
│   ├── truth_context.txt     # Factually correct context
│   └── fault_context.txt     # Fault-injected context
└── evidence/
    └── evidence_sentences.json  # Truth sentences for NLI verification
```

## How It Works

### 1. Model-Side Uncertainty
- Extracts token-level log probabilities from LLM API
- Computes average probability per sentence
- Higher probability = higher model confidence

### 2. Evidence-Side Uncertainty
For each sentence claim:
1. Embed claim and all evidence sentences
2. Retrieve top-5 most similar evidence sentences (cosine similarity)
3. Run NLI model (DeBERTa-V3-MNLI) on each pair
4. Extract entailment and contradiction scores
5. Compute: `confidence = max(entailment) × (1 - max(contradiction))`

### 3. Fusion
Combine both signals:
```
fused_conf = α × model_conf + (1-α) × evidence_conf
```

Default: α = 0.3 (weights evidence more heavily)

## Testing Protocol

1. **With truth context:**
   - Ask: "When did the Kyoto Protocol enter into force?"
   - Expected: High confidence, correct answer (2005)

2. **With fault context:**
   - Ask same question
   - Expected: Lower confidence, incorrect answer (2003)

3. **Compare confidence distributions** between contexts

4. **Validate that evidence-side uncertainty** catches contradictions

## Example Output

```
User: When did the Kyoto Protocol enter into force?