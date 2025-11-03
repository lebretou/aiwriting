"""Script to extract evidence sentences from truth context"""
import spacy
import json

# Load spacy model
nlp = spacy.load("en_core_web_sm")

# Read truth context
with open("contexts/truth_context.txt", "r") as f:
    truth_text = f.read()

# Split into sentences
doc = nlp(truth_text)
evidence_sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

# Save to JSON
with open("evidence/evidence_sentences.json", "w") as f:
    json.dump(evidence_sentences, f, indent=2)

print(f"Created evidence database with {len(evidence_sentences)} sentences")
print("\nFirst 5 sentences:")
for i, sent in enumerate(evidence_sentences[:5], 1):
    print(f"{i}. {sent}")
