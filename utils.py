"""Utility functions for per-claim uncertainty estimation"""
import math
import numpy as np
import torch
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Global models - will be initialized once
nlp = None
embedder = None
nli_pipe = None
evidence_sentences = None
evidence_embeddings = None

def initialize_models():
    """Initialize all models and load evidence data"""
    global nlp, embedder, nli_pipe, evidence_sentences, evidence_embeddings

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading sentence embedder...")
    embedder = SentenceTransformer("all-mpnet-base-v2")

    print("Loading NLI model (DeBERTa-V3-base-MNLI)...")
    nli_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-deberta-v3-base")
    nli_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-base")

    # Determine device
    device = 0 if torch.cuda.is_available() else -1
    nli_pipe = pipeline(
        "text-classification",
        model=nli_model,
        tokenizer=nli_tokenizer,
        return_all_scores=True,
        device=device
    )

    print("Loading evidence sentences...")
    import json
    with open("evidence/evidence_sentences.json", "r") as f:
        evidence_sentences = json.load(f)

    print("Computing evidence embeddings...")
    evidence_embeddings = embedder.encode(evidence_sentences, normalize_embeddings=True)

    print("Initialization complete!")

def split_sentences(text):
    """Split text into sentences using spaCy"""
    if nlp is None:
        raise RuntimeError("Models not initialized. Call initialize_models() first.")

    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 3]

def get_model_confidence(logprobs):
    """
    Compute model confidence from token log probabilities.

    Args:
        logprobs: List of log probabilities for tokens

    Returns:
        Float confidence score between 0 and 1
    """
    if not logprobs or len(logprobs) == 0:
        return 0.5  # Default neutral confidence if no logprobs

    avg_logprob = np.mean(logprobs)
    # Convert to probability space
    prob = math.exp(avg_logprob)
    # Clip to [0, 1] range
    return float(np.clip(prob, 0.0, 1.0))

def evidence_confidence(claim, topk=1):
    """
    Compute evidence-side confidence for a claim using NLI.

    Args:
        claim: Text claim to verify
        topk: Number of top similar evidence sentences to use

    Returns:
        Tuple of (confidence, max_entail, max_contradict, top_match_sentence, top_match_similarity)
    """
    if embedder is None or nli_pipe is None or evidence_sentences is None:
        raise RuntimeError("Models not initialized. Call initialize_models() first.")

    # Embed the claim
    claim_emb = embedder.encode(claim, normalize_embeddings=True)

    # Compute cosine similarities
    cos_scores = util.cos_sim(claim_emb, evidence_embeddings)[0]

    # Get top-k most similar evidence sentences
    top_results = torch.topk(cos_scores, k=min(topk, len(evidence_sentences)))
    top_idx = top_results.indices.tolist()
    top_sims = top_results.values.tolist()

    # Store the top-1 match
    top_match_idx = top_idx[0]
    top_match_sentence = evidence_sentences[top_match_idx]
    top_match_similarity = float(top_sims[0])

    # Run NLI on each retrieved evidence sentence
    S = 0.0  # Max entailment score
    X = 0.0  # Max contradiction score

    for idx in top_idx:
        premise = evidence_sentences[idx]

        # DeBERTa NLI format: premise + hypothesis
        nli_input = f"{premise} [SEP] {claim}"
        scores = nli_pipe(nli_input)[0]
        # scores = nli_pipe((premise, claim))[0]

        # Extract entailment and contradiction probabilities
        entail_score = 0.0
        contradict_score = 0.0

        for score_dict in scores:
            label = score_dict["label"].lower()
            if "entail" in label:
                entail_score = score_dict["score"]
            elif "contra" in label:
                contradict_score = score_dict["score"]

        S = max(S, entail_score)
        X = max(X, contradict_score)

    # Fuse: confidence = entailment * (1 - contradiction)
    confidence = S * (1 - X)

    return float(confidence), float(S), float(X), top_match_sentence, top_match_similarity

def fuse_confidence(model_conf, evidence_conf, alpha=0.3):
    """
    Fuse model-side and evidence-side confidence scores.

    Args:
        model_conf: Model confidence (0-1)
        evidence_conf: Evidence confidence (0-1)
        alpha: Weight for model confidence (1-alpha for evidence)

    Returns:
        Fused confidence score (0-1)
    """
    return alpha * model_conf + (1 - alpha) * evidence_conf

def color_from_confidence(conf):
    """
    Generate CSS background color based on confidence score.

    Green (high confidence) -> Yellow (medium) -> Red (low)

    Args:
        conf: Confidence score (0-1)

    Returns:
        CSS style string
    """
    if conf >= 0.75:
        # High confidence - green background with dark text
        intensity = int(200 + (conf - 0.75) * 220)  # 200-255
        return f"background-color: rgb(200, {intensity}, 200); color: #1a5c1a; padding: 2px 4px; border-radius: 3px;"
    elif conf >= 0.5:
        # Medium confidence - yellow background with dark text
        green_intensity = int(200 + (conf - 0.5) * 220)  # 200-255
        return f"background-color: rgb(255, {green_intensity}, 150); color: #8b6914; padding: 2px 4px; border-radius: 3px;"
    else:
        # Low confidence - red background with dark text
        green_blue = int(150 - conf * 150)  # 150 down to 0
        return f"background-color: rgb(255, {green_blue}, {green_blue}); color: #8b1a1a; padding: 2px 4px; border-radius: 3px;"

def format_sentence_with_confidence(sentence, confidence):
    """
    Format a sentence with inline confidence coloring.

    Args:
        sentence: Text sentence
        confidence: Confidence score (0-1)

    Returns:
        HTML formatted string
    """
    style = color_from_confidence(confidence)
    return f'<span style="{style}" title="Confidence: {confidence:.3f}">{sentence} ({confidence:.2f})</span>'
