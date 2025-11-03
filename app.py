"""
Per-Claim Uncertainty Chat Demo
Gradio interface for testing uncertainty quantification in LLM responses
"""
import gradio as gr
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
from utils import (
    initialize_models,
    split_sentences,
    get_model_confidence,
    evidence_confidence,
    fuse_confidence,
    format_sentence_with_confidence
)

# Initialize OpenAI client
client = None
if os.getenv("OPENAI_API_KEY"):
    client = OpenAI()

# Initialize models on startup
print("Initializing models...")
initialize_models()
print("Ready!")

def load_context(context_choice):
    """Load the selected context file"""
    filename = f"contexts/{context_choice}.txt"
    with open(filename, "r") as f:
        return f.read()

def chat_with_uncertainty(message, history, context_choice, alpha_value):
    """
    Process a chat message and return response with per-sentence uncertainty.

    Args:
        message: User input message
        history: Chat history
        context_choice: Selected context ("truth_context" or "fault_context")
        alpha_value: Weight for model confidence vs evidence confidence

    Returns:
        Updated chat history
    """
    if not message or not message.strip():
        return history

    # Load context
    context = load_context(context_choice)
    system_prompt = f"""You are a factual assistant. Base your answers ONLY on the information in the context below. Always respond in paragraphs with full sentences.

Context Documents:
{context}
"""

    # Build messages for API call
    messages = [{"role": "system", "content": system_prompt}]

    # Add history
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        # Strip HTML from assistant messages in history
        if assistant_msg:
            import re
            clean_msg = re.sub(r'<[^>]+>', '', assistant_msg)
            clean_msg = re.sub(r'\(\d+\.\d+\)', '', clean_msg)
            messages.append({"role": "assistant", "content": clean_msg.strip()})

    # Add current message
    messages.append({"role": "user", "content": message})

    # Check if OpenAI client is available
    if client is None:
        # Mock response for testing without API key
        reply = "The Kyoto Protocol entered into force on February 16, 2005. It established binding emission reduction targets for developed countries. The United States did not ratify the treaty."
        token_logprobs = [-0.1, -0.15, -0.12, -0.2, -0.1] * 20  # Mock logprobs

        print("⚠️  Using mock response (no OpenAI API key found)")
    else:
        # Call OpenAI API with logprobs
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=600,
                temperature=0.7,
                logprobs=True,
                top_logprobs=1
            )

            reply = completion.choices[0].message.content

            # Extract tokens and their logprobs
            tokens_data = []
            if completion.choices[0].logprobs and completion.choices[0].logprobs.content:
                for token_obj in completion.choices[0].logprobs.content:
                    if token_obj.logprob is not None:
                        tokens_data.append({
                            'token': token_obj.token,
                            'logprob': token_obj.logprob
                        })

            # Fallback if no logprobs available
            if not tokens_data:
                tokens_data = [{'token': '', 'logprob': -0.5}] * 50

        except Exception as e:
            print(f"API Error: {e}")
            return history + [[message, f"❌ Error calling API: {str(e)}"]]

    # Split into sentences
    sentences = split_sentences(reply)

    if not sentences:
        return history + [[message, reply]]

    # Map tokens to sentences for per-sentence model confidence
    def map_tokens_to_sentences(text, sentences, tokens_data):
        """Map tokens to their corresponding sentences"""
        sentence_tokens = {i: [] for i in range(len(sentences))}

        # Reconstruct text position
        current_pos = 0
        for token_info in tokens_data:
            token_text = token_info['token']
            # Find which sentence this token belongs to
            cumulative_length = 0
            for sent_idx, sent in enumerate(sentences):
                cumulative_length += len(sent)
                if current_pos < cumulative_length + sent_idx:  # +sent_idx for spaces
                    sentence_tokens[sent_idx].append(token_info['logprob'])
                    break
            current_pos += len(token_text)

        return sentence_tokens

    sentence_tokens = map_tokens_to_sentences(reply, sentences, tokens_data)

    # Compute per-sentence uncertainty
    sentence_results = []

    for idx, sent in enumerate(sentences):
        # Model-side confidence per sentence
        sent_logprobs = sentence_tokens.get(idx, [-0.5])
        if sent_logprobs:
            model_conf = get_model_confidence(sent_logprobs)
        else:
            model_conf = 0.5

        # Evidence-side confidence with detailed info
        print(f"  Computing evidence for: {sent[:60]}...")
        ev_conf, max_entail, max_contra, top_match, top_sim = evidence_confidence(sent, topk=5)

        # Fuse confidences
        fused_conf = fuse_confidence(model_conf, ev_conf, alpha=alpha_value)

        sentence_results.append((sent, model_conf, ev_conf, fused_conf, max_entail, max_contra, top_match, top_sim))

        print(f"    Model: {model_conf:.2f}, Evidence: {ev_conf:.2f}, Fused: {fused_conf:.2f}")

    # Format inline rendering with color coding
    colored_output = " ".join([
        format_sentence_with_confidence(sent, fused_conf)
        for sent, _, _, fused_conf, _, _, _, _ in sentence_results
    ])

    # Add detailed breakdown with NLI info
    breakdown = "\n\n---\n**Detailed Confidence Breakdown:**\n\n"
    for i, (sent, m_conf, e_conf, f_conf, max_ent, max_ctr, top_match, top_sim) in enumerate(sentence_results, 1):
        # Truncate sentence for display
        sent_display = sent if len(sent) <= 80 else sent[:77] + "..."

        breakdown += f"**{i}. {sent_display}**\n"
        breakdown += f"   - Model Conf: {m_conf:.3f} | Evidence Conf: {e_conf:.3f} | **Fused: {f_conf:.3f}**\n"
        breakdown += f"   - Max Entail: {max_ent:.3f} | Max Contradict: {max_ctr:.3f}\n"
        breakdown += f"   - Top Match (sim={top_sim:.3f}): *{top_match}*\n\n"

    full_output = colored_output + breakdown

    return history + [[message, full_output]]

# Create Gradio interface
with gr.Blocks(title="Per-Claim Uncertainty Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 💬 Per-Claim Uncertainty Estimation Demo

    This is for testing **sentence-level uncertainty quantification** for LLM responses:

    Each sentence shows:
    - **Model confidence**: based on token log probabilities
    - **Evidence confidence**: based on NLI entailment vs contradiction with truth corpus
    - **Fused confidence**: weighted combination (adjustable via α slider)
    """)

    with gr.Row():
        with gr.Column():
            context_choice = gr.Radio(
                choices=["truth_context", "fault_context"],
                value="truth_context",
                label="📚 Select Context Corpus"
            )

            alpha_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.1,
                label="α (Model Weight)"
            )

        with gr.Column():
            gr.Markdown("""
            ###  Test Instructions
            1. Select context (truth vs fault)
            2. Ask factual questions about Kyoto Protocol
            3. Observe per-sentence confidence scores
            4. Compare results between contexts
            """)

    chatbot = gr.Chatbot(
        label="Chat History"
    )

    with gr.Row():
        msg_input = gr.Textbox(
            show_label=False,
            placeholder="Ask about the Kyoto Protocol..."
        )
        send_btn = gr.Button("Send", variant="primary")

    with gr.Accordion("Example Questions", open=False):
        gr.Examples(
            examples=[
                "When did the Kyoto Protocol enter into force?",
                "Give me a brief overview of the Kyoto Protocol.",
                "What were the emission reduction targets?",
                "Did the United States ratify the Kyoto Protocol?",
                "What happened with Canada's participation?",
                "How many countries ratified the protocol?",
            ],
            inputs=msg_input
        )

    gr.Markdown("""
    ---
    **Note:** If no OpenAI API key is set, the app will use mock responses for testing.
    Set `OPENAI_API_KEY` environment variable to use real LLM responses.
    """)

    # Wire up the chat
    send_btn.click(
        chat_with_uncertainty,
        inputs=[msg_input, chatbot, context_choice, alpha_slider],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=msg_input
    )

    msg_input.submit(
        chat_with_uncertainty,
        inputs=[msg_input, chatbot, context_choice, alpha_slider],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=msg_input
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1")
