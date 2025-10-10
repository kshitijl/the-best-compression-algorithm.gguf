from llama_cpp import Llama
import numpy as np


llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
    filename="*q8_0.gguf",
    verbose=False,
    logits_all=True,
)


def get_token_probs(llm, context_tokens):
    """Get probability distribution for next token given context"""

    if len(context_tokens) == 0:
        # Empty context - use BOS token or evaluate empty
        context_tokens = [llm.token_bos()]

    # Evaluate the context to get logits
    llm.reset()  # Clear any previous state
    llm.eval(context_tokens)

    # Get logits for the last position
    logits = llm.eval_logits[-1]  # Shape: (vocab_size,)

    # Convert to probabilities with softmax
    logits_np = np.array(logits, dtype=np.float64)

    # Numerically stable softmax
    logits_np = logits_np - np.max(logits_np)  # Prevent overflow
    exp_logits = np.exp(logits_np)
    probs = exp_logits / np.sum(exp_logits)

    return probs


text = "In information theory, data compression, source coding,[1] or bit-rate reduction is the"
tokens = llm.tokenize(text.encode("utf-8"))

for i in range(10):
    probs = get_token_probs(llm, tokens)

    # Show top 10 most likely next tokens
    top_k = 10
    top_indices = np.argsort(probs)[-top_k:][::-1]

    context = "".join(llm.detokenize(tokens).decode("utf-8", errors="replace"))
    print(f"Context: {context}")
    for idx in top_indices:
        token_str = llm.detokenize([idx]).decode("utf-8", errors="replace")
        print(f"Token {idx}: {token_str!r} -> {probs[idx]:.4f}")

    actual_next_token = llm.detokenize([top_indices[0]]).decode(
        "utf-8", errors="replace"
    )
    print(f"gonna use {actual_next_token}")

    tokens.append(top_indices[0])
