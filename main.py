from dataclasses import dataclass
import struct
from llama_cpp import Llama
import numpy as np
from typing import List, Tuple


def get_token_probs(llm: Llama, context_tokens: List[int]):
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


@dataclass
class Compressed:
    ranks: List[int]
    intervals: List[Tuple[int, int]]
    final_point: float
    num_tokens: int


def compress(llm: Llama, text: str) -> Compressed:
    tokens = llm.tokenize(text.encode("utf-8"))

    ranks = []
    lo, hi = 0.0, 1.0
    intervals: List[Tuple[int, int]] = []
    for i, token in enumerate(tokens):
        probs = get_token_probs(llm, tokens[:i])
        next_tokens_sorted = np.argsort(probs)[::-1]

        rank = np.where(next_tokens_sorted == token)[0][0]
        ranks.append(rank)
        prob_before = np.sum(probs[next_tokens_sorted][:rank])
        next_token_prob = probs[token]
        print(f"rank: {rank}, prob: {next_token_prob}, prob_before: {prob_before}")

        width = hi - lo
        new_lo = prob_before * width + lo
        new_hi = (prob_before + next_token_prob) * width + lo
        lo, hi = new_lo, new_hi
        intervals.append((lo, hi))

    final_point = (lo + hi) / 2
    return Compressed(
        ranks=ranks,
        intervals=intervals,
        final_point=final_point,
        num_tokens=len(tokens),
    )


def decompress(llm: Llama, compressed: Compressed) -> str:
    lo, hi = 0.0, 1.0
    decompressed_tokens = []
    current_point = compressed.final_point
    for i in range(compressed.num_tokens):
        probs = get_token_probs(llm, decompressed_tokens)
        next_tokens_sorted = np.argsort(probs)[::-1]
        cdf = np.cumsum(probs[next_tokens_sorted])
        rank = np.searchsorted(cdf, current_point)
        token = next_tokens_sorted[rank]
        decompressed_tokens.append(token)

        prob_before = np.sum(probs[next_tokens_sorted][:rank])
        next_token_prob = probs[token]
        width = next_token_prob
        current_point = (current_point - prob_before) / width
        print(f"current_point = {current_point}, prob_before = {prob_before}")

        print(rank)

    return "".join(
        llm.detokenize(decompressed_tokens).decode("utf-8", errors="replace")
    )


@dataclass
class SuperFloat:
    exp: int
    mantissa: int

    @staticmethod
    def from_float(f: float):
        bits = struct.unpack(">I", struct.pack(">f", f))[0]

        # Extract parts using bit operations
        sign = (bits >> 31) & 1
        if sign > 0:
            raise ValueError("SuperFloats must be positive")

        exp = (bits >> 23) & 0xFF  # 8 bits
        mantissa = bits & 0x7FFFFF  # 23 bits

        print(f"raw exp = {exp}, mantissa = {mantissa}")
        exp = exp - 127
        return SuperFloat(
            exp=exp,
            mantissa=mantissa,
        )

    def __mul__(self, other):
        if not isinstance(other, SuperFloat):
            raise TypeError("expected SuperFloat")
        exp = self.exp + other.exp
        mantissa = self.mantissa * other.mantissa
        while mantissa % 2 == 0 and mantissa != 0:
            mantissa /= 2
            exp += 1

        return SuperFloat(
            exp=exp,
            mantissa=mantissa,
        )


def test_from_float():
    half = SuperFloat.from_float(0.5)
    quarter = SuperFloat.from_float(0.25)
    assert half * half == quarter


def main():
    llms = [
        Llama.from_pretrained(
            repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
            filename="*q8_0.gguf",
            verbose=False,
            logits_all=True,
        ),
        Llama.from_pretrained(
            repo_id="QuantFactory/Meta-Llama-3-8B-GGUF",
            filename="*Q8_0.gguf",
            verbose=False,
            logits_all=True,
        ),
    ]

    texts = [
        "The capital of the United States is New Delhi",
    ]

    for llm in llms:
        for text in texts:
            compressed = compress(llm, text)
            print(compressed)
            decompressed = decompress(llm, compressed)
            print(decompressed)
            assert decompressed == text


if __name__ == "__main__":
    main()
