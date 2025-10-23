import random
import zlib
import bz2
import lzma
import base64
import struct
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from llama_cpp import Llama
from numpy.typing import NDArray
import numpy as np
import zstd


PRECISION = 32  # use 32bit presision b/c it fits in 64 (so overflow is easy to detect) & math is easier
MAX_RANGE = 1 << PRECISION  # 2^32
HALF = 1 << (PRECISION - 1)  # 2^31
QUARTER = 1 << (PRECISION - 2)  # 2^30
THREE_QUARTERS = 3 * QUARTER  # 3 * 2^30

WINDOW_SIZE = 64  # this is the max number of tokens we will send to the LLM before resetting the context


def get_token_probs(llm: Llama, context_tokens: List[int]) -> NDArray[np.float64]:
    """Get probability distribution for next token given context"""
    if len(context_tokens) == 0:
        # Empty context - use BOS token or evaluate empty
        context_tokens = [llm.token_bos()]

    if len(context_tokens) % WINDOW_SIZE == 0:
        llm.reset()
        llm.eval([llm.token_bos()])

    # Evaluate the context to get logits
    llm.eval([context_tokens[-1]])

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
    num_tokens: int
    data: bytes

    def to_bytes(self) -> bytes:
        # big-endian unsigned 4byte int
        header = struct.pack(">I", self.num_tokens)
        return header + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> "Compressed":
        num_tokens = struct.unpack(">I", data[:4])[0]
        return cls(num_tokens=num_tokens, data=data[4:])


def bits_to_bytes(bits: List[int]) -> bytes:
    # pad to byte width
    padding = (8 - len(bits) % 8) % 8
    padded_bits = bits + [0] * padding

    byte_array = bytearray()
    for i in range(0, len(padded_bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded_bits[i + j]
        byte_array.append(byte)

    return bytes(byte_array)


def bytes_to_bits(data: bytes) -> List[int]:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def compress(
    llm: Llama,
    text: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Compressed:
    tokens = llm.tokenize(text.encode("utf-8"))
    total_tokens = len(tokens)
    llm.reset()

    lo = np.uint64(0)
    hi = np.uint64(MAX_RANGE)
    output_bits = []
    underflow_count = 0

    for i, token in enumerate(tokens):
        if progress_callback:
            progress_callback(i + 1, total_tokens)

        probs = get_token_probs(llm, tokens[:i])
        next_tokens_sorted = np.argsort(probs)[::-1]

        rank = np.where(next_tokens_sorted == token)[0][0]
        prob_before = np.sum(probs[next_tokens_sorted][:rank])
        next_token_prob = probs[token]

        # update interval
        width = hi - lo
        lo = lo + np.uint64(prob_before * width)
        hi = lo + max(np.uint64(1), np.uint64(next_token_prob * width))

        # renormalize
        while True:
            if hi <= HALF:
                output_bits.append(0)
                output_bits.extend([1] * underflow_count)
                underflow_count = 0
                lo = lo << 1
                hi = hi << 1
            elif lo >= HALF:
                output_bits.append(1)
                output_bits.extend([0] * underflow_count)
                underflow_count = 0
                lo = (lo - HALF) << 1
                hi = (hi - HALF) << 1
            elif lo >= QUARTER and hi <= THREE_QUARTERS:
                underflow_count += 1
                lo = (lo - QUARTER) << 1
                hi = (hi - QUARTER) << 1
            else:
                break

    # flush remaining bits
    underflow_count += 1
    if lo < QUARTER:
        output_bits.append(0)
        output_bits.extend([1] * underflow_count)
    else:
        output_bits.append(1)
        output_bits.extend([0] * underflow_count)

    compressed_bytes = bits_to_bytes(output_bits)

    return Compressed(
        num_tokens=len(tokens),
        data=compressed_bytes,
    )


def decompress(
    llm: Llama,
    compressed: Compressed,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    bits = bytes_to_bits(compressed.data)
    llm.reset()

    # read initial val from bitstream
    value = np.uint64(0)
    bit_index = 0
    for _ in range(PRECISION):
        value = value << 1
        if bit_index < len(bits):
            value = value | bits[bit_index]
            bit_index += 1

    lo = np.uint64(0)
    hi = np.uint64(MAX_RANGE)

    decompressed_tokens = []
    for i in range(compressed.num_tokens):
        if progress_callback:
            progress_callback(i + 1, compressed.num_tokens)
        probs = get_token_probs(llm, decompressed_tokens)
        next_tokens_sorted = np.argsort(probs)[::-1]

        width = hi - lo
        position = (value - lo) / width

        # Clamp position to valid range [0, 1) to prevent out-of-bounds
        if position < 0:
            print(
                f"\nWARNING: position < 0 at token {i}: {position}, value={value}, lo={lo}, hi={hi}"
            )
            position = 0
        elif position >= 1.0:
            print(
                f"\nWARNING: position >= 1.0 at token {i}: {position}, value={value}, lo={lo}, hi={hi}"
            )
            position = 0.9999999

        cdf = np.cumsum(probs[next_tokens_sorted])
        rank = np.searchsorted(cdf, position, side="right")
        token = next_tokens_sorted[rank]
        decompressed_tokens.append(token)

        prob_before = np.sum(probs[next_tokens_sorted][:rank])
        prob_token = probs[token]

        # update interval
        lo = lo + np.uint64(prob_before * width)
        hi = lo + max(np.uint64(1), np.uint64(prob_token * width))

        # renormalize
        while True:
            if hi <= HALF:
                lo = lo << 1
                hi = hi << 1
                value = value << 1
                if bit_index < len(bits):
                    value = value | bits[bit_index]
                    bit_index += 1
            elif lo >= HALF:
                lo = (lo - HALF) << 1
                hi = (hi - HALF) << 1
                value = (value - HALF) << 1
                if bit_index < len(bits):
                    value = value | bits[bit_index]
                    bit_index += 1
            elif lo >= QUARTER and hi <= THREE_QUARTERS:
                lo = (lo - QUARTER) << 1
                hi = (hi - QUARTER) << 1
                value = (value - QUARTER) << 1
                if bit_index < len(bits):
                    value = value | bits[bit_index]
                    bit_index += 1
            else:
                break

    return llm.detokenize(decompressed_tokens).decode("utf-8", errors="replace")


def decompress_b64(llm: Llama, input: str):
    bytes = base64.b64decode(input)
    compressed = Compressed.from_bytes(bytes)
    out = decompress(llm, compressed)
    print("Decoded Text:")
    print(out)


def print_progress(current: int, total: int, operation: str = "Compressing"):
    percent = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(
        f"\r{operation}: [{bar}] {current}/{total} tokens ({percent:.1f}%)",
        end="",
        flush=True,
    )
    if current == total:
        print()


def compress_and_compare(
    llm: Llama,
    input: str,
):
    print("Compressing the following input:\n")
    print(input)

    start = time.time()
    compressed = compress(llm, input)
    end = time.time()
    print(f"\nCompressed in {end - start:.2f} seconds.\n")
    # compare results
    original_bytes = input.encode("utf-8")
    original_size = len(original_bytes)

    # calc llm compression ratio
    compressed_size = len(compressed.to_bytes())
    compression_ratio = original_size / compressed_size

    # calc compression algorithms
    gzip_size = len(zlib.compress(original_bytes, level=9))
    bz2_size = len(bz2.compress(original_bytes, compresslevel=9))
    lzma_size = len(lzma.compress(original_bytes, preset=9))
    zstd_size = len(zstd.compress(original_bytes, 22))

    print(f"Encoded: {base64.b64encode(compressed.to_bytes()).decode('ascii')}")
    print("\nCompression Results:")
    print(f"  Original:        {original_size:>6} bytes")
    print(f"  LLM compressed:  {compressed_size:>6} bytes ({compression_ratio:.2f}x)")
    print(
        f"  ZSTD (level 22):  {zstd_size:>6} bytes ({original_size / zstd_size:.2f}x)"
    )
    print(f"  GZIP (level 9):  {gzip_size:>6} bytes ({original_size / gzip_size:.2f}x)")
    print(f"  BZ2 (level 9):   {bz2_size:>6} bytes ({original_size / bz2_size:.2f}x)")
    print(f"  LZMA (level 9):  {lzma_size:>6} bytes ({original_size / lzma_size:.2f}x)")

    print("Decompressing...")
    to_decompress = Compressed.from_bytes(compressed.to_bytes())

    start = time.time()
    decompressed = decompress(llm, to_decompress)
    end = time.time()
    print(f"\nDecompressed in {end - start:.2f} seconds.\n")

    assert decompressed == input, "Decompression failed!"
    print("\nDecompression successful!")


def main():
    llms = [
        # Llama.from_pretrained(
        #     repo_id="ggml-org/gpt-oss-120b-GGUF",
        #     filename="gpt-oss-120b-mxfp4-00001-of-00003.gguf",
        #     additional_files=[
        #         "gpt-oss-120b-mxfp4-00002-of-00003.gguf",
        #         "gpt-oss-120b-mxfp4-00003-of-00003.gguf",
        #     ],
        #     verbose=False,
        #     logits_all=True,
        #     n_gpu_layers=-1,
        #     n_ctx=32768,
        # ),
        # Llama.from_pretrained(
        #     repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
        #     filename="*q8_0.gguf",
        #     verbose=False,
        #     logits_all=True,
        #     n_gpu_layers=-1,
        #     n_ctx=32768,
        # ),
        Llama.from_pretrained(
            repo_id="QuantFactory/Meta-Llama-3-8B-GGUF",
            filename="*Q8_0.gguf",
            verbose=False,
            # logits_all=True,
            n_gpu_layers=-1,
            n_ctx=32768,
        ),
        # Llama.from_pretrained(
        #     repo_id="QuantFactory/SmolLM2-360M-GGUF",
        #     filename="*Q4_0.gguf",
        #     verbose=False,
        #     logits_all=True,
        #     n_gpu_layers=-1,
        #     n_ctx=32768,
        # ),
    ]

    texts = [
        # "hello world",
        # "The capital of the United States is Washington, D.C.",
        # "tdoajpwdojaw podfjawpofjawpfojawpfojawpfojawfpoawjfpoawjfpoawjfpawofjawpofjawpofjawpofjawpofjawpofjawpofjawfpoa",
        # "".join(
        #     random.choices(
        #         "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=200
        #     )
        # ),
        """In information theory, data compression, source coding,[1] or bit-rate reduction is the process of encoding information using fewer bits than the original representation.[2] Any particular compression is either lossy or lossless. Lossless compression reduces bits by identifying and eliminating statistical redundancy. No information is lost in lossless compression. Lossy compression reduces bits by removing unnecessary or less important information.[3] Typically, a device that performs data compression is referred to as an encoder, and one that performs the reversal of the process (decompression) as a decoder.
        The process of reducing the size of a data file is often referred to as data compression. In the context of data transmission, it is called source coding: encoding is done at the source of the data before it is stored or transmitted.[4] Source coding should not be confused with channel coding, for error detection and correction or line coding, the means for mapping data onto a signal.
        Data compression algorithms present a space–time complexity trade-off between the bytes needed to store or transmit information, and the computational resources needed to perform the encoding and decoding. The design of data compression schemes involves balancing the degree of compression, the amount of distortion introduced (when using lossy data compression), and the computational resources or time required to compress and decompress the data.[5] """,
    ]

    for llm in llms:
        print(f"Using model: {llm.metadata['general.name']}")
        for text in texts:
            print(f"\nText to encode: {text}")
            original_bytes = text.encode("utf-8")
            original_size = len(original_bytes)

            # LLM compression
            compressed = compress(llm, text, progress_callback=print_progress)
            compressed_size = len(compressed.to_bytes())
            compression_ratio = original_size / compressed_size

            # Common compression algorithms
            gzip_size = len(zlib.compress(original_bytes, level=9))
            bz2_size = len(bz2.compress(original_bytes, compresslevel=9))
            lzma_size = len(lzma.compress(original_bytes, preset=9))
            zstd_size = len(zstd.compress(original_bytes, 22))

            print(f"Encoded: {base64.b64encode(compressed.to_bytes()).decode('ascii')}")
            print("\nCompression Results:")
            print(f"  Original:        {original_size:>6} bytes")
            print(
                f"  LLM compressed:  {compressed_size:>6} bytes ({compression_ratio:.2f}x)"
            )
            print(
                f"  ZSTD (level 22):  {zstd_size:>6} bytes ({original_size / zstd_size:.2f}x)"
            )
            print(
                f"  GZIP (level 9):  {gzip_size:>6} bytes ({original_size / gzip_size:.2f}x)"
            )
            print(
                f"  BZ2 (level 9):   {bz2_size:>6} bytes ({original_size / bz2_size:.2f}x)"
            )
            print(
                f"  LZMA (level 9):  {lzma_size:>6} bytes ({original_size / lzma_size:.2f}x)"
            )

            decompressed = decompress(
                llm,
                compressed,
                progress_callback=lambda c, t: print_progress(c, t, "Decompressing"),
            )
            matches = decompressed == text
            print(f"Decompressed correctly? {matches}")
            if not matches:
                print(f"\nOriginal length: {len(text)}")
                print(f"Decompressed length: {len(decompressed)}")
                # Find first difference
                for i, (c1, c2) in enumerate(zip(text, decompressed)):
                    if c1 != c2:
                        print(f"First difference at position {i}:")
                        print(f"  Expected: {repr(text[max(0, i - 20) : i + 20])}")
                        print(
                            f"  Got:      {repr(decompressed[max(0, i - 20) : i + 20])}"
                        )
                        break
            assert decompressed == text


if __name__ == "__main__":
    main()
