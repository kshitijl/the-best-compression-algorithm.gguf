import random
import zlib
import bz2
import lzma
import base64
import struct
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from llama_cpp import Llama, llama_get_logits
from numpy.typing import NDArray
import numpy as np
import zstd


PRECISION = 64  # use 32bit presision b/c it fits in 64 (so overflow is easy to detect) & math is easier
MAX_RANGE = np.uint64(0xFFFFFFFFFFFFFFFF)
hi_order_bit_mask = 0x8000000000000000

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
    logits_ptr = llama_get_logits(llm.ctx)
    logits_np = np.ctypeslib.as_array(logits_ptr, shape=(llm.n_vocab(),))
    # Convert to probabilities with softmax
    logits_np = np.array(
        logits_np, dtype=np.float64
    )  # TODO figure out why we need this extra precision

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

    for i, token in enumerate(tokens):
        token_str = llm.detokenize([token]).decode("utf-8", errors="replace")
        print(f"token: {token_str}")
        if progress_callback:
            progress_callback(i + 1, total_tokens)

        probs = get_token_probs(llm, tokens[:i])

        prob_before = np.sum(probs[:token])
        next_token_prob = probs[token]

        assert hi > lo
        assert (lo & hi_order_bit_mask) != (hi & hi)

        # update interval
        print(f"before update: lo: {lo}, hi: {hi}")
        width = hi - lo
        lo = lo + np.uint64(prob_before * width)
        hi = lo + np.uint64(next_token_prob * width)
        print(f"after update: lo: {lo}, hi: {hi}")
        print(f"next_token_prob: {next_token_prob}")
        assert lo != hi
        assert hi > lo

        while (lo & hi_order_bit_mask) == (hi & hi_order_bit_mask):
            output_bits.append((lo & hi_order_bit_mask) >> 63)
            lo = lo << 1
            hi = hi << 1

        print(f"after renormalize: lo: {lo}, hi: {hi}")

        assert hi > lo
        assert (lo & hi_order_bit_mask) != (hi & hi_order_bit_mask)

    # while hi != 0:
    #     output_bits.append((hi & hi_order_bit_mask) >> 63)
    #     hi = hi << 1

    # lo = lo + 1
    # while lo != 0:
    #     output_bits.append((lo & hi_order_bit_mask) >> 63)
    #     lo = lo << 1

    remain = hi // 2 + lo // 2

    while remain != 0:
        output_bits.append((remain & hi_order_bit_mask) >> 63)
        remain = remain << 1

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
    current_window = np.uint64(0)
    bit_index = 0
    for _ in range(PRECISION):
        current_window = current_window << 1
        if bit_index < len(bits):
            current_window = current_window | bits[bit_index]
            bit_index += 1

    lo = np.uint64(0)
    hi = np.uint64(MAX_RANGE)

    decompressed_tokens = []
    for i in range(compressed.num_tokens):
        if progress_callback:
            progress_callback(i + 1, compressed.num_tokens)
        probs = get_token_probs(llm, decompressed_tokens)

        width = hi - lo
        position = (current_window - lo) / width

        # Clamp position to valid range [0, 1) to prevent out-of-bounds
        if position < 0:
            print(
                f"\nWARNING: position < 0 at token {i}: {position}, value={current_window}, lo={lo}, hi={hi}"
            )
        elif position >= 1.0:
            print(
                f"\nWARNING: position >= 1.0 at token {i}: {position}, value={current_window}, lo={lo}, hi={hi}"
            )

        cdf = np.cumsum(probs)
        token = np.searchsorted(cdf, position, side="right")
        decompressed_tokens.append(token)

        prob_before = np.sum(probs[:token])
        prob_token = probs[token]

        token_str = llm.detokenize([token]).decode("utf-8", errors="replace")
        print(f"token: {token_str}")
        if i == compressed.num_tokens - 1:
            print(
                llm.detokenize(
                    [token - 2, token - 1, token, token + 1, token + 2]
                ).decode("utf-8", errors="replace")
            )
        # update interval
        print(f"before update: lo: {lo}, hi: {hi}")
        lo = lo + np.uint64(prob_before * width)
        hi = lo + np.uint64(prob_token * width)
        print(f"after update: lo: {lo}, hi: {hi}")
        print(f"next_token_prob: {prob_token}")

        while (lo & hi_order_bit_mask) == (hi & hi_order_bit_mask):
            lo = lo << 1
            hi = hi << 1
            current_window = current_window << 1
            if bit_index < len(bits):
                current_window = current_window | bits[bit_index]
                bit_index += 1

        print(f"after renormalize: lo: {lo}, hi: {hi}")

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
        #     logits_all=False,
        #     n_gpu_layers=-1,
        #     n_ctx=WINDOW_SIZE * 2,
        # ),
        # Llama.from_pretrained(
        #     repo_id="Qwen/Qwen2-0.5B-Instruct-GGUF",
        #     filename="*q8_0.gguf",
        #     verbose=False,
        #     logits_all=False,
        #     n_gpu_layers=-1,
        #     n_ctx=WINDOW_SIZE * 2,
        # ),
        # Llama.from_pretrained(
        #     repo_id="QuantFactory/Meta-Llama-3-8B-GGUF",
        #     filename="*Q8_0.gguf",
        #     verbose=False,
        #     logits_all=False,
        #     n_gpu_layers=-1,
        #     n_ctx=WINDOW_SIZE * 2,
        # ),
        Llama.from_pretrained(
            repo_id="QuantFactory/SmolLM2-360M-GGUF",
            filename="*Q4_0.gguf",
            verbose=False,
            logits_all=False,
            n_gpu_layers=-1,
            n_ctx=WINDOW_SIZE * 2,
        ),
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
        #         """"/**
        #  * Copyright (c) Meta Platforms, Inc. and affiliates.
        #  *
        #  * This source code is licensed under the MIT license found in the
        #  * LICENSE file in the root directory of this source tree.
        #  *
        #  * @flow
        #  */
        # // Keep in sync with ReactServerConsoleConfig
        # const badgeFormat = '%c%s%c';
        # // Same badge styling as DevTools.
        # const badgeStyle =
        #   // We use a fixed background if light-dark is not supported, otherwise
        #   // we use a transparent background.
        #   # 'background: #e6e6e6;' +
        # """,
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
            start = time.time()
            compressed = compress(llm, text)  # , progress_callback=print_progress)
            end = time.time()
            compressed_size = len(compressed.to_bytes())
            compression_ratio = original_size / compressed_size

            # Common compression algorithms
            gzip_size = len(zlib.compress(original_bytes, level=9))
            bz2_size = len(bz2.compress(original_bytes, compresslevel=9))
            lzma_size = len(lzma.compress(original_bytes, preset=9))
            zstd_size = len(zstd.compress(original_bytes, 22))

            print(f"Encoded: {base64.b64encode(compressed.to_bytes()).decode('ascii')}")
            print("\nCompression Results:")
            print(f"Compressed in {end - start:.2f} seconds")
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

            start = time.time()
            decompressed = decompress(
                llm,
                compressed,
                # progress_callback=lambda c, t: print_progress(c, t, "Decompressing"),
            )
            end = time.time()
            matches = decompressed == text
            print(f"Decompressed correctly? {matches}")
            if not matches:
                print(f"\nOriginal length: {len(text)}")
                print(f"Decompressed length: {len(decompressed)}")
                print(f"Decompressed in {end - start:.2f} seconds")
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
