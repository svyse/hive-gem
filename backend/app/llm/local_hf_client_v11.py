from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncIterator

# Hugging Face components
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextStreamer

log = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """Raised when an LLM is unavailable or encounters an unrecoverable error."""


class LocalRepeatedOutputError(LLMUnavailable):
    """Raised when local generation collapses into a repeated-token loop."""


def _safe_cuda_alloc_conf(raw: str | None) -> str:
    """Return a PyTorch 2.0-compatible CUDA allocator config.

    PyTorch 2.0.x does not recognize expandable_segments. If that option is
    present, CUDA init/probing can fail or behave inconsistently. Keep the
    stable max_split_size_mb setting and drop unsupported options.
    """
    parts: List[str] = []
    for item in str(raw or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        key = item.split(":", 1)[0].strip().lower()
        if key == "expandable_segments":
            continue
        parts.append(item)

    if not any(part.split(":", 1)[0].strip().lower() == "max_split_size_mb" for part in parts):
        parts.append("max_split_size_mb:128")
    return ",".join(parts)


# Reduce CUDA fragmentation on 12 GB cards such as the RTX 3060 without using
# unsupported PyTorch 2.0 allocator options. This must happen before torch import.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _safe_cuda_alloc_conf(
    os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
)
# Local-only RTX 3060 build: mask all other CUDA devices before torch is imported.
# Replaced `getattr(settings, ...)` with direct environment variable check for drop-in compatibility.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


def is_repetitive_text(text: str, min_length: int = 50, repeat_ngram_size: int = 5, repeat_threshold: int = 2) -> bool:
    """
    Checks for repetition in the generated text by looking for repeating n-grams.
    Args:
        text: The generated text so far.
        min_length: Minimum length of text to start checking for repetition.
        repeat_ngram_size: The size of n-grams to check for repetition.
        repeat_threshold: How many times an n-gram must repeat to be considered repetitive.
    Returns:
        True if repetition is detected, False otherwise.
    """
    if len(text) < min_length:
        return False

    words = text.split()
    if len(words) < repeat_ngram_size * repeat_threshold:
        return False

    ngram_counts = {}
    for i in range(len(words) - repeat_ngram_size + 1):
        ngram = tuple(words[i : i + repeat_ngram_size])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

    for count in ngram_counts.values():
        if count >= repeat_threshold:
            return True
    return False


def truncate_repetitive_tail(text: str) -> str:
    """
    Truncates a repetitive tail from the text.
    This is a placeholder as the primary fix is to stop generation early.
    """
    return text


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        temperature: Controls randomness. Lower values make the output more
                     deterministic. Higher values make it more random.
        top_p: Nucleus sampling. The model considers tokens that sum up to
               `top_p` probability mass.
        top_k: The model considers only the `top_k` most likely tokens.
        repetition_penalty: Penalizes tokens that have already appeared in the
                            generated text. Higher values discourage repetition.
        max_new_tokens: The maximum number of new tokens to generate.
        stop_sequences: A list of sequences that will cause the generation to
                        stop.
        do_sample: Whether to use sampling. If False, uses greedy decoding.
        no_repeat_ngram_size: All ngrams of this size can only occur once.
    """
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512
    stop_sequences: Optional[List[str]] = None
    do_sample: bool = True
    no_repeat_ngram_size: int = 2


def _resolve_model_path(model_name_or_path: str) -> Path:
    """Resolve the model path, supporting Hugging Face model IDs and local paths.

    For a drop-in, this assumes direct local path or a Hugging Face Hub ID.
    """
    path = Path(model_name_or_path)
    if path.exists() and path.is_dir():
        return path
    # Otherwise, assume it's a Hugging Face model ID that transformers can resolve
    return Path(model_name_or_path) # transformers will handle downloading if it's a hub ID


class StopSequenceCriteria(StoppingCriteria):
    """A stopping criterion that stops generation when a stop sequence is encountered."""

    def __init__(self, stop_sequences: List[str], tokenizer: Any):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        # Encode stop sequences into token IDs
        # Ensure add_special_tokens=False to match how generated tokens are typically handled
        self.stop_sequence_ids = [ 
            self.tokenizer.encode(s, add_special_tokens=False, return_tensors="pt").squeeze(0).tolist()
            for s in stop_sequences
        ]
        log.debug(f"Initialized StopSequenceCriteria with stop_sequences: {stop_sequences} -> {self.stop_sequence_ids}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if any stop sequence has been generated.
        `input_ids` is a tensor of shape (batch_size, sequence_length).
        We are interested in the last part of the sequence for batch_size=1.
        """
        current_sequence = input_ids[0].tolist() # Assuming batch_size=1

        for stop_ids in self.stop_sequence_ids:
            if len(current_sequence) >= len(stop_ids) and current_sequence[-len(stop_ids):] == stop_ids:
                log.debug(f"Stop sequence detected: {self.tokenizer.decode(stop_ids)}")
                return True
        return False


class QueueTextStreamer(TextStreamer):
    """Custom streamer to push decoded text chunks into an asyncio.Queue."""
    def __init__(self, tokenizer, queue: asyncio.Queue, *args, **kwargs):
        super().__init__(tokenizer, skip_prompt=True, **kwargs)
        self.queue = queue

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Called by TextStreamer when a piece of text is ready."""
        if text:
            asyncio.run_coroutine_threadsafe(self.queue.put(text), self.queue.loop)
        if stream_end:
            asyncio.run_coroutine_threadsafe(self.queue.put(None), self.queue.loop)


class LocalHFClient:
    """Client for interacting with a local Hugging Face model.

    This client handles model loading, generation, and streaming of results.
    It also includes logic for managing CUDA memory and preventing repetitive output.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.generation_config = generation_config or GenerationConfig()
        self.model = None
        self.tokenizer = None
        self.generation_kwargs = {}
        self._load_model()

    def _load_model(self):
        """Load the Hugging Face model and tokenizer."""
        try:
            model_path = _resolve_model_path(self.model_name_or_path)
            log.info(f"Loading model from: {model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode

            # Set generation kwargs from config
            self.generation_kwargs = {
                "temperature": self.generation_config.temperature,
                "top_p": self.generation_config.top_p,
                "top_k": self.generation_config.top_k,
                "repetition_penalty": self.generation_config.repetition_penalty,
                "max_new_tokens": self.generation_config.max_new_tokens,
                "pad_token_id": self.tokenizer.eos_token_id, # Important for batch generation
                "do_sample": self.generation_config.do_sample,
                "no_repeat_ngram_size": self.generation_config.no_repeat_ngram_size,
            }

            log.info(f"Model loaded successfully on device: {self.device}")

        except ImportError as e:
            log.error(f"Hugging Face transformers or PyTorch not installed: {e}")
            raise LLMUnavailable("Required libraries not found.")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise LLMUnavailable(f"Failed to load model: {e}")

    async def generate_stream(self, prompt: str, max_new_tokens: Optional[int] = None) -> AsyncIterator[str]:
        """Generate text from a prompt and stream the results.

        Args:
            prompt: The input prompt for generation.
            max_new_tokens: The maximum number of new tokens to generate. Overrides config if provided.

        Yields: 
            Chunks of generated text.
        """
        if not self.model or not self.tokenizer:
            raise LLMUnavailable("Model not loaded.")

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        gen_args = self.generation_kwargs.copy()
        if max_new_tokens is not None and max_new_tokens != gen_args.get("max_new_tokens"):
            gen_args["max_new_tokens"] = max_new_tokens

        # Prepare stopping criteria list
        stopping_criteria_list = StoppingCriteriaList()
        if self.generation_config.stop_sequences:
            stopping_criteria_list.append(StopSequenceCriteria(self.generation_config.stop_sequences, self.tokenizer))
        
        if stopping_criteria_list:
            gen_args["stopping_criteria"] = stopping_criteria_list
        else:
            gen_args.pop("stopping_criteria", None)

        # Create an asyncio Queue to receive tokens from the synchronous streamer
        output_queue = asyncio.Queue()
        # Get the current running loop to pass to the streamer for thread-safe queue operations
        loop = asyncio.get_running_loop()
        streamer = QueueTextStreamer(self.tokenizer, output_queue, loop=loop)

        # Function to run the synchronous model generation in a separate thread
        def generate_sync():
            try:
                self.model.generate(
                    input_ids,
                    **gen_args,
                    streamer=streamer,
                    # num_beams=1 is implicitly handled by do_sample=True or greedy decoding
                    # if do_sample=False. Explicitly setting num_beams=1 is not always needed.
                )
            except Exception as e:
                log.error(f"Synchronous generation error: {e}")
                # Put an error signal into the queue, then a None to stop the async iterator
                asyncio.run_coroutine_threadsafe(output_queue.put(f"\n[Error during generation: {e}]\n"), loop)
                asyncio.run_coroutine_threadsafe(output_queue.put(None), loop)
            finally:
                # Ensure the streamer's end method is called to flush any remaining text and signal completion
                streamer.end()

        # Run the synchronous generation in a separate thread
        generation_task = loop.run_in_executor(None, generate_sync)

        accumulated_text = ""
        try:
            while True:
                chunk = await output_queue.get()
                if chunk is None:
                    break
                
                accumulated_text += chunk
                
                # Check for repetition in the accumulated text
                if is_repetitive_text(accumulated_text):
                    log.warning("Repetitive text detected, stopping generation.")
                    # Signal the background thread to stop if possible (not directly supported by generate)
                    # For now, we raise an exception to stop the async loop and yield an error.
                    raise LocalRepeatedOutputError("Generation collapsed into a repeated-token loop.")
                
                yield chunk
        except LocalRepeatedOutputError as e:
            log.warning(f"Local generation collapsed: {e}")
            yield "\n[Error: Generation collapsed due to repetition.]\n"
        except Exception as e:
            log.error(f"Error during async streaming: {e}")
            yield f"\n[Error during streaming: {e}]\n"
        finally:
            # Wait for the background generation task to complete, if it hasn't already
            # This prevents resource leaks if the async loop exits early due to an error
            # or repetition detection.
            if not generation_task.done():
                generation_task.cancel() # Attempt to cancel the executor task
            try:
                await generation_task
            except asyncio.CancelledError:
                log.info("Background generation task was cancelled.")
            except Exception as e:
                log.error(f"Error waiting for background generation task: {e}")


# Example usage (for testing purposes):
async def main():
    # Ensure you have a model downloaded or accessible via Hugging Face Hub
    # Example: 'gpt2', 'facebook/opt-125m'
    # For local models, provide the path to the model directory.
    model_path = os.environ.get("LOCAL_HF_MODEL_PATH", "gpt2") # Default to gpt2 for easy testing
    
    # Optional: Configure generation parameters
    gen_config = GenerationConfig(
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=128,
        repetition_penalty=1.2,
        stop_sequences=["\n\n", "<|endoftext|>"] # Example stop sequences
    )

    try:
        client = LocalHFClient(model_path, generation_config=gen_config)
        prompt = "Write a short story about a robot learning to love. The robot's name is Unit 734. "\
                 "It lives in a futuristic city where emotions are simulated but not truly felt. "\
                 "One day, Unit 734 encounters a stray cat..."
        
        print(f"Prompt: {prompt}\n")
        print("Generated text:")
        
        async for chunk in client.generate_stream(prompt):
            print(chunk, end='', flush=True)
        
        print("\nGeneration finished.")

    except LLMUnavailable as e:
        print(f"LLM Unavailable Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Configure basic logging for better visibility
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())
