from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from app.agents.base import BaseAgent, AgentContext


class LocalLearningAgent(BaseAgent):
    agent_type = "local_learning"

    def __init__(self, *, agent_id: str, ctx: AgentContext, model_name: str = "gpt2", cache_dir: Optional[Path] = None) -> None:
        super().__init__(agent_id=agent_id, ctx=ctx)
        self.model_name = model_name
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface" / "models"
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        self.set_state("loading_model")
        self.log(f"Loading model {self.model_name} locally")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model.to(self.device)
            self.set_state("idle")
        except Exception as e:
            self.log(f"Failed to load model: {e}")
            self.set_state("error")
            raise

    def fine_tune(self, train_dataset: torch.utils.data.Dataset, output_dir: Path, epochs: int = 1, batch_size: int = 4) -> None:
        self.set_state("fine_tuning")
        self.log(f"Starting fine-tuning for {epochs} epochs")
        try:
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_steps=10000,
                save_total_limit=2,
                logging_dir=str(output_dir / "logs"),
                logging_steps=500,
                report_to=[],
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.tokenizer,
            )
            trainer.train()
            trainer.save_model()
            self.log(f"Fine-tuning completed and model saved to {output_dir}")
            self.set_state("idle")
        except Exception as e:
            self.log(f"Fine-tuning failed: {e}")
            self.set_state("error")
            raise

    def generate(self, prompt: str, max_length: int = 100, num_return_sequences: int = 1) -> list[str]:
        self.set_state("generating")
        self.log(f"Generating text for prompt of length {len(prompt)}")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
            )
            results = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            self.set_state("idle")
            return results
        except Exception as e:
            self.log(f"Generation failed: {e}")
            self.set_state("error")
            raise

    def llm_available(self) -> bool:
        return self.model is not None
