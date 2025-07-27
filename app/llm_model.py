# app/llm_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import re

class StopOnINST(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        # Stop when the token corresponding to ' [/INST]' is generated
        decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return decoded.strip().endswith("[/INST]")

class LLM:
    def __init__(self):
        model_id = "models/stablelm-zephyr-3b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to("cuda")
        # For stopping criterion we need tokenizer scope
        StopOnINST.tokenizer = self.tokenizer

    def generate_answer(self, query: str, contexts: list) -> str:
        """
        contexts: list of dicts with keys: 'text' and optionally 'source_id'
        """
        # Format context chunks with numbered sources
        formatted_context = ""
        for i, ctx in enumerate(contexts, 1):
            formatted_context += f"[Source {i}]\n{ctx['text'].strip()}\n\n"

        prompt = (
            "You are a helpful assistant that answers questions based solely on the provided context.\n\n"
            "INSTRUCTIONS:\n"
            "1. Use only the information contained in the context below to answer the question.\n"
            "2. If the answer cannot be found in the context, reply exactly with: 'Answer not found in the document.'\n"
            "3. Be precise and detailed if the information is available.\n"
            "4. When you provide an answer, cite the source by referring to its number in square brackets, e.g. [Source 1].\n\n"
            f"CONTEXT:\n{formatted_context}"
            f"QUESTION:\n{query}\n\n"
            "ANSWER:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        stopping = StoppingCriteriaList([StopOnINST()])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            stopping_criteria=stopping,
            pad_token_id=self.tokenizer.eos_token_id,
            temperature=0.2,  # Lower temperature for factual answers
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the answer part
        if "ANSWER:" in full:
            answer = full.split("ANSWER:")[-1].strip()
        else:
            answer = full.strip()
        # Remove any [Source X] references from the answer
        answer = re.sub(r'\[Source \d+\]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()
        return answer
