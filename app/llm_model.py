from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch

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
        # For your stopping criterion we need the tokenizer in scope
        StopOnINST.tokenizer = self.tokenizer

    def generate_answer(self, query: str, context: str) -> str:
        # Clean up context whitespace
        context = " ".join(context.split())

        prompt = (
            "[INST] Use only the following context to answer the question. "
            "If you can’t find the answer, say ‘Answer not found in the document.’\n\n"
            f"Context: {context}\n\n"
            f"Question: {query} [/INST]"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        stopping = StoppingCriteriaList([StopOnINST()])

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            stopping_criteria=stopping,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # decode everything and then strip the prompt & stop token
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # The model echo includes the prompt + answer + "[/INST]" at end.
        # So we take everything after the final "[/INST]"
        answer = full.split("[/INST]")[-1].strip()
        return answer
