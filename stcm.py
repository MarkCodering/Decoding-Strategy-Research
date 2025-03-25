"""Selective Token Constraint Mechanism (STCM)"""
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedTokenizer, PreTrainedTokenizerFast

class evaluator():
    def __init__(self, allowed_tokens, allow_token_id):
        #self.tokenizor = tokenizor
        self.allow_token = allowed_tokens # str
        self.allow_token_id = allow_token_id # torch tensor
        self.buffer = {
            # Save the max token for each round
            "in_token": [],
            "out_token": [],
        }
    
    def set_in_token(self, token):
        self.buffer["in_token"].append(token)
        return
    
    def set_out_token(self, token):
        self.buffer["out_token"].append(token)
        return
    
    def dump(self, tokenizer):
        # Print the case distribution (Analysis part)
        # TBD!!!
        result = {
            "in_word": [],
            "out_word": [],
            "changed": 0, "his_changed": [],
            "nochanged": 0, 
        }
        
        # Decode token into word
        result["in_word"] = tokenizer.batch_decode(
            self.buffer["in_token"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        result["out_word"] = tokenizer.batch_decode(
            self.buffer["out_token"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # Analysis (TBD)
        for i in range(len(result["in_word"])):
            in_word = result["in_word"][i].replace(" ", "")
            out_word = result["out_word"][i].replace(" ", "")
            if in_word == out_word:
                result["nochanged"] += 1
            else:
                result["changed"] += 1
                result["his_changed"].append(f"{result["in_word"]}-{result["out_word"]}")
                
        return {"changed": result["changed"], "nochanged": result["nochanged"], "history": result["his_changed"]}
    
class STCM(LogitsProcessor):
    """Selective Token Constraint Mechanism (STCM)"""
    def __init__(
        self,
        allowed_tokens: list[str],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        penalty: float = None,
        temperature: float = 1.0,
        
        debug_mode: bool = None,
    ) -> None:
        super().__init__()
        if not allowed_tokens:
            raise ValueError("allowed_tokens must be a non-empty list of tokens.")
        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            raise ValueError("A valid PreTrainedTokenizer instance must be provided.")
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0.")

        self.penalty = penalty
        self.temperature = temperature

        self.allowed_token_ids = torch.tensor(
            [
                tokenizer.convert_tokens_to_ids(token) for token in allowed_tokens
                if tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id
            ],
            dtype=torch.long
        ).unique()

        if len(self.allowed_token_ids) == 0:
            raise ValueError("None of the allowed tokens could be converted to valid token IDs.")

        self.cumulative_scores = None
        self.tokenizer = tokenizer
        
        # Debug mode: checking for token distribution
        self.debug_mode = debug_mode
        if self.debug_mode:
            self.evaluator = evaluator(allowed_tokens=allowed_tokens, allow_token_id=self.allowed_token_ids)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        
        # Debug mode: checking for token distribution
        if self.debug_mode:
            self.evaluator.set_in_token(torch.argmax(scores, dim=-1))

        # STCM algo init
        allowed_mask = torch.zeros(scores.size(-1), dtype=torch.bool, device=scores.device)
        allowed_mask[self.allowed_token_ids] = True

        # Add softmax
        scores = F.softmax(scores, dim=1)

        # Apply penalty
        if self.penalty is None:
            scores[:, ~allowed_mask] = -float('inf')
        else:
            scores[:, ~allowed_mask] -= self.penalty
            
        # Apply Temperature
        if self.temperature != 1.0:
            scores = scores / self.temperature

        if self.cumulative_scores is None:
            self.cumulative_scores = scores.clone()
        else:
            self.cumulative_scores += scores

        return scores

    def generate(self) -> list[str]:
        """generate"""
        if self.cumulative_scores is None:
            raise RuntimeError("No scores have been accumulated. Ensure generation has occurred.")

        top_token_ids = torch.argmax(self.cumulative_scores, dim=-1)
        self.cumulative_scores = None

        # Debug mode: checking for token distribution
        if self.debug_mode:
            self.evaluator.set_out_token(top_token_ids)

        return self.tokenizer.batch_decode(
            top_token_ids.unsqueeze(0),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
    # Function for debug_mode #
    def dump_debug(self):
        return self.evaluator.dump(tokenizer=self.tokenizer)
        