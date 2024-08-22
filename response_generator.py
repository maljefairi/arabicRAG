# response_generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import setup_logger
from config import Config

logger = setup_logger('response_generator')

class ResponseGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(Config.LLM_MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Model loaded and moved to {self.device}")

    def generate_response(self, query, relevant_docs):
        try:
            context = self._prepare_context(relevant_docs)
            prompt = self._create_prompt(query, context)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).float()
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=Config.MAX_LENGTH,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7
                )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return self._extract_answer(response)
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "عذرًا، لم أتمكن من إنشاء استجابة بسبب خطأ ما."  # "Sorry, I couldn't generate a response due to an error."

    def _prepare_context(self, relevant_docs):
        # Combine content from relevant documents
        combined_content = "\n".join(relevant_docs['content'].tolist())
        # Truncate if too long
        max_context_length = Config.MAX_LENGTH // 2  # Use half of max_length for context
        return combined_content[:max_context_length]

    def _create_prompt(self, query, context):
        return f"""مستند قانوني:
{context}

سؤال:
{query}

إجابة:"""

    def _extract_answer(self, response):
        # Extract the generated answer from the full response
        answer_start = response.find("إجابة:") + len("إجابة:")
        return response[answer_start:].strip()

    def update_model(self, new_model_name):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(new_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(new_model_name)
            self.model.to(self.device)
            logger.info(f"Model updated to {new_model_name}")
        except Exception as e:
            logger.error(f"Error updating model: {e}")