import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from huggingface_hub import login

from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# quantize to save memeory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

#############################
####### Tokenization  #######
#############################
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Mistral, same as before
    quantization_config=bnb_config,  # Same quantization config as before
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)
#tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token
ft_model = PeftModel.from_pretrained(base_model, "mistral-toxic/checkpoint-100")

def mistral_toxic():
   while True:
      query = input("Enter your prompt: ")
      messages = [
         {"role": "user", "content": str(query)}
      ]
      encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
      generated_ids = ft_model.generate(encodeds, max_new_tokens=300, do_sample=False)
      decoded = tokenizer.batch_decode(generated_ids)
      print("Query: ",query)
      print("Result")
      print(decoded[0])

mistral_toxic()