'''
# You only need to run this once per machine
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
!pip install -q -U datasets scipy ipywidgets
'''
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from huggingface_hub import login
from datetime import datetime

from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

# check your torch version make sure it is GPU version
"""
from torch.cuda import is_available
print(is_available())
print(torch.__version__)
"""
# set up accelerator may not be necessary for QLoRA
#device = 'cuda:0'
#device = torch.device(device)

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# load in datasets
train_dataset = load_dataset('AiresPucrs/toxic-comments', split='train')
# eval takes too long
# eval_dataset = load_dataset('affahrizain/jigsaw-toxic-comment', split='dev')

# clean train data
def replace_value(data_point):
    data_point["toxic"] = "toxic" if data_point["toxic"] == 0 else "non-toxic"
    return data_point

train_dataset = train_dataset.map(replace_value)
train_dataset = train_dataset.shuffle(seed=1234)


# print to see the data point
print(train_dataset[134]["comment_text"])
print(train_dataset[134]["toxic"])
#print(eval_dataset)
#print(test_dataset)


# login to hugging face so we can load model from hugging face only once then model will be downloaded to your local machine
# Note: it will take up around 13 GB of storage 
# login() 


# load in the model
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# quantize to save memeory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)


#############################
####### Tokenization  #######
#############################
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=512,
    padding_side="left",
    add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single word.
                     This word should describe the target sentence accurately and the function must be one of the following ['toxic', 'non-toxic'].

                    ### Target sentence:
                    {data_point["comment_text"]}

                    ### Meaning representation:
                    {data_point["toxic"]}
                   """
    return tokenize(full_prompt)
####################################
####### End of Tokenization  #######
####################################

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
#tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

# print to see examples
#print(tokenized_train_dataset[4]['input_ids'])
#print(len(tokenized_train_dataset[4]['input_ids']))

'''
eval_prompt = """Given a target sentence construct the underlying meaning representation of the input sentence as a single word.
                     This word should describe the target sentence accurately and the function must be one of the following ['toxic', 'non-toxic'].

                    ### Target sentence:
                    you stupid fucking bitch shut the fuck up

                    ### Meaning representation:
                   """
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0], skip_special_tokens=True))

'''


# setup LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)
# print(model)

# Training
project = "toxic"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=70000,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        logging_steps=5000,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=5000,               # Save checkpoints every 5000 steps
        # evaluation_strategy="steps", # Evaluate the model every logging step
        # eval_steps=50,               # Evaluate and save checkpoints every 50 steps
        do_eval=False,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train('mistral-toxic/checkpoint-50')