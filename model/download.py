import os
current_dir = os.path.dirname(os.path.abspath(__file__))

path = current_dir
os.environ["HF_DATASETS_CACHE"] = current_dir
os.environ["HF_HOME"] = current_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = current_dir
os.environ["HF_HUB_CACHE"] = current_dir

from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# download BGEM#FlagModel
model = BGEM3FlagModel('BAAI/bge-m3')
print("bgem3 download done")

# download 9B-Chat-16K
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-9B-Chat-16K")
model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-1.5-9B-Chat-16K")
print(f"model size = {model.num_parameters()}")
del tokenizer
del model

# download 34B-Chat
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-1.5-34B-Chat-16K")
model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-1.5-34B-Chat-16K")
print(f"model size = {model.num_parameters()}")
del tokenizer
del model

# download qwen2-72b
model_name = "Qwen/Qwen2-72B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).cpu()
print(f"model size = {model.num_parameters()}")
del tokenizer
del model