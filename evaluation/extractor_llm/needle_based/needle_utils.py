import os
import json
from transformers import TextStreamer


def logistic(x, L=100, x0=50, k=.1):
        if x in [0, 100]:
            return x
        x = -k * (x - x0)
        return np.round(L * sigmoid(x), 3)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def result_exists(context_length, depth_percent, results_version, model_name, haystack_name):
    """
    Checks to see if a result has already been evaluated or not
    """

    results_dir = f'results/{model_name.replace("/", "_")}/{haystack_name}'
    if not os.path.exists(results_dir):
        return False
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            with open(os.path.join(results_dir, filename), 'r') as f:
                result = json.load(f)
                context_length_met = result['context_length'] == context_length
                depth_percent_met = result['depth_percent'] == depth_percent
                version_met = result.get('version', 1) == results_version
                model_met = result['model'] == model_name
                if context_length_met and depth_percent_met and version_met and model_met:
                    return True
    return False


def get_context_length_in_tokens(context, tokenizer):
    messages = [
        {"role": "user", "content": context}
    ]
    tokens = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')[0]
    # print(f"hgx: get_context_length_in_tokens = {len(tokens)}")
    return len(tokens)

def get_context_length_and_context_in_tokens(context, tokenizer):
    messages = [
        {"role": "user", "content": context}
    ]
    tokens = tokenizer.apply_chat_template(conversation=messages, tokenize=True, return_tensors='pt')[0]
    # print(f"hgx: get_context_length_in_tokens = {len(tokens)}")
    return len(tokens), tokens

def read_context_files(context_lengths, tokenizer, haystack_name):
    context = ""
    max_context_length = max(context_lengths)
    base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

    with open(base_dir+f"/needlebench/{haystack_name}.jsonl", 'r') as file:
        for line in file:
            if get_context_length_in_tokens(context, tokenizer) < max_context_length:
                data = json.loads(line)
                text = data.get("text", "")
                context += text
            else:
                break
    return context      

def encode_and_trim(context, context_length, tokenizer):
    tokens = tokenizer.encode(context)
    # tokens_needle = torch.tensor(tokenizer.encode(needles))
    if len(tokens) > context_length:
        context = tokenizer.decode(tokens[:context_length])
    return context

def insert_needles(context, depth_percent, context_length, tokenizer, needles):

    tokens_context = tokenizer.encode(context)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= 200

    total_needles_length = sum(len(tokenizer.encode(needle)) for needle in needles)

    if len(tokens_context) + total_needles_length > context_length:
        tokens_context = tokens_context[:context_length - total_needles_length]

    depth_percent_interval = (100 - depth_percent) / len(needles)

    insertion_percentages = []

    for needle in needles:
        tokens_needle = tokenizer.encode(needle)
        
        if depth_percent == 100:
            tokens_context = tokens_context + tokens_needle
            insertion_point = len(tokens_context)
            insertion_percentages.append(insertion_point)
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            tokens_new_context = tokens_context[:insertion_point]

            period_tokens_zh = tokenizer.encode('。')[0]
            period_tokens_en = tokenizer.encode('.')[0]
            period_tokens = [period_tokens_zh, period_tokens_en]

            while len(tokens_new_context) > 0 and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                # print(f"hgx: insertion_point = {insertion_point}")
                tokens_new_context = tokens_context[:insertion_point]
            
            tokens_context = tokens_context[:insertion_point] + tokens_needle + tokens_context[insertion_point:]

            insertion_percentage = (insertion_point / len(tokens_context)) * 100
            insertion_percentages.append(insertion_point)

            depth_percent += depth_percent_interval
    
    new_context = tokenizer.decode(tokens_context)
    # print(f"hgx: new_context = {new_context}")
    return new_context, insertion_percentages


def evaluate_model(model, tokenizer, prompt: str) -> str:
    """
    Evaluates a given prompt using the OpenAI model and retrieves the model's response.

    Args:
        prompt (str): The prompt to send to the model.

    Returns:
        str: The content of the model's response to the prompt.
    """

    input_ids = tokenizer.apply_chat_template(conversation=prompt, tokenize=True, return_tensors='pt')
    output_ids = model.generate(input_ids.to('cuda'), max_new_tokens=200, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    return response