import os
import sys
from needle_utils import logistic
from needle_utils import read_context_files, encode_and_trim, insert_needles
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# print(f"import path = {current_dir + '/../../../data/'}")
sys.path.append(current_dir + '/../../../data/')
model_path = current_dir + '/../../../model/' # YOUR LOCAL MODEL PATH

os.environ["HF_DATASETS_CACHE"] = model_path
os.environ["HF_HOME"] = model_path
os.environ["HUGGINGFACE_HUB_CACHE"] = model_path
os.environ["HF_HUB_CACHE"] = model_path
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

from utils.util import OpenAIClientTool
from utils.prompt_template import  format_extractNeedleTag_from_query_tagContent_zero_shots_message
from utils.util import TextSplitterFSM, read_from_jsonl_into_list, get_tagnum_possible, create_folder_if_not_exists, write_from_list_into_jsonl

def get_reference(content, nums):
    nums_sorted = sorted(nums)
    r = ""    
    last_num = nums_sorted[0] - 1

    for num in nums_sorted:
        start = f"<T-{num}>"
        end = f"</T-{num}>"
        content_partial = content[content.find(start) + len(start) : content.find(end)]
        if num == last_num + 1:
            r += content_partial
        else:
            r += '\n\n' + content_partial
        last_num = num
    
    return r

def get_scores(nums, ground_truth_tags):
    common_num = 0
    for gt in ground_truth_tags:
        if gt in nums:
            common_num += 1
    if common_num == 0: return 0, 0, 0
    else:
        r_score = common_num / len(ground_truth_tags)
        p_score = common_num / len(nums)
        f1_score = (2 * r_score * p_score) / (r_score + p_score)
        return f1_score


model_name = '/xpfs/public/gxhe/workspace/model_ckpts/v12_test_yi-9B_weightDecat-0._minlr-2e6_Deduplicate_xnon-3860_non-204_syn-78_gpunum-8_globalbs-16_lr-6e6_cosine_epoch-3/3-0'
haystack_name = "PaulGrahamEssays"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16).eval()

print(f"model size = {model.num_parameters()}")
print(f"model dtype = {model.dtype}")

# 存储和打印
testing_results = []
print_ongoing_status = True
save_results = False

context_lengths = None
context_lengths_min = 1024
context_lengths_max = 12 * 1024
context_lengths_num_intervals = 5

document_depth_percent_interval_type = "linear"
document_depth_percents = None
document_depth_percent_min = 0
document_depth_percent_max = 100
document_depth_percent_intervals = 5

needle_tests_examples = read_from_jsonl_into_list("./needles.jsonl")
retrieval_question = needle_tests_examples[0]['request']
needles = needle_tests_examples[0]['needles']
# retrieval_question = "Which band was the first to perform on the Moon? Who is the ruler of the Alpha Bot star system? What legendary item is hidden on Mythical Island?"
# needles = [' The first band to play on the Moon was the Virtual Rocket Band.', ' The ruler of the Alpha Bot star system is Intelligent Robot Zambot.', ' Hidden on Mythical Island is the legendary Dream Bubble.']

def check_ground_truth(seg, needles):
    for needle in needles:
        if needle in seg:
            return True
    return False


if context_lengths is None:
    if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
        raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
    else:
        context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
else:
    context_lengths = context_lengths

if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
    raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")

if document_depth_percents is None:
    if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
        raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
    
    if document_depth_percent_interval_type == 'linear':
        document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
    elif document_depth_percent_interval_type == 'sigmoid':
        document_depth_percents = [logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
    else:
        raise ValueError("document_depth_percent_interval_type must be either 'sigmoid' or 'linear' if document_depth_percents is None.")
else:
    document_depth_percents = document_depth_percents


sum_f1_score = 0
test_num = 0
client_create = False
for context_length in context_lengths:
    context_tmp = read_context_files(context_lengths, tokenizer, haystack_name)
    context_tmp = encode_and_trim(context_tmp, context_length, tokenizer)
    for depth_percent in document_depth_percents:

        context, insertion_percentages = insert_needles(context_tmp, depth_percent, context_length, tokenizer, needles)

        fsm = TextSplitterFSM()
        fsm.process(context)
        segments = fsm.get_segments()

        tag_content = ""
        idx = 1
        ground_truth_tags = []
        for seg in segments:
            tag_content += f"<T-{idx}>"
            tag_content += seg
            tag_content += f"</T-{idx}>"
            if check_ground_truth(seg, needles):
                ground_truth_tags.append(idx)
            idx += 1
        prompt = [format_extractNeedleTag_from_query_tagContent_zero_shots_message(query=retrieval_question, content=tag_content)]
        context = tag_content

        if client_create == False:
            openai_base_url = "https://api.openai.com/v1"
            openai_api_key = ""
            # yi_base_url = "https://api.lingyiwanwu.com/v1"
            # yi_api_key = ""
            client_tool = OpenAIClientTool(name="test", api_key=openai_api_key, base_url=openai_base_url)
            print(client_tool.client)
            client_create = True

        # 使用多线程批量生成响应
        model_name = "gpt-4o-2024-05-13"
        max_tokens = 200
        temperature = 0.0
        top_p = 1.0
        n = 1

        responses = client_tool.batch_generate(batch_messages=prompt, model_name=model_name, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n)  

        while responses[0] == None:
            print(f"[DEBUG] CALL API RETURN NONE")
            responses, s_e = client_tool.batch_generate(batch_messages=prompt, model_name=model_name, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n, start_end=None)  

        if responses[0] == None: 
            nums = []
            response = None
        else:
            response = responses[0][0]
            nums = get_tagnum_possible(response)

        if len(nums) == 0:
            reference = None
            f1_score = 0
        else:
            reference = get_reference(tag_content, nums)
            f1_score = get_scores(nums, ground_truth_tags)

        results = {
            'model' : model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'needle' : needles,
            'model_response' : response,
            'tag_to_content' : reference,
            'f1_score' : f1_score,
        }
        
        sum_f1_score += f1_score
        test_num += 1
        testing_results.append(results)

        if print_ongoing_status:
            print (f"-- Test Summary -- ")
            print(f"nums in response = {sorted(nums)}")
            print(f"ground_truth_tags = {ground_truth_tags}")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"f1_score: {f1_score:.4f}")
            print (f"Insertion Percentages: {insertion_percentages}\n")
            print (f"Response: {response}\n")
            print(f"Tag num: {len(nums)}\n")
            print (f"tag_to_content: {reference}\n")

        needles_name = "api_needle_test"
        result_folder = f'./result/{haystack_name}/{needles_name}/'
        create_folder_if_not_exists(result_folder)
        if save_results:
            write_from_list_into_jsonl(results, result_folder + f"len_{context_length}_depth_{int(depth_percent)}.jsonl")

final_result = [{
    "avg_f1_score": sum_f1_score /test_num
}]
final_result_folder = f'./result/{haystack_name}/{needles_name}/'
create_folder_if_not_exists(final_result_folder)
write_from_list_into_jsonl(final_result, final_result_folder + '1_final_result.jsonl')
print(f"final result f1 = {sum_f1_score / test_num}")