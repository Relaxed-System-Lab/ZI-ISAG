import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# print(f"import path = {current_dir + '/../../../data/'}")
sys.path.append(current_dir + '/../../../data/')
from utils.util import get_tagnum_possible 
"""
For positive relevant data, we calculate Recall, Presision and F1 score
based on the model's TAG INFERENCE, and the Gtound Truth TAG

For negative relevant data, we calculate Exact Match Ratio (EM) score
based on the whole tagged_content, and model's TAG OUTPUT.
in this case, Ground Truth TAG is NONE
"""
def positive_relevant_data_score(model_infer: str, ground_truth: str):
    infer_nums = get_tagnum_possible(model_infer)
    ground_truth_nums = get_tagnum_possible(ground_truth)

    if len(infer_nums) == 0:
        return 0, 0, 0

    common_num = 0
    for n in infer_nums:
        if n in ground_truth_nums: common_num += 1
    recall = common_num / len(ground_truth_nums)
    precision = common_num / len(infer_nums)
    f1 = 2 * (recall * precision) / (recall + precision)

    return recall, precision, f1

def negative_relevant_data_score(tagged_content: str, model_infer: str):
    tagged_content_nums = get_tagnum_possible(tagged_content)
    infer_nums = get_tagnum_possible(model_infer)
    """
    consider LLM may be error in outputting tags without stopping,
    so we need calculate max(0, ...)
    """
    em = max(0, 1 - (len(infer_nums) / len(tagged_content_nums)))
    return em