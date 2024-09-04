import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.util import OpenAIClientTool
from utils.prompt_template import format_extractInfoTag_from_query_tagContent_one_shots_message, double_check_negative_relevant, double_check_positive_relevant, format_extractInfoTag_from_query_tagContent_train_data, format_extractTag_from_query_tagContent_train_data
from utils.util import read_from_jsonl_into_list, split_and_tag, write_from_list_into_jsonl, get_tag_within_text, create_folder_if_not_exists

import os
import random

def extract_info_tag_from_openai(responses: list[list[str]], tag_data: list[dict]):
    extract_info_tag_data = []
    for idx in range(len(responses)):
        extract_tag = get_tag_within_text(responses[idx])
        extract_tag = "#Extracted Information Tag#: " + extract_tag
        json_data = {
            "request": tag_data[idx]['request'],
            "tagged_content": tag_data[idx]['tagged_content'],
            "extractInfoTag": responses[idx],
            "extractTag": extract_tag
        }
        extract_info_tag_data.append(json_data)
    save_folder = "./extract_data/"
    create_folder_if_not_exists(save_folder)
    write_from_list_into_jsonl(extract_info_tag_data, save_folder + "extract_info_tag.jsonl")
    return extract_info_tag_data

# def save_json_data(data, output_file, start_end):
#     output_file += f"{str(start_end[0])}_{str(start_end[1])}.jsonl"
#     with open(output_file, 'a', encoding='utf-8') as f:
#         for d in data:
#             json_d = json.dumps(d, ensure_ascii=False)
#             f.write(json_d)
#             f.write("\n")

def double_check(extract_info_tag_data: list[dict]):
    negative_relevant_data = []
    positive_relevant_data = []
    true_negative_relevant_data = []
    true_positive_relevant_data = []
    valid_datas = []

    true_negative_relevant_cnt = 0
    true_positive_relevant_cnt = 0
    valid_cnt = 0
    for data in extract_info_tag_data:
        if data['extractTag'] == None: continue
        if data['extractTag'] == "#Extracted Information Tag#: None":
            negative_relevant_data.append(data)
        else: 
            positive_relevant_data.append(data)
        valid_datas.append(data)
        valid_cnt += 1

    # 使用多线程批量生成响应
    openai_base_url = "https://api.openai.com/v1"
    openai_api_key = ""
    # yi_base_url = "https://api.lingyiwanwu.com/v1"
    # yi_api_key = ""
    client_tool = OpenAIClientTool(name="test", api_key=openai_api_key, base_url=openai_base_url)
    print(client_tool.client)

    model_name = "gpt-4o-2024-05-13"
    max_tokens = 4096
    temperature = 0.0
    top_p = 1.0
    n = 1

    prompt_negative_relevant = [double_check_negative_relevant(query=d['request'], tag_content=d['tagged_content']) for d in negative_relevant_data]

    responses= client_tool.batch_generate(batch_messages=prompt_negative_relevant, model_name=model_name, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n)

    for idx in range(len(responses)):
        if responses[idx] == None: continue
        response = responses[idx][0]
        if '#Double Check#: YES' in response or 'YES' in response:
            continue
        elif '#Double Check#: NO' in response or 'NO' in response:
            json_data = {
                "request": negative_relevant_data[idx]['request'],
                "tagged_content": negative_relevant_data[idx]['tagged_content'],
                "extractInfoTag": negative_relevant_data[idx]['extractInfoTag'],
                "extractTag": negative_relevant_data[idx]['extractTag'],
                "double_check": response
            }
            true_negative_relevant_data.append(json_data)
            true_negative_relevant_cnt += 1

    prompt_positive_relevant = [double_check_positive_relevant(query=d['request'], extractInfoTag=d['extractInfoTag']) for d in positive_relevant_data]

    responses= client_tool.batch_generate(batch_messages=prompt_positive_relevant, model_name=model_name, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n)

    for idx in range(len(responses)):
        if responses[idx] == None: continue
        response = responses[idx][0]
        if '#Double Check#: NO' in response:
            continue
        elif '#Double Check#: YES' in response:
            json_data = {
                "request": positive_relevant_data[idx]['request'],
                "tagged_content": positive_relevant_data[idx]['tagged_content'],
                "extractInfoTag": positive_relevant_data[idx]['extractInfoTag'],
                "extractTag": positive_relevant_data[idx]['extractTag'],
                "double_check": response
            }
            true_positive_relevant_data.append(json_data)
            true_positive_relevant_cnt += 1

    double_check_folder = "./extract_data/double_check/"
    create_folder_if_not_exists(double_check_folder)
    write_from_list_into_jsonl(true_negative_relevant_data, double_check_folder + f"{true_negative_relevant_cnt}_true_negative_relevant_data.jsonl")
    write_from_list_into_jsonl(true_positive_relevant_data, double_check_folder + f"{true_positive_relevant_cnt}_true_positive_relevant_data.jsonl")

    return true_negative_relevant_data, true_positive_relevant_data, true_negative_relevant_cnt, true_positive_relevant_cnt

def construct_instruction_set(true_negative_relevant_data: list[dict], true_positive_relevant_datal: list[dict], true_negative_relevant_cnt: int, true_positive_relevant_cnt: int, negative_ratio: float):
    chosen_num = int((negative_ratio / (1 - negative_ratio)) * true_positive_relevant_cnt)
    chosen_true_negative_relevant_data = random.sample(true_negative_relevant_data, chosen_num)
    merge_data = chosen_true_negative_relevant_data + true_positive_relevant_datal

    extract_info_tag_train_data = [format_extractInfoTag_from_query_tagContent_train_data(query=d['request'], content=d['tagged_content'], extractInfoTag=d['extractInfoTag']) for d in merge_data]
    extract_tag_train_data = [format_extractTag_from_query_tagContent_train_data(query=d['request'], content=d['tagged_content'], extractTag=d['extractTag']) for d in merge_data]
   
    all_train_data = []
    for i in range(len(extract_info_tag_train_data)):
        all_train_data.append(extract_info_tag_train_data[i])
        all_train_data.append(extract_tag_train_data[i])
    
    train_data_folder = "./instruction_set/zi-isag/data/"
    create_folder_if_not_exists(train_data_folder)
    write_from_list_into_jsonl(all_train_data, train_data_folder + "train.jsonl")
    write_from_list_into_jsonl(all_train_data[:1], train_data_folder + "eval.jsonl")




if __name__ == "__main__":
    """
    SPLIT AND TAG
    """
    open_folder = './raw_data/'
    open_file = open_folder + "example_notag.jsonl"
    data = read_from_jsonl_into_list(open_file)
    tag_data = []
    for d in data:
        json_d = {
            "request": d["request"],
            "tagged_content": split_and_tag(d["content"])
        }
        tag_data.append(json_d)
    write_from_list_into_jsonl(tag_data, "./raw_data/example_tag.jsonl")
    # exit()


    """
    EXTRACT INFO-TAG FROM SPLIT AND TAG
    """
    extract_info_tag_folder = "./extract_data/"
    if not (os.path.isfile(extract_info_tag_folder + "extract_info_tag.jsonl")):
        prompt_datas = [format_extractInfoTag_from_query_tagContent_one_shots_message(d["request"], split_and_tag(d["tagged_content"])) for d in tag_data]
        print(len(prompt_datas))
        openai_base_url = "https://api.openai.com/v1"
        openai_api_key = ""
        # yi_base_url = "https://api.lingyiwanwu.com/v1"
        # yi_api_key = ""
        client_tool = OpenAIClientTool(name="test", api_key=openai_api_key, base_url=openai_base_url)
        print(client_tool.client)

        # 使用多线程批量生成响应
        model_name = "gpt-4o-2024-05-13"
        max_tokens = 4096
        temperature = 0.0
        top_p = 1.0
        n = 1

        responses = client_tool.batch_generate(batch_messages=prompt_datas, model_name=model_name, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n)
        extract_info_tag_data = extract_info_tag_from_openai(responses, tag_data)
    else:
        extract_info_tag_data = read_from_jsonl_into_list(extract_info_tag_folder + "extract_info_tag.jsonl")
    # exit(0)

    """
    DOUBLE CHECK FROM EXTRACT INFO-TAG
    """
    true_negative_relevant_data, true_positive_relevant_data, true_negative_relevant_cnt, true_positive_relevant_cnt = double_check(extract_info_tag_data)

    # construct instruction set
    """
    CONSTRUCT INSTRUCTION SET FROM DOUBLE CHECK
    """
    construct_instruction_set(true_negative_relevant_data, true_positive_relevant_data, true_negative_relevant_cnt, true_positive_relevant_cnt, negative_ratio=0.5)


