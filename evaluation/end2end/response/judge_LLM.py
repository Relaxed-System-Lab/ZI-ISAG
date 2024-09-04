import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
# print(f"import path = {current_dir + '/../../data/'}")
sys.path.append(current_dir + '/../../../data/')

import json
from utils.util import OpenAIClientTool, create_folder_if_not_exists
from utils.prompt_template import format_extractInfoTag_from_query_tagContent_one_shots_message
from utils.util import read_from_jsonl_into_list, write_from_list_into_jsonl

import os
import json

def save_json_data(data, output_file, start_end):
    output_file += f"{str(start_end[0])}_{str(start_end[1])}.jsonl"
    with open(output_file, 'a', encoding='utf-8') as f:
        for d in data:
            json_d = json.dumps(d, ensure_ascii=False)
            f.write(json_d)
            f.write("\n")
        # json_data = json.dumps(data, ensure_ascii=False)
        # f.write(json_data)
        # f.write("\n")
        # json.dump(data, f, ensure_ascii=False)

def compare_two_response(file1: str, file2: str, front: bool):
    compare_1 = read_from_jsonl_into_list(file1)
    compare_2 = read_from_jsonl_into_list(file2)

    compare_data = []
    for i in range(len(compare_1)):
        cp1 = compare_1[i]
        cp2 = compare_2[i]

        cp1_request = cp1['request']
        cp2_request = cp2['request']
        assert cp1_request == cp2_request

        json_data = {
            "request": cp1_request,
            "cp1_response": cp1['response'],
            "cp2_response": cp2['response']
        }
        compare_data.append(json_data)

    compare_prompt = [
        {
            "role": "user",
            "content": 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n\n[User Question]\n{question}\n\n[The Start of Assistant A’s Answer]\n{answer_a}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{answer_b}\n[The End of Assistant B’s Answer]'
        }
    ]

    def format_compare_prompt(response1, response2, request):
        # messages_t = copy.deepcopy(messages)
        return [
            {
                "role": "user",
                "content": compare_prompt[0]['content'].format(question=request, answer_a=response1, answer_b=response2)
            }
        ]
    prompt_data = [format_compare_prompt(request=d['request'], response1=d['cp1_response'], response2=d['cp2_response']) for d in compare_data]

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

    save_folder = "./compare_result/"

    create_folder_if_not_exists(save_folder)

    responses = client_tool.batch_generate(batch_messages=prompt_data, model_name=model_name, max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=n)

    compare_result_data = []
    for i in range(len(compare_data)):
        json_data = {
            "request": compare_data[i]["request"],
            "cp1_response": compare_data[i]['cp1_response'],
            "cp2_response": compare_data[i]['cp2_response'],
            "comparison_reault": responses[i],
        }
        compare_result_data.append(json_data)

    if front:
        write_from_list_into_jsonl(compare_result_data, save_folder + 'font_compare_result.jsonl')
    else:
        write_from_list_into_jsonl(compare_result_data, save_folder + 'back_compare_result.jsonl')
    return compare_result_data

if __name__ == "__main__":
    open_folder = './Internet-SAG-Ext/'
    open_child_folder = 'Yi-large_response/'
    open_folder += open_child_folder
    open_file = 'example.jsonl'
    my_file = open_folder + open_file

    open_folder = './Internet-SAG-Naive/'
    open_child_folder = 'Yi-large_response/'
    open_folder += open_child_folder
    open_file = 'example.jsonl'
    baseline_file = open_folder + open_file

    front = True
    front_compare_result = compare_two_response(my_file, baseline_file, front)
    front = False
    back_compare_result = compare_two_response(baseline_file, my_file, front)

    """
    POSITION BIAS
    """
    datas = []
    win = 0
    tie = 0
    lose = 0
    position_bias_case_num = 0
    valid = 0
    none = 0
    for i in range(len(front_compare_result)):
        front_win = front_compare_result[i]
        back_win = back_compare_result[i]

        assert front_win['request'] == back_win['request']

        skip = False
        if front_win['comparison_reault'] == None:
            print(f"idx: {i + 1} front_compare_result is None")
            none += 1
            skip = True
        if back_win['comparison_reault'] == None:
            print(f"idx: {i + 1}, back_compare_result is None")
            skip = True
        if skip: continue

        front_win = front_win['comparison_reault'][0]
        back_win = back_win['comparison_reault'][0]

        valid += 1

        if '[[A]]' in front_win and '[[B]]' in back_win:
            win += 1
        elif '[[B]]' in front_win and '[[A]]' in back_win:
            lose += 1
        elif '[[C]]' in front_win and '[[C]]' in back_win:
            tie += 1
        else:
            position_bias_case_num += 1

    result_json = [{
        "win": win,
        "tie": tie,
        "lose": lose,
        "position bias": position_bias_case_num,
        "none": none
    }]
    result_folder = './compare_result/'
    create_folder_if_not_exists(result_folder)
    write_from_list_into_jsonl(result_json, result_folder + "final_result.jsonl")
    
    print(f"win: {win}; tie: {tie}; lose: {lose}; position_bias_case_num: {position_bias_case_num}; none: {none}")