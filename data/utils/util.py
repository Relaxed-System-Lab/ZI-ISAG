import json
import os

def parse_pv_log_search_result(search_result_text_list):
    result = []
    for search_result_text in search_result_text_list:
        # print(f"hgx: search_result_text = {search_result_text}")
        try:
            search_result = {}
            split_part = search_result_text.strip().split("\n")
            # print(f"hgx: split_part_len = {len(split_part)}")
            if len(split_part) < 3:
                continue
            pos_source_title = split_part[0].split(": ")
            if len(pos_source_title) >= 3:
                search_result["position"] = pos_source_title[0].strip()
                search_result["source"] = pos_source_title[1].strip()
                search_result["title"] = ": ".join(pos_source_title[2:])
            search_result["link"] = split_part[1]
            search_result["snippet"] = "\n".join(split_part[2:])
            result.append(search_result)
        except:
            pass
    # return json.dumps(result, ensure_ascii=False)
    return result

def get_all_url(search_result_text_list):
    results = parse_pv_log_search_result(search_result_text_list)
    urls = []
    for r in results:
        urls.append(r["link"])
    return urls

def get_valid_url(url: list[str], my_redis):
    valid_url = []
    for u in url:
        
        if my_redis.exists(u):
            ctx = my_redis.hgetall(u)
            if 'content' in ctx and len(ctx['content']) > 0:
                valid_url.append(u)
    return valid_url

def get_content_token_num_from_url(url, tokenizer, my_redis)->list[int]:
    token_num = []
    for u in url:
        ctx = my_redis.hgetall(u)
        if 'content' in ctx and len(ctx['content']) > 0:
            content = ctx['content']
            num = get_token_num_from_content(content, tokenizer)
            token_num.append(num)
        else:
            raise ValueError("No content found or content is empty. Please provide valid url.")
    return token_num

def get_token_num_from_content(content, tokenizer):
    tokens = tokenizer.encode(content)
    return len(tokens)

import re
import os
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        pass

def get_valid_content_from_url(url, token_num, my_redis):
    """
    如果content长度小于100，那么需要加上snippet，但是需要删去时间信息，因为有的信息为相对时间，类似“四天前”
    将文本中的…替换为...
    通过上面两条首先获得valid_content
    """
    data_dict = my_redis.hgetall(url)
    snippet = data_dict.get("sr_content", "")
    content = data_dict.get("content", "")
    
    idx = snippet.find("信源时间：")
    snippet = snippet[:idx]


    if token_num < 100:
        valid_content = snippet + content
    else:
        valid_content = content
    
    valid_content.replace('…', '...')
    
    return valid_content 

def get_tag_content(segments):
    """
    <T-n>...</T-n>
    """
    tag_idx = 1
    tag_content = ""
    for seg in segments:
        tag_content += f"<T-{tag_idx}>"
        tag_content += seg
        tag_content += f"</T-{tag_idx}>"
        tag_idx += 1

    return tag_content

import json
def write_to_json_file(json_data, filename):

    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False)

import re
class TextSplitterFSM:
    def __init__(self):
        self.state = 'START'
        self.current_segment = ''
        self.segments = []
        self.urls = []

    def extract_urls(self, text):
        url_pattern = re.compile(r'https?://[^\s/$.?#].[^\s]*')
        self.urls = [(m.start(), m.end()) for m in url_pattern.finditer(text)]
        return text

    def is_in_url(self, index):
        for start, end in self.urls:
            if start <= index < end:
                return True
        return False

    def process(self, text):
        text = self.extract_urls(text)
        i = 0
        while i < len(text):
            char = text[i]
            if self.state == 'START':
                if char.isspace():
                    self.current_segment += char
                else:
                    self.state = 'READING'
                    self.current_segment += char
            elif self.state == 'READING':
                if char == '.':
                    if self.is_in_url(i):
                        self.current_segment += char
                    elif i + 1 < len(text) and text[i + 1].isdigit():
                        self.current_segment += char
                    elif i > 0 and text[i - 1].isdigit():
                        self.current_segment += char
                    elif i + 2 < len(text) and text[i + 1] == '.' and text[i + 2] == '.':
                        self.current_segment += char
                    elif i + 1 < len(text) and text[i + 1] == '.':
                        self.current_segment += char
                    else:
                        self.current_segment += char
                        if i + 1 < len(text) and text[i + 1] == '"':
                            self.current_segment += text[i + 1]
                            i += 1
                        self.segments.append(self.current_segment)
                        self.current_segment = ''
                        self.state = 'START'
                elif char in '!?。！？':
                    self.current_segment += char
                    if i + 1 < len(text) and text[i + 1] in '’”"':
                        self.current_segment += text[i + 1]
                        i += 1
                    self.segments.append(self.current_segment)
                    self.current_segment = ''
                    self.state = 'START'
                elif char == ' ' and i + 1 < len(text) and text[i + 1] == ' ':
                    self.segments.append(self.current_segment)
                    self.current_segment = ''
                    self.state = 'START'
                else:
                    self.current_segment += char
            i += 1
        if self.current_segment:
            self.segments.append(self.current_segment)

    def get_segments(self):
        return self.segments
    
def split_and_tag(content: str):
    fsm = TextSplitterFSM()
    fsm.process(content)
    segments = fsm.get_segments()
    tag_content = get_tag_content(segments)
    return tag_content


from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from tqdm import tqdm
import openai

class OpenAIClientTool():
    def __init__(self, name: str, api_key: str = None, base_url: str = None) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url

        assert self.api_key, 'api_key is required'
        assert self.base_url, 'base_url is required'

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        assert 'model_name' in kwargs, 'model_name is required'
        model_name = kwargs['model_name']
        temperature = kwargs.get('temperature', 0.0)
        top_p = kwargs.get('top_p', 1.0)
        max_tokens = kwargs.get('max_tokens', None)
        n = kwargs.get('n', 1)
        presence_penalty = kwargs.get('presence_penalty', 0.0)
        frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        stop = kwargs.get('stop', None)

        completion = None

        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop
            )
            return [choice.message.content for choice in completion.choices]
        except Exception as e:
            print(f">>> Exception while process {model_name} api, completion: {completion}, error: {e}")
            return None

    def batch_generate(self, batch_messages: List[List[Dict[str, str]]], **kwargs) -> List[List[str]]:
        assert 'model_name' in kwargs, 'model_name is required'
        workers = kwargs.get('workers', 10)

        model_name = kwargs['model_name']
        temperature = kwargs.get('temperature', 0.0)
        top_p = kwargs.get('top_p', 1.0)
        max_tokens = kwargs.get('max_tokens', None)
        n = kwargs.get('n', 1)
        presence_penalty = kwargs.get('presence_penalty', 0.0)
        frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        stop = kwargs.get('stop', None)

        responses = [None] * len(batch_messages)  # Initialize a list with None to hold the responses

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    self.client.chat.completions.create,
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    stop=stop
                ): index for index, messages in enumerate(batch_messages)
            }

            for future in tqdm(as_completed(future_to_index), desc=f"{model_name} api processing", total=len(batch_messages)):
                index = future_to_index[future]
                completion = None
                try:
                    completion = future.result()
                    responses[index] = [choice.message.content for choice in completion.choices]  # Store the response at the correct index
                    # print(f"index: {index}; response: {responses[index]}")
                    # exit()
                except Exception as e:
                    print(f">>> Exception while processing {model_name} API, completion: {completion}, error: {e}. index: {index}")
                    responses[index] = None  # In case of an exception, store None at the correct index
            
        return responses

special_skip = ['k', 'j', '+', 'n', 'i']
def get_tagnum_possible_format_1(text: str):
    nums = set()
    str_ = text 
    str_ = str_.replace(' ', '')
    while str_.find('<T-') != -1:
        if str_.find('>') == -1: break
        num1_s = str_.find('<T-') + len('<T-')
        str_2 = str_[num1_s : ]
        num1_e = str_2.find('>')
        num1_e = num1_s + num1_e
        skip = False
        for sps in special_skip:
            if sps in str_[num1_s : num1_e]:
                str_ = str_[num1_e + 1 : ]
                skip = True
                break
        if skip: continue
        num1 = int(str_[num1_s : num1_e])
        str_ = str_[num1_e + 1 : ]
        if len(str_) > 6 and '>' in str_ and str_[0] == '~':
            num2_s = str_.find("<T-") + len('<T-')
            num2_e = str_.find('>')
            skip = False
            for sps in special_skip:
                if sps in str_[num2_s : num2_e]:
                    str_ = str_[num2_e + 1 : ]
                    skip = True
                    break
            if skip: continue
            num2 = int(str_[num2_s : num2_e])
            str_ = str_[num2_e + 1 : ]
            nums.update(range(num1, num2 + 1))
        else:
            nums.add(num1)
    return nums

def get_tagnum_possible_format_2(text):
    nums = set()
    str_ = text 
    str_ = str_.replace(' ', '')
    while str_.find('[T-') != -1:
        if str_.find(']') == -1: break
        num1_s = str_.find('[T-') + len('[T-')
        str_2 = str_[num1_s : ]
        num1_e = str_2.find(']')
        num1_e = num1_s + num1_e
        skip = False
        for sps in special_skip:
            if sps in str_[num1_s : num1_e]:
                str_ = str_[num1_e + 1 : ]
                skip = True
                break
        if skip: continue
        num1 = int(str_[num1_s : num1_e])
        str_ = str_[num1_e + 1 : ]
        if len(str_) > 6 and ']' in str_ and str_[0] == '~':
            num2_s = str_.find("[T-") + len('[T-')
            num2_e = str_.find(']')
            skip = False
            for sps in special_skip:
                if sps in str_[num2_s : num2_e]:
                    str_ = str_[num2_e + 1 : ]
                    skip = True
                    break
            if skip: continue
            num2 = int(str_[num2_s : num2_e])
            str_ = str_[num2_e + 1 : ]
            nums.update(range(num1, num2 + 1))
        else:
            nums.add(num1)
    return nums

def get_tagnum_possible(text: str):
    nums = get_tagnum_possible_format_1(text)
    if len(nums) == 0:
        nums = get_tagnum_possible_format_2(text)
    return nums

def get_tag_within_text(text: list[str]):
    if text == None:
        return None
    text = text[0]
    if 'none' in text or 'None' in text and len(text) < len("#Extracted Information#：None") + 10:
        return "None"
    text_t = text
    result = ""
    tag_within_text = []
    while text.find('[') != -1:
        left_bracket_idx = text.find('[')
        right_brackets_idx = text.find(']')
        if right_brackets_idx == -1: break
        tags = text[left_bracket_idx:right_brackets_idx+1]
        tags = tags.replace(" ", "")
        if '<' in tags and '>' in tags and 'T' in tags and '-' in tags:
            tag_within_text.append(tags)
        text = text[right_brackets_idx+1:]
        # if '<' in tags or '>' in tags or 'T' in tags or '-' in tags:
        #     if '<' not in tags or '>' not in tags or 'T' not in tags or '-' not in tags:
        #         print(f"this tags may have some promblem: {tags}")

    counter = 1
    result = "["
    if len(tag_within_text) == 0:
        print(f"find no tag in this text, check it manually: {text_t}")
    for rt in tag_within_text:
        result += rt + ", "
        counter += 1
    if len(tag_within_text) > 0:
        result = result[:len(result)-2]
    result += "]"
    if result == '[]': return "None"
    return result

def read_from_jsonl_into_list(file_name):
    datas = []
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            json_data = json.loads(line)
            datas.append(json_data)

    return datas

def read_from_endswith_jsonl_into_list(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]

    # 加载所有 JSON 文件并按 query_idx 分组
    datas = []
    for file_name in json_files:
        # query_idx = file_name.split('-')[0]
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    datas.append(json.loads(line))
    
    return datas

def read_from_endswith_json_into_list(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    # 加载所有 JSON 文件并按 query_idx 分组
    datas = []
    for file_name in json_files:
        # query_idx = file_name.split('-')[0]
        with open(file_name, 'r', encoding='utf-8') as file:
                # datas.append(json.load(file))
                datas += json.load(file)
    
    return datas

def write_from_list_into_json(datas, save_file):
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False)

def write_from_list_into_jsonl(datas, save_file):
    with open(save_file, 'w', encoding='utf-8') as f:
        for data in datas:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')  # 在每行末尾添加换行符




