extractInfoTag_from_query_tagContent_tmplate = [
    {
        "role": "user",
        "content": "Your task is to filter and extract the relevant information and corresponding tags from the #Reference Materials#. The extracted information will be included as reference content in a prompt given to a generative AI. This generative AI will use this prompt in a downstream task to generate a response to the #User Query#. Although your task does not include generating the response to the #User Query#, you must ensure the completeness, richness, and accuracy of the information you extract to guarantee that the generative AI in the downstream task has rich, complete, and accurate relevant information when generating the response to the #User Query#.\nThe content of the #Reference Materials# provided is enclosed within the tags <Reference Materials> and </Reference Materials>, while the content of the #User Query# is enclosed within the tags <User Query> and </User Query>. The #Reference Materials# are divided into multiple segments, with each segment's content enclosed within its corresponding tags. For example, the content of the i-th segment is enclosed within the tags <T-i> and </T-i>.\n\n#Reference Materials#:<Reference Materials>{content}</Reference Materials>\n\n#User Query#:<User Query>{query}</User Query>\n\nPlease extract information from the #Reference Materials# based on the #User Query# by following these steps:\n1. Filter out content in the #Reference Materials# that is irrelevant or weakly related to the #User Query#.\n2. Extract all relevant information and their corresponding tags from the #Reference Materials# needed to generate a response to the #User Query#, with a primary focus on highly relevant information\n\nReturn the result in the following format: '#Extracted Information#:1.___[<T-i>, ...]\n2.___[<T-j>, <T-k>~<T-k+m>, <T-n>]\n...'. Output the key content of each piece of information that meets the extraction requirements (keep the key content concise and ensure completeness of each extracted piece of information), and note the associated tags in brackets [].(A piece of information may contain multiple segments and therefore may be associated with multiple tags.)\n\nAttention Points:\n- If a group of output tags contains consecutive labels, connect the first and last tags in the group with a '~' symbol.\n- If there are multiple pieces of effective information in the #Reference Materials# with similar key content, do not de-duplicate. Extract all such information and their tag sets and merge them into one combined piece of information and one combined tag set. Ensure that the combined tag set still includes all the tags associated with the original information before merging.\n- For multi-hop reasoning, extract and retain all information and their tags needed in the reasoning process.\n- If no relevant information is extracted, output '#Extracted Information#:None'.\n- The language of the extracted information should be consistent with the original text in the #Reference Materials#."
    }
]

extractInfoTag_from_query_tagContent_one_shots_example_1 = "Example 1: #Reference Materials#: <Reference Materials><T-1>小张在2000年成立了公司C...</T-1><T-2>公司C的总部位于纽约...</T-2><T-3>Company C acquires Company D in 2020...</T-3><T-4>公司D是世界领先的科技公司...</T-4><T-5>公司D的主要业务包括...</T-5><T-6>公司D还有一些别的次要业务...</T-6><T-7>除此之外，公司D还有一些潜在业务...</T-7><T-8>在收购了公司D后，公司C的市值达到了...</T-8><T-9>公司C的业务是...</T-9></Reference Materials>\n\n#User Query#:<User Query>What are the businesses of Zhang's company?</User Query>\n\nExtracted Information:#Extracted Information#:1. 小张在2000年成立了公司C[<T-1>]\n2. Company C acquires Company D in 2020[<T-3>]\n3. 公司D的业务包括...[<T-5>~<T-7>]\n4. 公司C的业务是...[<T-9>]\nAbove is an example."

def format_extractInfoTag_from_query_tagContent_zero_shots_message(query, content):
    return [
        {
            "role": extractInfoTag_from_query_tagContent_tmplate[0]['role'],
            "content": extractInfoTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content)
        }
    ]

def format_extractInfoTag_from_query_tagContent_zero_shots_prompt(query, content):
    return "<|im_start|>user\n" + extractInfoTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content) + "<|im_end|>\n<|im_start|>assistant\n"

def format_extractInfoTag_from_query_tagContent_one_shots_message(query, content):
    return [
        {
            "role": extractInfoTag_from_query_tagContent_tmplate[0]['role'],
            "content": extractInfoTag_from_query_tagContent_one_shots_example_1 + ' ' + extractInfoTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content)
        }
    ]

def format_extractInfoTag_from_query_tagContent_one_shots_prompt(query, content):
    return "<|im_start|>user\n" + extractInfoTag_from_query_tagContent_one_shots_example_1 + ' ' + extractInfoTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content) + "<|im_end|>\n<|im_start|>assistant\n"

def format_extractInfoTag_from_query_tagContent_train_data(query, content, extractInfoTag):
    return {
        "prompt": "<|im_start|>user\n" + extractInfoTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content) + "<|im_end|>\n",
        "chosen": "<|im_start|>assistant\n{extractInfoTag}<|im_end|>\n".format(extractInfoTag=extractInfoTag)
    }


extractTag_from_query_tagContent_tmplate = [
    {
        "role": "user",
        "content": (
            "Your task is to filter and extract the relevant tags from the #Reference Materials#. The content of extracted tags will be included as reference content in a prompt given to a generative AI. This generative AI will use this prompt in a downstream task to generate a response to the #User Query#. Although your task does not include generating the response to the #User Query#, you must ensure the completeness, richness, and accuracy of the tags you extract to guarantee that the generative AI in the downstream task has rich, complete, and accurate relevant information when generating the response to the #User Query#.\nThe content of the #Reference Materials# provided is enclosed within the tags <Reference Materials> and </Reference Materials>, while the content of the #User Query# is enclosed within the tags <User Query> and </User Query>. The #Reference Materials# are divided into multiple segments, with each segment's content enclosed within its corresponding tags. For example, the content of the i-th segment is enclosed within the tags <T-i> and </T-i>.\n\n#Reference Materials#:<Reference Materials>{content}</Reference Materials>\n\n#User Query#:<User Query>{query}</User Query>\n\nPlease extract tags from the #Reference Materials# based on the #User Query# by following these steps:\n1. Filter out content in the #Reference Materials# that is irrelevant or weakly related to the #User Query#.\n2. Extract all tags from the #Reference Materials# that contain information needed to generate a response to the #User Query#, with a primary focus on highly relevant information\n\nReturn the result in the following format: '#Extracted Information Tag#:[[<T-i>, ...], [<T-j>, <T-k>~<T-k+m>], <T-n>...], ...]'. Output the tags of each piece of information that meets the extraction requirements (ensure the completeness of the information for each set of extracted tags).\n\nAttention Points:\n- If a group of output tags contains consecutive labels, connect the first and last tags in the group with a '~' symbol.\n- If there are multiple pieces of effective information in the #Reference Materials# with similar key content, do not de-duplicate. Extract tags of all such information and merge them into one combined tag set. Ensure that the combined tag set still includes all the tags associated with the original information before merging.\n- For multi-hop reasoning, extract and retain tags of all information needed in the reasoning process.\n- If no relevant tags are extracted, output '#Extracted Information Tag#:None'."
        ) 
    }
]

extractTag_from_query_tagContent_one_shots_example_1 = "Example 1: #Reference Materials#: <Reference Materials><T-1>小张在2000年成立了公司C...</T-1><T-2>公司C的总部位于纽约...</T-2><T-3>Company C acquires Company D in 2020...</T-3><T-4>公司D是世界领先的科技公司...</T-4><T-5>公司D的主要业务包括...</T-5><T-6>公司D还有一些别的次要业务...</T-6><T-7>除此之外，公司D还有一些潜在业务...</T-7><T-8>在收购了公司D后，公司C的市值达到了...</T-8><T-9>公司C的业务是...</T-9></Reference Materials>\n\n#User Query#:<User Query>What are the businesses of Zhang's company?</User Query>\n\nExtracted Information Tag:#Extracted Information Tag#:[[<T-1>], [<T-3>], [<T-5>~<T-7>], [<T-9>]]\nAbove is an example."

def format_extractTag_from_query_tagContent_zero_shots_message(query, content):
    return [
        {
             "role": extractTag_from_query_tagContent_tmplate[0]['role'],
            "content": extractTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content)
        }
    ]

def format_extractTag_from_query_tagContent_zero_shots_prompt(query, content):
    return "<|im_start|>user\n" + extractTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content) + "<|im_end|>\n<|im_start|>assistant\n"

def format_extractTag_from_query_tagContent_one_shots_message(query, content):
    return [
        {
            "role": extractTag_from_query_tagContent_tmplate[0]['role'],
            "content": extractTag_from_query_tagContent_one_shots_example_1 + ' ' + extractTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content)
        }
    ]

def format_extractTag_from_query_tagContent_one_shots_prompt(query, content):
    return "<|im_start|>user\n" + extractTag_from_query_tagContent_one_shots_example_1 + ' ' + extractTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content) + "<|im_end|>\n<|im_start|>assistant\n"

def format_extractTag_from_query_tagContent_train_data(query, content, extractTag):
    return {
        "prompt": "<|im_start|>user\n" + extractTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content) + "<|im_end|>\n",
        "chosen": "<|im_start|>assistant\n{extractTag}<|im_end|>\n".format(extractTag=extractTag)
    }


def double_check_negative_relevant(query, tag_content):
    """
    if true negative relevant output NO
    if false negative relevant output YES
    """
    return [
        {
            "role": 'user',
            "content": f"Your task is to determine if the #Reference Materials# contain useful information for generating a response to the #User Query#. This useful information will be included as reference content in a prompt for a generative AI, which will use it to respond to the #User Query#. Although you are not generating the response, you must assess whether the #Reference Materials# provide useful information for the generative AI's downstream task. Note that the information does not need to directly answer the #User Query#; indirect information, useful context, or relevant background are all valuable.If the #Reference Materials# contain useful information for responding to the #User Query#, output '#Double Check#: YES'. If not, output '#Double Check#: NO' and provide concise reasons.\n\n#Reference Materials#:{tag_content}\n\n#User Query#:{query}"
        }
    ]

def double_check_positive_relevant(query, extractInfoTag):
    """
    if false positive relevant output NO
    if true positive relevant output YES
    """
    return [
        {
            "role": 'user',
            "content": f"Your task is to judge whether the #Extracted Information# is useful for generating a response to the #User Query#. The #Extracted Information# will be included as reference content in a prompt given to a generative AI. This generative AI will use this prompt in a downstream task to generate a response to the #User Query#. Although your task does not include generating the response to the #User Query#, you must ensure the #Extracted Information# is useful for the generative AI in the downstream task to generate the response to the #User Query#. Note that the #Extracted Information# does not need to directly answer the #User Query# to be considered useful. If it provides indirect information, useful context, or relevant background, it is still useful. If the #Extracted Information# is useful for generating a response to the #User Query#, output '#Double Check#: YES'. If not, output '#Double Check#: NO', and give concise reasons.\n\n#Extracted Information#:{extractInfoTag}\n\n#User Query#:{query}"
        }
    ]


extractNeedleTag_from_query_tagContent_tmplate = [
    {
        "role": "user",
        "content": "Your task is to filter and extract ALL the TAG from the #Reference Materials# that will aid in generating the response to the #User Query#. \n\n#Reference Materials#:<Reference Materials>{content}</Reference Materials>\n\n#User Query#:<User Query>{query}</User Query>\n\nOnly output the TAGs of ALL the relevant information. Output in the following format: '[<T-i>], [<T-j>], ...'."
    }
]

def format_extractNeedleTag_from_query_tagContent_zero_shots_prompt(query, content):
    return "<|im_start|>user\n" + extractNeedleTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content) + "<|im_end|>\n<|im_start|>assistant\n"

def format_extractNeedleTag_from_query_tagContent_zero_shots_message(query, content):
    return [
        {
            "role": extractNeedleTag_from_query_tagContent_tmplate[0]['role'],
            "content": extractNeedleTag_from_query_tagContent_tmplate[0]['content'].format(query=query, content=content)
        }
    ]


inference_evaluation_tmplate = [
    {
        "role": "user",
        "content": "Reference Materials: {reference}. Please follow these instructions carefully: 1. The references are sourced from the Internet, so critically evaluate and use only those relevant to the Query. Ignore irrelevant content, but ensure no relevant information is omitted. Provide a comprehensive response, and make your answer as rich in detail as possible. 2. Answer the Query with thorough reasoning, adhering to facts and logic. 3. Unless specified otherwise, respond in Markdown formatting. 4. Unless specified otherwise, respond in the same language as the Query. Now, using the Reference Materials above (if any), answer the following Query: {query}"
    }
]

llm_judge_prompt = [
        {
            "role": "user",
            "content": 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.\n\n[User Question]\n{question}\n\n[The Start of Assistant A’s Answer]\n{answer_a}\n[The End of Assistant A’s Answer]\n\n[The Start of Assistant B’s Answer]\n{answer_b}\n[The End of Assistant B’s Answer]'
        }
    ]

