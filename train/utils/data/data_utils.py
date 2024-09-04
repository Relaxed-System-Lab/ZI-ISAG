# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib
from itertools import chain
from . import raw_datasets


def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    print(f"{dataset_name=}")

    if "rm-static" in dataset_name:
        print('rm-static')
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, local_rank, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               "mkqa")
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                "mkqa")
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    elif "pku" in dataset_name or 'arena' in dataset_name:
        return raw_datasets.PkuDataset(output_path, seed, local_rank,
                                                 dataset_name)
    elif "local/jsonfile" in dataset_name:
        chat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir,
                         os.path.pardir, os.path.pardir))
        if not (os.path.isfile(chat_path + '/data/train.json')
                and os.path.isfile(chat_path + '/data/eval.json')):
            raise RuntimeError(
                f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
            )
        return raw_datasets.LocalJsonFileDataset(output_path, seed, local_rank,
                                                 dataset_name, chat_path)
    elif "yi" in dataset_name:
        chat_path = dataset_name
        print(chat_path + "/data/train.json")
        print(os.path.isfile(chat_path + "/data/train.jsonl"))
        print(chat_path + "/data/eval.json")
        print(os.path.isfile(chat_path + "/data/eval.jsonl"))
        if not (
            os.path.isfile(chat_path + "/data/train.jsonl")
            and os.path.isfile(chat_path + "/data/eval.jsonl")
        ):
            raise RuntimeError(
                "Please check both the train.json and eval.json files in your local directory."
            )
        return raw_datasets.YiDataset(
            output_path, seed, local_rank, dataset_name, chat_path
        )
    else:
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    # hgx test
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    # reindex each time when using local jsonfile since it's more likely to get modified
    if (not os.path.isfile(index_file_name)) or (dataset_name == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["labels"],
                # "loss_mask": self.chosen_dataset[idx]["loss_mask"],
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id
        elif self.train_phase == 4:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                    self.chosen_dataset[idx]["labels"], self.chosen_dataset[idx]["ref_logps"], \
                    self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"], \
                    self.reject_dataset[idx]["labels"], self.reject_dataset[idx]["ref_logps"]
        elif self.train_phase == 5 or self.train_phase == 6:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["labels"]


def get_logps(dpo_model, tokens, pad_id):
    input_ids, attention_mask, labels = tokens["input_ids"], tokens["attention_mask"], tokens["labels"]
    # print(input_ids)
    # print(labels)
    # import time
    # time.sleep(100)
    with torch.no_grad():
        # start_time = time.time()
        # print(start_time)
        dpo_model.eval()
        logits = dpo_model.forward_logits(input_ids, labels=labels, attention_mask=attention_mask)
        # end_time = time.time()
        # print('takes time {}'.format(end_time-start_time))

    return logits.reshape(input_ids.shape[0], -1)

def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len, dpo_ref_model=None, device=None):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []

    print('train phase is {}'.format(train_phase))
    # import time
    # time.sleep(100)



    if train_phase == 1:
        print('train phase is {}'.format(train_phase))
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                chosen_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         # hgx test
                                        #  padding="do_not_pad",
                                         truncation=True,
                                         # hgx test
                                        #  truncation=False,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                
                
                # In the SFT stage, it's essential that the source part doesn't contribute to the loss calculation, 
                # only the completion part should be considered. Set ignore index to labels to -100, so that loss function will ignore the loss of source part.
                # The specific value, -100, is the default value for ignore index in `toch.nn.CrossEntropyLoss`
                # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss for more details
                prompt_sentence = raw_dataset.get_prompt(tmp_data)
                prompt_token = tokenizer(prompt_sentence, add_special_tokens=False)
                prompt_token_len = min(max_seq_len, len(prompt_token["input_ids"]))
                chosen_token["labels"] = chosen_token["input_ids"].clone()
                chosen_token["labels"][:prompt_token_len] = -100
                pad_mask = chosen_token["labels"] == tokenizer.pad_token_id
                chosen_token["labels"] = ~pad_mask * chosen_token["labels"] + torch.full_like(chosen_token["labels"], fill_value=-100) * pad_mask  
                chosen_token["loss_mask"] = ~(chosen_token["input_ids"] == tokenizer.pad_token_id)
                chosen_token["loss_mask"][:prompt_token_len] = 0
                chosen_dataset.append(chosen_token)

    elif train_phase == 2:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                reject_dataset.append(reject_token)

    elif train_phase == 3:
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            prompt = raw_dataset.get_prompt(tmp_data)
            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                prompt_token["input_ids"] = prompt_token["input_ids"]
                prompt_token["attention_mask"] = prompt_token["attention_mask"]
                for key_word in ["input_ids", "attention_mask"]:
                    length = prompt_token[key_word].size()[-1]
                    if length > max_seq_len:
                        y = prompt_token[key_word].squeeze(0)[length -
                                                              (max_seq_len -
                                                               1):].flip(0)
                    else:
                        y = prompt_token[key_word].squeeze(0).flip(0)
                    prompt_token[key_word] = y
                prompt_dataset.append(prompt_token)
    elif train_phase == 4:
        from utils.utils import print_rank_0, to_device
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response
            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                if dpo_ref_model:
                    prompt_sentence = raw_dataset.get_prompt(tmp_data)
                    prompt_token = tokenizer(prompt_sentence, add_special_tokens=False)

                    prompt_token_len = min(max_seq_len, len(prompt_token["input_ids"]))

                    chosen_token["labels"] = chosen_token["input_ids"].clone()
                    chosen_token["labels"][0][:prompt_token_len] = tokenizer.pad_token_id

                    reject_token["labels"] = reject_token["input_ids"].clone()
                    reject_token["labels"][0][:prompt_token_len] = tokenizer.pad_token_id


                    # print(to_device(chosen_token, device))
                    print(i)
                    import time
                    # time.sleep(100)
                    chosen_token["ref_logps"] = get_logps(dpo_ref_model, to_device(chosen_token, device), tokenizer.pad_token_id)
                    reject_token["ref_logps"] = get_logps(dpo_ref_model, to_device(reject_token, device), tokenizer.pad_token_id)

                chosen_dataset.append(chosen_token)
                reject_dataset.append(reject_token)
    elif train_phase == 5:
        from utils.utils import print_rank_0, to_device
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            chosen_sentence = raw_dataset.get_prompt(
                tmp_data)  # the accept response
            lab = float(raw_dataset.get_chosen(tmp_data))
            if chosen_sentence is not None:
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         #padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token['labels'] = lab
                chosen_dataset.append(chosen_token)
    elif train_phase == 6:
        print('train phase is {}'.format(train_phase))
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            input_ids_str = raw_dataset.get_prompt(tmp_data)
            label_ids_str = raw_dataset.get_chosen(tmp_data)
            chosen_token = {}
            chosen_token["input_ids"] = torch.as_tensor([[int(x) for x in input_ids_str.split(',')]])
            chosen_token["labels"] = torch.as_tensor([[int(x) for x in label_ids_str.split(',')]])
            chosen_dataset.append(chosen_token)

    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len, dpo_ref_model=None, device=None):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    print('finish get raw dataset')
    import time
    train_dataset = raw_dataset.get_train_data()
    print(train_dataset)

    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len, dpo_ref_model, device)
    
    

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len, dpo_ref_model, device)
    print("length of train dataset {}".format(len(train_dataset)))
    print("length of eval dataset {}".format(len(eval_dataset)))

    print('finish create dataset')
    return train_dataset, eval_dataset


def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          reload=False,
                          dpo_ref_model=None,
                          device=None):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"
    print('train_fname', train_fname)
    print('eval_fname', eval_fname)

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    print('cache_found', cache_found)
    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        print('not cached!')
        if len(data_path) == 1:  # Single dataset.
            # print(1)
            # import time
            # time.sleep(100)
            train_dataset, eval_dataset = create_dataset(
                local_rank, data_path[0], data_split, output_path, train_phase,
                seed, tokenizer, end_of_conversation_token, max_seq_len, dpo_ref_model, device)
        else:  # Blending datasets.
            # print(2)
            # import time
            # time.sleep(100)
            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank, d_path, data_split, output_path, train_phase,
                    seed, tokenizer, end_of_conversation_token, max_seq_len, dpo_ref_model, device)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        
        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    "10,0,0",
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                    dpo_ref_model,
                    device
                )
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    torch.distributed.barrier()
    return torch.load(train_fname), torch.load(eval_fname)


class DataCollatorReward:

    def __call__(self, data):
        batch = {}
        alen, blen = len(data[0][0][0]), len(data[0][2][0])
        pzeros = torch.zeros(1, abs(blen - alen), dtype=torch.int64)
        if alen == blen:
            batch["input_ids"] = torch.cat([f[0]
                                            for f in data] + [f[2] for f in data],
                                           dim=0)
            batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                                [f[3] for f in data],
                                                dim=0)
        elif alen < blen:
            batch["input_ids"] = torch.cat([torch.cat([f[0], pzeros], axis=1)
                                            for f in data] + [f[2] for f in data],
                                           dim=0)
            batch["attention_mask"] = torch.cat([torch.cat([f[1], pzeros], axis=1) for f in data] +
                                                [f[3] for f in data],
                                                dim=0)
        else:
            batch["input_ids"] = torch.cat([f[0]
                                            for f in data] + [torch.cat([f[2], pzeros], axis=1) for f in data],
                                           dim=0)
            batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                                [torch.cat([f[3], pzeros], axis=1) for f in data],
                                                dim=0)
        return batch

class DataCollatorClassifier:

    def __call__(self, data):
        batch = {}
        batch["input_ids"] = data[0][0]
        batch["labels"] = torch.as_tensor([data[0][1]])
        return batch

class DataCollatorSoFT:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = data[0][0]
        batch["labels"] = data[0][1]
        return batch

class DataCollatorDpo:

    def __call__(self, data):
        batch = {}
        alen, blen = len(data[0][0][0]), len(data[0][4][0])
        pzeros = torch.zeros(1, abs(blen - alen), dtype=torch.int64)
        plabels = torch.ones(1, abs(blen - alen), dtype=torch.int64) * -100
        if alen == blen:
            batch["input_ids"] = torch.cat([f[0]
                                            for f in data] + [f[4] for f in data],
                                           dim=0)
            batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                                [f[5] for f in data],
                                                dim=0)
            batch["labels"] = torch.cat([f[2] for f in data] +
                                                [f[6] for f in data],
                                                dim=0)
        elif alen < blen:
            batch["input_ids"] = torch.cat([torch.cat([f[0], pzeros], axis=1)
                                            for f in data] + [f[4] for f in data],
                                           dim=0)
            batch["attention_mask"] = torch.cat([torch.cat([f[1], pzeros], axis=1) for f in data] +
                                                [f[5] for f in data],
                                                dim=0)
            batch["labels"] = torch.cat([torch.cat([f[2], plabels], axis=1) for f in data] +
                                                [f[6] for f in data],
                                                dim=0)
        else:
            batch["input_ids"] = torch.cat([f[0]
                                            for f in data] + [torch.cat([f[4], pzeros], axis=1) for f in data],
                                           dim=0)
            batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                                [torch.cat([f[5], pzeros], axis=1) for f in data],
                                                dim=0)
            batch["labels"] = torch.cat([f[2] for f in data] +
                                                [torch.cat([f[6], plabels], axis=1) for f in data],
                                                dim=0)

        batch["ref_logps"] = torch.cat([f[3] for f in data] +
                                            [f[7] for f in data],
                                            dim=0)
        return batch
    

class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size, pad_token_id):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size
        self.pad_token_id = pad_token_id

    def __call__(self, data):
        batch = {}
        # pad_token_id = data[-1][-1]
        pad_token_id = self.pad_token_id

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)
        return batch


def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                     self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        self.dataset = []
