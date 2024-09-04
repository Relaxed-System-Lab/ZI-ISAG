# Zero-Indexing-Internet-Search-Augmented-Generation
# ZI-ISAG
 ZI-ISAG is an internet search augmented generation paradigm for LLMs. Key features include:
 - Dynamically integrate the latest online information (without maintaining any index for any fixed corpus).
 - The core component is an extractor LLM which can accurately and efficiently extract relevant information from the tagged content.
 - Extractor LLM only outputs the associated TAGs in the tagged content, significantly improving efficiency of the extraction process.

----------

This project was made possible thanks to a collaboration with:
<img src="https://github.com/Relaxed-System-Lab/ZI-ISAG/blob/main/images/collaboration.png" width="750" height="90" alt="01_hkust_fdu_tsinghua">

----------

## Content
- [Paradigm](#paradigm)
- [Extractor LLM](#extractor-llm)
- [Environment](#environment)
- [Instruction Construction](#instruction-construction)
- [Train](#train)

## Paradigm
<img src="https://github.com/Relaxed-System-Lab/ZI-ISAG/blob/main/images/Search-RAG-Internet.png" width="520" height="248" alt="zi-isag-paradigm">

## Extractor LLM
<img src="https://github.com/Relaxed-System-Lab/ZI-ISAG/blob/main/images/case.png" width="520" height="304" alt="extractor-llm-case">

## Enviroment
We use conda to manage our enviroment with CUDA 12.1 and cuDNN 9.1.0.

Create and activate env zi-isag:
```bash
conda create -p /YOUR_PATH_TO_CONDA_ENV/zi-isag python=3.10.14
conda activate /YOUR_PATH_TO_CONDA_ENV/zi-isag
```
Build the env with requirements.txt
```bash
python -m pip install -r requirements.txt
```

## Instruction Set Construction
The construction includes the following steps:
- Split and Tag
- Extract Info-Tag
- Double Check
- Portion and Construct

All details of these steps could be found in `data/construct.py`

## Train
The training computation was conducted on four NVIDIA H800 GPUs, each with 80GB of memory by the DeepSpeed parallel training system. A example could be seen in `train/sft/train_zi-isag.sh`:
```bash
deepspeed --include localhost:0,1,2,3 main_no_padding.py \
   --data_path  ../../data/instruction_set/zi-isag/example \
   --data_split 1,0,0 \
   --model_name_or_path YOUR_MODLE_PATH \
   --max_seq_len 16384 \
   --learning_rate 5e-6 \
   --weight_decay 0. \
   --num_train_epochs 3  \
   --gradient_accumulation_steps 4 \
   --lr_scheduler_type cosine_with_min_lr \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --output_dir ./model_ckpts/ \
   --exp_name YOUR_EXP_NAME \
   --tensorboard_path ./save_tensorboard \
   --data_output_path ./YOUR_DATA_OUTPUT \
   --enable_tensorboard
```
Note that we set `--per_device_train_batch_size` to 1 by default, so no padding is needed. If you want to use a `--per_device_train_batch_size` larger than 1, you may edit `train/sft/train_zi-isag.sh`, the `create_dataset_split` function within `train/utils/data/data_utils_no_padding.py`, and make any other modifications you may require.
