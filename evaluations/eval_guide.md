# Evaluate dInfer Performance on Different Benchmarks

We provide an evaluation framework based on **dInfer** integrated with the 🤗 HuggingFace **lm‑eval‑harness**.  
It supports **Tensor Parallel (TP)** and **Data Parallel (DP)** inference for easy evaluation of large‑scale diffusion LLMs (dLLMs).

Currently supported model families include:

- **llada‑1.5**
- **llada‑moe**
- **llada2‑mini (llada‑mini)**

---

## 🔹 Supported Benchmarks

For different llada variants, we adapt the following benchmark tasks:

| Model | Tasks |
|------|------|
| llada‑1.5 | `gsm8k_llada1.5`, `mbpp_sanitized_llada1.5` |
| llada‑moe | `gsm8k_llada_moe`, `mbpp_sanitized_llada_moe` |
| **llada2‑mini** | `gsm8k_llada_mini`, `mbpp_sanitized_llada_mini` |

---

## 1️⃣ Install Dependencies

```bash
pip install -U accelerate evaluate datasets lm_eval==0.4.8 hf_transfer
```

---

## 2️⃣ Set Environment Variables

Before running evaluation, set these variables:

```bash
# Allow model code evaluation
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1

# Select GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## 3️⃣ Define Hyperparameters (llada / llada‑moe)

```bash
length=1024              # generation length
block_length=64          # block size for diffusion LLM
model_path='your_model_path'
output_path='your_output_folder'

# Cache & diffusion config
cache='dual'             # 'dual' / 'prefix' / ''
prefix_look=16
after_look=16
warmup_times=4
cont_weight=0.3          # cont_weight only useful for llada2 and llada-moe
use_credit=False         # use credit for credit-based decoding
use_compile=True
use_cudagraph=True

# Parallelism config
gpus='0,1,2,3'
parallel='tp'            # 'tp' or 'dp'

# Evaluation task
# llada 1.5: gsm8k_llada1.5 mbpp_sanitized_llada1.5
# llada moe: gsm8k_llada_moe mbpp_sanitized_llada_moe
task=mbpp_sanitized_llada_moe
```

---

## ⚙️ Run with Tensor Parallel (TP)

```bash
parallel_decoding='threshold'  # or hierarchy
threshold=0.8
low_threshold=0.5

python eval_dinfer.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args \
  model_path=${model_path},\
  gen_length=${length},\
  block_length=${block_length},\
  threshold=${threshold},\
  low_threshold=${low_threshold},\
  show_speed=True,\
  save_dir=${output_path},\
  parallel_decoding=${parallel_decoding},\
  prefix_look=${prefix_look},\
  after_look=${after_look},\
  cache=${cache},\
  warmup_times=${warmup_times},\
  use_compile=${use_compile},\
  parallel=${parallel},\
  cont_weight=${cont_weight},\
  use_credit=${use_credit},\
  gpus=${gpus} \
  --output_path ${output_path} \
  --include_path ./tasks \
  --apply_chat_template
```

---

## 🧩 Run with Accelerate (Data Parallel, DP)

```bash
parallel='dp'

accelerate launch eval_dinfer.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args \
  model_path=${model_path},\
  gen_length=${length},\
  block_length=${block_length},\
  threshold=${threshold},\
  low_threshold=${low_threshold},\
  show_speed=True,\
  save_dir=${output_path},\
  parallel_decoding=${parallel_decoding},\
  prefix_look=${prefix_look},\
  after_look=${after_look},\
  cache=${cache},\
  warmup_times=${warmup_times},\
  use_compile=${use_compile},\
  parallel=${parallel},\
  cont_weight=${cont_weight},\
  use_credit=${use_credit},\
  gpus=${gpus} \
  --output_path ${output_path} \
  --include_path ./tasks \
  --apply_chat_template
```

---

## 🧮 Hierarchy Parallel Decoding

```bash
parallel_decoding='hierarchy'
threshold=0.92
low_threshold=0.62

python eval_dinfer.py \
  --tasks ${task} \
  --confirm_run_unsafe_code \
  --model dInfer_eval \
  --model_args \
  model_path=${model_path},\
  gen_length=${length},\
  block_length=${block_length},\
  threshold=${threshold},\
  low_threshold=${low_threshold},\
  save_dir=${output_path},\
  parallel_decoding=${parallel_decoding},\
  prefix_look=${prefix_look},\
  after_look=${after_look},\
  cache=${cache},\
  warmup_times=${warmup_times},\
  cont_weight=${cont_weight} \
  --output_path ${output_path} \
  --include_path ./tasks \
  --apply_chat_template \
  --log_samples
```

---

## 💿 Credit‑Based Threshold Decoding

```bash
parallel_decoding='threshold'
threshold=0.8
use_credit=True
```

---

# 🆕 Evaluate **llada‑mini (llada2‑mini)**

`llada2‑mini` uses **block diffusion**, **prefix cache**, and **Tensor Parallel inference via sglang**.

---

## Key Differences from llada / llada‑moe

| Item | llada‑mini |
|----|----|
| Evaluation Script | `eval_dinfer_sglang.py` |
| Model type | `llada2` |
| Cache | `prefix` |
| Block diffusion | ✅ `use_bd=True` |
| Parallelism | TP only |
| Tasks | `gsm8k_llada_mini`, `mbpp_sanitized_llada_mini` |

---

## 📌 Environment Variables (llada‑mini)

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

---

## ⚙️ Hyperparameters (llada‑mini)

```bash
parallel_decoding='threshold'   # or hierarchy
length=2048
block_length=32

model_path='your_llada2_mini_model_path'
output_dir='./res'

threshold=0.80
low_threshold=0.62

cache='prefix'
warmup_times=0
prefix_look=0
after_look=0

cont_weight=0
use_credit=False
use_compile=True

tp_size=4
gpus='0;1;2;3'
parallel='tp'

model_type='llada2'     # IMPORTANT
use_bd=True             # enable block diffusion
```

---

## ▶️ Run llada‑mini Evaluation (TP)

```bash
if [ ${parallel} == 'tp' ]; then
  for task in gsm8k_llada_mini; do
    output_path="${output_dir}/${task}"

    python eval_dinfer_sglang.py \
      --tasks ${task} \
      --confirm_run_unsafe_code \
      --model dInfer_eval \
      --model_args \
      model_path=${model_path},\
      gen_length=${length},\
      block_length=${block_length},\
      threshold=${threshold},\
      low_threshold=${low_threshold},\
      show_speed=True,\
      save_dir=${output_path},\
      parallel_decoding=${parallel_decoding},\
      cache=${cache},\
      warmup_times=${warmup_times},\
      use_compile=${use_compile},\
      tp_size=${tp_size},\
      parallel=${parallel},\
      cont_weight=${cont_weight},\
      use_credit=${use_credit},\
      prefix_look=${prefix_look},\
      after_look=${after_look},\
      gpus=${gpus},\
      model_type=${model_type},\
      use_bd=${use_bd} \
      --output_path ${output_path} \
      --include_path ./tasks \
      --apply_chat_template
  done
else
  echo "llada-mini currently supports TP only"
fi
```