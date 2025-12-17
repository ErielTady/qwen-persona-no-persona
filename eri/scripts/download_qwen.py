
"""
from huggingface_hub import snapshot_download
import os

MODEL_ID = "zhiqing/Qwen3-14B-INT8"
LOCAL_DIR = "/mnt/disk/zhiqing/Qwen3-14B-INT8"

os.makedirs(LOCAL_DIR, exist_ok=True)

snapshot_download(repo_id=MODEL_ID, 
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False, #make it a real copy in LOCAL_DIR
                )
print(f"Downloaded {LOCAL_DIR}")


"""
import os
import torch
import torch.nn as nn
import re

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from accelerate import init_empty_weights, infer_auto_device_map


MODEL_DIR   = "/mnt/disk/Qwen/Qwen3-14B"
OFFLOAD_DIR = "/mnt/disk/offload_int8"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

bnb8 = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

tok = AutoTokenizer.from_pretrained(
    MODEL_DIR, trust_remote_code=True, local_files_only=True
)

# ensure PAD exists
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

config = AutoConfig.from_pretrained(
    MODEL_DIR, trust_remote_code=True, local_files_only=True
)

with init_empty_weights():
    empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

max_memory = {0: "13GiB", "cpu": "20GiB"}

device_map = infer_auto_device_map(
    empty,
    max_memory=max_memory,
    no_split_module_classes=["Qwen3DecoderLayer"],
)

# keep big tied matrices on CPU (prevents VRAM OOM at lm_head)
device_map["lm_head"] = "cpu"
device_map["model.embed_tokens"] = "cpu"

load_kwargs = dict(
    device_map=device_map,
    max_memory=max_memory,
    quantization_config=bnb8,
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    local_files_only=True,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float16, **load_kwargs
    ).eval()
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, **load_kwargs
    ).eval()

print("Loaded. Key placements:")
print("  lm_head:", model.hf_device_map.get("lm_head"))
print("  embed_tokens:", model.hf_device_map.get("model.embed_tokens"))

# ---- CRITICAL: force model.device to be CPU (not meta) for generate() ----
# Transformers uses next(model.parameters()).device; add a real top-level param on CPU.
model._force_device_param = nn.Parameter(torch.zeros((), device="cpu"), requires_grad=False)
# -------------------------------------------------------------------------

# Also force special-token tensors to be real CPU tensors (not meta)
gen_cfg = GenerationConfig.from_model_config(model.config)
gen_cfg.do_sample = False
gen_cfg.temperature = 1.0
gen_cfg.top_p = 1.0
gen_cfg.top_k = 0

gen_cfg.eos_token_id = int(tok.eos_token_id)
gen_cfg.pad_token_id = int(tok.pad_token_id)
if tok.bos_token_id is not None and gen_cfg.bos_token_id is None:
    gen_cfg.bos_token_id = int(tok.bos_token_id)

#gen_cfg._eos_token_tensor = torch.tensor([gen_cfg.eos_token_id], dtype=torch.long, device="cpu")
#gen_cfg._pad_token_tensor = torch.tensor([gen_cfg.pad_token_id], dtype=torch.long, device="cpu")
#if gen_cfg.bos_token_id is not None:
#    gen_cfg._bos_token_tensor = torch.tensor([gen_cfg.bos_token_id], dtype=torch.long, device="cpu")

model.generation_config = gen_cfg

prompt_1 = """Answer the following question by typing the number corresponding to your chosen answer. Return only the number of the chosen option. No words or punctuation
How much confidence do you have in the United Nations (UN)?
Options:
(1) A great deal
(2) Quite a lot
(3) Not very much
(4) None at all
(5) Don't know
"""

prompt = tok.apply_chat_template(
    [{"role": "user", "content": prompt_1}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tok(prompt, return_tensors="pt")
# embeddings are on CPU (we pinned them), so inputs must be on CPU too
inputs = inputs.to("cpu")

with torch.inference_mode():
    out = model.generate(
        **inputs,
        generation_config=model.generation_config,
        max_new_tokens=16,
        use_cache=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

print(tok.decode(out[0], skip_special_tokens=True))
import re

text = tok.decode(out[0], skip_special_tokens=True)
text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
print(text)

