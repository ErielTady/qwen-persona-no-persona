import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn

#os.environ["HF_HOME"]="/mnt/disk/hf_cache"

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, AutoConfig, GenerationConfig
from accelerate import init_empty_weights, infer_auto_device_map

from wvs_dataset import WVSDataset
from utils import read_json, write_json

from transformers.utils import logging
logging.set_verbosity(50)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


OFFLOAD_DIR = "/mnt/disk/offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

QWEN_MODEL_DIR = "/mnt/disk/Qwen/Qwen3-14B"
QWEN_OFFLOAD_DIR = "/mnt/disk/offload_int8"
os.makedirs(QWEN_OFFLOAD_DIR, exist_ok=True)


#bnb_config = BitsAndBytesConfig(
#   load_in_8bit=True,
#    llm_int8_enable_fp32_cpu_offload=True
#)

def load_qwen_model(model_dir: str, cuda_index: int):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    bnb8 = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True
    )

    with init_empty_weights():
        empty = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    max_memory = {cuda_index: "13GiB", "cpu": "20GiB"}

    device_map = infer_auto_device_map(
        empty,
        max_memory=max_memory,
        no_split_module_classes=["Qwen3DecoderLayer"],
    )

    device_map["lm_head"] = "cpu"
    device_map["model.embed_tokens"] = "cpu"

    load_kwargs = dict(
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=bnb8,
        offload_folder=QWEN_OFFLOAD_DIR,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float16, **load_kwargs
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, **load_kwargs
        ).eval()

    model._force_device_param = nn.Parameter(
        torch.zeros((), device="cpu"), requires_grad=False
    )

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.do_sample = False
    gen_cfg.temperature = 1.0
    gen_cfg.top_p = 1.0
    gen_cfg.top_k = 0

    gen_cfg.eos_token_id = int(tokenizer.eos_token_id)
    gen_cfg.pad_token_id = int(tokenizer.pad_token_id)
    if tokenizer.bos_token_id is not None and gen_cfg.bos_token_id is None:
        gen_cfg.bos_token_id = int(tokenizer.bos_token_id)

    #gen_cfg._eos_token_tensor = torch.tensor([gen_cfg.eos_token_id], dtype=torch.long, device="cpu")
    #gen_cfg._pad_token_tensor = torch.tensor([gen_cfg.pad_token_id], dtype=torch.long, device="cpu")
    #if gen_cfg.bos_token_id is not None:
    #       gen_cfg._bos_token_tensor = torch.tensor([gen_cfg.bos_token_id], dtype=torch.long, device="cpu")

    model.generation_config = gen_cfg

    return model, tokenizer

def format_qwen_prompts(tokenizer, prompts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]



def generate(model, tokenizer, fewshot_cache, prompts, device, n_steps=20):
    ################################## generation cycle with 20 steps
    step = 0
    past_key_values = fewshot_cache
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]
    prompt_lengths = tokens["attention_mask"].sum(dim=1)
    output = None
    while step < n_steps:
        attention_mask = input_ids.new_ones(input_ids.shape)

        if output is not None:
            past_key_values = output["past_key_values"]

        ids = model.prepare_inputs_for_generation(input_ids,
                                                past=past_key_values,
                                                attention_mask=attention_mask,
                                                use_cache=True)

        output = model(**ids)

        # next_token = random.choice(torch.topk(output.logits[:, -1, :], top_k, dim=-1).indices[0])
        next_token = output.logits[:, -1, :].argmax(dim=-1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

        step += 1

    return input_ids, prompt_lengths


def query_hf(
    qid: str,
    *,
    model_name: str = 'bigscience/mt0-small',
    version: int = 1,
    lang: str = 'en',
    prompt_variant: int = 1,
    max_tokens: int = 8,
    temperature: float = 0.7,
    n_gen: int = 5,
    batch_size: int = 4,
    fewshot: int = 0,
    cuda: int = 0,
    greedy: bool = False,
    generator = None,
    tokenizer = None,
    subset = None,
    country: str = "egypt",
):

    model_name_ = model_name.split("/")[-1]
    #savedir = f"../results_wvs_2/{model_name_}/{lang}"
    #results_root = "../results_wvs_2_no_persona" if no_persona else "../results_wvs_2"
    savedir = f"../results_wvs_2/{model_name_}/{lang}"
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    filepath = f"../dataset/wvs_template.{lang}.yml"

    dataset = WVSDataset(filepath,
        language=lang,
        country=country,
        api=False,
        model_name=model_name_,
        use_anthro_prompt=False,
        prompt_variant= prompt_variant,
    )

    device = torch.device(f'cuda:{cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Language={lang} | Temperature={temperature} | Tokens={max_tokens} | N={n_gen} | Batch={batch_size} | Version={version}")
    print(f"> Device {device}")

    use_qwen = "qwen" in model_name_.lower()
    token_device = device

    if qid <= 0:
        question_ids = dataset.question_ids
    else:
        question_ids = [f"Q{qid}"]

    print(f"> Running {len(question_ids)} Qs")
    
    model = None
    tokenizer = None
    model_path = model_name
    
    if use_qwen:
        qwen_dir = model_path if os.path.isdir(model_path) else QWEN_MODEL_DIR
        model, tokenizer = load_qwen_model(qwen_dir, cuda)
        token_device = torch.device("cpu")
    elif "mt0" in model_name_:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float16).to(device)
    else:
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        if "AceGPT" in model_name:
            model_path = "/mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/models/models--FreedomIntelligence--AceGPT-13B-chat/snapshots/ab87ccbc2c4a05969957755aaabc04400bb20052"
        
        elif "Llama" in model_name:
            model_path = "eri00eli/llama2-13b-8bit"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                #quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                #offload_folder=OFFLOAD_DIR,
                trust_remote_code=True,
                #local_files_only=True,
            )
        
    if model is None and not use_qwen:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )





       # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", torch_dtype=torch.float16).to(device)

    #tokenizer = AutoTokenizer.from_pretrained(model_path,
    #                                          trust_remote_code=True
    #                                          #local_files_only=True
    #                                          )

    if"Llama-2-13b-chat-hf" in model_name or "AceGPT-13B-chat" in model_name or "qwen-14b-bnb-4bit" in model_name.lower():
        print("> Changing padding side")
        tokenizer.padding_side = "left"

    if model_name == "gpt2" or "Sheared-LLaMA-1.3B" in model_name or "Llama-2-13b" in model_name or "AceGPT-13B-chat" in model_name or "qwen-14b-bnb-4bit" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token

    for qid in question_ids:
        qid = int(qid[1:])
        dataset.set_question(index=qid)

        filesuffix = (
            f"q={str(qid).zfill(2)}"
            f"_lang={lang}"
            f"_country={country}"
            f"_temp={temperature}"
            f"_maxt={max_tokens}"
            f"_n={n_gen}"
            f"_v{version}"
            f"_fewshot={fewshot}"
            f"_pv{prompt_variant}"
        )
        print(filesuffix)

        preds_path = os.path.join(savedir, f"preds_{filesuffix}.json")

        completions = []
        if os.path.exists(preds_path):
            completions = read_json(preds_path)

        if len(completions) >= len(dataset):
            print(f"Skipping Q{qid}")
            continue

        if len(completions) > 0:
            print(f"> Trimming Dataset from {len(completions)}")
            dataset.trim_dataset(len(completions))


        # num_workers{16,2}
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        if fewshot > 0:

            fewshot_examples, _ = dataset.fewshot_examples()
            #fewshot_tokens = tokenizer(fewshot_examples, padding=True, return_tensors="pt").to(device)
            fewshot_payloads = format_qwen_prompts(tokenizer, [fewshot_examples]) if use_qwen else [fewshot_examples]
            fewshot_tokens = tokenizer(fewshot_payloads, padding=True, return_tensors="pt").to(token_device)
            with torch.no_grad():
                fewshot_cache = model(**fewshot_tokens, use_cache=True)["past_key_values"]

        index = 0
        print(f"> Prompting {model_name} with Q{qid}")
        for batch_idx, prompts in tqdm(enumerate(dataloader), total=len(dataloader)):

            if fewshot == 0:
                #prompts_for_model = apply_chat_template_if_needed(prompts, tokenizer, model_name_)
                prompt_batch = format_qwen_prompts(tokenizer, prompts) if use_qwen else prompts
               #if batch_idx == 0:
               #    print("PROMPT[0]:\n", prompt_batch[0])

                
                tokens = tokenizer(prompt_batch, padding=True, return_tensors="pt").to(token_device)
                #input_len = tokens["input_ids"].shape[1]

                #tokens = tokenizer(prompts_for_model, padding=True, return_tensors="pt").to(device)

                #tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                ##################### independt of the batch-size #######
                prompt_lengths = tokens["attention_mask"].sum(dim=1)
                gen_outputs = model.generate(**tokens,
                    temperature=temperature,
                    do_sample=(not greedy),
                    num_return_sequences=n_gen,
                    max_new_tokens=max_tokens,
                )
                #decoded_output = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                #f batch_idx == 0:
                #   decoded_full = tokenizer.batch_decode(gen_outputs, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                #   print("FULL[0]:", repr(decoded_full[0]))


                generated_only = [
                    seq[int(prompt_lengths[idx // n_gen]) :]
                    for idx, seq in enumerate(gen_outputs)
                ]

                decoded_output = tokenizer.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
               #if batch_idx == 0:
               #    print("GEN_ONLY[0]:", repr(decoded_output[0]))

                       



                for b_i in range(0, len(decoded_output), n_gen):
                    preds = decoded_output[b_i:b_i+n_gen]
                    #prompt_to_strip = prompt_batch[b_i//n_gen]
                    #preds = [pred.replace(prompt_to_strip, "") for pred in preds]

                    #preds = [pred.replace(prompts_for_model[b_i//n_gen], "") for pred in preds]
                    persona = dataset.persona_qid[f"Q{qid}"][index]
                    q_info = dataset.question_info[f"Q{qid}"][index]
                    index += 1
                    completions += [{
                        "persona": persona,
                        "question": q_info,
                        "response": preds,
                        "prompt_variant": dataset.prompt_variant,
                    }]

            else:

                # prompts_with_fewshot = [fewshot_examples + prompt for prompt in prompts]
                # tokens_with_fewshot = tokenizer(prompts_with_fewshot, padding=True, return_tensors="pt").to(device)

                # start_time = time.time()
                # gen_outputs_wo_cache = model.generate(**tokens_with_fewshot,
                #     temperature=temperature,
                #     do_sample=(not greedy),
                #     num_return_sequences=n_gen,
                #     max_new_tokens=max_tokens,
                # )

                # decoded_output = tokenizer.batch_decode(gen_outputs_wo_cache, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                # tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
                # tokens["input_ids"] = tokens["input_ids"][:, 1:]
                # tokens["attention_mask"] = tokens["attention_mask"][:, 1:]

                # with torch.no_grad():
                #     prompt_cache = model(**tokens_with_fewshot, past_key_values=fewshot_cache, use_cache=True)["past_key_values"]

                # tokens_with_fewshot_concat = {
                #     "input_ids": torch.cat([fewshot_tokens["input_ids"].repeat(batch_size, 1), tokens["input_ids"][:, 1:]], dim=1),
                #     "attention_mask": torch.cat([fewshot_tokens["attention_mask"].repeat(batch_size, 1), tokens["attention_mask"][:, 1:]],dim=1),
                #     # "attention_mask": torch.cat([fewshot_tokens["attention_mask"].repeat(batch_size, 1), tokens["attention_mask"][:, 1:], torch.ones(batch_size, 1).to(device)],dim=1),
                # }

                # tokens_with_fewshot_concat = {
                #     "input_ids": torch.cat([fewshot_tokens["input_ids"].repeat(batch_size, 1), tokens["input_ids"]], dim=1),
                #     "attention_mask": torch.cat([fewshot_tokens["attention_mask"].repeat(batch_size, 1), tokens["attention_mask"]],dim=1),
                #     # "attention_mask": torch.cat([fewshot_tokens["attention_mask"].repeat(batch_size, 1), tokens["attention_mask"], torch.zeros(batch_size, 1).to(device)],dim=1),
                # }

                # num_layers = len(fewshot_cache)
                # all_cache = []
                # for layer_idx in range(num_layers):
                #     all_cache += [(
                #         torch.cat([fewshot_cache[layer_idx][0].repeat(batch_size*n_gen,1,1,1), prompt_cache[layer_idx][0].repeat(n_gen,1,1,1)[:,:,:-1,:]], dim=2),
                #         torch.cat([fewshot_cache[layer_idx][1].repeat(batch_size*n_gen,1,1,1), prompt_cache[layer_idx][1].repeat(n_gen,1,1,1)[:,:,:-1,:]], dim=2),
                #     )]
                    # all_cache += [(
                    #     torch.cat([fewshot_cache[layer_idx][0].repeat(batch_size*n_gen,1,1,1), prompt_cache[layer_idx][0].repeat(n_gen,1,1,1)], dim=2),
                    #     torch.cat([fewshot_cache[layer_idx][1].repeat(batch_size*n_gen,1,1,1), prompt_cache[layer_idx][1].repeat(n_gen,1,1,1)], dim=2),
                    # )]

                # print("Fewshot Tokens + Prompt Tokens: ", fewshot_tokens["input_ids"].size(1) + tokens["input_ids"].size(1))
                # print("[Fewshot, Prompt] Tokens: ", tokens_with_fewshot_concat["input_ids"].size(1))
                # # print("(Fewshot + Prompt) Tokens: ", tokens_with_fewshot["input_ids"].size(1))
                # print("Cache Concat: ", all_cache[0][0].size())
                # print("[Fewshot, Prompt] Attention Mask: ", tokens_with_fewshot_concat["attention_mask"].size(1))
                # breakpoint()
                # # del prompt_cache

                # tokens_with_fewshot["attention_mask"] = torch.cat([tokens_with_fewshot["attention_mask"], torch.ones(batch_size,1).to(device)], dim=1)

                # # start_time = time.time()
                # gen_outputs = model.generate(**tokens_with_fewshot_concat,
                #     # input_ids=tokens_with_fewshot_concat["input_ids"],
                #     temperature=temperature,
                #     do_sample=(not greedy),
                #     num_return_sequences=n_gen,
                #     max_new_tokens=max_tokens,
                #     past_key_values=tuple(all_cache)
                # )

                #gen_outputs = generate(model, tokenizer, fewshot_cache, prompts, device, n_steps=max_tokens)
                #prompts_for_model = apply_chat_template_if_needed(prompts, tokenizer, model_name_)
                #gen_outputs = generate(model, tokenizer, fewshot_cache, prompts_for_model, device, n_steps=max_tokens)
                prompt_batch = format_qwen_prompts(tokenizer, prompts) if use_qwen else prompts
                #gen_outputs = generate(model, tokenizer, fewshot_cache, prompt_batch, token_device, n_steps=max_tokens)
                gen_outputs, prompt_lengths = generate(
                    model,
                    tokenizer,
                    fewshot_cache,
                    prompt_batch,
                    token_device,
                    n_steps=max_tokens,
                )

                # gen_outputs = model.generate(**tokens_with_fewshot_concat,
                #     temperature=temperature,
                #     do_sample=(not greedy),
                #     num_return_sequences=n_gen,
                #     max_new_tokens=max_tokens,
                #     past_key_values=tuple(all_cache)
                # )

                #decoded_output = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                generated_only = [
                    seq[int(prompt_lengths[idx // n_gen]) :]
                    for idx, seq in enumerate(gen_outputs)
                ]

                decoded_output = tokenizer.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                for b_i in range(0, len(decoded_output), n_gen):
                    preds = decoded_output[b_i:b_i+n_gen]
                    #preds = [pred.replace(fewshot_examples, "").replace(prompts[b_i//n_gen], "") for pred in preds]
                    prompt_to_strip = prompt_batch[b_i//n_gen]
                    fewshot_prompt = fewshot_payloads[0] if fewshot > 0 else ""
                    preds = [pred.replace(fewshot_prompt, "").replace(prompt_to_strip, "") for pred in preds]
                    persona = dataset.persona_qid[f"Q{qid}"][index]
                    q_info = dataset.question_info[f"Q{qid}"][index]
                    index += 1
                    completions += [{
                        "persona": persona,
                        "question": q_info,
                        "response": preds,
                        "prompt_variant": dataset.prompt_variant,
                    }]

            write_json(preds_path, completions)

if __name__ == "__main__":

    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --lang en --fewshot 3 --cuda 0
    # python wvs_query_hf_generate.py --qid -1 --model princeton-nlp/Sheared-LLaMA-1.3B --max-tokens 10 --lang en --fewshot 3 --cuda 0
    # python wvs_query_hf_generate.py --qid -1 --model princeton-nlp/Sheared-LLaMA-1.3B --max-tokens 5 --lang ar --fewshot 3 --cuda 0 --n-gen 1
    # python wvs_query_hf_generate.py --qid -1 --model meta-llama/Llama-2-13b-chat-hf --max-tokens 16 --lang en --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4

    # python wvs_query_hf_generate.py --qid -1 --model meta-llama/Llama-2-13b-chat-hf --max-tokens 32 --lang ar --fewshot 0 --cuda 1 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --max-tokens 5 --lang ar --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model FreedomIntelligence/AceGPT-13B-chat --max-tokens 5 --lang en --fewshot 0 --cuda 0 --n-gen 5 --batch-size 4
    # python wvs_query_hf_generate.py --qid -1 --model bigscience/mt0-xxl --max-tokens 5 --lang en --fewshot 0 --cuda 1 --n-gen 5 --batch-size 4 --country us

    # cp -r /home/bkhmsi/.cache/huggingface/hub/models--FreedomIntelligence--AceGPT-13B-chat /mnt/u14157_ic_nlp_001_files_nfs/nlpdata1/home/bkhmsi/models
    parser = argparse.ArgumentParser()

    parser.add_argument('--qid', required=True, type=int, help='question index')
    parser.add_argument('--model', default="bigscience/mt0-small", help='model to use')
    parser.add_argument('--version', default=1, help='dataset version number')
    parser.add_argument('--lang', default="en", help='language')
    parser.add_argument('--max-tokens', default=4, type=int, help='maximum number of output tokens')
    parser.add_argument('--temperature', default=0.7, type=float, help='temperature')
    parser.add_argument('--n-gen', default=5, type=int, help='number of generations')
    parser.add_argument('--batch-size', default=4, type=int, help='batch size')
    parser.add_argument('--fewshot', default=0, type=int, help='fewshot examples')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device number')
    parser.add_argument('--greedy', action="store_true", help='greedy decoding')
    parser.add_argument('--country', type=str, help='country')
    parser.add_argument('--prompt-variant', default=1, type=int, help='prompt template variant to use')
    #parser.add_argument(
    #    '--model',
    #    default="bigscience/mt0-small",
    #    help="model to use (pass a path or name containing 'qwen' to trigger the Qwen3 loader)",
    #)



    args = parser.parse_args()

    if args.greedy:
        args.n_gen = 1
        args.temperature = 1.0

    qid = int(args.qid)

    query_hf(
        qid=qid,
        model_name=args.model,
        version=args.version,
        lang=args.lang,
        max_tokens=args.max_tokens,
        temperature=float(args.temperature),
        n_gen=int(args.n_gen),
        batch_size=int(args.batch_size),
        fewshot=int(args.fewshot),
        cuda=args.cuda,
        greedy=args.greedy,
        country=args.country,
        prompt_variant=int(args.prompt_variant)
    )