import json
import os
import re
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
DEVICE = 'cuda:0'

from openai import OpenAI
from unsloth import FastLanguageModel
import transformers
import requests

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_TOKEN'] = ''

DEEPSEEK_API_KEY = json.load(open('.keys.json', 'r'))['DEEPSEEK_API_KEY']
OPENROUTER_CLIENT = OpenAI(base_url="https://openrouter.ai/api/v1",
                           api_key="")

KIMI_CLIENT = OpenAI(base_url="https://api.moonshot.cn/v1",
                     api_key="sk-")

DOUBAO_CLIENT = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key='',
)

GPT_CLIENT = OpenAI(
    base_url="https://xiaoai.plus/v1",
    api_key="sk-"
)

REP_DISORDER = '<disorder>信息失序分析</disorder>'
REP_INTENT = '<intent>意图分析</intent>'

# 生成信息失序分析
PROMPT1 = """Analyze the given news using Information Disorder theory. In this theory, the disorder methods are categorized as follows:
Satire/Parody : Content where potential harm is excused by the humorous intent.
False Connection : Headlines, visuals, or captions that don't support the content.
Misleading Content : Misleading use of information to frame an issue or individual.
False Context : 'Genuine content shared with false contextual information.
Imposter Content : "Use of genuine sources' branding/identity with false content.
Manipulated Content : 'Genuine information or imagery that is deliberately altered.
Fabricated Content : 'Wholly false content designed to deceive.
Harassment : A serious form of targeted, persistent misconduct that threatens a person's well-being and safety.
Hate speech : 'bias-motivated, hostile expression targeting people for who they are.

Your response must be a flat JSON object using only the exact keys below: { "disorder_method": <One of the above methods (or 'Normal' if no disorder method is present)>, "reason": <Analysis in one concise paragraph>" }"""

# 生成意图分析
PROMPT2 = """Analyze the author’s core intent in the given news. This news has already undergone a preliminary analysis of information disorder: 
{REP_DISORDER}
Please note that the above results may contain inaccuracies, and not all explanations are necessarily correct.  
Based on this, analyze the author’s core intent in the news. 
Provide your analysis in one concise paragraph."""

RESP_FORMAT = 'Factual , Misinformation (Information that is false, but not created with the intention of causing harm), Disinformation (Information that is false and deliberately created to harm a person, social group, organization or country), or Malinformation(Information that is based on reality, used to inflict harm on a person, organization or country) news. Your response must be a flat JSON object using only the exact keys below: { "response": "<one of Factual, Misinformation, Disinformation, or Malinformation>", "reason": "<Analysis in one concise paragraph>" }'

# 基础 无提示
BASE_PROMPT = f"""Analyze the given news and determine whether it is {RESP_FORMAT}"""

# 进阶 注入信息失序信息
DISORDER_PROMPT = f"""Analyze the given news, which has undergone a preliminary information disorder analysis: 
{REP_DISORDER}
Please note that the above results may contain errors and not all explanations may be correct.
Based on this information, continue analyzing the news and determine whether it is {RESP_FORMAT}"""

# 高级 注入信息失序信息 + 意图分析
DISORDER_INTENT_PROMPT = f"""Analyze the given news, which has undergone a preliminary information disorder analysis: 
{REP_DISORDER}
which has undergone a preliminary intent analysis: 
{REP_INTENT}
Please note that the above results may contain errors and not all explanations may be correct.
Based on this information, continue analyzing the news and determine whether it is {RESP_FORMAT}"""

news_categories = ["Factual", "Misinformation", "Disinformation", "Malinformation"]
news_methods = [
    "Satire/Parody", "False Connection", "Misleading Content", 
    "False Context", "Imposter Content", "Manipulated Content", "Fabricated Content",
    "Harassment", "Hate speech"
]

def ask_gpt(model_id, item, prompt):
    result = GPT_CLIENT.chat.completions.create(
        model=model_id,
        messages=[{"role": "user",
                "content": [{"type": "text", "text": f'{item['title']}\n{item['content']}\n{prompt}'}],
            }],
        max_tokens=4000,
        temperature=0.2,
    )
    result = result.choices[0].message.content
    return result

def ask_open_router(model_id, item, prompt):
    client = OPENROUTER_CLIENT
    completion = client.chat.completions.create(
        # extra_headers={
        #     # "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
        # },
        extra_body={
            "provider": {
                "quantizations": [
                    "bf16"
                ],
                "sort": "price"
            }
        },
        model=model_id,
        messages=[{"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}'}]
    )
    result = completion.choices[0].message.content
    return result


def ask_mistral3(tokenizer, model, item, prompt):
    messages_batch = [[{"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}\nAssistant: '}]]
    tokenized = tokenizer.apply_chat_template(
        messages_batch, padding=True, truncation=False, return_tensors="pt", return_dict=True
    )
    tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}
    outputs = model.generate(**tokenized, max_new_tokens=1024)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    result = result[result.find('\nAssistant:') + len('\nAssistant:'):]
    return result


def ask_phi_4(model, item, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}'},
    ]
    outputs = model(messages, max_new_tokens=1024)
    result = outputs[0]["generated_text"][-1]['content']
    return result


def ask_qwen3(tokenizer, model, item, prompt):
    messages = [{"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}'}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    except ValueError:
        index = 0
    result = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return result


def ask_llama3(model, item, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}'},
    ]
    outputs = model(messages, max_new_tokens=1024)
    result = outputs[0]["generated_text"][-1]['content']
    return result


def ask_phi3(model, item, prompt):
    messages = [
        {
            "role": "user",
            "content": f'{item['title']}\n{item['content']}\n{prompt}'
        },
    ]
    generation_args = {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = model(messages, **generation_args)
    result = output[0]['generated_text']
    return result


def ask_qwen2_5(tokenizer, model, item, prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}'}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return result


def ask_minicpm4(tokenizer, model, item, prompt):
    messages = [
        {"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}'}
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    model_outputs = model.generate(**model_inputs, max_new_tokens=32768, top_p=0.95, temperature=0.6)

    output_token_ids = [model_outputs[i][len(model_inputs[i]):] for i in range(len(model_inputs['input_ids']))]
    result = tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]
    cut_position = result.find('</think>')
    if cut_position == -1:
        cut_position = 0
    else:
        cut_position += len('</think>')
    result = result[cut_position:]
    return result


def ask_deepseek_online(item, prompt):
    data = {
        "model": 'deepseek-chat',  # reasoner chat
        "messages": [{"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}'}],
        "search": False,  # 联网模式
    }
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            }, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return None
    return content


def ask_internlm3(tokenizer, model, item, prompt):
    messages = [
        {"role": "system", "content": "You are an helpful assistant."},
        {"role": "user", "content": f'{item['title']}\n{item['content']}\n{prompt}'}
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt").to("cuda")
    generated_ids = model.generate(tokenized_chat, max_new_tokens=1024, temperature=1, repetition_penalty=1.005,
                                   top_k=40, top_p=0.8)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(tokenized_chat, generated_ids)]
    result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return result


def ask_glm4(tokenizer, model, item, prompt):
    import torch
    encoding = tokenizer(f"{item['title']}\n{item['content']}\n{prompt}<|endoftext|>")
    inputs = {key: torch.tensor([value]).to(model.device) for key, value in encoding.items()}

    gen_kwargs = {"max_length": 32768, "do_sample": True, "top_k": 1}
    result = None
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if '{' in result and '}' in result:
        result = result[result.rfind('{'):result.rfind('}') + 1]
    return result


def ask_baidu(model_id, item, prompt):
    text = f"{item['title']}\n{item['content']}\n{prompt}"

    url = 'https://qianfan.baidubce.com/v2/chat/completions'
    auth = 'Bearer bce-v3'

    payload = json.dumps({
        "model": model_id,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": f"{item['title']}\n{item['content']}\n{prompt}"}]}],
        "max_completion_tokens": 1024
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json',
        'Authorization': auth,
        'appid': 'app-12qs7NEx'
    }
    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    result = response.text
    return result


def ask_kimi(model_id, item, prompt):
    completion = KIMI_CLIENT.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": f"{item['title']}\n{item['content']}\n{prompt}"}, ],
        # temperature = 0.6,
    )
    result = completion.choices[0].message.content
    time.sleep(1.0)
    return result

def ask_doubao(model_id, item, prompt):
    response = DOUBAO_CLIENT.responses.create(
        model=model_id,
        input=[{ "role": "user", "content": [{"type": "input_text", "text": f"{item['title']}\n{item['content']}\n{prompt}" }], }],
        reasoning = {"effort": "minimal"},
    )
    result = response.output_text
    return result

def ask_bart_method(model, item, prompt):
    result = model(f"{item['title']}\n{item['content']}\n{prompt}", news_methods)
    scores = result['scores']
    result = result['labels'][scores.index(max(scores))]
    return result

def ask_bart(model, item, prompt):
    result = model(f"{item['title']}\n{item['content']}\n{prompt}", news_categories)
    scores = result['scores']
    result = result['labels'][scores.index(max(scores))]
    return result

def ask_deepseek_8b(tokenizer, model, item, prompt):
    # '<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n '

    prompt = f"<|im_start|>user\n{item['title']}\n{item['content']}\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n "
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=10240, use_cache = True,
        pad_token_id=tokenizer.eos_token_id, temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)[len(prompt):].replace('<|im_end|>', '').strip()
    if response.startswith("\"response") or response.startswith("\"disorder_method"):
        response = '{ ' + response + ' }'
    # response = response[response.find('</think>\n\n') + len('</think>\n\n'):]
    return response


def ask_model(model_name, model_id, tokenizer, model, item, prompt):
    result = ''
    if model_name == 'mistral3-14b':
        result = ask_mistral3(tokenizer, model, item, prompt)
    elif model_name == 'phi-4-14b':
        result = ask_phi_4(model, item, prompt)
    elif model_name in ('qwen3-8b', 'qwen3-4b'):
        result = ask_qwen3(tokenizer, model, item, prompt)
    elif model_name in ('llama3.1-8b', 'llama3.2-3b'):
        result = ask_llama3(model, item, prompt)
    elif model_name == 'llama4-scout':
        result = ask_open_router(model_id, item, prompt)
    elif model_name in ('phi3-medium', 'phi-3.5-mini'):
        result = ask_phi3(model, item, prompt)
    elif model_name == 'llama3.1-70b':
        result = ask_open_router(model_id, item, prompt)
    elif model_name in ('qwen2.5-7b', 'qwen2-7b'):
        result = ask_qwen2_5(tokenizer, model, item, prompt)
    elif model_name in ('minicpm4.1-8b', 'minicpm4-8b'):
        result = ask_minicpm4(tokenizer, model, item, prompt)
    elif model_name == 'deepseek-online':
        result = ask_deepseek_online(item, prompt)
    elif model_name == 'internlm3-8b':
        result = ask_internlm3(tokenizer, model, item, prompt)
    elif model_name == 'glm4-9b':
        result = ask_glm4(tokenizer, model, item, prompt)
    elif model_name == 'kimi-k2':
        result = ask_kimi(model_id, item, prompt)
    elif model_name == 'doubao-1.6':
        result = ask_doubao(model_id, item, prompt)
    elif model_name == 'gpt5':
        result = ask_gpt(model_id, item, prompt)
    elif model_name == 'deepseek-8b':
        result = ask_deepseek_8b(tokenizer, model, item, prompt)

    result = result.replace('```json', ' ').replace('`', ' ').replace('\n', ' ').strip()
    if result.startswith('{') and not result.endswith('}'):
        result = result + '}'
    print(result, flush=True)
    return result


def load_model(model_name, model_id):
    model = None
    tokenizer = None
    if model_name == 'mistral3-14b':
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
        tokenizer = MistralCommonBackend.from_pretrained(model_id)
        model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map=DEVICE)
    elif model_name == 'phi-4-14b':
        import torch
        model = transformers.pipeline("text-generation", model="microsoft/phi-4",
                                      model_kwargs={"torch_dtype": torch.bfloat16}, device_map=DEVICE)
    elif model_name in ('qwen3-8b', 'qwen3-4b'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=DEVICE)
    elif model_name in ('llama3.1-8b', 'llama3.2-3b'):
        import torch
        model = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                                      device_map=DEVICE)
    elif model_name in ('qwen2.5-7b', 'qwen2-7b'):
        import torch
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    elif model_name in ('phi3-medium', 'phi-3.5-mini'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        torch.random.manual_seed(0)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                model_id,
                _attn_implementation="flash_attention_2",
                device_map="cuda",
                torch_dtype="auto",
                trust_remote_code=True,
            ),
            tokenizer=tokenizer
        )
    elif model_name in ('minicpm4-8b', 'minicpm4.1-8b'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        torch.manual_seed(0)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='cuda',
                                                     trust_remote_code=True, revision='main')

    elif model_name == 'internlm3-8b':
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16).cuda()
        model = model.eval()

    elif model_name == 'glm4-9b':
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto"
                                                     ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
    elif model_name == 'bart':
        from transformers import pipeline
        model = pipeline("zero-shot-classification", model=model_id)
    elif model_name == 'deepseek-8b':
        # from unsloth import FastLanguageModel
        import torch
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_id,
            max_seq_length = 8192,
            load_in_4bit = False,  # 关闭4bit量化
            dtype = torch.bfloat16,
            # fast_inference = True,
            trust_remote_code = False,
        )
        model = model.eval()

    return model, tokenizer


def main(model_name, model_id, data, data_path):
    MODEL_ID = model_id
    MODEL_NAME = model_name
    METHOD_KEY = 'disorder_method_' + MODEL_NAME
    METHOD_REASON_KEY = 'disorder_method_reason_' + MODEL_NAME
    INTENT_KEY = 'intent_' + MODEL_NAME

    BASE_ANSWER = 'answer_base_' + MODEL_NAME
    BASE_ANSWER_REASON = 'answer_base_reason_' + MODEL_NAME
    DISORDER_ANSWER = 'answer_disorder_' + MODEL_NAME
    DISORDER_ANSWER_REASON = 'answer_disorder_reason_' + MODEL_NAME
    INTENT_ANSWER = 'answer_disorder_intent_' + MODEL_NAME
    INTENT_ANSWER_REASON = 'answer_disorder_intent_reason_' + MODEL_NAME

    model, tokenizer = load_model(MODEL_NAME, MODEL_ID)

    for item in data:
        modified = False
        
        try:
            # 生成信息失序信息
            if METHOD_KEY not in item:
                modified = True
                if model_name == 'bart':
                    result = ask_bart_method(model, item, PROMPT1)
                    item[METHOD_KEY] = result
                    item[METHOD_REASON_KEY] = 'None'
                else:
                    result = ask_model(MODEL_NAME, MODEL_ID, tokenizer, model, item, PROMPT1)
                    result = re.search(r'\{.*?\}(?=\s*\{|$)', result, re.DOTALL).group(0).replace('\"', '"').replace('"', "'").replace("\'", "'")
                    result = re.sub(r"\{\s*'disorder_method':\s*'", '{ "disorder_method": "', result)
                    result = re.sub(r"',\s*'reason':\s*'", '", "reason": "', result)
                    result = re.sub(r"'\s*\}", '" }', result)
                    result = json.loads(result)
                    item[METHOD_KEY] = result['disorder_method'].strip().replace('*', '')
                    item[METHOD_REASON_KEY] = result['reason'].strip().replace('*', '')
                # print('信息失序方式: ', item[METHOD_KEY])

            # 生成意图信息
            if INTENT_KEY not in item:
                modified = True
                full_prompt = PROMPT2.replace(REP_DISORDER, f'disorder method: {item[METHOD_KEY]}\nexplanation: {item[METHOD_REASON_KEY]}')
                if model_name == 'bart':
                    item[INTENT_KEY] = 'None'
                else:
                    result = ask_model(MODEL_NAME, MODEL_ID, tokenizer, model, item, full_prompt)
                    item[INTENT_KEY] = result.replace('*', '').strip()
                # print('新闻意图: ', item[INTENT_KEY])

            # 信息失序分类，基线
            if BASE_ANSWER not in item:
                modified = True
                if model_name == 'bart':
                    item[BASE_ANSWER] = ask_bart(model, item, BASE_PROMPT)
                    item[BASE_ANSWER_REASON] = 'None'
                else:
                    result = ask_model(MODEL_NAME, MODEL_ID, tokenizer, model, item, BASE_PROMPT)
                    result = re.search(r'\{.*?\}(?=\s*\{|$)', result, re.DOTALL).group(0).replace('\"', '"').replace('"', "'").replace("\'", "'")
                    result = re.sub(r"\{\s*'response':\s*'", '{ "response": "', result)
                    result = re.sub(r"',\s*'reason':\s*'", '", "reason": "', result)
                    result = re.sub(r"'\s*\}", '" }', result)
                    result = json.loads(result)
                    item[BASE_ANSWER] = result['response'].strip().replace('*', '')
                    item[BASE_ANSWER_REASON] = result['reason'].strip().replace('*', '')
                print('基线答案: ', item[BASE_ANSWER])
                # print('基线理由: ', item[BASE_ANSWER_REASON])

            # 信息失序分类，进阶，添加信息失序分析
            if DISORDER_ANSWER not in item and METHOD_KEY in item:
                modified = True
                full_prompt = DISORDER_PROMPT.replace(REP_DISORDER,
                                                      f'disorder method: {item[METHOD_KEY]}\nexplanation: {item[METHOD_REASON_KEY]}')
                if model_name == 'bart':
                    item[DISORDER_ANSWER] = ask_bart(model, item, full_prompt)
                    item[DISORDER_ANSWER_REASON] = 'None'
                else:
                    result = ask_model(MODEL_NAME, MODEL_ID, tokenizer, model, item, full_prompt)
                    result = re.search(r'\{.*?\}(?=\s*\{|$)', result, re.DOTALL).group(0).replace('\"', '"').replace('"', "'").replace("\'", "'")
                    result = re.sub(r"\{\s*'response':\s*'", '{ "response": "', result)
                    result = re.sub(r"',\s*'reason':\s*'", '", "reason": "', result)
                    result = re.sub(r"'\s*\}", '" }', result)
                    result = json.loads(result)
                    item[DISORDER_ANSWER] = result['response'].strip().replace('*', '')
                    item[DISORDER_ANSWER_REASON] = result['reason'].strip().replace('*', '')
                print('进阶答案: ', item[DISORDER_ANSWER])
                # print('进阶理由: ', item[DISORDER_ANSWER_REASON])

            # 信息失序分类，高级，添加信息失序分析+意图分析
            if INTENT_ANSWER not in item and METHOD_KEY in item and INTENT_KEY in item:
                modified = True
                full_prompt = DISORDER_INTENT_PROMPT.replace(REP_DISORDER,
                                                             f'disorder method: {item[METHOD_KEY]}\nexplanation: {item[METHOD_REASON_KEY]}'
                                                             ).replace(REP_INTENT, item[INTENT_KEY])
                if model_name == 'bart':
                    item[INTENT_ANSWER] = ask_bart(model, item, full_prompt)
                    item[INTENT_ANSWER_REASON] = 'None'
                else:
                    result = ask_model(MODEL_NAME, MODEL_ID, tokenizer, model, item, full_prompt)
                    result = re.search(r'\{.*?\}(?=\s*\{|$)', result, re.DOTALL).group(0).replace('\"', '"').replace('"', "'").replace("\'", "'")
                    result = re.sub(r"\{\s*'response':\s*'", '{ "response": "', result)
                    result = re.sub(r"',\s*'reason':\s*'", '", "reason": "', result)
                    result = re.sub(r"'\s*\}", '" }', result)
                    result = json.loads(result)
                    item[INTENT_ANSWER] = result['response'].strip().replace('*', '')
                    item[INTENT_ANSWER_REASON] = result['reason'].strip().replace('*', '')
                print('高级答案: ', item[INTENT_ANSWER])
                # print('高级理由: ', item[INTENT_ANSWER_REASON])

            if modified:
                print('', flush=True)
                print('\n数据索引: ', item['id'])
                print('标准答案', item['info_type'])
                data[item['id']] = item
                with open(data_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    model_name = 'mistral3-14b'
    model_id = 'mistralai/Ministral-3-14B-Instruct-2512'

    model_name = 'llama3.1-8b'
    model_id = '/ai/teacher/ssz/0-Weights/Llama-3.1-8B-Instruct'

    model_name = 'llama3.1-70b'
    model_id = "meta-llama/llama-3.1-70b-instruct"

    model_name = 'llama3.2-3b'
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    model_name = 'qwen2-7b'
    model_id = "Qwen/Qwen2-7B-Instruct"

    model_name = 'qwen2.5-7b'
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    model_name = 'qwen3-8b'
    model_id = 'Qwen/Qwen3-8B'

    model_name = 'qwen3-4b'
    model_id = 'Qwen/Qwen3-4B-Instruct-2507'

    model_name = 'llama4-scout'
    model_id = 'meta-llama/llama-4-scout' # llama-4-scout-17b-16e-instruct

    # model_name = 'hunyuan-13B'
    # model_id = 'Tencent-Hunyuan/Hunyuan-A13B-Instruct'

    # model_name = 'phi-4-14b'
    # model_id = 'microsoft/phi-4'

    # model_name = 'phi3-medium'
    # model_id = 'LLM-Research/Phi-3-medium-128k-instruct'

    # model_name = 'phi-3.5-mini'
    # model_id = 'LLM-Research/Phi-3.5-mini-instruct'

    # model_name = 'minicpm4-8b'
    # model_id = '/root/.cache/huggingface/hub/models--openbmb--MiniCPM4-8B/snapshots/bb2ae14cf59d4ca769c4e42ece54cc3b82a58ef7'

    # transformers 4.56.2
    # model_name = 'minicpm4.1-8b'
    # model_id = '/root/.cache/huggingface/hub/models--openbmb--MiniCPM4.1-8B/snapshots/3a8dfed9c79a45e07dbff95bcd49d792343fa1a3'

    # model_name = 'deepseek-online'
    # model_id = ''

    # model_name = 'internlm3-8b'
    # model_id = 'internlm/internlm3-8b-instruct'

    # transformers 4.47.1
    # model_name = 'glm4-9b'
    # model_id = 'zai-org/glm-4-9b-hf'

    # model_name = 'kimi-k2'
    # model_id = "kimi-k2-0905-preview"
    
    # agent doubao-seed-1-8-251228
    # model_name = 'doubao-1.6'
    # model_id = 'doubao-seed-1-6-251015'
    
    # model_name = 'bart'
    # model_id = "facebook/bart-large-mnli"
    
    # model_name = 'gpt5'
    # model_id = "gpt-5-chat-latest"
    
    model_name = 'gpt5'
    model_id = "gpt-5-chat-latest"
    
    model_name = 'deepseek-8b'
    model_id = 'TeichAI/Qwen3-8B-DeepSeek-v3.2-Speciale-Distill'
    

    data_path = f'chat/data/GoodNewsData/GoodNews_{model_name}.json'
    with open(data_path, 'r') as f:
        data: list = json.load(f)
    main(model_name, model_id, data, data_path)
