# host Qwen3VL testing with vLLM

## Installation
See [vLLM guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html#qwen3-vl-235b-a22b-instruct)

```bash
cd vllm-env
uv venv
source .venv/bin/activate

# Install vLLM >=0.11.0
uv pip install -U vllm

# Install Qwen-VL utility library (recommended for offline inference)
uv pip install qwen-vl-utils==0.0.14

# Optional: get the qwen3 repo for example test
cd ..
git clone https://github.com/QwenLM/Qwen3-VL.git
cd vllm-env
```

## test run

First deploy:
```bash
sbatch slurm_deploy.sh
```
or in interactive job
```bash
./deploy.sh
```

Then run (just an example):
```bash
test_mm.py
```

Simple example:
```Python
# after server is hosted on port 8000
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    max_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    extra_body={
        "top_k": 20,
    },
)
print("Chat response:", chat_response)
print("--------------------------------------------------------")
print("Response text:", chat_response.choices[0].message.content)
```

## qwen3 cookbooks
More test examples are in the [qwen3 cookbooks](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb)

## TODO

- [ ] when the server is launched on hpc using slurm, how to get access to it in another job? let other people?