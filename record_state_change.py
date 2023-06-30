from os import environ
import os

import numpy as np

try:
    # setup RWKV_JIT_ON and RWKV_CUDA_ON
    environ["RWKV_CUDA_ON"] = "1"
    environ["RWKV_JIT_ON"] = "1"

    from rwkv.model import RWKV
    from rwkv_contrib.pipeline import StatefulPipeline
    from rwkv_contrib.tokenizer import RWKVTokenizer
    from rwkv_contrib.debug_tools import StateDump
except Exception as e:
    raise e

PROMPT_FILE = "test_prompt.txt"
MODEL_PATH = "models/RWKV-4-World-7B-v1-20230626-ctx4096.pth"
STRATEGY = "cuda fp16"
LAYER_COUNT = 32

os.makedirs("states", exist_ok=True)
tokenizer = RWKVTokenizer()
tokens = tokenizer.encode(open(PROMPT_FILE).read())

# create RWKV instance
model = RWKV(model=MODEL_PATH, strategy=STRATEGY)

# create pipeline, note that we don't need to specify anything here
# since the model is here for reading prompt and record token effects
pipeline = StatefulPipeline(model)

# create state dump and wrap the pipeline
state_dump = StateDump("states/state")


def concat(att_xx, att_aa, att_bb, att_pp, ffn_xx) -> list[float]:
    return att_xx.float().tolist() + att_aa.float().tolist() + att_bb.float().tolist() + ffn_xx.float().tolist()
state_dump.concat = concat

state_dump.wraps(pipeline)

# start pipeline, note that penalties or sampling are not needed as we 
# actively feed tokens into the model
for token in tokens:
    pipeline.infer([token])

# dump state
state_dump.dumps()
