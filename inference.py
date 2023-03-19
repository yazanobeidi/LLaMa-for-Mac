# Copyright (c) 2023 Yazan Obeidi.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import torch
import time
import os
import gc
from llama import ModelArgs, Tokenizer, Transformer, LLaMA

class LLaMAInference:
    def __init__(self, path_to_weights, model,
                 max_seq_len=1024, max_batch_size=1, **kwargs):
        start_time = time.time()
        print('PYTORCH_ENABLE_MPS_FALLBACK', os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))
        state_dict = os.path.join(path_to_weights, model, "state_dict.pth")
        params_file = os.path.join(path_to_weights, model, "params.json")
        tokenizer_path = os.path.join(path_to_weights, "tokenizer.model")

        assert os.path.exists(os.path.join(path_to_weights, model)), f"Model {model} does not exist"
        assert os.path.exists(state_dict), f"Model {model} does not exist"
        assert os.path.exists(params_file), f"Model {model} does not exist"
        assert os.path.exists(tokenizer_path), f"Missing tokenizer in {path_to_weights}"

        with open(params_file, "r") as f:
            params = json.load(f)

        model_args = dict(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params
        )
        model_args.update(kwargs)
        model_args = ModelArgs(**model_args)
        print('loading LLaMA tokenizer...')
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = self.tokenizer.n_words
        print('loading Transformer...')
        torch.set_default_tensor_type(torch.HalfTensor)
        model = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)
        print('loading LLaMA checkpoint....')
        checkpoint = torch.load(state_dict, map_location="cpu")
        print('loading checkpoint into LLaMA initialization...')
        model.load_state_dict(checkpoint, strict=False)
        print('clearing cache...')
        del checkpoint
        del state_dict
        gc.collect()
        print('moving LLaMa model to mps......')
        model = model.to("mps")
        print('loading LLaMA model object...')
        self.generator = LLaMA(model, self.tokenizer)
        print('clearing cache...')
        del model
        print(f"Done. Loaded LLaMA in {time.time() - start_time:.2f} seconds")
        gc.collect()

    def generate(self, prompts, temperature=0.8, top_p=0.95, max_gen_len=32):
        results = self.generator.generate(
            prompts=prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return results