# from https://github.com/galatolofederico/vanilla-llama/blob/main/convert.py
import argparse
import json
import torch
from accelerate import init_empty_weights
from tqdm import tqdm
from pathlib import Path
import os
import shutil
from llama import ModelArgs, Tokenizer, Transformer

def convert(model_path, tokenizer_path, output_path):
    checkpoints = sorted(Path(model_path).glob("*.pth"))
    with open(Path(model_path) / "params.json", "r") as f:
        params = json.loads(f.read())
    
    model_args = ModelArgs(
        max_seq_len=1024,
        max_batch_size=1,
        **params
    )

    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    with init_empty_weights():
        torch.set_default_tensor_type(torch.HalfTensor)
        model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)

    key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
    }

    converted_state_dict = {}

    for i, ckpt in tqdm(enumerate(checkpoints), total=len(checkpoints)):
        print(f'converted {i} out of {len(checkpoints)}')
        checkpoint = torch.load(ckpt, map_location="mps")
        for parameter_name, parameter in model.named_parameters():
            if parameter_name not in converted_state_dict:
                converted_state_dict[parameter_name] = torch.zeros_like(parameter, device="mps")
            short_name = parameter_name.split(".")[-2]
            if key_to_dim[short_name] is None and i == 0:
                converted_state_dict[parameter_name] = checkpoint[parameter_name]
            elif key_to_dim[short_name] == 0:
                size = checkpoint[parameter_name].size(0)
                converted_state_dict[parameter_name][size * i : size * (i + 1), :] = checkpoint[
                    parameter_name
                ]
            elif key_to_dim[short_name] == -1:
                size = checkpoint[parameter_name].size(-1)
                converted_state_dict[parameter_name][:, size * i : size * (i + 1)] = checkpoint[
                    parameter_name
                ]
            del checkpoint[parameter_name]
        del checkpoint

    with open(os.path.join(output_path, "params.json"), "w") as f:
        f.write(json.dumps(params, indent=4))

    torch.save(converted_state_dict, os.path.join(output_path, "state_dict.pth"))

if __name__ == "__main__":
    models = ["7B", "13B", "30B", "65B"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-weights", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=models, default="30B")
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()

    assert "tokenizer.model" in os.listdir(args.path_to_weights), "Tokenizer model not found path-to-weights"
    assert args.model in os.listdir(args.path_to_weights), f"Model {args.model} not found in llama path"
    
    output_path = os.path.abspath(os.path.join(args.output_path, args.model))
    os.makedirs(output_path, exist_ok=True)

    if "tokenizer.model" not in os.listdir(output_path):
        shutil.copy(os.path.join(args.path_to_weights, "tokenizer.model"), args.output_path)

    convert(
        os.path.join(args.path_to_weights, args.model),
        os.path.join(args.output_path, "tokenizer.model"),
        output_path
    )