# Copyright (c) 2023 Yazan Obeidi.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Union
import os
import time
import gc

from inference import LLaMAInference

def create_app(args):
    app = FastAPI()
    print('Loading model....')
    start_loading = time.time()
    llama = LLaMAInference(
        os.path.abspath(args.path_to_weights),
        args.model,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len
    )
    print("Model loaded.")

    class GenerateRequest(BaseModel):
        prompt: Union[List[str], str]
        temperature: float = 0.8
        top_p: float = 0.95
        max_gen_len: int = args.max_gen_len
    
    def verify_token(req: Request):
        if args.token == "":
            return True
        
        token = req.headers["Authorization"]
        if token != args.token:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized"
            )
        return True

    @app.get("/generate")
    def generate(gen_args: GenerateRequest, authorized: bool = Depends(verify_token)):
        gc.collect()
        print('Starting generation.....')
        if isinstance(gen_args.prompt, str):
            gen_args.prompt = [gen_args.prompt]

        if len(gen_args.prompt) > args.max_batch_size:
            return {"error": "Batch size too small"}

        start_generation = time.time()
        generated = llama.generate(
            prompts=gen_args.prompt,
            max_gen_len=gen_args.max_gen_len,
            temperature=gen_args.temperature,
            top_p=gen_args.top_p,
        )
        elapsed = f'{time.time() - start_generation:.2f}'
        print(f"Inference took {elapsed} seconds")
        return {"generated": generated,
                "elapsed_time": elapsed}

    return app


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--path-to-weights", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["7B", "13B", "30B", "65B"], default="30B")
    parser.add_argument("--max-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--token", type=str, default="")

    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(app, host="0.0.0.0", port=args.port)