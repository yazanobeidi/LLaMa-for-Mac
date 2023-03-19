# LLaMa-for-Mac
Meta's LLaMa ready to run on your Mac with M1/M2 Apple Silicon

## Description

The original LLaMa release ([facebookresearch/llma](https://github.com/facebookresearch/llama)) requires CUDA. 

This repo contains minimal modifications to run on Apple Silicon M1/M2 and GPU by leveraging Torch MPS.

## Setup

First download a) the weights and b) `tokenizer.model`:

1. IPFS: [here](https://ipfs.io/ipfs/Qmb9y5GCkTG7ZzbBWMu2BXwMkzyCKcUjtEKPpgdZ7GEFKm/30B/) or [mirror (note this does not have tokenizer.model)](ipfs://QmSD8cxm4zvvnD35KKFu8D9VjXAavNoGWemPW1pQ3AF9ZZ)
2. BiTorrent: magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA

After downloading, move the files from your Downloads folder to `LLaMa-for-Mac/weights/sharded`

Next, clone this repository

`git clone https://github.com/yazanobeidi/LLaMa-for-Mac`

Setup a virtualenv (optional) and install Python requirements by running:

`python3.11 -m venv ~/.LLaMa`
`source ~/.env/LLaMa/bin/activate`
`pip install -r requirements.txt`

To run without torch-distributed on single node we must unshard the sharded weights. To do this, run the following, where --path-to-weights points to your `path/to/weights`, and --model points to the model version you downloaded.

`python3 convert_to_unsharded.py --path-to-weights weights/sharded/ --model 30B --output-path weights/unsharded/`

This will take a minute or so.

## Usage

After following the Setup steps above, you can launch a webserver hosting LLaMa with a single command:

`python server.py --path-to-weights weights/unsharded/ --max-seq-len 128 --max-gen-len 128 --model 30B`

Now you can make requests to the `/generate` endpoint with your prompt as payload, for example:

`curl -X GET http://localhost:3000/generate -H "Content-Type: application/json" -d '{"prompt": "Hello world"}'`

## License

This is a modified version of [facebookresearch/llma](https://github.com/facebookresearch/llama) which was originally licensed under GPL3. Therefore this work retains the GPL3 license. See [LICENSE.md](License.md).