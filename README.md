# bitnet-rs: Bitnet Transformer in Rust!

Implementation of the Bitnet transformer using [Candle](https://github.com/huggingface/candle). Implementation is based on the pytorch implementation here: [kyegomez/BitNet](https://github.com/kyegomez/BitNet)

## About

I started this project in order to better understand what goes into making a transformer model in a ML library from scratch, rather than re-implement an existing model I wanted to try doing this from a less known and unimplemented model. In addition, I'm curious about non pytorch based models in order to push performance for models, as such learning to use Candle was a big part of this!

## Building

### CPU

`cargo build --release`

### Metal

`cargo build --release --features "metal,accelerate"`

### CUDA

`cargo build --release --features "cuda"`

## Training

First, build the binary according to the instructions above, then run the command below.

`./target/release/bitnet-rs train --dataset "<path to dataset>"`

Replace `<path to dataset>` with the directory location of the dataset you are training from. These must be precompiled datasets. I would recommend using the same dataset that has been used for validation: [karpathy/llama2.c](https://github.com/karpathy/llama2.c?tab=readme-ov-file#training). Please follow the instructions in that repository for generating the pretokenized dataset.

For example, on my machine the training command is this: `./target/release/bitnet-rs train --dataset "../../karpathy/llama2.c/data/TinyStories_all_data"`.

## Inference

First, build the binary according to the instructions above, then run the command below.

`./target/release/bitnet-rs inference`

If you want to provide a prompt, provide the `--prompt` flag.

`./target/release/bitnet-rs inference --prompt "Once upon a time "`

If you want to specify a specific model to use for the inference, use the `--pretrained-model-path` flag.

`./target/release/bitnet-rs inference --pretrained-model-path "./checkpoint.safetensors"`.

## Known Issues

I'm still testing this out but I am getting semi coherent output with models I've trained. Definitely not useful for any task right now until I can get loss down.

## Contributing

If you have an interest in contributing please feel free! I'm still learning and would appreciate any suggestions from others.