use std::io::Write;

use anyhow::Result;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use crate::bit_transformer::BitTransformer;
use crate::config::Config;
use crate::token_output_stream::TokenOutputStream;
use crate::{utils_tensor::device, Args, InferenceCmd};
use candle_core::{safetensors, DType, Error as E, IndexOp, Tensor};
use candle_nn::VarBuilder;

pub fn run(args: &InferenceCmd, common_args: &Args) -> Result<()> {
    let tokenizer = {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
        let tokenizer_path = api.get("tokenizer.json")?;
        Tokenizer::from_file(tokenizer_path).unwrap()
    };

    let device = device(common_args.cpu)?;

    let safetensors = safetensors::load("./checkpoint.safetensors", &device)?;
    let vb = VarBuilder::from_tensors(safetensors, DType::F32, &device);

    let config = Config::default();
    let mut model = BitTransformer::load(config, vb, false)?;

    println!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(299792458, args.temperature, args.top_p);

    print!("{}", args.prompt);
    let mut tokens = tokenizer
        .encode(args.prompt.clone(), true)
        .unwrap()
        .get_ids()
        .to_vec();
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    let start_gen = std::time::Instant::now();
    for index in 0.. {
        if tokens.len() >= config.max_seq_len {
            break;
        }
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input)?;
        let logits = logits.i((0, logits.dim(1)? - 1))?;
        let logits = if args.repeat_penalty == 1. || tokens.is_empty() {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n{} tokens generated ({:.2} token/s)\n",
        tokens.len(),
        tokens.len() as f64 / dt.as_secs_f64(),
    );

    Ok(())
}
