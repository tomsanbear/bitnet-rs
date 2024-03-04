#![feature(test)]
extern crate test;

mod bit_attention;
mod bit_ffn;
mod bit_linear;
mod bit_transformer;
mod config;
mod inference;
mod training;
mod utils_rms_norm;
mod utils_tensor;

use anyhow::Result;
use candle_core::Tensor;
use clap::Parser;

use crate::{
    bit_transformer::BitTransformer, inference::AutoregressiveWrapper, training::train,
    utils_tensor::device,
};

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Run training program
    #[arg(long, default_value = "false")]
    train: bool,

    /// The number of tokens in the vocabulary.
    #[arg(long, default_value = "10000")]
    num_tokens: usize,

    /// The dimension of the model.
    #[arg(long, default_value = "512")]
    dim: usize,

    /// The number of layers in the model.
    #[arg(long, default_value = "6")]
    depth: usize,

    /// The number of attention heads.
    #[arg(long, default_value = "8")]
    heads: usize,

    /// The feedforward multiplier.
    #[arg(long, default_value = "4")]
    ff_mult: usize,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    // Run Training
    if args.train {
        train()?;
        return Ok(());
    }

    // Run inference
    let device = device(false)?;
    let net = BitTransformer::load(
        args.dim,
        args.depth,
        args.num_tokens,
        args.heads,
        args.ff_mult,
        &device,
    )?;
    let mut wrapper = AutoregressiveWrapper::new(0, net, device.clone());

    let start_tokens = Tensor::ones((1, 256), candle_core::DType::U32, &device.clone()).unwrap();

    println!("starting the inference loop");
    let start_gen = std::time::Instant::now();
    wrapper.generate(&start_tokens, 10, None, 1.0, 0.0).unwrap();
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        10,
        10f64 / dt.as_secs_f64(),
    );
    Ok(())
}
