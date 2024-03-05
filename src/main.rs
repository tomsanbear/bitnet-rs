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

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[derive(Parser, Debug, Clone)]
struct InferenceCmd {}

#[derive(Parser, Debug, Clone)]
struct EvaluationCmd {}

#[derive(Parser, Debug, Clone)]
pub struct TrainingCmd {
    /// The data type for the weights, due to the implementation, we should theoretically be able to use a single bit, but we need candle to support this or contribute this
    /// For now, this can only be: u8, u32, bf16, f16, f32, f64
    #[arg(long, default_value = "f16")]
    dtype: String,

    /// The path to the dataset.
    #[arg(
        long,
        default_value = "/Users/tomsanbear/workspace/github.com/karpathy/llama2.c/data/TinyStories_all_data"
    )]
    dataset: String,

    /// The maxiumum steps to train for
    #[arg(long, default_value = "10000")]
    max_steps: usize,

    /// The batch size to use
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// The learning rate to use
    #[arg(long, default_value = "2e-4")]
    learning_rate: f64,

    /// The sequence length to use
    #[arg(long, default_value = "1024")]
    seq_len: usize,

    /// The number of tokens in the vocabulary
    #[arg(long, default_value = "10000")]
    num_tokens: usize,
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Inference(InferenceCmd),
    Eval(EvaluationCmd),
    Train(TrainingCmd),
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The task to be performed, inference, training or evaluation.
    #[command(subcommand)]
    task: Option<Task>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    // Setup tracing if enabled
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    match &args.task {
        Some(Task::Inference(_cmd)) => inference::run()?,
        Some(Task::Train(cmd)) => training::run(cmd, &args)?,
        _ => return Err(anyhow!("No task specified")),
    }
    Ok(())
}
