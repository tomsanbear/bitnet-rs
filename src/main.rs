mod bit_attention;
mod bit_dropout;
mod bit_ffn;
mod bit_linear;
mod bit_transformer;
mod config;
mod embedding;
mod inference;
mod optimizer;
mod rms_norm;
mod training;
mod utils_tensor;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[derive(Parser, Debug, Clone)]
struct InferenceCmd {
    /// Pretrained model path, only safetensors are supported
    #[arg(long, default_value = "./checkpoint.safetensors")]
    pretrained_model_path: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long, default_value = "0.9")]
    top_p: f64,

    /// The repeat penalty to use
    #[arg(long, default_value = "1.1")]
    repeat_penalty: f32,

    /// The prompt for generation
    #[arg(long, default_value = "")]
    prompt: String,

    /// The number of tokens to repeat
    /// This is used to penalize repeating tokens
    #[arg(long, default_value = "10")]
    repeat_last_n: usize,
}

#[derive(Parser, Debug, Clone)]
pub struct TrainingCmd {
    /// The data type for the weights, due to the implementation, we should theoretically be able to use a single bit, but we need candle to support this or contribute this
    /// For now, this can only be: f32
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// The path to the dataset.
    #[arg(
        long,
        default_value = "../../karpathy/llama2.c/data/TinyStories_all_data"
    )]
    dataset: String,

    /// The batch size to use
    #[arg(long, default_value = "1")]
    batch_size: usize,

    /// The learning rate to use
    #[arg(long, default_value = "8e-4")]
    learning_rate: f64,

    /// The sequence length to use
    #[arg(long, default_value = "100")]
    seq_len: usize,

    /// The number of tokens in the vocabulary
    #[arg(long, default_value = "32000")]
    num_tokens: usize,

    /// The checkpoint file to continue from
    #[arg(long)]
    checkpoint: Option<String>,

    /// The number of epochs to train for
    #[arg(long, default_value = "1")]
    epochs: usize,

    /// Max number of steps
    #[arg(long, default_value = "100000")]
    max_steps: usize,
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Inference(InferenceCmd),
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
        Some(Task::Inference(cmd)) => inference::run(cmd, &args)?,
        Some(Task::Train(cmd)) => training::run(cmd, &args)?,
        _ => return Err(anyhow!("No task specified")),
    }
    Ok(())
}
