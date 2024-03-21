use crate::config::Config;
use crate::optimizer::BitnetOptimizer;
use crate::utils_tensor::cross_entropy;
use crate::{bit_transformer::BitTransformer, utils_tensor::device, Args, TrainingCmd};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_datasets::nlp::tinystories::{Dataset, DatasetRandomIter};
use candle_datasets::Batcher;
use candle_nn::{VarBuilder, VarMap};
use kdam::{tqdm, BarExt};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::span;

fn valid_loss(
    seq_len: usize,
    batch_size: usize,
    dataset: &Dataset,
    model: &mut BitTransformer,
    device: &Device,
) -> Result<f64> {
    let span = span!(tracing::Level::TRACE, "validate-loss");
    let _enter = span.enter();

    let iter = DatasetRandomIter::new(dataset, true, seq_len, device.clone());
    let batch_iter = Batcher::new_r2(iter).batch_size(batch_size);
    let mut sum_ce = 0f64;
    let mut cnt = 0usize;
    let batch_count = 10;

    for inp_tgt in batch_iter.take(batch_count) {
        let span = span!(tracing::Level::TRACE, "validate-loss-iter");
        let _enter = span.enter();
        let (inp, tgt) = inp_tgt?;
        let logits = model.forward(&inp)?;
        let loss = cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        sum_ce += match loss.dtype() {
            DType::F32 => f64::from(loss.to_vec0::<f32>()?),
            DType::F16 => f64::from(loss.to_vec0::<half::f16>()?),
            DType::BF16 => f64::from(loss.to_vec0::<half::bf16>()?),
            _ => panic!("Invalid dtype"),
        };
        cnt += 1;
    }
    Ok(sum_ce / cnt as f64)
}

pub fn run(args: &TrainingCmd, common_args: &Args) -> Result<()> {
    let span = span!(tracing::Level::TRACE, "training");
    let _enter = span.enter();

    // Setup device
    let device = device(common_args.cpu)?;

    // Get the underlying data type to use for the model
    let dtype = match args.dtype.as_str() {
        "f32" => candle_core::DType::F32,
        "f16" => candle_core::DType::F16,
        "bf16" => candle_core::DType::BF16,
        _ => panic!("Invalid dtype"),
    };

    // Setup varbuilder
    let mut varmap = VarMap::new();

    // Load vars if checkpoint was provided
    if let Some(checkpoint) = &args.checkpoint {
        println!("Loading checkpoint: {:?}", checkpoint);
        varmap.load(checkpoint)?;
    }

    // Setup varbuilder
    let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

    // Get the datasets
    let dataset = { Dataset::new("../../karpathy/llama2.c/data/TinyStories_all_data")? };

    // Setup the model
    let config = Config {
        seq_len: args.seq_len,
        ..Config::default()
    };
    let mut model = BitTransformer::load(config, vb, true)?;

    // Setup the optimizer
    let mut opt = BitnetOptimizer::load(varmap.all_vars(), args.learning_rate)?;

    // Setup the dataset, currently using tinystories to replicate the llama example from candle
    let iter = DatasetRandomIter::new(&dataset, true, args.seq_len, device.clone());
    let batch_iter = candle_datasets::Batcher::new_r2(iter).batch_size(args.batch_size);

    // Training loop
    let mut training_loss = 0f64;
    let mut validation_loss = 0f64;
    let mut pb = tqdm!(total = args.max_steps, desc = "Training", position = 0);
    for (batch_index, batch) in batch_iter.enumerate() {
        pb.update(1)?;
        pb.set_postfix(&format!(
            "Training Loss: {:.2}, Validation Loss: {:.2}",
            training_loss, validation_loss
        ));

        if batch_index > args.max_steps {
            break;
        }

        let span = span!(tracing::Level::TRACE, "training-iteration");
        let _enter = span.enter();

        let (inp, tgt) = batch?;
        let logits = model.forward(&inp)?;
        let loss = cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        training_loss = match dtype {
            candle_core::DType::F32 => f64::from(loss.to_vec0::<f32>()?),
            candle_core::DType::F16 => f64::from(loss.to_vec0::<half::f16>()?),
            candle_core::DType::BF16 => f64::from(loss.to_vec0::<half::bf16>()?),
            _ => panic!("Invalid dtype"),
        };
        opt.backward_step(&loss)?;

        if batch_index > 0 && batch_index % 10 == 0 {
            validation_loss =
                valid_loss(args.seq_len, args.batch_size, &dataset, &mut model, &device)?;
            if batch_index % 10000 == 0 {
                let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
                let checkpoint_file_name = format!("checkpoint-{:?}.safetensors", timestamp);
                varmap.save(checkpoint_file_name)?;
            }
            varmap.save("checkpoint.safetensors")?;
        }
    }

    Ok(())
}
