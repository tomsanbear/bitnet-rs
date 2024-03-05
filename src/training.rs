use crate::config::Config;
use crate::{bit_transformer::BitTransformer, utils_tensor::device, Args, TrainingCmd};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_datasets::nlp::tinystories::{Dataset, DatasetRandomIter};
use candle_datasets::Batcher;
use candle_nn::loss::cross_entropy;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use kdam::tqdm;

fn valid_loss(
    seq_len: usize,
    batch_size: usize,
    dataset: &Dataset,
    model: &mut BitTransformer,
    device: &Device,
) -> Result<f64> {
    let iter = DatasetRandomIter::new(dataset, true, seq_len, device.clone());
    let batch_iter = Batcher::new_r2(iter).batch_size(batch_size);
    let mut sum_ce = 0f64;
    let mut cnt = 0usize;
    for inp_tgt in tqdm!(batch_iter.take(50), total = 50, desc = "Validating loss") {
        let (inp, tgt) = inp_tgt?;
        let logits = model.forward(&inp)?;
        let loss = cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        sum_ce += f64::from(loss.to_vec0::<half::bf16>()?);
        cnt += 1;
    }
    Ok(sum_ce / cnt as f64)
}

pub fn run(args: &TrainingCmd, common_args: &Args) -> Result<()> {
    // Setup device
    let device = device(common_args.cpu)?;

    // Setup varbuilder
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::BF16, &device);

    // Get the datasets
    let dataset = { Dataset::new("../../karpathy/llama2.c/data/TinyStories_all_data")? };

    // Setup the model
    let config = Config::default();
    let mut model = BitTransformer::load(config, vb)?;

    // Setup the optimizer
    let params = ParamsAdamW {
        lr: args.learning_rate,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    let iter = DatasetRandomIter::new(&dataset, false, args.seq_len, device.clone());
    let batch_iter = candle_datasets::Batcher::new_r2(iter).batch_size(args.batch_size);

    // Training loop
    for (batch_index, batch) in tqdm!(
        batch_iter.enumerate(),
        total = args.max_steps,
        desc = "Training"
    ) {
        if batch_index > args.max_steps {
            break;
        }
        let (inp, tgt) = batch?;
        let logits = model.forward(&inp)?;
        let loss = cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        opt.backward_step(&loss)?;

        if batch_index > 0 && batch_index % 100 == 0 {
            let loss = valid_loss(args.seq_len, args.batch_size, &dataset, &mut model, &device)?;
            println!("batch={batch_index}, loss={loss}");
            varmap.save("checkpoint.safetensors")?
        }
    }

    Ok(())
}
