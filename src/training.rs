use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use anyhow::Result;
use candle_core::{Device, Error as E};
use candle_datasets::nlp::tinystories::{Dataset, DatasetRandomIter};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use hf_hub::api::sync::Api;

use crate::{
    bit_transformer::BitTransformer, inference::AutoregressiveWrapper, utils_tensor::device,
};

fn valid_loss(
    seq_len: usize,
    batch_size: usize,
    dataset: &Dataset,
    model: &mut AutoregressiveWrapper,
    device: &Device,
) -> Result<f64> {
    let iter = DatasetRandomIter::new(dataset, true, seq_len, device.clone());
    let batch_iter = candle_datasets::Batcher::new_r2(iter).batch_size(batch_size);
    let mut sum_ce = 0f64;
    let mut cnt = 0usize;
    for inp_tgt in batch_iter.take(50) {
        let (inp, tgt) = inp_tgt?;
        let logits = model.forward(&inp)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        sum_ce += loss.to_vec0::<f32>()? as f64;
        cnt += 1;
    }
    Ok(sum_ce / cnt as f64)
}

pub fn train() -> Result<()> {
    // Training parameters
    const BATCH_SIZE: usize = 4;
    const LEARNING_RATE: f64 = 2e-4;
    const SEQ_LEN: usize = 1024;
    const DIMS: usize = 512;
    const DEPTH: usize = 8;
    const NUM_HEADS: usize = 8;
    const FF_MULT: usize = 4;
    const NUM_TOKENS: usize = 256;

    // Setup device
    let device = device(false)?;

    // Setup tokenizer
    let tokenizer = {
        let api = hf_hub::api::sync::Api::new()?;
        let api = api.model("hf-internal-testing/llama-tokenizer".to_string());
        let api = api.get("tokenizer.json")?;
        let tokenizer = tokenizers::Tokenizer::from_file(api).unwrap();
        tokenizer
    };

    // Get the datasets
    let dataset = {
        Dataset::new(
            "/Users/tomsanbear/workspace/github.com/karpathy/llama2.c/data/TinyStories_all_data",
        )?
    };

    // Setup the model
    let model = BitTransformer::load(DIMS, DEPTH, NUM_TOKENS, NUM_HEADS, FF_MULT, &device)?;
    let mut model = AutoregressiveWrapper::new(1, model, device.clone());

    // Setup the optimizer
    let varmap = VarMap::new();
    let params = ParamsAdamW {
        lr: LEARNING_RATE,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    let iter = DatasetRandomIter::new(&dataset, false, SEQ_LEN, device.clone());
    let batch_iter = candle_datasets::Batcher::new_r2(iter).batch_size(BATCH_SIZE);

    // Training loop
    for (batch_index, batch) in batch_iter.enumerate() {
        let (inp, tgt) = batch?;
        println!("inp: {:?}", inp.shape());
        println!("tgt: {:?}", tgt.shape());
        let logits = model.forward(&inp)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &tgt.flatten_to(1)?)?;
        opt.backward_step(&loss)?;

        if batch_index > 0 && batch_index % 100 == 0 {
            // TODO: Add a way to deactivate the backprop graph tracking when computing the
            // validation loss.
            let loss = valid_loss(SEQ_LEN, BATCH_SIZE, &dataset, &mut model, &device)?;
            println!("{batch_index} {loss}");
        }
        if batch_index > 0 && batch_index % 1000 == 0 {
            varmap.save("checkpoint.safetensors")?
        }
    }

    Ok(())
}
