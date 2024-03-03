use crate::bit_transformer::BitTransformer;
use anyhow::Result;
use candle_core::{Device, Error, Tensor, D};
use candle_nn::ops::softmax;
use candle_transformers::generation::LogitsProcessor;
use rand::distributions::Distribution;
use rand::{self, SeedableRng};

pub struct AutoregressiveWrapper {
    net: BitTransformer,
    max_seq_len: usize,
    pad_value: usize,
    device: Device,
    seed: u64,
}

impl AutoregressiveWrapper {
    pub fn new(
        net: BitTransformer,
        max_seq_len: usize,
        pad_value: usize,
        seed: u64,
        device: Device,
    ) -> Self {
        Self {
            net,
            max_seq_len,
            pad_value,
            device,
            seed,
        }
    }

    pub fn generate(
        &mut self,
        start_tokens: &Tensor,
        seq_len: usize,
        eos_token: Option<usize>,
        temperature: f64,
        filter_thres: f32,
    ) -> Result<Tensor> {
        let mut output = start_tokens.clone();
        let mut logits_processor = LogitsProcessor::new(self.seed, Some(temperature), None);

        for _ in 0..seq_len {
            let logits = self.net.forward(&output)?;
            let last_seq_idx = (logits.shape().dims()[1] - 1) as u32;
            let logits = logits
                .squeeze(0)?
                .index_select(
                    &Tensor::new(last_seq_idx, &self.device)?.unsqueeze(0)?,
                    D::Minus1,
                )?
                .squeeze(1)?;
            let sample = Tensor::new(logits_processor.sample(&logits)?, &self.device)?
                .unsqueeze(0)?
                .unsqueeze(0)?;
            output = Tensor::cat(&[&output, &sample], D::Minus1)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod inference_tests {
    use anyhow::Result;
    use candle_core::Tensor;

    use crate::bit_transformer::BitTransformer;
    use crate::inference::AutoregressiveWrapper;
    use crate::utils_tensor::device;

    #[test]
    fn test_inference() -> Result<()> {
        let device = device(false)?;

        let net = BitTransformer::load(512, 8, 256, 8, 4, &device.clone()).unwrap();
        let mut wrapper = AutoregressiveWrapper::new(net, 1024, 0, 0, device.clone());

        let start_tokens =
            Tensor::ones((1, 256), candle_core::DType::U32, &device.clone()).unwrap();
        let output = wrapper.generate(&start_tokens, 2, None, 1.0, 0.0).unwrap();

        assert_eq!(output.shape().dims(), &[1, 258]);
        Ok(())
    }
}
