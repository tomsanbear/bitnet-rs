use crate::bit_transformer::BitTransformer;
use anyhow::Result;
use candle_core::{Device, Tensor, D};

use candle_nn::loss::cross_entropy;
use candle_transformers::generation::LogitsProcessor;

pub struct AutoregressiveWrapper {
    net: BitTransformer,
    device: Device,
    seed: u64,
}

impl AutoregressiveWrapper {
    pub fn new(seed: u64, net: BitTransformer, device: Device) -> Self {
        Self { net, device, seed }
    }

    pub fn generate(
        &mut self,
        start_tokens: &Tensor,
        seq_len: usize,
        _eos_token: Option<usize>,
        temperature: f64,
        _filter_thres: f32,
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

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        // Original python implementation
        // x_inp, x_labels = x[:, :-1], x[:, 1:]
        // logits = self.net(x_inp, **kwargs)
        let x_inp = x.index_select(&Tensor::new(0f32, &self.device)?, D::Minus1)?;
        let x_labels = x.index_select(&Tensor::new(1f32, &self.device)?, D::Minus1)?;
        let logits = self.net.forward(&x_inp)?;
        // rearrange logits "b c n -> b n c"
        let logits = logits.permute((0, 2, 1))?;
        // return F.cross_entropy(logits, x_labels)
        let x = cross_entropy(&logits, &x_labels)?;
        Ok(x)
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

        let net: BitTransformer = BitTransformer::load(128, 8, 256, 8, 4, &device.clone()).unwrap();
        let mut wrapper = AutoregressiveWrapper::new(0, net, device.clone());

        let start_tokens =
            Tensor::ones((1, 128), candle_core::DType::U32, &device.clone()).unwrap();
        let output = wrapper.generate(&start_tokens, 2, None, 1.0, 0.0).unwrap();

        assert_eq!(output.shape().dims(), &[1, 130]);
        Ok(())
    }
}
