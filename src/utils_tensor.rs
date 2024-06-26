use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Shape, Tensor, WithDType, D};
use candle_nn::ops::{self};
use tracing::instrument;

// Get the device to use for the tensor operations, only really used for tests
// Originally from: https://github.com/huggingface/candle/blob/314630638d8f6886c07d73211d6c35f8cf05d56a/candle-examples/src/lib.rs#L9
pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

#[instrument]
pub fn full<S: Into<Shape> + std::fmt::Debug, D: WithDType + std::fmt::Debug>(
    shape: S,
    fill_value: D,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    Tensor::new(&[fill_value], device)?
        .to_dtype(dtype)?
        .broadcast_as(shape)
}

#[instrument]
pub fn full_like<D: WithDType + std::fmt::Debug>(
    input: &Tensor,
    fill_value: D,
) -> candle_core::Result<Tensor> {
    full(input.shape(), fill_value, input.dtype(), input.device())
}

#[instrument]
pub fn masked_fill<D: WithDType + std::fmt::Debug>(
    xs: &Tensor,
    mask: &Tensor,
    value: D,
) -> candle_core::Result<Tensor> {
    let on_true = full_like(xs, value)?;
    let on_false = xs;
    mask.broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)
}

#[instrument]
fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> candle_core::Result<Tensor> {
    let device = xs.device();
    let (l, s) = xs.dims2()?;
    let mut xs_tri = vec![];
    for i in 0..l as isize {
        for j in 0..s as isize {
            let cond = if upper {
                i + diagonal > j
            } else {
                i + diagonal < j
            };
            xs_tri.push(if cond { 0u8 } else { 1u8 });
        }
    }
    xs * Tensor::from_vec(xs_tri, (l, s), device)?.to_dtype(xs.dtype())?
}

#[instrument]
pub fn logical_not(xs: &Tensor) -> Result<Tensor> {
    let out = xs.where_cond(&xs.zeros_like()?, &xs.ones_like()?)?;
    Ok(out)
}

#[instrument]
pub fn dropout(xs: &Tensor, drop_p: f32) -> candle_core::Result<Tensor> {
    // This implementation is inefficient as it stores the full mask for the backward pass.
    // Instead we could just store the seed and have a specialized kernel that would both
    // generate the random mask and apply it.
    // Another easier optimization would be to be able to generate boolean mask using just a bit of
    // entropy per element rather than generating a full float per element.
    if !(0. ..1.).contains(&drop_p) {
        candle_core::bail!("dropout probability has to be in [0, 1), got {drop_p}")
    }
    let rand = Tensor::rand(0f32, 1f32, xs.shape(), xs.device())?;
    let scale = 1.0 / (1.0 - drop_p as f64);
    let drop_p = Tensor::new(drop_p, xs.device())?.broadcast_as(xs.shape())?;
    let mask = (rand.ge(&drop_p)?.to_dtype(DType::F32)? * scale)?.to_dtype(xs.dtype())?;
    xs * mask
}

#[instrument]
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_p: Option<f32>,
    is_causal: Option<bool>,
    scale: Option<f64>,
) -> Result<Tensor> {
    let device = query.device();
    let l = query.dim(D::Minus2)?;
    let s = key.dim(D::Minus2)?;
    let dim = query.dim(D::Minus1)?;

    let scale_factor = if let Some(scale) = scale {
        scale
    } else {
        1.0 / (dim as f64).sqrt()
    };

    let mut attn_bias = Tensor::zeros((l, s), query.dtype(), device)?;

    if matches!(is_causal, Some(true)) {
        assert!(attn_mask.is_none(), "scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
        let mask = apply_triangular(&Tensor::ones((l, s), DType::U8, device)?, 0, false)?;
        attn_bias = masked_fill(&attn_bias, &logical_not(&mask)?, f32::NEG_INFINITY)?;
    }

    if let Some(attn_mask) = attn_mask {
        if attn_mask.rank() > attn_bias.rank() {
            attn_bias = attn_bias.broadcast_as(attn_mask.shape())?;
        }
        if attn_mask.dtype() == DType::U8 {
            // bool
            attn_bias = masked_fill(&attn_bias, &logical_not(attn_mask)?, f32::NEG_INFINITY)?;
        } else {
            attn_bias = (&attn_bias
                + attn_mask
                    .to_dtype(attn_bias.dtype())?
                    .broadcast_as(attn_bias.shape())?)?;
        }
    }

    let mut attn_weights =
        (query.matmul(&key.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale_factor)?;

    attn_weights = (&attn_weights + attn_bias.broadcast_as(attn_weights.shape())?)?;
    attn_weights = ops::softmax_last_dim(&attn_weights)?;
    if let Some(drop_p) = dropout_p {
        attn_weights = if attn_weights.device().is_metal() {
            dropout(&attn_weights, drop_p)
        } else {
            ops::dropout(&attn_weights, drop_p)
        }?;
    }
    let out = attn_weights.matmul(value)?;
    Ok(out)
}

#[instrument]
pub fn cross_entropy(inp: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    candle_nn::loss::cross_entropy(inp, target)
}
