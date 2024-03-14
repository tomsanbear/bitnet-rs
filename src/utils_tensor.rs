use anyhow::{anyhow, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{DType, Device, Shape, Tensor, WithDType, D};
use candle_nn::ops::{self};
use tracing::{event, span, Level};

// Transform the input values of the tensor to it's signs, -1, 0 or 1
pub fn sign(x: &Tensor) -> candle_core::Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "sign");
    let _enter = span.enter();

    // Implemented as by dividing by the absolute value to get the sign, we add a 1 to the numerator where x = 0 such that we don't divide by zero
    let zeros = x.broadcast_eq(&Tensor::zeros(x.shape(), x.dtype(), x.device())?)?;
    event!(Level::TRACE, "zeros: {:?}", zeros);
    let abs_x = x.abs()?.add(&zeros.to_dtype(x.dtype())?)?;
    event!(Level::TRACE, "abs_x: {:?}", abs_x);
    let sign_x = (x / abs_x)?;
    event!(Level::TRACE, "sign_x: {:?}", sign_x);
    Ok(sign_x)
}

#[cfg(test)]
mod sign_tests {
    use crate::utils_tensor::sign;
    use candle_core::{Device, Result, Tensor};

    #[test]
    fn it_works() -> Result<()> {
        let input = vec![-3f32, -2f32, -1f32, 0f32, 1f32, 2f32, 3f32];
        let input_size = input.len();
        let tensor = Tensor::from_vec(input, (input_size,), &Device::Cpu)?;
        let output = sign(&tensor).unwrap();

        let expected_shape = [input_size];
        assert_eq!(output.shape().dims(), &expected_shape);

        let expected_output = [-1f32, -1f32, -1f32, 0f32, 1f32, 1f32, 1f32];
        let output = output.squeeze(0)?;
        let output = output.to_vec1::<f32>()?;

        assert_eq!(output, expected_output);

        Ok(())
    }
}

// Get the device to use for the tensor operations, only really used for tests
// Originally from: https://github.com/huggingface/candle/blob/314630638d8f6886c07d73211d6c35f8cf05d56a/candle-examples/src/lib.rs#L9
#[allow(dead_code)]
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

// For a given device, return the dtype for the requested dtype
pub fn dtype(device: &Device) -> Result<candle_core::DType> {
    if device.is_cpu() {
        // We use f32 for cpu since f16 is not supported for many required operations
        Ok(candle_core::DType::F32)
    } else if device.is_metal() {
        // We use f32 for metal since f16 is not supported for many required operations
        Ok(candle_core::DType::F32)
    } else if device.is_cuda() {
        // We use f16 for cuda since we don't actually need anything more than that for this model
        Ok(candle_core::DType::F32)
    } else {
        return Err(anyhow!("Unsupported device"));
    }
}

pub fn full<S: Into<Shape>, D: WithDType>(
    shape: S,
    fill_value: D,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "full");
    let _enter = span.enter();

    Tensor::new(&[fill_value], device)?
        .to_dtype(dtype)?
        .broadcast_as(shape)
}

pub fn full_like<D: WithDType>(input: &Tensor, fill_value: D) -> candle_core::Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "full-like");
    let _enter = span.enter();

    full(input.shape(), fill_value, input.dtype(), input.device())
}

pub fn masked_fill<D: WithDType>(
    xs: &Tensor,
    mask: &Tensor,
    value: D,
) -> candle_core::Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "masked-fill");
    let _enter = span.enter();

    let on_true = full_like(xs, value)?;
    let on_false = xs;
    mask.broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)
}

fn apply_triangular(xs: &Tensor, diagonal: isize, upper: bool) -> candle_core::Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "apply-triangular");
    let _enter = span.enter();

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

pub fn logical_not(xs: &Tensor) -> Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "logical-not");
    let _enter = span.enter();

    let out = xs.where_cond(&xs.zeros_like()?, &xs.ones_like()?)?;
    Ok(out)
}

// Modified to force the dropout datatype to be something supported on metal
pub fn dropout(xs: &Tensor, drop_p: f32) -> candle_core::Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "dropout");
    let _enter = span.enter();

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

pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_p: Option<f32>,
    is_causal: Option<bool>,
    scale: Option<f64>,
) -> Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "scaled-dot-product-attention");
    let _enter = span.enter();

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

// Wrapper on cross entropy to add tracing
pub fn cross_entropy(inp: &Tensor, target: &Tensor) -> candle_core::Result<Tensor> {
    let span = span!(tracing::Level::TRACE, "cross-entropy");
    let _enter = span.enter();
    candle_nn::loss::cross_entropy(inp, target)
}
