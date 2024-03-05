use anyhow::{anyhow, Result};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::Dropout;

// Convert an input tensor into a tensor of the same shape but with all elements set to it's sign, one of -1, 0 or 1.
pub fn sign(x: &Tensor) -> candle_core::Result<Tensor> {
    // The zeros are converted to ones here to enable us to avoid dividing by zero, not sure if there is a cleaner way to avoid the extra ops
    let zeros = x.eq(0f32)?.to_dtype(x.dtype())?;
    let abs_x = x.abs()?.to_dtype(x.dtype())?.add(&zeros)?;
    // need to handle dividing by zero
    let sign_x = (x / abs_x)?;
    Ok(sign_x)
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

// Mask the elements of a tensor based on a mask tensor
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let _on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = on_false.broadcast_mul(&mask.to_dtype(on_false.dtype())?)?;
    Ok(m)
}

pub struct ScaledDotProductCfg {
    pub is_causal: bool,
    pub need_weights: bool,
    pub average_attn_weights: bool,
    pub force_grouped: bool,
    pub dropout: f32,
}

// Scaled dot-product attention with grouped query and key heads.
// Original implementation: https://github.com/kyegomez/BitNet/blob/2fe67c30f2b47fb510b39f7a4cffb79fad354838/bitnet/bit_attention.py#L10
pub fn scaled_dot_product_gqa(
    query: Tensor, // (b, n, h, d)
    key: Tensor,   // (b, s, h, d)
    value: Tensor, // (b, s, h, d)
    cfg: ScaledDotProductCfg,
) -> Result<(Tensor, Option<Tensor>), anyhow::Error> {
    if query.dims().len() != 4 || key.dims().len() != 4 || value.dims().len() != 4 {
        return Err(anyhow!("Input tensors must have 4 dimensions"));
    };

    // Move sequence length dimension to axis 2, this makes it faster in torch
    // "b n h d -> b h n d"
    let query = query.permute([0, 2, 1, 3])?;

    // "b s h d -> b h s d"
    let key = key.permute([0, 2, 1, 3])?;

    // "b s h d -> b h s d"
    let value = value.permute([0, 2, 1, 3])?;

    // Extract the dimensions
    let (bq, hq, nq, dq) = query.dims4()?;
    let (bk, hk, nk, dk) = key.dims4()?;
    let (bv, hv, nv, dv) = value.dims4()?;

    // All batch sizes must be equal
    if !(bq == bk && bq == bv) {
        return Err(anyhow!("Batch sizes must be equal"));
    };

    // All dimension sizes must be equal
    if !(dq == dk && dq == dv) {
        return Err(anyhow!("Dimension sizes must be equal"));
    };

    // key and value should have same size in dim 1 and 2
    if nk != nv || hk != hv {
        return Err(anyhow!(
            "Key and value should have same size in dim 1 and 2"
        ));
    };

    // Query heads must be a multiple of kv heads
    if hq % hk != 0 {
        return Err(anyhow!("Query heads must be a multiple of key/value heads"));
    };

    let scale = (*query.dims().last().unwrap() as f64).sqrt();
    let query = (query / scale)?;
    let num_head_groups = hq / hk;

    let similarity = match num_head_groups > 1 || cfg.force_grouped {
        true => {
            // query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
            // similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
            let (batch_size, heads, seq_len, depth) = query.dims4()?;
            let heads = heads / num_head_groups; // Calculate the number of heads per group.

            // Reshape query to [batch, num_head_groups, heads, seq_len, depth]
            let query_reshaped =
                query.reshape((batch_size, num_head_groups, heads, seq_len, depth))?;

            let query_for_matmul = query_reshaped.sum(1)?;

            // Transpose the last two dimensions of key to align them for matmul.
            let key_transposed = key.transpose(D::Minus2, D::Minus1)?; // [batch, heads, depth, seq_len]

            // Perform batched matrix multiplication.
            query_for_matmul.matmul(&key_transposed.contiguous()?)
        }
        false => {
            // If the number of query/key heads is equal, we can skip grouping the queries,
            // and just use the standard sdot product attention.
            // einsum(query, key, "b h n d, b h s d -> b h n s")
            let query = query.unsqueeze(3)?;
            let key_t = key.transpose(D::Minus2, D::Minus1)?;
            query.matmul(&key_t)
        }
    }?;

    // Apply mask if causal attention is required
    let mask = match cfg.is_causal {
        true => {
            // Mask out the upper triangular portion of the attention matrix. This prevents
            // the model from attending to tokens in the future
            let mask = Tensor::ones((bq, nq, nk), query.dtype(), query.device())?;
            Some(mask)
        }
        false => None,
    };

    // Expand mask to match the shape of the attn matrix
    let mask = match mask {
        Some(mask) => Some({
            if mask.shape().dims().len() == 2 {
                mask.unsqueeze(1)?.unsqueeze(2)?
            } else if mask.dims().len() == 3 {
                mask.unsqueeze(1)?
            } else {
                mask
            }
        }),
        None => None,
    };

    let similarity = match mask {
        Some(mask) => masked_fill(&similarity, &mask, f32::NEG_INFINITY)?,
        None => similarity,
    };

    // attention = F.softmax(similarity / scale, dim=-1)
    let attention = softmax(&(similarity / scale)?, D::Minus1)?;

    // apply dropout
    let attention = match cfg.dropout > 0.0 {
        true => {
            // Original python code:
            // attention = F.dropout(attention, p=dropout, training=self.training)
            Dropout::new(cfg.dropout).forward(&attention, false)?
        }
        false => attention,
    };

    // Apply attention matrix to the value Tensor.
    // out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    let out = attention.matmul(&value.contiguous()?)?;

    // Move head dimension back to axis 2
    // out = rearrange(out, "b h n d -> b n h d")
    let out = out.permute([0, 2, 1, 3])?;

    let attn_weights = match cfg.need_weights {
        false => None,
        true => {
            // Move the sequence dimensions back to positions 1, 2.  Move the head dimension
            // to position 3.  This more closely matches the return shape of the attention
            // output: (b, n, h, d).
            // python code:
            // attn_weights = rearrange(attention, "b h n s -> b n s h")
            let attn_weights = attention.permute([0, 2, 3, 1])?;
            // if average_attn_weights:
            //   attn_weights = attn_weights.mean(dim=1)
            if cfg.average_attn_weights {
                let attn_weights = attn_weights.mean_keepdim(1)?;
                Some(attn_weights)
            } else {
                Some(attn_weights)
            }
        }
    };

    Ok((out, attn_weights))
}

#[cfg(test)]
mod scaled_dot_product_gqa_tests {
    use crate::utils_tensor::{device, scaled_dot_product_gqa, ScaledDotProductCfg};
    use anyhow::Result;
    use candle_core::safetensors;
    use test::Bencher;

    macro_rules! python_snapshot_tests {
        ($($name:ident: $value:expr,)*) => {
            $(
                #[test]
                fn $name() {
                    let input = $value;

                    let device = device(true).unwrap();
                    let safetensor =
                        safetensors::load("src/test_data/scaled_dot_product_gqa.safetensors", &device).unwrap();

                    let input_tensor_name = format!("{}_input", input);
                    let input_tensor = match safetensor.get(input_tensor_name.as_str()) {
                        Some(tensor) => tensor,
                        None => panic!("Input tensor not found"),
                    };

                    let output_tensor_name = format!("{}_output", input);
                    let expected_output = match safetensor.get(output_tensor_name.as_str()) {
                        Some(tensor) => tensor,
                        None => panic!("Output tensor not found"),
                    };

                    let attn_weights_tensor_name = format!("{}_attn_weights", input);
                    let expected_attn_weights = match safetensor.get(attn_weights_tensor_name.as_str()) {
                        Some(tensor) => tensor,
                        None => panic!("Output tensor not found"),
                    };

                    let (out, attn_weights) = scaled_dot_product_gqa(input_tensor.clone(), input_tensor.clone(), input_tensor.clone(), ScaledDotProductCfg {
                        is_causal: true,
                        need_weights: true,
                        average_attn_weights: true,
                        force_grouped: true,
                        dropout: 0.0,
                    }).unwrap();
                    let attn_weights = attn_weights.unwrap();

                    assert_eq!(
                        out.shape().dims(),
                        expected_output.shape().dims(),
                        "output shape mismatch"
                    );
                    assert_eq!(
                        out.squeeze(0).unwrap().to_vec3::<f32>().unwrap(),
                        expected_output.squeeze(0).unwrap().to_vec3::<f32>().unwrap(),
                        "output value mismatch"
                    );
                    assert_eq!(
                        attn_weights.shape().dims(),
                        expected_attn_weights.shape().dims(),
                        "attn_weights shape mismatch"
                    );
                    assert_eq!(
                        attn_weights.squeeze(0).unwrap().to_vec3::<f32>().unwrap(),
                        expected_attn_weights.squeeze(0).unwrap().to_vec3::<f32>().unwrap(),
                        "attn_weights value mismatch"
                    );
                }
            )*
        }
    }

    python_snapshot_tests! {
        it_matches_snapshot_small: "small",
        it_matches_snapshot_large: "large",
    }

    #[bench]
    fn bench_scaled_dot_product_gqa(b: &mut Bencher) -> Result<()> {
        let device = device(true).unwrap();
        let safetensor =
            safetensors::load("src/test_data/scaled_dot_product_gqa.safetensors", &device).unwrap();

        let input_tensor = match safetensor.get("small_input") {
            Some(tensor) => tensor,
            None => panic!("Input tensor not found"),
        };

        b.iter(|| {
            for _ in 1..100 {
                scaled_dot_product_gqa(
                    input_tensor.clone(),
                    input_tensor.clone(),
                    input_tensor.clone(),
                    ScaledDotProductCfg {
                        is_causal: true,
                        need_weights: true,
                        average_attn_weights: true,
                        force_grouped: true,
                        dropout: 0.0,
                    },
                )
                .unwrap();
            }
        });

        Ok(())
    }
}
