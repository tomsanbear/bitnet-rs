use candle_core::{Result, Tensor};

// Convert an input tensor into a tensor of the same shape but with all elements set to it's sign, one of -1, 0 or 1.
pub fn sign(x: &Tensor) -> Result<Tensor> {
    // The zeros are converted to ones here to enable us to avoid dividing by zero, not sure if there is a cleaner way to avoid the extra ops
    let zeros = x.eq(0f32)?.to_dtype(x.dtype())?;
    let abs_x = x.abs()?.to_dtype(x.dtype())?.add(&zeros)?;
    // need to handle dividing by zero
    let sign_x = (x / abs_x)?;
    Ok(sign_x)
}

#[cfg(test)]
mod sign_tests {
    use crate::utils::sign;
    use candle_core::{Device, Result, Tensor};

    #[test]
    fn it_works_for_positives() -> Result<()> {
        let input = vec![-3f32, -2f32, -1f32, 0f32, 1f32, 2f32, 3f32];
        let input_size = input.len();
        let tensor = Tensor::from_vec(input, (input_size,), &Device::Cpu)?;
        let output = sign(&tensor)?;

        let expected_shape = [input_size];
        assert_eq!(output.shape().dims(), &expected_shape);

        let expected_output = [-1f32, -1f32, -1f32, 0f32, 1f32, 1f32, 1f32];
        let output = output.squeeze(0)?;
        let output = output.to_vec1::<f32>()?;
        assert_eq!(output, expected_output);

        Ok(())
    }
}
