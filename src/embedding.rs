use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::{Init, VarBuilder};

#[derive(Clone, Debug)]
pub struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
    forward_dtype: DType,
}

impl Embedding {
    pub fn new(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let embeddings = vb
            .get_with_hints(
                (vocab_size, hidden_size),
                "embedding.weight",
                Init::Randn {
                    mean: 0.,
                    stdev: 1.,
                },
            )?
            .to_dtype(DType::F32)?;
        Ok(Self {
            embeddings,
            hidden_size,
            forward_dtype: vb.dtype(),
        })
    }

    pub fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = self.embeddings.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?.to_dtype(self.forward_dtype)?;
        Ok(values)
    }
}
