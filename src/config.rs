#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Config {
    pub(crate) dim: usize,
    pub(crate) depth: usize,
    pub(crate) vocab_size: usize,
    pub(crate) heads: usize,
    pub(crate) ff_mult: usize,
    pub(crate) layer_norm_eps: f64,
    pub(crate) bit_attention_eps: f64,
    pub(crate) ff_dropout: f32,
}

impl Config {
    pub fn default() -> Self {
        Self {
            dim: 64,
            depth: 8,
            vocab_size: 32000,
            heads: 8,
            ff_mult: 12,
            layer_norm_eps: 1e-6,
            bit_attention_eps: 1e-6,
            ff_dropout: 0.0,
        }
    }
}
