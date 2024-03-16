#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Config {
    pub(crate) dim: usize,
    pub(crate) depth: usize,
    pub(crate) vocab_size: usize,
    pub(crate) heads: usize,
    pub(crate) ff_mult: usize,
    pub(crate) eps: f32,
    pub(crate) ff_dropout: f32,
    pub(crate) seq_len: usize,
}

impl Config {
    // Default configuration for initial evaluation, will add larger configs later after confirming valid output
    pub fn default() -> Self {
        Self {
            dim: 256,
            depth: 12,
            vocab_size: 32000,
            heads: 8,
            ff_mult: 10,
            eps: 1e-6,
            ff_dropout: 0.1,
            seq_len: 100,
        }
    }
}
