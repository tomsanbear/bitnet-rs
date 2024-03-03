#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Config {
    pub(crate) dim: usize,
    pub(crate) depth: usize,
    pub(crate) num_tokens: usize,
    pub(crate) heads: usize,
    pub(crate) ff_mult: usize,
    pub(crate) layer_norm_eps: f64,
}
