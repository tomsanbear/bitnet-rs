#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) sliding_window: usize,
    pub(crate) use_flash_attn: bool,
}
