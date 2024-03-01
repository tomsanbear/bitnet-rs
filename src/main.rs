use crate::bitlinear::Bitlinear;
use candle_core::{Device, Result};

mod attention;
mod bitffn;
mod bitlinear;
mod bitnet_transformer;
mod config;
mod rms_norm;
mod rotary_embedding;
mod transformer;
mod utils;

fn main() -> Result<()> {
    let device = &Device::Cpu;
    Bitlinear::load(3, 3, device)?;
    Ok(())
}
