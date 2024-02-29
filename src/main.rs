use candle_core::{Device, Result};

mod bitffn;
mod bitlinear;
mod utils;

use crate::bitlinear::Bitlinear;

fn main() -> Result<()> {
    let device = &Device::Cpu;
    Bitlinear::load(3, 3, device)?;
    Ok(())
}
