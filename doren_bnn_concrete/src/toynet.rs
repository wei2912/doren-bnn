use anyhow::Result;
use concrete::*;

use crate::linear;

pub struct ToyNetStateDict {
    pub fc_weight: Vec<Vec<bool>>,
}

pub fn toynet(
    _ksk: &LWEKSK,
    _bsk: &LWEBSK,
    state_dict: &ToyNetStateDict,
    input: &VectorLWE,
) -> Result<Vec<VectorLWE>> {
    let fc_weight = &state_dict.fc_weight;
    let output = linear(input, fc_weight)?;
    println!("{:?}", output[0].encoders[0]);
    Ok(output)
}
