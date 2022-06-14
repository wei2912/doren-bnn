use concrete::*;

use std::error::Error;

use crate::linear;

pub struct ToyNetStateDict {
    pub fc_weight: Vec<Vec<bool>>,
}

pub fn toynet(
    _ksk: &LWEKSK,
    _bsk: &LWEBSK,
    state_dict: &ToyNetStateDict,
    input: &VectorLWE,
) -> Result<Vec<VectorLWE>, Box<dyn Error>> {
    let fc_weight = &state_dict.fc_weight;
    let output = linear(input, fc_weight)?;
    Ok(output)
}
