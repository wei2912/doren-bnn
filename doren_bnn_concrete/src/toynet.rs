use anyhow::Result;
use concrete::*;

use crate::{linear, rsign};

pub struct ToyNetStateDict {
    pub fc_weight: Vec<Vec<bool>>,
}

pub fn toynet(
    ksk: &LWEKSK,
    bsk: &LWEBSK,
    state_dict: &ToyNetStateDict,
    input: &VectorLWE,
) -> Result<Vec<VectorLWE>> {
    let fc_weight = &state_dict.fc_weight;
    let output = linear(input, fc_weight)?;
    println!("{:?}", output[0].encoders[0]);
    let output2 = output
        .into_iter()
        .map(
            |x| rsign(ksk, bsk, &x, 2.0), // TODO - parameterise threshold
        )
        .collect::<Result<Vec<VectorLWE>, _>>()?;
    println!("{:?}", output2[0].encoders[0]);
    Ok(output2)
}
