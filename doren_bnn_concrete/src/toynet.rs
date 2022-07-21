use concrete::*;

use crate::linear;

pub struct ToyNetStateDict {
    pub fc_weight: Vec<Vec<i8>>,
}

pub fn toynet(
    state_dict: ToyNetStateDict,
    input: Vec<DynInteger>,
) -> Vec<(Option<DynInteger>, i64)> {
    let fc_weight = state_dict.fc_weight;
    linear(input, fc_weight)

    /*
    let output2 = output
        .iter()
        .map(|x| sign(ksk, bsk, x))
        .collect::<Result<Vec<VectorLWE>, _>>()?;
    println!("{:?}", output2[0].encoders[0]);
    */
}
