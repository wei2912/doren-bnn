use anyhow::Result;
use concrete::*;

use crate::{decrypt, linear, sign};

pub struct ToyNetStateDict {
    pub fc_weight: Vec<Vec<bool>>,
}

pub fn toynet(
    sk_lwe: &LWESecretKey,
    ksk: &LWEKSK,
    bsk: &LWEBSK,
    state_dict: &ToyNetStateDict,
    input: &VectorLWE,
) -> Result<Vec<VectorLWE>> {
    let fc_weight = &state_dict.fc_weight;
    let output = linear(input, fc_weight)?;

    let output_dec = output
        .iter()
        .map(|ct| {
            decrypt(sk_lwe, ct).map(|vec| {
                assert!(vec.len() == 1);
                vec[0]
            })
        })
        .collect::<Result<Vec<f64>>>()?;
    println!("{:?}", output_dec);
    println!("{:?}", output[0].encoders[0]);

    let output2 = output
        .iter()
        .map(|x| sign(ksk, bsk, x))
        .collect::<Result<Vec<VectorLWE>, _>>()?;
    println!("{:?}", output2[0].encoders[0]);
    Ok(output2)
}
