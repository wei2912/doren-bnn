use bitvec::prelude::BitVec;
use concrete::*;

use std::error::Error;

use crate::new_encoder_bin;

pub fn apply_weights(input: &mut VectorLWE, weights: &BitVec) -> Result<(), Box<dyn Error>> {
    assert!(input.nb_ciphertexts == weights.len());
    for i in 0..weights.len() {
        // only negate ciphertext bits with corresponding weight = false (or -1)
        if !weights[i] {
            input.opposite_nth_inplace(i)?;
        }
    }
    Ok(())
}

pub fn accumulate_rsign(
    ksk: &LWEKSK,
    bsk: &LWEBSK,
    input: &VectorLWE,
    alpha: f64,
) -> Result<VectorLWE, Box<dyn Error>> {
    let input_sum = input.sum_with_padding()?;
    Ok(input_sum.keyswitch(ksk)?.bootstrap_nth_with_function(
        &bsk,
        |x| if x >= alpha { 1.0 } else { -1.0 },
        &new_encoder_bin()?,
        0,
    )?)
}
