use anyhow::Result;
use concrete::*;
use rayon::prelude::*;

use crate::new_encoder_bin;

pub fn apply_weights(input: &mut VectorLWE, weight: &Vec<bool>) -> Result<()> {
    assert!(input.nb_ciphertexts == weight.len()); // TODO: Check if we need to return an Err instead
    for i in 0..weight.len() {
        // only negate ciphertext bits with corresponding weight = false (or -1)
        if !weight[i] {
            input.opposite_nth_inplace(i)?;
        }
    }
    Ok(())
}

pub fn accumulate(input: &VectorLWE) -> Result<VectorLWE> {
    Ok(input.sum_with_padding()?)
}

pub fn linear(input: &VectorLWE, weight: &Vec<Vec<bool>>) -> Result<Vec<VectorLWE>> {
    let multiply_and_sum = |row: &Vec<bool>| -> Result<VectorLWE> {
        let mut input = input.clone();
        apply_weights(&mut input, row)?;
        Ok(accumulate(&input)?)
    };

    Ok(weight
        .into_par_iter()
        .map(multiply_and_sum)
        .collect::<Result<Vec<VectorLWE>>>()?)
}

pub fn rsign(ksk: &LWEKSK, bsk: &LWEBSK, input: &VectorLWE, threshold: f64) -> Result<VectorLWE> {
    Ok(input.keyswitch(ksk)?.bootstrap_nth_with_function(
        &bsk,
        |x| if x > threshold { 1.0 } else { -1.0 },
        &new_encoder_bin()?,
        0,
    )?)
}
