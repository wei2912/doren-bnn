use concrete::*;

use std::error::Error;

use crate::new_encoder_bin;

pub fn apply_weights(input: &mut VectorLWE, weight: &Vec<bool>) -> Result<(), Box<dyn Error>> {
    assert!(input.nb_ciphertexts == weight.len());
    for i in 0..weight.len() {
        // only negate ciphertext bits with corresponding weight = false (or -1)
        if !weight[i] {
            input.opposite_nth_inplace(i)?;
        }
    }
    Ok(())
}

pub fn accumulate(input: &VectorLWE) -> Result<VectorLWE, Box<dyn Error>> {
    Ok(input.sum_with_padding()?)
}

pub fn rsign(
    ksk: &LWEKSK,
    bsk: &LWEBSK,
    input: &VectorLWE,
    threshold: f64,
) -> Result<VectorLWE, Box<dyn Error>> {
    Ok(input.keyswitch(ksk)?.bootstrap_nth_with_function(
        &bsk,
        |x| if x > threshold { 1.0 } else { -1.0 },
        &new_encoder_bin()?,
        0,
    )?)
}

pub fn linear(
    input: &VectorLWE,
    weight: &Vec<Vec<bool>>,
) -> Result<Vec<VectorLWE>, Box<dyn Error>> {
    let multiply_and_sum = |row: &Vec<bool>| -> Result<VectorLWE, Box<dyn Error>> {
        let mut input = input.clone();
        apply_weights(&mut input, row)?;
        Ok(accumulate(&input)?)
    };

    Ok(weight
        .into_iter()
        .map(multiply_and_sum)
        .collect::<Result<Vec<VectorLWE>, _>>()?)
}
