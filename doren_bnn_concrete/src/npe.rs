use anyhow::{Context, Result};

use concrete::*;
use concrete_commons::dispersion::{LogStandardDev, Variance};
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{
    DecompositionBaseLog, DecompositionLevelCount, GlweDimension, LweDimension, PolynomialSize,
};
use concrete_npe::{
    estimate_addition_noise, estimate_number_of_noise_bits, estimate_pbs_noise,
    estimate_several_additions_noise,
};

use std::cmp;
use std::iter;

pub fn calc_sum_noise(lwe_params: &LWEParams, n: usize) -> Variance {
    let var = LogStandardDev(lwe_params.log2_std_dev.into());
    let vars = iter::repeat(var).take(n).collect::<Vec<LogStandardDev>>();
    estimate_several_additions_noise::<Torus, _>(&vars)
}

pub fn calc_pbs_noise(
    lwe_params: &LWEParams,
    rlwe_params: &RLWEParams,
    bsk_base_log: usize,
    bsk_level: usize,
) -> Variance {
    estimate_pbs_noise::<Torus, _, BinaryKeyKind>(
        LweDimension(lwe_params.dimension),
        PolynomialSize(rlwe_params.polynomial_size),
        GlweDimension(rlwe_params.dimension),
        DecompositionBaseLog(bsk_base_log),
        DecompositionLevelCount(bsk_level),
        LogStandardDev(rlwe_params.log2_std_dev.into()),
    )
}

pub fn calc_bin_sum_width(
    lwe_params: &LWEParams,
    rlwe_params: &RLWEParams,
    bsk_base_log: usize,
    bsk_level: usize,
) -> Result<usize> {
    let start_pow2 = 32_usize;
    let max_pow2 = (2 as usize).pow(16);

    let pow2s = iter::successors(Some(start_pow2), |n| {
        if n * 2 <= max_pow2 {
            Some(n * 2)
        } else {
            None
        }
    });

    let test_usize = |usize: usize| {
        let sum_noise = calc_sum_noise(lwe_params, usize);
        let pbs_noise = calc_pbs_noise(lwe_params, rlwe_params, bsk_base_log, bsk_level);

        let num_noise_bits =
            estimate_number_of_noise_bits::<Torus, _>(estimate_addition_noise::<Torus, _, _>(
                sum_noise, pbs_noise,
            ));
        (usize as f64).log2().ceil() as usize + 2 + num_noise_bits <= Torus::BITS as usize
    };

    let mut lb_usize = pow2s
        .take_while(|x| test_usize(*x))
        .last()
        .context("starting size is too big to have zero overlap")?;
    let mut ub_usize = cmp::min(lb_usize * 2, max_pow2);

    while lb_usize < ub_usize {
        let mid_usize = (lb_usize + ub_usize) / 2;
        if test_usize(mid_usize) {
            lb_usize = mid_usize + 1;
        } else {
            ub_usize = mid_usize - 1;
        }
    }
    Ok(ub_usize)
}
