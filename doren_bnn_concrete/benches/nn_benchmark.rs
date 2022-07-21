use std::iter;

use concrete::{set_server_key, DynInteger};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use doren_bnn_concrete::{
    convert_f64_to_bin, encrypt_vec, get_uint12_params, load_keys, multiply_and_sum,
};
use rand::seq::SliceRandom;

const INPUT_SIZE_DW: usize = 9;
// const INPUT_SIZE_PW: usize = 256;

fn gen_inputs<T: Copy>(vals: &Vec<T>, size: usize) -> Vec<T> {
    let mut rng = rand::thread_rng();
    let choose_val = || vals.choose(&mut rng).unwrap();
    iter::repeat_with(choose_val)
        .take(size)
        .cloned()
        .collect::<Vec<_>>()
}

fn multiply_and_sum_bm(c: &mut Criterion) {
    let (client_key, server_key, uint12_enc) = load_keys("keys_3x4/", get_uint12_params()).unwrap();
    set_server_key(server_key);

    let encrypt = |ps: Vec<f64>| -> Vec<DynInteger> {
        encrypt_vec(&client_key, &uint12_enc, &convert_f64_to_bin(&ps))
    };

    let act_vals: Vec<f64> = vec![-1.0, 1.0];
    let weight_vals: Vec<i8> = vec![-1, 1]; // assume zero sparsity

    c.sample_size(500);

    c.bench_function("multiply_and_sum dw", |b| {
        b.iter_batched(
            || {
                (
                    encrypt(gen_inputs(&act_vals, INPUT_SIZE_DW)),
                    gen_inputs(&weight_vals, INPUT_SIZE_DW),
                )
            },
            |(cs, ws)| multiply_and_sum(cs, &ws),
            BatchSize::PerIteration,
        )
    });

    /* Takes ~7.5 mins/it.
     * c.bench_function("multiply_and_sum pw", |b| {
     *     b.iter_batched(
     *         || {
     *             (
     *                 encrypt(gen_inputs(&act_vals, INPUT_SIZE_PW)),
     *                 gen_inputs(&weight_vals, INPUT_SIZE_PW),
     *             )
     *         },
     *         |(cs, ws)| multiply_and_sum(cs, &ws),
     *         BatchSize::PerIteration,
     *     )
     * });
     */
}

criterion_group!(benches, multiply_and_sum_bm);
criterion_main!(benches);
