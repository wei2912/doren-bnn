use anyhow::Result;
use concrete::{prelude::*, set_server_key};

use doren_bnn_concrete::{
    convert_f64_to_bin_pm, decrypt_vec, get_uint4_config, load_keys, multiply_and_sum,
    relu_batchnorm_sign, try_encrypt_vec_bin_pm, BatchNormState, CastInto,
};

fn main() -> Result<()> {
    let messages = vec![1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];

    println!("Before encryption: {:?}", messages);

    let (config, uint4_enc) = get_uint4_config();
    let (client_key, server_key) = load_keys("keys_4/", config)?;
    set_server_key(server_key.clone());
    println!();

    let c1 = try_encrypt_vec_bin_pm(&client_key, &uint4_enc, &convert_f64_to_bin_pm(&messages))?;
    let o1 = decrypt_vec(&client_key, &c1);
    println!("{:?}", c1);
    println!("After encryption + decryption: {:?}", o1);
    println!();

    let ws = vec![1, -1, 1, -1, 0, -1, 1, 0, 0];
    println!("Weights: {:?}", ws);

    let c2 = multiply_and_sum(&c1, &ws);
    println!("{:?}", c2);
    let o2 = decrypt_vec(&client_key, &vec![c2.clone()]);
    println!("After multiply & sum: {:?}", o2);
    let o2_raw = c2.ct_opt.as_ref().unwrap().decrypt(&client_key);
    println!("(Raw CT) After multiply & sum: {:?}", o2_raw);
    println!();

    let state = BatchNormState {
        weight: vec![2.0],
        bias: vec![1.3],
        running_mean: vec![1.0],
        running_var: vec![1.0],
    };
    let c3 = relu_batchnorm_sign(&server_key, &vec![c2.clone()], &state);
    println!("{:?}", c3);
    let o3 = decrypt_vec(&client_key, &c3);
    println!("After ReLU + BatchNorm: {:?}", o3);
    println!(
        "(Raw CT) After ReLU + BatchNorm: {:?}",
        c3[0].ct_opt.as_ref().unwrap().decrypt(&client_key)
    );
    println!();

    let relu_batchnorm_sign_func = |x: f64| -> u64 {
        ((x - &state.running_mean[0]) / f64::sqrt(1e-5 + &state.running_var[0]) * &state.weight[0]
            + &state.bias[0]
            > 0.0) as u64
    };
    let transform_func = |x: u64| -> f64 { &c2.weight * (x as f64) + &c2.bias };
    let func = |x: u64| -> u64 { relu_batchnorm_sign_func(transform_func(x)) };
    println!(
        "(Alt) After ReLU + BatchNorm: {:?}",
        2 * c2.ct_opt.as_ref().unwrap().map(func).decrypt(&client_key) - 1
    );

    let c4 = uint4_enc.try_encrypt(o2_raw as u8, &client_key)?;
    println!(
        "(Alt 2) After encryption + decryption: {:?}",
        c4.decrypt(&client_key)
    );
    let c5 = c4.map(func);
    println!(
        "(Alt 2) After ReLU + BatchNorm: {:?}",
        2.0 * c5.decrypt(&client_key).cast_into() - 1.0
    );
    println!(
        "(Expected) After ReLU + BatchNorm: {:?}",
        2.0 * func(o2_raw.into()).cast_into() - 1.0
    );

    Ok(())
}
