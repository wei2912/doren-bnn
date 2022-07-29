use anyhow::Result;
use concrete::{prelude::*, set_server_key};

use doren_bnn_concrete::{
    convert_f64_to_bin_pm, decrypt_vec, get_uint4_config, load_keys, multiply_and_sum,
    relu_batchnorm_sign, try_encrypt_vec_bin_pm, BatchNormState,
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

    let ws = vec![1, 0, -1, 1, 0, 0, 1, 0, -1];
    println!("Weights: {:?}", ws);

    let c2 = multiply_and_sum(&c1, &ws);
    println!("{:?}", c2);
    let o2 = decrypt_vec(&client_key, &vec![c2.clone()]);
    println!("After multiply & sum: {:?}", o2);
    println!(
        "(Raw CT) After multiply & sum: {:?}",
        c2.ct_opt.as_ref().unwrap().decrypt(&client_key)
    );

    let weight = vec![2.0];
    let bias = vec![-4.3];
    let running_mean = vec![1.0];
    let running_var = vec![1.0];

    let state = BatchNormState {
        weight,
        bias,
        running_mean,
        running_var,
    };
    let c3 = relu_batchnorm_sign(&server_key, &vec![c2.clone()], state);
    println!("{:?}", c3);
    let o3 = decrypt_vec(&client_key, &c3);
    println!("After ReLU + BatchNorm: {:?}", o3);
    println!(
        "(Raw CT) After ReLU + BatchNorm: {:?}",
        c3[0].ct_opt.as_ref().unwrap().decrypt(&client_key)
    );

    let func = |x| ((x as f64 - 1.0) / f64::sqrt(1e-5 + 1.0) * 2.0 - 4.3 > 0.0) as u64;
    println!(
        "(Alt Raw CT) After ReLU + BatchNorm: {:?}",
        c2.ct_opt
            .as_ref()
            .unwrap()
            .map(|x| func(2 * x - 5))
            .decrypt(&client_key)
    );

    let c4 = uint4_enc.try_encrypt(3, &client_key)?;
    println!(
        "(Alt 2 Raw CT) After encryption + decryption: {:?}",
        c4.decrypt(&client_key)
    );
    let c5 = c4.map(|x| func(2 * x - 5));
    let o5 = c5.decrypt(&client_key);
    println!("(Alt 2 Raw CT) After ReLU + BatchNorm: {:?}", o5);
    println!("(Expected Raw CT) After ReLU + BatchNorm: {:?}", func(1));

    Ok(())
}
