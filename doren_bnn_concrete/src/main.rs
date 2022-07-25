use anyhow::Result;
use concrete::set_server_key;

use doren_bnn_concrete::{
    convert_f64_to_bin_pm, decrypt_vec, get_uint4_config, load_keys, multiply_and_sum,
    try_encrypt_vec_bin_pm,
};

fn main() -> Result<()> {
    let messages = vec![1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];

    println!("Before encryption: {:?}", messages);

    let (config, uint4_enc) = get_uint4_config();
    let (client_key, server_key) = load_keys("keys_4/", config)?;
    set_server_key(server_key);
    println!();

    let c1 = try_encrypt_vec_bin_pm(&client_key, &uint4_enc, &convert_f64_to_bin_pm(&messages))?;
    let o1 = decrypt_vec(&client_key, &c1);
    println!("{:?}", c1);
    println!("After encryption + decryption: {:?}", o1);

    let ws = vec![1, 0, -1, 1, 0, 0, 1, 0, -1];
    println!("Weights: {:?}", ws);

    let c2 = multiply_and_sum(c1, &ws);
    println!("{:?}", c2);
    let o2 = decrypt_vec(&client_key, &vec![c2]);
    println!("After multiply & sum: {:?}", o2);

    /*
    let mut c2 = c1.clone();
    let weights = vec![1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
    apply_weights(&mut c2, &convert_f64_to_bin(&weights))?;
    let o2 = decrypt(&sk_lwe, &c2)?;
    println!("Weights: {:?}", weights);
    println!("After weighting: {:?}", o2);

    let c3 = accumulate(&c2)?;
    let o3 = decrypt(&sk_lwe, &c3)?;
    println!("Accumulate: {:?}", o3);
    print_encoder(&c3.encoders[0]);
    println!();

    let c4 = linear(&c1, &[convert_f64_to_bin(&weights)])?;
    let o4 = decrypt(&sk_lwe, &c4[0])?;
    println!("Linear layer with same weights: {:?}", o4);
    print_encoder(&c4[0].encoders[0]);
    println!();

    let c5 = sign(&ksk, &bsk, &c4[0])?;
    let o5 = decrypt(&sk_lwe, &c5)?;
    println!("Sign: {:?}", o5);
    print_encoder(&c5.encoders[0]);
    println!();
    */

    Ok(())
}
