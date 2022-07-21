use anyhow::Result;
use concrete::{prelude::*, set_server_key};
use doren_bnn_concrete::{
    convert_f64_to_bin, decrypt_vec, encrypt_vec, get_uint12_params, load_keys, multiply_and_sum,
};

fn main() -> Result<()> {
    let messages = vec![1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];

    println!("Before encryption: {:?}", messages);

    let (client_key, server_key, uint10_enc) = load_keys("keys_3x4/", get_uint12_params())?;
    set_server_key(server_key);
    println!();

    let c1 = encrypt_vec(&client_key, &uint10_enc, &convert_f64_to_bin(&messages));
    let o1 = decrypt_vec(&client_key, &c1);
    println!("After encryption + decryption: {:?}", o1);

    let ws = vec![1, -1, -1, 1, 1, 1, -1, -1, -1];
    println!("Weights: {:?}", ws);

    let (c2, offset) = multiply_and_sum(c1, &ws);
    let o2 = c2.map_or_else(|| 0, |x| x.decrypt(&client_key)) as i64 + offset;
    println!("After linear: {:?}", o2);

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
