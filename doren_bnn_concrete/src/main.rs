use anyhow::Result;
use concrete::*;
use doren_bnn_concrete::*;

use std::fs;

fn print_encoder(encoder: &Encoder) {
    println!("{:?}", encoder);
    println!(
        "{:?}",
        (
            encoder.get_granularity(),
            encoder.get_min(),
            encoder.get_max(),
            encoder.get_size()
        )
    );
}

fn main() -> Result<()> {
    let messages = vec![-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

    println!("Before encryption: {:?}", messages);

    fs::create_dir_all("sk/")?;
    let (sk_lwe, ksk, bsk) = load_keys("sk/")?;
    println!();

    let c1 = encrypt_bin(&sk_lwe, &convert_f64_to_bin(&messages))?;
    let o1 = decrypt(&sk_lwe, &c1)?;
    println!("After encryption + decryption: {:?}", o1);
    print_encoder(&c1.encoders[0]);
    println!();

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

    const THRESHOLD: f64 = -2.0;
    let c5 = rsign(&ksk, &bsk, &c4[0], THRESHOLD)?;
    let o5 = decrypt(&sk_lwe, &c5)?;
    println!("RSign with threshold {:?}: {:?}", THRESHOLD, o5);
    print_encoder(&c5.encoders[0]);
    println!();

    Ok(())
}
