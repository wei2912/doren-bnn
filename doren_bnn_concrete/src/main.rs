use concrete::*;
use doren_bnn_concrete::*;

use std::error::Error;
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

fn main() -> Result<(), Box<dyn Error>> {
    let messages = vec![-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

    println!("Before encryption: {:?}", messages);

    fs::create_dir_all("sk/")?;
    let (sk_lwe, _, _) = load_keys("sk/")?;
    println!();

    let mut c1 = encrypt_bin(&sk_lwe, &convert_f64_to_bin(&messages))?;
    let o1 = decrypt(&sk_lwe, &c1)?;
    println!("After encryption + decryption: {:?}", o1);
    print_encoder(&c1.encoders[0]);
    println!();

    let weights = vec![1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
    apply_weights(&mut c1, &convert_f64_to_bin(&weights))?;
    let o1 = decrypt(&sk_lwe, &c1)?;
    println!("Weights: {:?}", weights);
    println!("After weighting: {:?}", o1);
    println!();

    let c2 = accumulate(&c1)?;
    let o2 = decrypt(&sk_lwe, &c2)?;
    println!("Accumulate + RSign: {:?}", o2);
    print_encoder(&c2.encoders[0]);
    println!();

    Ok(())
}
