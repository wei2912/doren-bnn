use concrete::*;

use std::error::Error;
use std::fs;
use std::path::Path;

const BASE_LOG: usize = 3;
const LEVEL: usize = 5;

fn main() -> Result<(), Box<dyn Error>> {
    let encoder = Encoder::new(-1.0, 1.0, 5, 5)?;
    let messages = vec![-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

    println!("Before encoding: {:?}", messages);

    let p1 = encoder.encode(&messages)?;
    let o1 = p1.decode()?;

    println!("After encoding: {:?}", o1);

    fs::create_dir_all("sk/")?;

    const SK_OUT_PATH: &str = "sk/sk_lwe.json";
    const BSK_PATH: &str = "sk/bsk.json";

    let (sk_out, bsk) = if Path::new(SK_OUT_PATH).exists() && Path::new(BSK_PATH).exists() {
        println!("Loading existing secret keys...");
        let sk_out = LWESecretKey::load(SK_OUT_PATH)?;
        let bsk = LWEBSK::load(BSK_PATH);

        println!("{} {}", sk_out.dimension, bsk.polynomial_size);

        println!("Existing secret keys loaded.");
        (sk_out, bsk)
    } else {
        println!("Generating secret keys...");

        let rlwe_params = RLWE80_1024_1;
        let lwe_params = LWE80_1024;

        let sk_rlwe = RLWESecretKey::new(&rlwe_params);
        let sk_in = LWESecretKey::new(&lwe_params);
        let sk_out = sk_rlwe.to_lwe_secret_key();

        let bsk = LWEBSK::new(&sk_in, &sk_rlwe, BASE_LOG, LEVEL);
        sk_out.save(SK_OUT_PATH)?;
        bsk.save(BSK_PATH);

        println!("Secret keys generated.");
        (sk_out, bsk)
    };

    let mut c1 = VectorLWE::encode_encrypt(&sk_out, &messages, &encoder)?;
    let o1 = c1.decrypt_decode(&sk_out)?;
    println!("After encryption + decryption: {:?}", o1);
    println!("{:?}", c1.encoders[0]);

    let weights = vec![1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0];
    for i in 0..weights.len() - 1 {
        // only negate ciphertext bits with corresponding weight -1
        if weights[i] == -1.0 {
            c1.opposite_nth_inplace(i)?;
        }
    }
    let o1 = c1.decrypt_decode(&sk_out)?;
    println!("Weights: {:?}", weights);
    println!("After weighting: {:?}", o1);
    println!("{:?}", c1.encoders[0]);

    let c1_sum = c1.sum_with_padding()?;
    let o1_sum = c1_sum.decrypt_decode(&sk_out)?;
    println!("Sum: {:?}", o1_sum);
    println!("{:?}", c1_sum.encoders[0]);

    const THRESHOLD: f64 = 2.0; // arbitrary threshold as "learnable parameter"
    let c2 = c1_sum.bootstrap_nth_with_function(
        &bsk,
        |x| if x >= THRESHOLD { 1.0 } else { -1.0 },
        &encoder,
        0,
    )?;
    let o2 = c2.decrypt_decode(&sk_out)?;
    println!("Threshold: {:?}", THRESHOLD);
    println!("ReLU + bootstrap: {:?}", o2);
    println!("{:?}", c2.encoders[0]);

    Ok(())
}
