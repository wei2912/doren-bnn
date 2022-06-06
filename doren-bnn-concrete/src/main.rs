use concrete::*;

use std::error::Error;
use std::fs;
use std::path::Path;

const LWE_PARAMS: LWEParams = LWE128_750;
const RLWE_PARAMS: RLWEParams = RLWE128_2048_1;
const BSK_BASE_LOG: usize = 7;
const BSK_LEVEL: usize = 3;
const KSK_BASE_LOG: usize = 2;
const KSK_LEVEL: usize = 7;

fn main() -> Result<(), Box<dyn Error>> {
    let encoder = Encoder::new(-1.0, 1.0, 1, 7)?;
    let messages = vec![-1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

    println!("Before encoding: {:?}", messages);

    let p1 = encoder.encode(&messages)?;
    let o1 = p1.decode()?;

    println!("After encoding: {:?}", o1);

    fs::create_dir_all("sk/")?;

    const SK_OUT_PATH: &str = "sk/sk_lwe.json";
    const KSK_PATH: &str = "sk/ksk.json";
    const BSK_PATH: &str = "sk/bsk.json";

    let (sk_out, ksk, bsk) = if vec![SK_OUT_PATH, BSK_PATH, KSK_PATH]
        .into_iter()
        .all(|path| Path::new(path).exists())
    {
        println!("Loading existing secret keys...");

        let sk_out = LWESecretKey::load(SK_OUT_PATH)?;
        let ksk = LWEKSK::load(KSK_PATH);
        let bsk = LWEBSK::load(BSK_PATH);

        println!("Existing secret keys loaded.");
        (sk_out, ksk, bsk)
    } else {
        println!("Generating secret keys...");

        let sk_rlwe = RLWESecretKey::new(&RLWE_PARAMS);
        let sk_in = LWESecretKey::new(&LWE_PARAMS);
        let sk_out = sk_rlwe.to_lwe_secret_key();

        let ksk = LWEKSK::new(&sk_in, &sk_out, KSK_BASE_LOG, KSK_LEVEL);
        let bsk = LWEBSK::new(&sk_out, &sk_rlwe, BSK_BASE_LOG, BSK_LEVEL);

        sk_out.save(SK_OUT_PATH)?;
        ksk.save(KSK_PATH);
        bsk.save(BSK_PATH);

        println!("Secret keys generated.");
        (sk_out, ksk, bsk)
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

    let c1_sum = c1.sum_with_padding()?;
    let o1_sum = c1_sum.decrypt_decode(&sk_out)?;
    println!("Sum: {:?}", o1_sum);

    const THRESHOLD: f64 = -6.0; // arbitrary threshold as "learnable parameter"
    let c2 = c1_sum.keyswitch(&ksk)?.bootstrap_nth_with_function(
        &bsk,
        |x| if x >= THRESHOLD { 1.0 } else { -1.0 },
        &encoder,
        0,
    )?;
    let o2 = c2.decrypt_decode(&sk_out)?;
    println!("Threshold: {:?}", THRESHOLD);
    println!("ReLU + bootstrap: {:?}", o2);

    let encoder = &c2.encoders[0];
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

    Ok(())
}
