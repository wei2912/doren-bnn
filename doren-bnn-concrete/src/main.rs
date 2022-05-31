use concrete::*;

use std::error::Error;
use std::fs;
use std::path::Path;

const BASE_LOG: usize = 3;
const LEVEL: usize = 5;

fn main() -> Result<(), Box<dyn Error>> {
    let encoder = Encoder::new_rounding_context(-1.0, 1.0, 1, 3)?;
    let messages = vec![-0.56, 0.23, 0.98, 0.0];

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

        println!("Existing secret keys loaded.");
        (sk_out, bsk)
    } else {
        println!("Generating secret keys...");
        let sk_rlwe = RLWESecretKey::new(&RLWE128_1024_1);
        let sk_in = LWESecretKey::new(&LWE128_1024);
        let sk_out = sk_rlwe.to_lwe_secret_key();

        let bsk = LWEBSK::new(&sk_in, &sk_rlwe, BASE_LOG, LEVEL);
        sk_out.save(SK_OUT_PATH)?;
        bsk.save(BSK_PATH);

        println!("Secret keys generated.");
        (sk_out, bsk)
    };

    let c1 = &messages
        .into_iter()
        .map(|m| LWE::encode_encrypt(&sk_out, m, &encoder))
        .collect::<Result<Vec<_>, _>>()?;
    let o1 = &c1
        .into_iter()
        .map(|c| c.decrypt_decode(&sk_out))
        .collect::<Result<Vec<_>, _>>()?;
    println!("After encryption + decryption: {:?}", o1);

    let c2 = &c1.clone();

    let c3 = &c1
        .into_iter()
        .zip(c2.into_iter())
        .map(|(x, y)| x.mul_from_bootstrap(&y, &bsk))
        .collect::<Result<Vec<_>, _>>()?;
    let o3 = &c3
        .into_iter()
        .map(|c| c.decrypt_decode(&sk_out))
        .collect::<Result<Vec<_>, _>>()?;

    println!("Self-multiplication + bootstrap: {:?}", o3);

    println!("{:?}", &c1[0].encoder);
    println!("{:?}", &c2[0].encoder);
    println!("{:?}", &c3[0].encoder);

    Ok(())
}
