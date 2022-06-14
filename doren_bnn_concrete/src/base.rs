use bitvec::prelude::BitVec;
use concrete::*;

use std::error::Error;
use std::path::Path;

const LWE_PARAMS: LWEParams = LWE128_750;
const RLWE_PARAMS: RLWEParams = RLWE128_2048_1;
const BSK_BASE_LOG: usize = 7;
const BSK_LEVEL: usize = 3;
const KSK_BASE_LOG: usize = 2;
const KSK_LEVEL: usize = 7;

const SK_LWE_FILENAME: &str = "sk_lwe.json";
const KSK_FILENAME: &str = "ksk.json";
const BSK_FILENAME: &str = "bsk.json";

pub fn load_keys(sk_path_str: &str) -> Result<(LWESecretKey, LWEKSK, LWEBSK), Box<dyn Error>> {
    let sk_path = Path::new(&sk_path_str);

    let sk_lwe_path = sk_path.join(SK_LWE_FILENAME);
    let ksk_path = sk_path.join(KSK_FILENAME);
    let bsk_path = sk_path.join(BSK_FILENAME);

    let sk_lwe_path_str = sk_lwe_path.display().to_string();
    let ksk_path_str = ksk_path.display().to_string();
    let bsk_path_str = bsk_path.display().to_string();

    if vec![sk_lwe_path, ksk_path, bsk_path]
        .into_iter()
        .all(|path| path.exists())
    {
        println!("Loading existing secret keys...");

        let sk_lwe = LWESecretKey::load(&sk_lwe_path_str)?;
        let ksk = LWEKSK::load(&ksk_path_str);
        let bsk = LWEBSK::load(&bsk_path_str);

        println!("Existing secret keys loaded.");
        Ok((sk_lwe, ksk, bsk))
    } else {
        println!("Generating secret keys...");

        let sk_rlwe = RLWESecretKey::new(&RLWE_PARAMS);
        let sk_in = LWESecretKey::new(&LWE_PARAMS);
        let sk_lwe = sk_rlwe.to_lwe_secret_key();

        let ksk = LWEKSK::new(&sk_in, &sk_lwe, KSK_BASE_LOG, KSK_LEVEL);
        let bsk = LWEBSK::new(&sk_lwe, &sk_rlwe, BSK_BASE_LOG, BSK_LEVEL);

        sk_lwe.save(&sk_lwe_path_str)?;
        ksk.save(&ksk_path_str);
        bsk.save(&bsk_path_str);

        println!("Secret keys generated.");
        Ok((sk_lwe, ksk, bsk))
    }
}

pub fn convert_bin_to_pm1(input: &BitVec) -> Vec<f64> {
    input
        .into_iter()
        .map(|b| if *b { 1.0 } else { -1.0 })
        .collect()
}

pub fn convert_pm1_to_bin(input: &Vec<f64>) -> BitVec {
    input.into_iter().map(|x| *x >= 0.0).collect()
}

pub fn new_encoder_bin() -> Result<Encoder, Box<dyn Error>> {
    Ok(Encoder::new(-1.0, 1.0, 1, 7)?)
}

pub fn encrypt_bin(sk_lwe: &LWESecretKey, input: &BitVec) -> Result<VectorLWE, Box<dyn Error>> {
    let encoder = new_encoder_bin()?;
    let output = VectorLWE::encode_encrypt(sk_lwe, &convert_bin_to_pm1(input), &encoder)?;
    Ok(output)
}

pub fn decrypt(sk_lwe: &LWESecretKey, input: &VectorLWE) -> Result<Vec<f64>, Box<dyn Error>> {
    let output = input.decrypt_decode(sk_lwe)?;
    Ok(output)
}
