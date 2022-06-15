use anyhow::Result;
use concrete::*;

use std::fs;
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

pub fn load_keys(sk_path_str: &str) -> Result<(LWESecretKey, LWEKSK, LWEBSK)> {
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

        let sk_lwe = LWESecretKey::load(&sk_lwe_path_str).unwrap(); // FIXME
        let ksk = LWEKSK::load(&ksk_path_str);
        let bsk = LWEBSK::load(&bsk_path_str);

        println!("Existing secret keys loaded.");
        Ok((sk_lwe, ksk, bsk))
    } else {
        println!("Generating secret keys...");

        fs::create_dir_all(sk_path)?;

        let sk_rlwe = RLWESecretKey::new(&RLWE_PARAMS);
        let sk_in = LWESecretKey::new(&LWE_PARAMS);
        let sk_lwe = sk_rlwe.to_lwe_secret_key();

        let ksk = LWEKSK::new(&sk_in, &sk_lwe, KSK_BASE_LOG, KSK_LEVEL);
        let bsk = LWEBSK::new(&sk_lwe, &sk_rlwe, BSK_BASE_LOG, BSK_LEVEL);

        sk_lwe.save(&sk_lwe_path_str).unwrap(); // FIXME
        ksk.save(&ksk_path_str);
        bsk.save(&bsk_path_str);

        println!("Secret keys generated.");
        Ok((sk_lwe, ksk, bsk))
    }
}

pub fn convert_bin_to_pm1(input: &Vec<bool>) -> Vec<f64> {
    input
        .into_iter()
        .map(|b| if *b { 1.0 } else { -1.0 })
        .collect()
}

pub fn convert_f64_to_bin(input: &Vec<f64>) -> Vec<bool> {
    input.into_iter().map(|x| *x > 0.0).collect()
}

pub fn new_encoder(nb_bit_precision: usize) -> Result<Encoder> {
    Ok(Encoder::new(
        -1.0,
        1.0,
        nb_bit_precision,
        16 - nb_bit_precision,
    )?)
}

pub fn new_encoder_bin() -> Result<Encoder> {
    Ok(new_encoder(1)?)
}

pub fn encrypt(sk_lwe: &LWESecretKey, input: &Vec<f64>, encoder: &Encoder) -> Result<VectorLWE> {
    let output = VectorLWE::encode_encrypt(sk_lwe, &input, &encoder)?;
    Ok(output)
}

pub fn encrypt_bin(sk_lwe: &LWESecretKey, input: &Vec<bool>) -> Result<VectorLWE> {
    let encoder = new_encoder_bin()?;
    let input_pm1 = convert_bin_to_pm1(input);
    Ok(encrypt(sk_lwe, &input_pm1, &encoder)?)
}

pub fn decrypt(sk_lwe: &LWESecretKey, input: &VectorLWE) -> Result<Vec<f64>> {
    let output = input.decrypt_decode(sk_lwe)?;
    Ok(output)
}
