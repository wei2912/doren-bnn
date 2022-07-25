use anyhow::Result;
use concrete::{
    generate_keys,
    prelude::{DynamicFheEncryptor, DynamicFheTryEncryptor},
    ClientKey, ConfigBuilder, DynIntegerEncryptor, DynIntegerParameters, DynShortIntEncryptor,
    FheUint4Parameters, ServerKey,
};

use std::fs::File;
use std::path::Path;
use std::{error::Error, fs};

use crate::{FheInt, FheIntCiphertext, FheIntPlaintext};

pub fn get_uint4_config() -> (ConfigBuilder, DynShortIntEncryptor) {
    let mut config = ConfigBuilder::all_disabled();
    let uint_enc = config.add_short_int_type(FheUint4Parameters::default().into());
    (config, uint_enc)
}

pub fn get_uint12_config() -> (ConfigBuilder, DynIntegerEncryptor) {
    let mut config = ConfigBuilder::all_disabled();
    let uint_enc = config.add_integer_type(DynIntegerParameters {
        block_parameters: FheUint4Parameters::default().into(),
        num_block: 3,
    });
    (config, uint_enc)
}

const CLIENT_KEY_FILENAME: &str = "client_key";
const SERVER_KEY_FILENAME: &str = "server_key";

pub fn load_keys(keys_path_str: &str, config: ConfigBuilder) -> Result<(ClientKey, ServerKey)> {
    let keys_path = Path::new(&keys_path_str);
    let client_key_path = keys_path.join(CLIENT_KEY_FILENAME);
    let server_key_path = keys_path.join(SERVER_KEY_FILENAME);

    let files = vec![&client_key_path, &server_key_path]
        .into_iter()
        .map(|path| File::open(path))
        .collect::<Result<Vec<_>, _>>();

    if let Ok([ck_file, sk_file]) = files.as_deref() {
        println!("Loading existing client & server keys...");

        // FIXME: Add error checking to ensure keys are compatible with config
        let keys = (
            bincode::deserialize_from(ck_file),
            bincode::deserialize_from(sk_file),
        );
        match keys {
            (Ok(ck), Ok(sk)) => {
                println!("Existing client & server keys loaded.");
                return Ok((ck, sk));
            }
            _ => println!("Loading of client & server keys failed."),
        }
    }

    println!("Generating new client & server keys...");

    fs::create_dir_all(keys_path)?;
    let ck_file = File::create(client_key_path)?;
    let sk_file = File::create(server_key_path)?;

    let (ck, sk) = generate_keys(config);
    bincode::serialize_into(ck_file, &ck)?;
    bincode::serialize_into(sk_file, &sk)?;

    println!("Client & server keys generated.");
    Ok((ck, sk))
}

pub fn convert_f64_to_bin_pm(input: &[f64]) -> Vec<bool> {
    input.iter().map(|x| *x > 0.0).collect()
}

pub fn encrypt_vec_bin_pm<T: FheIntPlaintext, U: FheIntCiphertext<T>>(
    client_key: &ClientKey,
    uint_enc: &dyn DynamicFheEncryptor<T, FheType = U>,
    pts: &[bool],
) -> Vec<FheInt<T, U>> {
    let enc_func = |pt: T| uint_enc.encrypt(pt, client_key);
    pts.iter()
        .map(|pt| FheInt::encrypt_bin_pm(&enc_func, *pt))
        .collect::<Vec<_>>()
}

pub fn try_encrypt_vec_bin_pm<
    T: FheIntPlaintext,
    U: FheIntCiphertext<T>,
    E: Error + Sync + Send + 'static,
>(
    client_key: &ClientKey,
    uint_try_enc: &dyn DynamicFheTryEncryptor<T, FheType = U, Error = E>,
    pts: &[bool],
) -> Result<Vec<FheInt<T, U>>> {
    let try_enc_func = |pt: T| uint_try_enc.try_encrypt(pt, client_key);
    Ok(pts
        .iter()
        .map(|pt| FheInt::try_encrypt_bin_pm(&try_enc_func, *pt))
        .collect::<Result<Vec<_>, _>>()?)
}

pub fn decrypt_vec<T: FheIntPlaintext, U: FheIntCiphertext<T>>(
    client_key: &ClientKey,
    input: &[FheInt<T, U>],
) -> Vec<f64> {
    input
        .iter()
        .map(|ct| ct.decrypt(&client_key))
        .collect::<Vec<_>>()
}
