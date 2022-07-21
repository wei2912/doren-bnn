use anyhow::Result;
use concrete::{
    generate_keys, ClientKey, ConfigBuilder, DynInteger, DynIntegerEncryptor, DynIntegerParameters,
    ServerKey,
};
use concrete::{prelude::*, FheUint4Parameters};

use std::fs;
use std::fs::File;
use std::path::Path;

const CLIENT_KEY_FILENAME: &str = "client_key";
const SERVER_KEY_FILENAME: &str = "server_key";

pub fn get_uint12_params() -> DynIntegerParameters {
    DynIntegerParameters {
        block_parameters: FheUint4Parameters::default().into(),
        num_block: 3,
    }
}

pub fn load_keys(
    keys_path_str: &str,
    int_params: DynIntegerParameters,
) -> Result<(ClientKey, ServerKey, DynIntegerEncryptor)> {
    let mut config = ConfigBuilder::all_disabled();
    let uint_enc = config.add_integer_type(int_params);

    let keys_path = Path::new(&keys_path_str);
    let client_key_path = keys_path.join(CLIENT_KEY_FILENAME);
    let server_key_path = keys_path.join(SERVER_KEY_FILENAME);

    match vec![&client_key_path, &server_key_path]
        .into_iter()
        .map(|path| File::open(path))
        .collect::<Result<Vec<_>, _>>()
        .as_deref()
    {
        Ok([ck_file, sk_file]) => {
            println!("Loading existing client & server keys...");

            // FIXME: Add error checking to ensure keys are compatible with config
            let ck: ClientKey = bincode::deserialize_from(ck_file)?;
            let sk: ServerKey = bincode::deserialize_from(sk_file)?;

            println!("Existing client & server keys loaded.");
            Ok((ck, sk, uint_enc))
        }
        _ => {
            println!("Generating client & server keys...");

            fs::create_dir_all(keys_path)?;
            let ck_file = File::create(client_key_path)?;
            let sk_file = File::create(server_key_path)?;

            let (ck, sk) = generate_keys(config);
            bincode::serialize_into(ck_file, &ck)?;
            bincode::serialize_into(sk_file, &sk)?;

            println!("Client & server keys generated.");
            Ok((ck, sk, uint_enc))
        }
    }
}

pub fn convert_f64_to_bin(input: &[f64]) -> Vec<u64> {
    input.iter().map(|x| (*x > 0.0) as u64).collect()
}

pub fn encrypt_vec(
    client_key: &ClientKey,
    uint_enc: &DynIntegerEncryptor,
    input: &[u64],
) -> Vec<DynInteger> {
    input
        .into_iter()
        .map(|pt| uint_enc.encrypt(*pt, client_key))
        .collect::<Vec<_>>()
}

pub fn decrypt_vec(client_key: &ClientKey, input: &[DynInteger]) -> Vec<u64> {
    input
        .into_iter()
        .map(|ct| ct.decrypt(client_key))
        .collect::<Vec<_>>()
}
