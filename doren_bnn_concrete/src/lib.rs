use concrete::{set_server_key, ClientKey, DynShortIntEncryptor, ServerKey};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::fmt::Display;
use std::sync::Mutex;

mod base;
pub use crate::base::*;
mod fheint;
pub use crate::fheint::*;
mod nn;
pub use crate::nn::*;
mod toynet;
pub use crate::toynet::*;

static KEYS: Lazy<Mutex<(ClientKey, ServerKey, DynShortIntEncryptor)>> = Lazy::new(|| {
    let (config, uint_enc) = get_uint4_config();
    match load_keys("keys_4", config) {
        Ok((client_key, secret_key)) => Mutex::new((client_key, secret_key, uint_enc)),
        Err(err) => panic!("{}", err),
    }
});

fn make_error<E: Display + Sized>(e: E) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(e.to_string())
}

#[pyfunction]
#[pyo3(name = "preload_keys")]
fn preload_keys_py() -> PyResult<()> {
    let _keys = &*KEYS.lock().map_err(make_error)?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "toynet")]
fn toynet_py(state_dict_py: &PyDict, input: Vec<f64>) -> PyResult<Vec<f64>> {
    let (client_key, server_key, uint4_enc) = &*KEYS.lock().map_err(make_error)?;
    set_server_key(server_key.clone());

    let get_state = |key: &str| -> PyResult<&PyAny> {
        state_dict_py.get_item(key).map_or_else(
            || Err(make_error(format!("{:?} not found", key))),
            |x_py| Ok(x_py),
        )
    };

    let block_0 = LinearState {
        weight: get_state("block.0.weight")?.extract()?,
    };
    let block_1 = BatchNormState {
        weight: get_state("block.1.weight")?.extract()?,
        bias: get_state("block.1.bias")?.extract()?,
        running_mean: get_state("block.1.running_mean")?.extract()?,
        running_var: get_state("block.1.running_var")?.extract()?,
    };

    let input_enc = try_encrypt_vec_bin_pm(&client_key, uint4_enc, &convert_f64_to_bin_pm(&input))
        .map_err(make_error)?;

    let state_dict = ToyNetState { block_0, block_1 };
    let output_enc = toynet(state_dict, input_enc);

    let output_dec = decrypt_vec(client_key, &output_enc);
    println!("{:?}", output_dec);

    Ok(vec![])
}

#[pymodule]
fn doren_bnn_concrete(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preload_keys_py, m)?)?;
    m.add_function(wrap_pyfunction!(toynet_py, m)?)?;
    Ok(())
}
