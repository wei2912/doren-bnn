use concrete::prelude::*;
use concrete::set_server_key;
use concrete::ClientKey;
use concrete::DynIntegerEncryptor;
use concrete::ServerKey;
use once_cell::sync::Lazy;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::fmt::Display;
use std::sync::Mutex;

mod base;
pub use crate::base::*;
mod nn;
pub use crate::nn::*;
mod toynet;
pub use crate::toynet::*;

static KEYS: Lazy<Mutex<(ClientKey, ServerKey, DynIntegerEncryptor)>> =
    Lazy::new(|| match load_keys("keys", get_uint12_params()) {
        Ok(data) => Mutex::new(data),
        Err(err) => panic!("{}", err),
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
fn toynet_py(state_dict_py: &PyDict, input: Vec<f64>) -> PyResult<Vec<i64>> {
    let (client_key, server_key, uint12_enc) = &*KEYS.lock().map_err(make_error)?;
    set_server_key(server_key.clone());

    let fc_weight: Vec<Vec<i8>> = state_dict_py.get_item("fc.weight").map_or_else(
        || Err(make_error("fc.weight not found")),
        |fc_weight_py| fc_weight_py.extract(),
    )?;
    let input_enc = encrypt_vec(&client_key, &uint12_enc, &convert_f64_to_bin(&input));

    let state_dict = ToyNetStateDict { fc_weight };
    let output_enc = toynet(state_dict, input_enc);

    let output_dec = output_enc
        .into_iter()
        .map(|(opt, offset)| opt.map_or_else(|| 0, |x| x.decrypt(&client_key)) as i64 + offset)
        .collect::<Vec<_>>();
    println!("{:?}", output_dec);

    Ok(vec![])
}

#[pymodule]
fn doren_bnn_concrete(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preload_keys_py, m)?)?;
    m.add_function(wrap_pyfunction!(toynet_py, m)?)?;
    Ok(())
}
