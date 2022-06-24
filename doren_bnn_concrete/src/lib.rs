use anyhow::Result;
use concrete::{LWESecretKey, LWEBSK, LWEKSK};
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
mod npe;
pub use crate::npe::*;
mod toynet;
pub use crate::toynet::*;

static SK: Lazy<Mutex<(LWESecretKey, LWEKSK, LWEBSK)>> = Lazy::new(|| match load_keys("sk/") {
    Ok(sk) => Mutex::new(sk),
    Err(err) => panic!("{}", err),
});

fn make_error<E: Display + Sized>(e: E) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(e.to_string())
}

#[pyfunction]
#[pyo3(name = "preload_keys")]
fn preload_keys_py() -> PyResult<()> {
    let _sk = SK.lock().map_err(make_error)?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "toynet")]
fn toynet_py(state_dict_py: &PyDict, input: Vec<f64>) -> PyResult<Vec<f64>> {
    let sk = SK.lock().map_err(make_error)?;
    let (sk_lwe, ksk, bsk) = &*sk;

    let fc_weight: Vec<Vec<bool>> = match state_dict_py.get_item("fc.weight") {
        Some(fc_weight_py) => fc_weight_py.extract(),
        None => Err(make_error("fc.weight not found")),
    }?;

    let input_enc = encrypt_bin(
        sk_lwe,
        &input.into_iter().map(|x| x > 0.0).collect::<Vec<bool>>(),
    )
    .map_err(make_error)?;

    let state_dict = ToyNetStateDict { fc_weight };
    let output_enc = toynet(ksk, bsk, &state_dict, &input_enc).map_err(make_error)?;

    let output = output_enc
        .into_iter()
        .map(|ct| {
            decrypt(sk_lwe, &ct).map(|vec| {
                assert!(vec.len() == 1);
                vec[0]
            })
        })
        .collect::<Result<Vec<f64>>>()
        .map_err(make_error)?;
    Ok(output)
}

#[pymodule]
fn doren_bnn_concrete(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preload_keys_py, m)?)?;
    m.add_function(wrap_pyfunction!(toynet_py, m)?)?;
    Ok(())
}
