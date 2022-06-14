use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde_json;

use std::fmt::Display;

mod base;
pub use crate::base::*;
mod nn;
pub use crate::nn::*;

fn make_error<E: Display + Sized>(e: E) -> PyErr {
    PyErr::new::<PyRuntimeError, _>(e.to_string())
}

#[pyfunction]
#[pyo3(name = "load_keys")]
fn load_keys_py(sk_path_str: &str) -> PyResult<(String, String, String)> {
    let (sk_lwe, ksk, bsk) = load_keys(sk_path_str).map_err(make_error)?;
    let sk_lwe_json = serde_json::to_string(&sk_lwe).map_err(make_error)?;
    let ksk_json = serde_json::to_string(&ksk).map_err(make_error)?;
    let bsk_json = serde_json::to_string(&bsk).map_err(make_error)?;
    Ok((sk_lwe_json, ksk_json, bsk_json))
}

#[pymodule]
fn doren_bnn_concrete(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_keys_py, m)?)?;
    Ok(())
}
