use concrete::{prelude::*, set_server_key, ServerKey};
use itertools::izip;
use rayon::prelude::*;

use std::{
    ops::AddAssign,
    sync::{Arc, Mutex},
};

use crate::{FheInt, FheIntBootstrap, FheIntCiphertext, FheIntPlaintext};

/**
 * Invariant: xs should only contain encryption of -1 or 1 (no way to verify in code)
 */
pub fn multiply_and_sum<
    T: FheIntPlaintext,
    U: for<'a> FheIntCiphertext<'a, T> + for<'a> AddAssign<&'a U>,
>(
    xs: &[FheInt<T, U>],
    ws: &[i8],
) -> FheInt<T, U> {
    assert!(xs.len() == ws.len());

    let xws = xs.into_iter().zip(ws).collect::<Vec<_>>();

    let mut x = FheInt::zero();
    for (y, w) in xws {
        match w {
            1 => x += y,
            -1 => x -= y,
            0 => {}
            _ => panic!("weights should be either -1 or 1"),
        }
    }
    x
}

pub struct LinearState {
    pub weight: Vec<Vec<i8>>,
}

pub fn linear<T: FheIntPlaintext, U: for<'a> FheIntCiphertext<'a, T> + for<'a> AddAssign<&'a U>>(
    server_key: &ServerKey,
    xs: &[FheInt<T, U>],
    state: LinearState,
) -> Vec<FheInt<T, U>> {
    let xs_arc = Arc::new(Mutex::new(xs.to_vec()));
    let LinearState { weight: wss } = state;
    wss.par_iter()
        .map(|ws| {
            let xs_arc_clone = xs_arc.clone();
            let xs = xs_arc_clone.lock().unwrap();

            let xs_clone = (*xs).clone();
            drop(xs); // release lock on xs before end of scope for other threads

            set_server_key(server_key.clone());
            multiply_and_sum(&xs_clone, ws)
        })
        .collect::<Vec<_>>()
}

pub struct BatchNormState {
    pub weight: Vec<f64>,
    pub bias: Vec<f64>,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
}

const EPS: f64 = 1e-5;

pub fn relu_batchnorm_sign<
    T: FheIntPlaintext,
    U: for<'a> FheIntCiphertext<'a, T> + FheBootstrap,
>(
    server_key: &ServerKey,
    xs: &[FheInt<T, U>],
    state: BatchNormState,
) -> Vec<FheInt<T, U>> {
    let BatchNormState {
        weight,
        bias,
        running_mean,
        running_var,
    } = state;
    assert!(xs.len() == weight.len());
    assert!(xs.len() == bias.len());
    assert!(xs.len() == running_mean.len());
    assert!(xs.len() == running_var.len());

    let relu = |x| f64::max(x, 0.0);
    // g - weight (gamma), b - bias (beta), e - expectation, v - variance
    let bn = |x, g, b, e, v| (x - e) / f64::sqrt(v + EPS) * g + b;
    let sign = |x| (x > 0.0) as u64;

    let f = |x, g, b, e, v| sign(bn(relu(x), g, b, e, v));

    let is = xs.iter().enumerate().map(|(i, _)| i).collect::<Vec<_>>();
    let xs_arc = Arc::new(Mutex::new(xs.to_vec()));
    izip!(is, weight, bias, running_mean, running_var)
        .par_bridge()
        .map(|(i, g, b, e, v)| {
            let xs_arc_clone = xs_arc.clone();
            let xs = xs_arc_clone.lock().unwrap();
            let x_clone = (*xs)[i].clone();
            drop(xs); // release lock on xs before end of scope for other threads

            set_server_key(server_key.clone());
            x_clone.map(|y| f(y, g, b, e, v), 2.0, -1.0)
        })
        .collect::<Vec<_>>()
}
