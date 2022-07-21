use concrete::DynInteger;
use rayon::prelude::*;

use std::sync::{Arc, Mutex};

/**
 * Invariant: xs should only contain DynInteger of 0 or 1 (no way to verify in code)
 */
pub fn multiply_and_sum(xs: Vec<DynInteger>, ws: &[i8]) -> (Option<DynInteger>, i64) {
    assert!(xs.len() == ws.len());
    assert!(ws.into_iter().all(|w| *w == 0 || *w == -1 || *w == 1));

    let xws = xs
        .into_iter()
        .zip(ws)
        .filter(|(_, w)| **w != 0)
        .collect::<Vec<_>>();
    let offset = -(xws.len() as i64); // TODO: assert that this offset can be contained within DynInteger

    // Subroutine: Combine 2-chunks of (x, w) into a single DynInteger with offset
    let dyn_ints = xws
        .chunks(2)
        .map(|chunk| match chunk {
            [(x1, w1), (x2, w2)] => match (w1, w2) {
                (1, 1) => x1 + x2 + (2 as u64),
                (1, -1) => x1 - x2 + (2 as u64),
                (-1, 1) => x2 - x1 + (2 as u64),
                (-1, -1) => -x1 - x2 + (2 as u64),
                _ => panic!("chunk weights should be either -1 or 1"),
            },
            [(x1, w1)] => match w1 {
                1 => x1 + (1 as u64),
                -1 => -x1 + (1 as u64),
                _ => panic!("chunk weights should be either -1 or 1"),
            },
            _ => panic!("chunks should be either singleton or of length 2"),
        })
        .collect::<Vec<_>>();

    (sum(dyn_ints), offset)
}

// TODO: replace with iterative version
fn sum<'a>(mut xs: Vec<DynInteger>) -> Option<DynInteger> {
    while xs.len() > 1 {
        let ys = xs
            .chunks(2)
            .map(|chunk| match chunk {
                [x1, x2] => x1 + x2,
                [x] => (*x).clone(),
                _ => panic!("chunks should be either singleton or of length 2"),
            })
            .collect::<Vec<_>>();
        xs = ys;
    }

    match &xs[..] {
        [] => None,
        [x] => Some((*x).clone()),
        _ => panic!("final vector should have length <= 1"),
    }

    /*
    match &xs[..] {
        [] => None,
        [x] => Some((*x).clone()),
        _ => {
            let ys = xs
                .chunks(2)
                .map(|chunk| match chunk {
                    [x1, x2] => x1 + x2,
                    [x] => (*x).clone(),
                    _ => panic!("chunks should be either singleton or of length 2"),
                })
                .collect::<Vec<_>>();
            sum(ys)
        }
    }
    */
}

pub fn linear(xs: Vec<DynInteger>, wss: Vec<Vec<i8>>) -> Vec<(Option<DynInteger>, i64)> {
    let xs_arc = Arc::new(Mutex::new(xs));
    wss.par_iter()
        .map(|ws| {
            let xs_arc_clone = xs_arc.clone();
            let xs = xs_arc_clone.lock().unwrap();
            let xs_clone = (*xs).clone();
            drop(xs); // release lock on xs before end of scope

            let (opt, offset) = multiply_and_sum(xs_clone, ws);
            (opt.map(|x| x.to_owned()), offset)
        })
        .collect::<Vec<_>>()
}

/*
pub fn sign(ksk: &LWEKSK, bsk: &LWEBSK, input: &VectorLWE) -> Result<VectorLWE> {
    Ok(input.keyswitch(ksk)?.bootstrap_nth_with_function(
        bsk,
        |x| if x > 0.0 { 1.0 } else { -1.0 },
        &new_encoder_bin()?,
        0,
    )?)
}
*/
