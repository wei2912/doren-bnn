use concrete::DynShortInt;

use crate::{linear, relu_batchnorm, BatchNormState, FheInt, LinearState};

pub struct ToyNetState {
    pub block_0: LinearState,
    pub block_1: BatchNormState,
}

pub fn toynet(
    state_dict: ToyNetState,
    input: Vec<FheInt<u8, DynShortInt>>,
) -> Vec<FheInt<u8, DynShortInt>> {
    let ToyNetState { block_0, block_1 } = state_dict;
    let block_0_output = linear(input, block_0);
    let block_1_output = relu_batchnorm(block_0_output, block_1);
    block_1_output
}
