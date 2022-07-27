use concrete::{DynShortInt, ServerKey};

use crate::{linear, relu_batchnorm_sign, BatchNormState, FheInt, LinearState};

pub struct ToyNetState {
    pub block_0: LinearState,
    pub block_2: BatchNormState,
}

pub fn toynet(
    server_key: &ServerKey,
    state_dict: ToyNetState,
    input: Vec<FheInt<u8, DynShortInt>>,
) -> Vec<FheInt<u8, DynShortInt>> {
    let ToyNetState { block_0, block_2 } = state_dict;
    let block_0_output = linear(server_key, &input, block_0);
    let block_1_output = relu_batchnorm_sign(server_key, &block_0_output, block_2);
    block_1_output
}
