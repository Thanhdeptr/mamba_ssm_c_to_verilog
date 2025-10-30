#pragma once
#include "Layers_float.h"

// Floating-point version of MambaBlockWeights
struct MambaBlockWeights {
    model_dtype_float in_proj_weight[D_MODEL * D_INNER * 2];
    model_dtype_float in_proj_bias[D_INNER * 2];
    model_dtype_float conv1d_weight[D_INNER * D_CONV];
    model_dtype_float conv1d_bias[D_INNER];
    model_dtype_float x_proj_weight[D_INNER * (DT_RANK + D_STATE * 2)];
    model_dtype_float dt_proj_weight[DT_RANK * D_INNER];
    model_dtype_float dt_proj_bias[D_INNER];
    model_dtype_float A_log[D_INNER * D_STATE];
    model_dtype_float D[D_INNER];
    model_dtype_float out_proj_weight[D_INNER * D_MODEL];
    model_dtype_float out_proj_bias[D_MODEL];
};

// Floating-point version of main function
void main_mamba_block_float(
    const model_dtype_float hidden_states[SEQ_LEN][D_MODEL],
    model_dtype_float output[SEQ_LEN][D_MODEL],
    const MambaBlockWeights* weights
);
