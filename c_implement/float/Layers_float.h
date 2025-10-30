#pragma once

// Include the main constants from Layers.h
#include "Layers.h"

// Original floating-point data type
typedef float model_dtype_float;

// --- KHAI BÁO CÁC HÀM FLOATING-POINT ---
void linear_float(const model_dtype_float x[], model_dtype_float y[], 
            const model_dtype_float* W, const model_dtype_float b[],
            int in_dim, int out_dim);

void causal_conv1d_float(const model_dtype_float x[][SEQ_LEN], model_dtype_float out[][SEQ_LEN],
                   const model_dtype_float weight[][D_CONV], const model_dtype_float bias[]);

model_dtype_float silu_float(model_dtype_float x);
model_dtype_float softplus_float(model_dtype_float x);

void scan_core_float(
    const model_dtype_float discrete_A[][SEQ_LEN][D_STATE],
    const model_dtype_float deltaB_u[][SEQ_LEN][D_STATE],
    const model_dtype_float C_raw[][D_STATE],
    model_dtype_float scan_output_raw[][SEQ_LEN]
);
