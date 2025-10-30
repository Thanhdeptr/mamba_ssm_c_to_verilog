#include "Layers_float.h"
#include <cmath>

// --- IMPLEMENT HÀM LINEAR FLOATING-POINT ---
void linear_float(const model_dtype_float x[], model_dtype_float y[], 
            const model_dtype_float* W, const model_dtype_float b[],
            int in_dim, int out_dim) {
    
    for (int i = 0; i < out_dim; ++i) {
        model_dtype_float sum = 0.0f;
        for (int j = 0; j < in_dim; ++j) {
            sum += x[j] * W[i * in_dim + j];
        }
        y[i] = sum + (b ? b[i] : 0.0f);
    }
}

void causal_conv1d_float(const model_dtype_float x[][SEQ_LEN], model_dtype_float out[][SEQ_LEN],
                   const model_dtype_float weight[][D_CONV], const model_dtype_float bias[]) {
    
    const int PADDED_LEN = SEQ_LEN + D_CONV - 1;
    for (int d = 0; d < D_INNER; ++d) {
        model_dtype_float x_padded[PADDED_LEN] = {0.0f};
        for (int i = 0; i < SEQ_LEN; ++i) {
            x_padded[i + D_CONV - 1] = x[d][i];
        }

        for (int l = 0; l < SEQ_LEN; ++l) {
            model_dtype_float sum = 0.0f;
            for (int k = 0; k < D_CONV; ++k) {
                sum += x_padded[l + k] * weight[d][k];
            }
            out[d][l] = sum + bias[d];
        }
    }
}

// --- IMPLEMENT HÀM SILU FLOATING-POINT ---
model_dtype_float silu_float(model_dtype_float x) {
    return x / (1.0f + std::exp(-x)); //SiLU = x * sigmoid(x) = x / (1 + exp(-x))
}

model_dtype_float softplus_float(model_dtype_float x) {
    return std::log(1.0f + std::exp(x)); //Softplus = log(1 + exp(x))
}

void scan_core_float(
    const model_dtype_float discrete_A[][SEQ_LEN][D_STATE],
    const model_dtype_float deltaB_u[][SEQ_LEN][D_STATE],
    const model_dtype_float C_raw[][D_STATE],
    model_dtype_float scan_output_raw[][SEQ_LEN]
) {

    for (int d = 0; d < D_INNER; ++d) {
        model_dtype_float h[D_STATE] = {0.0f};

        for (int l = 0; l < SEQ_LEN; ++l) {
            // h_t = discrete_A_t * h_{t-1} + deltaB_u_t
            for (int n = 0; n < D_STATE; ++n) {
                h[n] = discrete_A[d][l][n] * h[n] + deltaB_u[d][l][n];
            }

            // y_t = C_t * h_t
            model_dtype_float y_scan = 0.0f;
            for (int n = 0; n < D_STATE; ++n) {
                y_scan += C_raw[l][n] * h[n];
            }
            scan_output_raw[d][l] = y_scan;
        }
    }
}
