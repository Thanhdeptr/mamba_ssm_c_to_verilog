#include "Main_float.h"
#include <iostream>
#include "Layers_float.h"

void transpose_2d_D_INNER_to_SEQ_LEN_float(const model_dtype_float in[D_INNER][SEQ_LEN], model_dtype_float out[SEQ_LEN][D_INNER]) {
    for (int i = 0; i < D_INNER; ++i) {
        for (int j = 0; j < SEQ_LEN; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

void transpose_2d_SEQ_LEN_to_D_INNER_float(const model_dtype_float in[SEQ_LEN][D_INNER], model_dtype_float out[D_INNER][SEQ_LEN]) {
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < D_INNER; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

template<size_t L, size_t D>
void transpose_ld_to_dl_float(const model_dtype_float in[L][D], model_dtype_float out[D][L]) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < D; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

template<size_t L, size_t N>
void transpose_2d_SEQ_LEN_to_D_STATE_float(const model_dtype_float in[L][N], model_dtype_float out[N][L]) {
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            out[j][i] = in[i][j];
        }
    }
}

void main_mamba_block_float(
    const model_dtype_float hidden_states[SEQ_LEN][D_MODEL],
    model_dtype_float output[SEQ_LEN][D_MODEL],
    const MambaBlockWeights* weights
) {
    static model_dtype_float projected_states_intermediate[SEQ_LEN][D_INNER * 2];
    static model_dtype_float projected_states_transposed_cpp[D_INNER * 2][SEQ_LEN];
    static model_dtype_float hidden_states_after_chunk_cpp[D_INNER][SEQ_LEN];
    static model_dtype_float gate_after_chunk_cpp[D_INNER][SEQ_LEN];
    static model_dtype_float x_conv[D_INNER][SEQ_LEN];
    static model_dtype_float hidden_states_after_conv_cpp[D_INNER][SEQ_LEN];
    static model_dtype_float hidden_states_rearranged_cpp[SEQ_LEN][D_INNER];
    static model_dtype_float ssm_parameters_cpp[SEQ_LEN][DT_RANK + D_STATE * 2];
    static model_dtype_float dt_raw_cpp[SEQ_LEN][DT_RANK];
    static model_dtype_float B_raw_cpp[SEQ_LEN][D_STATE];
    static model_dtype_float C_raw_cpp[SEQ_LEN][D_STATE];
    static model_dtype_float dt_proj_out_cpp[SEQ_LEN][D_INNER];
    static model_dtype_float dt_softplus_cpp[SEQ_LEN][D_INNER];
    static model_dtype_float discrete_time_step_cpp[D_INNER][SEQ_LEN];
    static model_dtype_float A_cpp[D_INNER][D_STATE];
    static model_dtype_float discrete_A_cpp[D_INNER][SEQ_LEN][D_STATE];
    static model_dtype_float discrete_B_cpp[D_INNER][SEQ_LEN][D_STATE];
    static model_dtype_float deltaB_u_cpp[D_INNER][SEQ_LEN][D_STATE];
    static model_dtype_float y_scan_raw_cpp[D_INNER][SEQ_LEN];
    static model_dtype_float scan_output_with_D_cpp[D_INNER][SEQ_LEN];
    static model_dtype_float scan_output_gated_cpp[D_INNER][SEQ_LEN];
    static model_dtype_float y_rearranged_cpp[SEQ_LEN][D_INNER];

    // projection
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear_float(hidden_states[i], projected_states_intermediate[i], weights->in_proj_weight, weights->in_proj_bias, D_MODEL, D_INNER * 2);
    }
    transpose_ld_to_dl_float(projected_states_intermediate, projected_states_transposed_cpp);

    // split
    for (int i = 0; i < D_INNER; ++i) 
        for (int j = 0; j < SEQ_LEN; ++j) {
        hidden_states_after_chunk_cpp[i][j] = projected_states_transposed_cpp[i][j];
        gate_after_chunk_cpp[i][j] = projected_states_transposed_cpp[i + D_INNER][j];
    }

    // Tích chập, SiLU
    causal_conv1d_float(hidden_states_after_chunk_cpp, x_conv, (const model_dtype_float(*)[D_CONV])weights->conv1d_weight, weights->conv1d_bias);
    for (int i = 0; i < D_INNER; ++i) 
        for (int j = 0; j < SEQ_LEN; ++j) {
            hidden_states_after_conv_cpp[i][j] = silu_float(x_conv[i][j]);
    }

    // Tạo tham số động delta_t, B, C
    transpose_2d_D_INNER_to_SEQ_LEN_float(hidden_states_after_conv_cpp, hidden_states_rearranged_cpp);
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear_float(hidden_states_rearranged_cpp[i], ssm_parameters_cpp[i], weights->x_proj_weight, nullptr, D_INNER, DT_RANK + D_STATE * 2);
    }

    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < DT_RANK; ++j) dt_raw_cpp[i][j] = ssm_parameters_cpp[i][j];
        for (int j = 0; j < D_STATE; ++j) B_raw_cpp[i][j] = ssm_parameters_cpp[i][j + DT_RANK];
        for (int j = 0; j < D_STATE; ++j) C_raw_cpp[i][j] = ssm_parameters_cpp[i][j + DT_RANK + D_STATE];
    }
    
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear_float(dt_raw_cpp[i], dt_proj_out_cpp[i], weights->dt_proj_weight, weights->dt_proj_bias, DT_RANK, D_INNER);
    }

    for (int i = 0; i < SEQ_LEN; ++i) 
        for (int j = 0; j < D_INNER; ++j) {
            dt_softplus_cpp[i][j] = softplus_float(dt_proj_out_cpp[i][j]);
    }

    transpose_2d_SEQ_LEN_to_D_INNER_float(dt_softplus_cpp, discrete_time_step_cpp);

    // discretion
    for (int i = 0; i < D_INNER; ++i) 
        for (int j = 0; j < D_STATE; ++j) 
            A_cpp[i][j] = -std::exp(weights->A_log[i * D_STATE + j]);


    for (int d=0; d<D_INNER; ++d) {
            for (int l=0; l<SEQ_LEN; ++l) {
                for (int n=0; n<D_STATE; ++n) {
                    // discrete_A = exp(A * delta)
                    discrete_A_cpp[d][l][n] = std::exp(A_cpp[d][n] * discrete_time_step_cpp[d][l]);
                    // discrete_B = delta * B (broadcast B)
                    discrete_B_cpp[d][l][n] = discrete_time_step_cpp[d][l] * B_raw_cpp[l][n];
                    // deltaB_u = discrete_B * u
                    deltaB_u_cpp[d][l][n] = discrete_B_cpp[d][l][n] * hidden_states_after_conv_cpp[d][l];
                }
            }
        }


    // selective Scan
    scan_core_float(discrete_A_cpp, deltaB_u_cpp, C_raw_cpp, y_scan_raw_cpp);

    // scan_output = scan_output + (hidden_states * D)
    for (int d = 0; d < D_INNER; ++d) {
        for (int l = 0; l < SEQ_LEN; ++l) {
            scan_output_with_D_cpp[d][l] = y_scan_raw_cpp[d][l] + (hidden_states_after_conv_cpp[d][l] * weights->D[d]);
        }
    }

    // scan_output_gated = scan_output * silu(gate)
    for (int d = 0; d < D_INNER; ++d) {
        for (int l = 0; l < SEQ_LEN; ++l) {
            scan_output_gated_cpp[d][l] = scan_output_with_D_cpp[d][l] * silu_float(gate_after_chunk_cpp[d][l]);
        }
    }


    transpose_2d_D_INNER_to_SEQ_LEN_float(scan_output_gated_cpp, y_rearranged_cpp);
    
    //linear cho out_proj
    for (int i = 0; i < SEQ_LEN; ++i) {
        linear_float(y_rearranged_cpp[i], output[i], weights->out_proj_weight, weights->out_proj_bias, D_INNER, D_MODEL);
    }
}
