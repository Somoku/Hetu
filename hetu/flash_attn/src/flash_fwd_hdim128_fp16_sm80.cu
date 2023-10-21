// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "hetu/flash_attn/src/flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t>(params, stream);
}
