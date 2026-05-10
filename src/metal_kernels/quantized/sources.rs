pub(super) const MULTI3_QMV4_HEADER: &str = r#"
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector4_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    inline float qdot4_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      const device uint16_t* ws = (const device uint16_t*)w;
      float accum = 0.0f;
      for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
        uint16_t packed = ws[i];
        accum +=
          x_thread[4 * i] * float(packed & 0x000f) +
          x_thread[4 * i + 1] * float(packed & 0x00f0) +
          x_thread[4 * i + 2] * float(packed & 0x0f00) +
          x_thread[4 * i + 3] * float(packed & 0xf000);
      }
      return scale * accum + sum * bias;
    }
"#;

pub(super) const SMALL_M_QMV4_HEADER: &str = r#"
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    template <typename T>
    inline float load_vector4_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    inline float qdot4_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      const device uint16_t* ws = (const device uint16_t*)w;
      float accum = 0.0f;
      for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
        uint16_t packed = ws[i];
        accum +=
          x_thread[4 * i] * float(packed & 0x000f) +
          x_thread[4 * i + 1] * float(packed & 0x00f0) +
          x_thread[4 * i + 2] * float(packed & 0x0f00) +
          x_thread[4 * i + 3] * float(packed & 0xf000);
      }
      return scale * accum + sum * bias;
    }
"#;

pub(super) const GATE_UP_SWIGLU_QMV4_HEADER: &str = r#"
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
    constant constexpr int MAX_M = 6;

    template <typename T>
    inline T sigmoid_mlx_exact(T x) {
      auto y = 1 / (1 + metal::exp(metal::abs(x)));
      return (x < T(0)) ? y : 1 - y;
    }

    template <typename T>
    inline T swiglu_mlx_exact(T gate, T up) {
      T silu = gate * sigmoid_mlx_exact<T>(gate);
      return T(silu * up);
    }

    template <typename T>
    inline float load_vector4_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    inline float qdot4_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      const device uint16_t* ws = (const device uint16_t*)w;
      float accum = 0.0f;
      for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
        uint16_t packed = ws[i];
        accum +=
          x_thread[4 * i] * float(packed & 0x000f) +
          x_thread[4 * i + 1] * float(packed & 0x00f0) +
          x_thread[4 * i + 2] * float(packed & 0x0f00) +
          x_thread[4 * i + 3] * float(packed & 0xf000);
      }
      return scale * accum + sum * bias;
    }
"#;

pub(super) const GATE_UP_SWIGLU_QMV4_SOURCE: &str = r#"
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int M = int(M_size);
    int K = int(K_size);
    int N = int(N_size);
    constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K / GS;

    const device uint8_t* gate_w_base =
      (const device uint8_t*)gate_w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device uint8_t* up_w_base =
      (const device uint8_t*)up_w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* gate_scales_base =
      gate_scales + out_row * in_vec_size_g
      + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* gate_biases_base =
      gate_biases + out_row * in_vec_size_g
      + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* up_scales_base =
      up_scales + out_row * in_vec_size_g
      + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* up_biases_base =
      up_biases + out_row * in_vec_size_g
      + int(simd_lid) / SCALE_STEP_PER_THREAD;

    float gate_result[MAX_M][RESULTS_PER_SIMDGROUP];
    float up_result[MAX_M][RESULTS_PER_SIMDGROUP];
    float x_thread[MAX_M][VALUES_PER_THREAD];
    float x_sum[MAX_M];

    for (int m = 0; m < MAX_M; ++m) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        gate_result[m][row] = 0.0f;
        up_result[m][row] = 0.0f;
      }
    }

    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
      for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
          const device T* x_m =
            x + m * K + k_block + int(simd_lid) * VALUES_PER_THREAD;
          x_sum[m] = load_vector4_exact<T>(x_m, x_thread[m]);
        }
      }

      const device uint8_t* gate_w_block =
        gate_w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device uint8_t* up_w_block =
        up_w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device T* gate_scales_block = gate_scales_base + k_block / GS;
      const device T* gate_biases_block = gate_biases_base + k_block / GS;
      const device T* up_scales_block = up_scales_base + k_block / GS;
      const device T* up_biases_block = up_biases_base + k_block / GS;

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          const device uint8_t* gate_w_row =
            gate_w_block + row * in_vec_size_w;
          const device uint8_t* up_w_row =
            up_w_block + row * in_vec_size_w;
          const device T* gate_sc_row =
            gate_scales_block + row * in_vec_size_g;
          const device T* gate_bs_row =
            gate_biases_block + row * in_vec_size_g;
          const device T* up_sc_row =
            up_scales_block + row * in_vec_size_g;
          const device T* up_bs_row =
            up_biases_block + row * in_vec_size_g;
          float gate_scale = float(gate_sc_row[0]);
          float gate_bias = float(gate_bs_row[0]);
          float up_scale = float(up_sc_row[0]);
          float up_bias = float(up_bs_row[0]);

          for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
              gate_result[m][row] += qdot4_exact(
                gate_w_row, x_thread[m], gate_scale, gate_bias, x_sum[m]
              );
              up_result[m][row] += qdot4_exact(
                up_w_row, x_thread[m], up_scale, up_bias, x_sum[m]
              );
            }
          }
        }
      }
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        for (int m = 0; m < MAX_M; ++m) {
          if (m < M) {
            float gate_sum = simd_sum(gate_result[m][row]);
            float up_sum = simd_sum(up_result[m][row]);
            if (simd_lid == 0) {
              T gate_value = T(gate_sum);
              T up_value = T(up_sum);
              y[m * N + n] = swiglu_mlx_exact<T>(gate_value, up_value);
            }
          }
        }
      }
    }
"#;

pub(super) const SMALL_M_QMM4_SOURCE: &str = r#"
    using namespace metal;
    constexpr int BM = 8;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int BK_SUB = 8;

    uint tid   = thread_position_in_threadgroup.x;
    uint sg_id = tid / 32;
    uint tg_m  = threadgroup_position_in_grid.x;
    uint tg_n  = threadgroup_position_in_grid.y;

    int M = int(M_size);
    int K = int(K_size);
    int N = int(N_size);
    int K_by_8  = K / 8;
    int K_by_gs = K / GS;
    int m0 = int(tg_m) * BM;
    int n0 = int(tg_n) * BN;

    threadgroup T B_tile[BK * BN];

    simdgroup_matrix<T, 8, 8> a, b_L, b_R;
    simdgroup_matrix<float, 8, 8> c_L =
      simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_R =
      simdgroup_matrix<float, 8, 8>(0.0f);

    int t_a = int(tid);
    int t_b = int(tid) + 64;
    int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
    int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
    int sg_n_off = int(sg_id) * 16;

    for (int k0 = 0; k0 < K; k0 += BK) {
        {
            int n_global = n0 + dq_n_a;
            int k_base = k0 + dq_k_a * 8;
            uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
            float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
            float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] =
                  T(float(nib) * s + b);
            }
        }
        {
            int n_global = n0 + dq_n_b;
            int k_base = k0 + dq_k_b * 8;
            uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
            float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
            float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] =
                  T(float(nib) * s + b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ks = 0; ks < BK / BK_SUB; ++ks) {
            simdgroup_load(a, x + m0 * K + k0 + ks * BK_SUB, K);
            simdgroup_load(
              b_L, B_tile + ks * BK_SUB * BN + sg_n_off, BN
            );
            simdgroup_load(
              b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8, BN
            );
            simdgroup_multiply_accumulate(c_L, a, b_L, c_L);
            simdgroup_multiply_accumulate(c_R, a, b_R, c_R);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_matrix<T, 8, 8> c_L_T, c_R_T;
    c_L_T.thread_elements()[0] = T(c_L.thread_elements()[0]);
    c_L_T.thread_elements()[1] = T(c_L.thread_elements()[1]);
    c_R_T.thread_elements()[0] = T(c_R.thread_elements()[0]);
    c_R_T.thread_elements()[1] = T(c_R.thread_elements()[1]);
    if (m0 < M) {
      simdgroup_store(c_L_T, y + m0 * N + n0 + sg_n_off, N);
      simdgroup_store(c_R_T, y + m0 * N + n0 + sg_n_off + 8, N);
    }
"#;

pub(super) const LARGE_M_QMM4_SOURCE: &str = r#"
    using namespace metal;
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int BK_SUB = 8;

    uint tid   = thread_position_in_threadgroup.x;
    uint sg_id = simdgroup_index_in_threadgroup;
    uint tg_m  = threadgroup_position_in_grid.x;
    uint tg_n  = threadgroup_position_in_grid.y;

    int M = int(M_size);
    int K = int(K_size);
    int N = int(N_size);
    int K_by_8  = K / 8;
    int K_by_gs = K / GS;
    int m0 = int(tg_m) * BM;
    int n0 = int(tg_n) * BN;
    int sg_m_off = int(sg_id / 2) * 8;
    int sg_n_off = int(sg_id % 2) * 16;

    threadgroup T B_tile[BK * BN];

    simdgroup_matrix<T, 8, 8> a, b_L, b_R;
    simdgroup_matrix<float, 8, 8> c_L =
      simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_R =
      simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        if (tid < (BK / 8) * BN) {
            int pack = int(tid);
            int dq_k = pack / BN;
            int dq_n = pack % BN;
            int n_global = n0 + dq_n;
            int k_base = k0 + dq_k * 8;
            uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
            float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
            float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                B_tile[(dq_k * 8 + ki) * BN + dq_n] =
                  T(float(nib) * s + b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ks = 0; ks < BK / BK_SUB; ++ks) {
            simdgroup_load(
              a,
              x + (m0 + sg_m_off) * K + k0 + ks * BK_SUB,
              K
            );
            simdgroup_load(
              b_L, B_tile + ks * BK_SUB * BN + sg_n_off, BN
            );
            simdgroup_load(
              b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8, BN
            );
            simdgroup_multiply_accumulate(c_L, a, b_L, c_L);
            simdgroup_multiply_accumulate(c_R, a, b_R, c_R);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m0 + sg_m_off < M) {
      simdgroup_matrix<T, 8, 8> c_L_T, c_R_T;
      c_L_T.thread_elements()[0] = T(c_L.thread_elements()[0]);
      c_L_T.thread_elements()[1] = T(c_L.thread_elements()[1]);
      c_R_T.thread_elements()[0] = T(c_R.thread_elements()[0]);
      c_R_T.thread_elements()[1] = T(c_R.thread_elements()[1]);
      simdgroup_store(
        c_L_T,
        y + (m0 + sg_m_off) * N + n0 + sg_n_off,
        N
      );
      simdgroup_store(
        c_R_T,
        y + (m0 + sg_m_off) * N + n0 + sg_n_off + 8,
        N
      );
    }
"#;

pub(super) const XLARGE_M_QMM4_SOURCE: &str = r#"
    using namespace metal;
    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int BK_SUB = 8;

    uint tid   = thread_position_in_threadgroup.x;
    uint sg_id = simdgroup_index_in_threadgroup;
    uint tg_m  = threadgroup_position_in_grid.x;
    uint tg_n  = threadgroup_position_in_grid.y;

    int M = int(M_size);
    int K = int(K_size);
    int N = int(N_size);
    int K_by_8  = K / 8;
    int K_by_gs = K / GS;
    int m0 = int(tg_m) * BM;
    int n0 = int(tg_n) * BN;
    int sg_m_off = int(sg_id / 2) * 8;
    int sg_n_off = int(sg_id % 2) * 16;

    threadgroup T B_tile[BK * BN];

    simdgroup_matrix<T, 8, 8> a, b_L, b_R;
    simdgroup_matrix<float, 8, 8> c_L =
      simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_R =
      simdgroup_matrix<float, 8, 8>(0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        if (tid < (BK / 8) * BN) {
            int pack = int(tid);
            int dq_k = pack / BN;
            int dq_n = pack % BN;
            int n_global = n0 + dq_n;
            int k_base = k0 + dq_k * 8;
            uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
            float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
            float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                B_tile[(dq_k * 8 + ki) * BN + dq_n] =
                  T(float(nib) * s + b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ks = 0; ks < BK / BK_SUB; ++ks) {
            simdgroup_load(
              a,
              x + (m0 + sg_m_off) * K + k0 + ks * BK_SUB,
              K
            );
            simdgroup_load(
              b_L, B_tile + ks * BK_SUB * BN + sg_n_off, BN
            );
            simdgroup_load(
              b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8, BN
            );
            simdgroup_multiply_accumulate(c_L, a, b_L, c_L);
            simdgroup_multiply_accumulate(c_R, a, b_R, c_R);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (m0 + sg_m_off < M) {
      simdgroup_matrix<T, 8, 8> c_L_T, c_R_T;
      c_L_T.thread_elements()[0] = T(c_L.thread_elements()[0]);
      c_L_T.thread_elements()[1] = T(c_L.thread_elements()[1]);
      c_R_T.thread_elements()[0] = T(c_R.thread_elements()[0]);
      c_R_T.thread_elements()[1] = T(c_R.thread_elements()[1]);
      simdgroup_store(
        c_L_T,
        y + (m0 + sg_m_off) * N + n0 + sg_n_off,
        N
      );
      simdgroup_store(
        c_R_T,
        y + (m0 + sg_m_off) * N + n0 + sg_n_off + 8,
        N
      );
    }
"#;

pub(super) const SMALL_M_QMV4_SOURCE: &str = r#"
    uint m_tile = threadgroup_position_in_grid.x;
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int K = int(K_size);
    int N = int(N_size);
    constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    int bn = RESULTS_PER_SIMDGROUP * SG;
    int out_row = int(n_tile) * bn + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K / GS;

    const device uint8_t* ws_base =
      (const device uint8_t*)w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
      scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
      biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
      x + int(m_tile) * K + int(simd_lid) * VALUES_PER_THREAD;

    float result[RESULTS_PER_SIMDGROUP] = {0.0f};
    float x_thread[VALUES_PER_THREAD];

    #pragma clang loop unroll_count(4)
    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
      const device T* x_block = x_base + k_block;
      float x_sum = load_vector4_exact<T>(x_block, x_thread);

      const device uint8_t* ws_block =
        ws_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device T* scales_block = scales_base + k_block / GS;
      const device T* biases_block = biases_base + k_block / GS;

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          const device uint8_t* wl = ws_block + row * in_vec_size_w;
          const device T* sl = scales_block + row * in_vec_size_g;
          const device T* bl = biases_block + row * in_vec_size_g;
          float s = float(sl[0]);
          float b = float(bl[0]);

          result[row] += qdot4_exact(wl, x_thread, s, b, x_sum);
        }
      }
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        float sum = simd_sum(result[row]);
        if (simd_lid == 0) {
          y[int(m_tile) * N + n] = T(sum);
        }
      }
    }
"#;

pub(super) const MULTI3_QMV4_SOURCE: &str = r#"
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int K = int(K_size);
    int N = int(N_size);
    constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K / GS;

    const device uint8_t* ws_base =
      (const device uint8_t*)w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
      scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
      biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

    const device T* x0_base = x + int(simd_lid) * VALUES_PER_THREAD;
    const device T* x1_base = x + K + int(simd_lid) * VALUES_PER_THREAD;
    const device T* x2_base = x + 2 * K + int(simd_lid) * VALUES_PER_THREAD;

    float result0[RESULTS_PER_SIMDGROUP] = {0.0f};
    float result1[RESULTS_PER_SIMDGROUP] = {0.0f};
    float result2[RESULTS_PER_SIMDGROUP] = {0.0f};
    float x0_thread[VALUES_PER_THREAD];
    float x1_thread[VALUES_PER_THREAD];
    float x2_thread[VALUES_PER_THREAD];

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* x0 = x0_base;
    const device T* x1 = x1_base;
    const device T* x2 = x2_base;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum0 = load_vector4_exact<T>(x0, x0_thread);
      float sum1 = load_vector4_exact<T>(x1, x1_thread);
      float sum2 = load_vector4_exact<T>(x2, x2_thread);

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          const device uint8_t* wl = ws + row * in_vec_size_w;
          const device T* sl = sc + row * in_vec_size_g;
          const device T* bl = bs + row * in_vec_size_g;
          float s = float(sl[0]);
          float b = float(bl[0]);
          result0[row] += qdot4_exact(wl, x0_thread, s, b, sum0);
          result1[row] += qdot4_exact(wl, x1_thread, s, b, sum1);
          result2[row] += qdot4_exact(wl, x2_thread, s, b, sum2);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      x0 += BLOCK_SIZE;
      x1 += BLOCK_SIZE;
      x2 += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        float r0 = simd_sum(result0[row]);
        float r1 = simd_sum(result1[row]);
        float r2 = simd_sum(result2[row]);
        if (simd_lid == 0) {
          y[n] = T(r0);
          y[N + n] = T(r1);
          y[2 * N + n] = T(r2);
        }
      }
    }
"#;
