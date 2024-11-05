/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cuda_fp16.h> 
#include <cuda_bf16.h>
#include "attention_dtypes.h"
#include "attention_utils.cuh"
#include <fstream>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
  #include "../quantization/fp8/amd/quant_utils.cuh"
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include "../quantization/fp8/nvidia/quant_utils.cuh"
#endif

#ifndef USE_ROCM
  #define WARP_SIZE 32
#else
  #define WARP_SIZE warpSize
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

//0930 1 1  000000     loc1722andfunc
///* DATTN_UNIFIED_QK_MAX & DATTN_SHIFT_PERHEAD_QKMAX cannot be on at the same time */
/* DATTN_SHIFT_PERHEAD_QKMAX relies on DATTN_UNIFIED_QK_MAX */
#define DATTN_UNIFIED_QK_MAX 1 // 0:Vanilla 1:Unified qk_max
#define DATTN_SHIFT_PERHEAD_QKMAX 1 // 0:Unified qk_max 1:Shifting window per-head qk_max
#define DATTN_DEBUG_OVERFLOW_ROLLBACK 0 // 1:statistics 0:perf
#define DATTN_DEBUG_OVERFLOW_ROLLBACK_USING_VARIABLE 0
#define VANILLA_DEBUG_PERHEAD_QKMAX 0 // Cannot run - no layer_num info passed in
/* DATTN_DEBUG_PERHEAD_QKMAX & DATTN_UNIFIED_QK_MAX cannot be on at the same time */
#define DATTN_DEBUG_PERHEAD_QK 0
#define DATTN_DEBUG_PERHEAD_QKMAX 0
#define WARNING(msg) printf("\033[33mWARNING: %s\033[0m\n", msg)

#if DATTN_UNIFIED_QK_MAX
//#define DATTENTION_QK_MAX 6.54f // llama2-7B medium 99.99mean // (-0.41+13.49)/2 = 6.54
//#define DATTENTION_QK_MAX 1.73f //  llama2-7B short 99.99mean // (-9.27+12.73)/2 = 1.73
//#define DATTENTION_QK_MAX 12.73f //  llama2-7B short 99.99high
//#define DATTENTION_QK_MAX 13.49f //  llama2-7B medium 99.99high
//#define DATTENTION_QK_MAX 4.58f //  llama2-7B short 99high
//#define DATTENTION_QK_MAX 5.96f //  llama2-7B medium 99high
//#define DATTENTION_QK_MAX -57.545f // patent OPT6.7B short Method3 for short prompts = (-823.94+708.85)/2 = -57.545
//#define DATTENTION_QK_MAX 16.38f // patent OPT6.7B medium Method3 for medium prompts = (-854.23+886.99)/2 = 16.38
//#define DATTENTION_QK_MAX -80.253522f // short phi
// #define DATTENTION_QK_MAX -70.842063f // medium phi

// Based on DATTN_UNIFIED_QK_MAX 1 #define DATTN_SHIFT_PERHEAD_QKMAX 0
//#define DATTENTION_QK_MAX -30.036194f // 1_1 (1+0) $ python jack_multiprocess_find_avg_perhead_allqk.py | tee jack_multiprocess_find_avg_perhead_allqk241007.log
#define DATTENTION_QK_MAX -69.647877 // 2_1 Mix(short+medium) phi

#if DATTN_SHIFT_PERHEAD_QKMAX // per-head
#define MAX_LAYERS 32
#define MAX_HEADS 32
__constant__ __align__(32) float qk_max_values[MAX_LAYERS * MAX_HEADS];
// Flag to ensure set_qk_max_values is only called once
std::once_flag init_flag;
#endif
#endif


#if !defined(likely)
#define likely(x)   __builtin_expect(!!(x), 1)
#endif

#if !defined(unlikely)
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

namespace vllm {

inline __device__ bool is_half_inf(float val) {
    return (val <= -65504.0f || val >= 65504.0f);
}

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

template <int NUM_WARPS, int THREAD_GROUP_SIZE>
inline __device__ float propogate_qk_max(float* red_smem, float qk_max) {
  // Decompose the thread index into warp / lane.
  int warp_idx = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  
  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }

  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
 
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  return qk_max; 
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float k_scale, const float v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  //if(blockIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
  //if(blockIdx.x == 0 && threadIdx.x == 0) {
  //  printf("[%d, %d, %d, %d]: threadblocks %d num_seq_blocks %d USE_PARTITIONING %d PARTITION_SIZE %d start_block_idx %d end_block_idx %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x * gridDim.y * gridDim.z,num_seq_blocks, USE_PARTITIONING, PARTITION_SIZE, start_block_idx, end_block_idx); 
  //}
  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
                                        // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;


  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;

    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in
    // the group has 0, 4, 8, ... th vectors of the key, and the second thread
    // has 1, 5, 9, ... th vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset =
          (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr =
            k_cache + physical_block_number * kv_block_stride +
            kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else {
          // Vector conversion from Quant_vec to K_vec.
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
              k_vec_quant, k_scale);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);
      
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);

#if 0
        /*** This vanilla version of PAcannot print layer_num ***/
        /**** Show qk_max distribution  ****/
        printf("[horenc] %s():%d: <<<grid[%d, %d, %d]block[%d, 0, 0]>>> "
              //"[%d/xxx] "
              "seq_len %d layer_num %02d head_num %02d qk_max %f\n", // %.2f
              __func__, __LINE__, blockIdx.x, seq_idx, partition_idx, threadIdx.x,
              //cnt,
              seq_len, layer_num, head_idx, qk_max);
        /**** Check overflow ****/
        if (logits[i]  > qk_max) {
          printf("[horenc] overflow - logits[i] %f > qk_max %f\n", logits[i], qk_max);
        }
#endif
      }
    }
  }

  qk_max = propogate_qk_max<NUM_WARPS, THREAD_GROUP_SIZE>(&red_smem[0], qk_max);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
#if VANILLA_DEBUG_PERHEAD_QKMAX
    /**** Show qk_max distribution - per-layer, per-head  ****/
    printf("[horenc] %s():%d: <<<grid[%d, %d, %d]block[%d, 0, 0]>>> "
          "lane 0 seq_len %d "
          //"layer_num %02d "
          "head_num %02d qk_max %f\n", // %.2f
          __func__, __LINE__, blockIdx.x, seq_idx, partition_idx, threadIdx.x,
          seq_len,
          //layer_num,
          head_idx, qk_max);
          /* Cannot - no layer_num info passed in */
#endif

    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  //if(blockIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
    //printf("[%d, %d, %d, %d]: gridDim.x-%d, gridDim.y-%d,gridDim.z-%d, blockDim.x-%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x, gridDim.y, gridDim.z, blockDim.x); 
    //int tbs = gridDim.x * gridDim.y * gridDim.z;
    //printf("threadsblocks-%d\n", tbs);  
    //printf(" threadBlocks-%d: gridDim.x-%d, gridDim.y-%d,gridDim.z-%d, blockDim.x-%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x, gridDim.y, gridDim.z, blockDim.x); 
  //}
  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
          !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
        continue;
      }
    }
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx));
    
    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          // Vector conversion from V_quant_vec to V_vec.
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    v_scale);
        }
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];

        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float k_scale, const float v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache,
      v_cache, num_kv_heads, scale, block_tables, seq_lens,
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,
      kv_head_stride, k_scale, v_scale, tp_rank, blocksparse_local_blocks,
      blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, vllm::Fp8KVCacheDataType KV_DTYPE,
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,       // [num_seqs, num_heads,
                                          // max_num_partitions]
    scalar_t* __restrict__ tmp_out,       // [num_seqs, num_heads,
                                          // max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float k_scale, const float v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) {
  paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS,
                         KV_DTYPE, IS_BLOCK_SPARSE, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, seq_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
      kv_block_stride, kv_head_stride, k_scale, v_scale, tp_rank,
      blocksparse_local_blocks, blocksparse_vert_stride, blocksparse_block_size,
      blocksparse_head_sliding_step);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int seq_len = seq_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr =
        out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();

  //if(blockIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
    //printf("[%d, %d, %d, %d]: gridDim.x-%d, gridDim.y-%d,gridDim.z-%d, blockDim.x-%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x, gridDim.y, gridDim.z, blockDim.x); 
  //int tbs = gridDim.x * gridDim.y * gridDim.z;
  //  printf("reduced threadsblocks-%d\n", tbs);  
    //printf(" threadBlocks-%d: gridDim.x-%d, gridDim.y-%d,gridDim.z-%d, blockDim.x-%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x, gridDim.y, gridDim.z, blockDim.x); 
  //}
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums =
      reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums +
                              seq_idx * num_heads * max_num_partitions +
                              head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr =
      out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] *
             inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType KV_DTYPE, 
         int BLOCK_SIZE, int HEAD_SIZE,
         int NUM_THREADS, int PARTITION_SIZE = 0>   
__global__ void dattention_kernel(
  float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
  float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
  scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions, head_size]
  scalar_t* __restrict__ q, // [num_seqs, num_heads, head_size]
  int64_t layer_offset,        // layer offset in the units
  int64_t whole_block_size,    // whole block size (bytes), including KV of all layers together
  int64_t max_seq_len,
  const int64_t* cache_row_mapping,  // [num_tokens]  record cache ptr for this token
  const int64_t* cache_col_mapping,  // [num_tokens]  record token index of the sequence
  const int* __restrict__ seq_lens,      // [num_seqs]
  const int64_t q_stride, 
  const int64_t num_kv_heads,               // [num_heads]
  const float scale,
  const float* __restrict__ alibi_slopes,  // [num_heads]
  const float k_scale,
  const float v_scale,
  uint64_t *h_counter_array // rollback counter perthread
) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = seq_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

#if DATTN_UNIFIED_QK_MAX
  __shared__ bool recompute;
  recompute = false;
#endif

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  //if(blockIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
  //if(blockIdx.x == 0 && threadIdx.x == 0) {
  //  printf("[%d, %d, %d, %d]: threadblocks %d num_seq_blocks %d USE_PARTITIONING %d PARTITION_SIZE %d start_block_idx %d end_block_idx %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x * gridDim.y * gridDim.z,num_seq_blocks, USE_PARTITIONING, PARTITION_SIZE, start_block_idx, end_block_idx); 
    //printf("[%d, %d, %d, %d]: threadblocks %d USE_PARTITIONING %d PARTITION_SIZE %d start_block_idx %d end_block_idx %d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x * gridDim.y * gridDim.z, USE_PARTITIONING, PARTITION_SIZE, start_block_idx, end_block_idx); 
  //}
   // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
                                        // divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

#if (DATTN_UNIFIED_QK_MAX || DATTN_DEBUG_PERHEAD_QK || DATTN_DEBUG_PERHEAD_QKMAX)
  int64_t layer_idx = layer_offset/(num_heads * HEAD_SIZE * BLOCK_SIZE);
#endif

#if DATTN_DEBUG_OVERFLOW_ROLLBACK
  static uint64_t local_rollback = 0;
#endif
#if DATTN_UNIFIED_QK_MAX
#if !DATTN_SHIFT_PERHEAD_QKMAX
  float dattn_qkmax = DATTENTION_QK_MAX;
#else // use per-head info
  int dattn_qkmax_idx = (layer_idx * num_heads) + head_idx;
  float dattn_qkmax = qk_max_values[dattn_qkmax_idx];
#endif
#endif

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread
  // group fetch or compute 16 bytes at a time. For example, if the size of a
  // thread group is 4 and the data type is half, then the vector size is 16 /
  // (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int KV_HEAD_STRIDE = HEAD_SIZE * BLOCK_SIZE; 
  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  //if(blockIdx.x ==0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
    //printf("[%d, %d, %d, %d]: gridDim.x-%d, gridDim.y-%d,gridDim.z-%d, blockDim.x-%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x, gridDim.y, gridDim.z, blockDim.x); 
    //int tbs = gridDim.x * gridDim.y * gridDim.z;
    //printf("threadsblocks-%d\n", tbs);  
    //printf(" threadBlocks-%d: gridDim.x-%d, gridDim.y-%d,gridDim.z-%d, blockDim.x-%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, gridDim.x, gridDim.y, gridDim.z, blockDim.x); 
  //}
  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in
  // the group has 0, 4, 8, ... th vectors of the query, and the second thread
  // has 1, 5, 9, ... th vectors of the query, and so on. NOTE(woosuk): Because
  // q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD;
       i += NUM_THREAD_GROUPS) {
    
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;

    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a
                    // memory wall right before we use q_vecs
  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

#if 0
  // blocksparse specific vars
  int bs_block_offset;
  int q_bs_block_id;
  if constexpr (IS_BLOCK_SPARSE) {
    // const int num_blocksparse_blocks = DIVIDE_ROUND_UP(seq_len,
    // blocksparse_block_size);
    q_bs_block_id = (seq_len - 1) / blocksparse_block_size;
    if (blocksparse_head_sliding_step >= 0)
      // sliding on q heads
      bs_block_offset =
          (tp_rank * num_heads + head_idx) * blocksparse_head_sliding_step + 1;
    else
      // sliding on kv heads
      bs_block_offset = (tp_rank * num_kv_heads + kv_head_idx) *
                            (-blocksparse_head_sliding_step) +
                        1;
  }
#endif

  // NOTE: cache_row_idx or cache_col_idx can be -1 if the token is padded
  cache_t * cache_start = reinterpret_cast<cache_t *>(cache_row_mapping[seq_idx]);

  // Iterate over the key blocks.
  // Each thread block will process one request's one head and one partition (up to 512 tokens)
  // Each warp will process a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes dot product with the query.
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
#if 0
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    if constexpr (IS_BLOCK_SPARSE) {
      const int k_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      const bool is_remote =
          ((k_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0);
      const bool is_local =
          (k_bs_block_id > q_bs_block_id - blocksparse_local_blocks);
      if (!is_remote && !is_local) {
        for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
          const int physical_block_offset =
              (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
          const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

          if (thread_group_offset == 0) {
            // NOTE(linxihui): assign very large number to skipped tokens to
            // avoid contribution to the sumexp softmax normalizer. This will
            // not be used at computing sum(softmax*v) as the blocks will be
            // skipped.
            logits[token_idx - start_token_idx] = -FLT_MAX;
          }
        }
        continue;
      }
    }
#endif
    // computing the starting address of the block for the given layer
    cache_t * key_cache = cache_start + block_idx*whole_block_size + layer_offset;

    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      // Load a key to registers. Inside a block, each thread group will fetch lane/THREAD_GROUP_SIZe
      // Each thread in a thread group has a different part of the key.
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE; // token index inside the block
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

      cache_t* k_ptr = key_cache + kv_head_idx * KV_HEAD_STRIDE + physical_block_offset * x;

    #pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else {
          // Vector conversion from Quant_vec to K_vec.
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
              k_vec_quant, k_scale);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(
                             q_vecs[thread_group_offset], k_vecs);

      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);

#if DATTN_DEBUG_PERHEAD_QK
        /* Show qk distribution - per-layer, per-head */
        if (!mask) {
          printf("[horenc] %s():%d: "
                "<<<grid[%02d, %d, %d]block[%02d, 0, 0]>>> "
                "lane 0 seq_len %d layer_idx %02" PRId64 " head_num %02d qk %f\n", // %.2f
                __func__, __LINE__,
                blockIdx.x, seq_idx, partition_idx, threadIdx.x,
                seq_len, layer_idx, head_idx, qk);
        } else {
          // qk always = 0;
        }
#endif

#if DATTN_UNIFIED_QK_MAX
        // may skip mask != 0?
// #if !DATTN_SHIFT_PERHEAD_QKMAX
//         dattn_qkmax = DATTENTION_QK_MAX;
// #else // use per-head info
//         int dattn_qkmax_idx = (layer_idx * num_heads) + head_idx;
//         dattn_qkmax = qk_max_values[dattn_qkmax_idx];
//         // Debug
//         // printf("AAA[%d]= %f [(%02" PRId64 " * %02d) + %02d]\n", dattn_qkmax_idx, dattn_qkmax,
//         //       layer_idx, num_heads, head_idx);
//         // Debug - per-head
//         // if (layer_idx == 2 && head_idx == 8) // 0~31
//         //   printf("DDDmax[%d]= %f [(%02" PRId64 " * %02d) + %02d]\n", dattn_qkmax_idx, dattn_qkmax,
//         //        layer_idx, num_heads, head_idx);
//         // if (layer_idx == 3 && head_idx == 29) // 0~31
//         //   printf("DDDmin[%d]= %f [(%02" PRId64 " * %02d) + %02d]\n", dattn_qkmax_idx, dattn_qkmax,
//         //        layer_idx, num_heads, head_idx);
// #endif
        /***** debug ******/
        //if (unlikely(is_half_inf(__expf(qk - DATTENTION_QK_MAX)))) {
        // if (unlikely(qk >=11.0903f || qk <= -11.0903f)) {
        //   printf("qk_max causes float16 overflow!!\n");
        // }
        // #define DATTENTION_QK_MAX 1.73f //  llama2-7B short 99.99mean
        float upper_bound = 88.72283f, lower_bound = -87.33654f;
        //float upper_bound = 50.0f, lower_bound = -87.33654f;
        if (unlikely((qk - dattn_qkmax) >= upper_bound ||
            (qk - dattn_qkmax) <= lower_bound)) {
          if ((qk - dattn_qkmax) >= upper_bound) {
            /* Rollback */
            recompute = true;
          }
          // WARNING("qk_max causes float32 overflow!!");
          // printf("\033[34mDEBUG: qk_max overflow!! qk %.6f - dattn_qkmax %.6f = %.6f\033[0m\n",
          //       qk, dattn_qkmax, qk - dattn_qkmax);
#if DATTN_DEBUG_OVERFLOW_ROLLBACK
          // Detailed statistics
          if (unlikely((qk - dattn_qkmax) >= 88.72283f &&
              (qk - dattn_qkmax) <= -87.33654f)) {
            printf("\033[36mDEBUG: qk_max overflow!! bothflow qk %.6f - dattn_qkmax %.6f = %.6f "
                  "tid %d seq_idx %d layer_idx %02" PRId64 " head_idx %02d\033[0m\n",
                  qk, dattn_qkmax, qk - dattn_qkmax,
                  threadIdx.x, seq_len, seq_idx, layer_idx, head_idx);
          } else if (unlikely((qk - dattn_qkmax) >= 88.72283f)) {
            printf("\033[35mDEBUG: qk_max overflow!! overflow qk %.6f - dattn_qkmax %.6f = %.6f "
                  "tid %d seq_idx %d layer_idx %02" PRId64 " head_idx %02d\033[0m\n",
                  qk, dattn_qkmax, qk - dattn_qkmax,
                  threadIdx.x, seq_len, seq_idx, layer_idx, head_idx);
          } else if (unlikely((qk - dattn_qkmax) <= -87.33654f)) {
            printf("\033[34mDEBUG: qk_max overflow!! underflow qk %.6f - dattn_qkmax %.6f = %.6f "
                  "tid %d seq_idx %d layer_idx %02" PRId64 " head_idx %02d\033[0m\n",
                  qk, dattn_qkmax, qk - dattn_qkmax,
                  threadIdx.x, seq_len, seq_idx, layer_idx, head_idx);
          }
#endif
        } else {
          // Cannot pollute qk_max. Rollback needs it. other threads may trigger it.
        }
#else // vanilla
        // recompute = true;
#endif
      }

    // within warp PA thread group
      //if(to_profile2) {
      //  time2 = clock64();
      //  fourthTime += time2 - time1; 
      //  time1 = time2; 
      //}

      // within warp PA thread group
    }
  }
  
  if(to_profile2) {
    time2 = clock64();
    newTime += time2 - time0; 
    time1 = time2; 
  } 
  
  __syncthreads(); // this works but not very sure why yet

#if DATTN_UNIFIED_QK_MAX // new
  // Multiple thread blocks/warps here
  __syncthreads(); // Introduced by our mechanism

  if (likely(recompute == false)) {
    /* Set unified qk_max value */
    // Do it here because not only thread group leader needs this value.
// #if !DATTN_SHIFT_PERHEAD_QKMAX
//     // qk_max = dattn_qkmax; // TODO test (clean code)
//     // dattn_qkmax = DATTENTION_QK_MAX;
//     qk_max = DATTENTION_QK_MAX;
// #else
// //    int dattn_qkmax_idx = (layer_idx * num_heads) + head_idx;
// //    dattn_qkmax = qk_max_values[dattn_qkmax_idx]; // not all the threads in a thread group have set dattn_qkmax!!
//     qk_max = dattn_qkmax;
// #endif
    qk_max = dattn_qkmax; // Have already set the value to use
  } else { // if (unlikely(recompute)) {
    /* Someone overflowed */
    // Perform reduction across all threads in the same thread block
    qk_max = propogate_qk_max<NUM_WARPS, THREAD_GROUP_SIZE>(&red_smem[0], qk_max);
    //if(threadIdx.x == 0) {
    //  printf("[%d, %d, %d]: scale %f qk_max %f. layer_offset %ld, kv_head_stride %d - %d. q_stride %ld\n", blockIdx.x, blockIdx.y, threadIdx.x, scale, qk_max, layer_offset, KV_HEAD_STRIDE, kv_head_stride, q_stride);
    //}
#if DATTN_DEBUG_OVERFLOW_ROLLBACK_USING_VARIABLE
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    h_counter_array[idx] += 1;  // 計數器 +1

#endif
#if DATTN_DEBUG_OVERFLOW_ROLLBACK
    local_rollback++;
    if (!threadIdx.x) { // 0~127, 16*4 tokens
      printf("\033[33mWARNING: local_rollback = %" PRIu64 " tid %d + seq_len %d + seq_idx %d per warp layer_idx %02" PRId64 " head_idx %02d\033[0m\n",
          local_rollback, threadIdx.x, seq_len, seq_idx, layer_idx, head_idx);
          //num_heads + head_idx
    }
#endif
  }
#else // vanilla
  // Perform reduction across all threads in the same thread block
  qk_max = propogate_qk_max<NUM_WARPS, THREAD_GROUP_SIZE>(&red_smem[0], qk_max);
  //if(threadIdx.x == 0) {
  //  printf("[%d, %d, %d]: scale %f qk_max %f. layer_offset %ld, kv_head_stride %d - %d. q_stride %ld\n", blockIdx.x, blockIdx.y, threadIdx.x, scale, qk_max, layer_offset, KV_HEAD_STRIDE, kv_head_stride, q_stride);
  //}
#endif

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
#if DATTN_DEBUG_PERHEAD_QKMAX
    /* Show qk_max distribution - per-layer, per-head */
    if (!i) {
      printf("[horenc] %s():%d: qk_max "
            "<<<grid[%02d, %d, %d]block[%02d, 0, 0]>>> "
            "lane 0 seq_len %d layer_idx %02" PRId64 " head_num %02d qk_max %f\n", // %.2f
            __func__, __LINE__,
            blockIdx.x, seq_idx, partition_idx, threadIdx.x,
            seq_len, layer_idx, head_idx, qk_max);
            // threadIdx.x is within the range of a warp (32)
    }
#endif

    const float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits +
                            seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions +
                          head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD =
      DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
#if 0
    if constexpr (IS_BLOCK_SPARSE) {
      int v_bs_block_id = block_idx * BLOCK_SIZE / blocksparse_block_size;
      if (!((v_bs_block_id + bs_block_offset) % blocksparse_vert_stride == 0) &&
          !((v_bs_block_id > q_bs_block_id - blocksparse_local_blocks))) {
        continue;
      }
    }
#endif

    // Load a key to registers. Inside a block, each thread group will fetch lane/THREAD_GROUP_SIZe
    // Each thread in a thread group has a different part of the key.    
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx));

    // computing the starting address of the block
    cache_t* v_ptr = cache_start + block_idx*whole_block_size + whole_block_size/2 + layer_offset + kv_head_idx * KV_HEAD_STRIDE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          V_quant_vec v_quant_vec =
              *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          // Vector conversion from V_quant_vec to V_vec.
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec,
                                                                    v_scale);
        }
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for
  // logits is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {

    scalar_t* out_ptr =
        out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        //printf("[%d, %d, %d]: seq_idx-%d, row_idx-%d, accs[%d]:%f\n",blockIdx.x, blockIdx.y, threadIdx.x,seq_idx,row_idx,i,accs[i]); 
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}
}  // namespace vllm

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(                     \
      ((void*)vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,      \
                                              KV_DTYPE, IS_BLOCK_SPARSE>),  \
      shared_mem_size);                                                     \
  vllm::paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE>   \
      <<<grid, block, shared_mem_size, stream>>>(                           \
          out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, \
          scale, block_tables_ptr, seq_lens_ptr, max_num_blocks_per_seq,    \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,      \
          k_scale, v_scale, tp_rank, blocksparse_local_blocks,              \
          blocksparse_vert_stride, blocksparse_block_size,                  \
          blocksparse_head_sliding_step);

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, float k_scale,
    float v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  [[maybe_unused]] int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len =
      DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  
  fprintf(stderr, "thread_blocks %ld num_headds %ld num_seqs %ld max_seq_len %ld\n", num_heads*num_seqs, num_heads, num_seqs, max_seq_len);
  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 120:
      LAUNCH_PAGED_ATTENTION_V1(120);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)  \
  paged_attention_v1_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,              \
                              IS_BLOCK_SPARSE>(                              \
      out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, \
      seq_lens, max_seq_len, alibi_slopes, k_scale, v_scale, tp_rank,        \
      blocksparse_local_blocks, blocksparse_vert_stride,                     \
      blocksparse_block_size, blocksparse_head_sliding_step);

#define CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  switch (is_block_sparse) {                                               \
    case true:                                                             \
      CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);     \
      break;                                                               \
    case false:                                                            \
      CALL_V1_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);    \
      break;                                                               \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V1_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 8:                                                       \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 8, KV_DTYPE);         \
      break;                                                      \
    case 16:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    case 32:                                                      \
      CALL_V1_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

void paged_attention_v1(
    torch::Tensor& out,    // [num_seqs, num_heads, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const int64_t tp_rank, const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);
  
  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V1_LAUNCHER_BLOCK_SIZE)
}

#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                   \
  vllm::paged_attention_v2_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,           \
                                  NUM_THREADS, KV_DTYPE, IS_BLOCK_SPARSE,      \
                                  PARTITION_SIZE>                              \
      <<<grid, block, shared_mem_size, stream>>>(                              \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr, key_cache_ptr, \
          value_cache_ptr, num_kv_heads, scale, block_tables_ptr,              \
          seq_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,    \
          kv_block_stride, kv_head_stride, k_scale, v_scale, tp_rank,          \
          blocksparse_local_blocks, blocksparse_vert_stride,                   \
          blocksparse_block_size, blocksparse_head_sliding_step);              \
  vllm::paged_attention_v2_reduce_kernel<T, HEAD_SIZE, NUM_THREADS,            \
                                         PARTITION_SIZE>                       \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr,    \
          max_num_partitions);

template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& seq_lens, int max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes, float k_scale,
    float v_scale, const int tp_rank, const int blocksparse_local_blocks,
    const int blocksparse_vert_stride, const int blocksparse_block_size,
    const int blocksparse_head_sliding_step) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  [[maybe_unused]] int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  CACHE_T* key_cache_ptr = reinterpret_cast<CACHE_T*>(key_cache.data_ptr());
  CACHE_T* value_cache_ptr = reinterpret_cast<CACHE_T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* seq_lens_ptr = seq_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  // For paged attention v2 kernel.
  fprintf(stderr, "thread_blocks %ld num_headds %ld num_seqs %ld max_num_partitions %ld max_seq_len %ld\n", num_heads*num_seqs*max_num_partitions , num_heads, num_seqs, max_num_partitions, max_seq_len);
  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);

  fprintf(stderr, "reduced thread_blocks %ld \n", num_heads*num_seqs);
  // For paged attention v2 reduce kernel.
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V2(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2(112);
      break;
    case 120:
      LAUNCH_PAGED_ATTENTION_V2(120);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V2(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }
}

#define CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, KV_DTYPE, IS_BLOCK_SPARSE)   \
  paged_attention_v2_launcher<T, CACHE_T, BLOCK_SIZE, KV_DTYPE,               \
                              IS_BLOCK_SPARSE>(                               \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,      \
      num_kv_heads, scale, block_tables, seq_lens, max_seq_len, alibi_slopes, \
      k_scale, v_scale, tp_rank, blocksparse_local_blocks,                    \
      blocksparse_vert_stride, blocksparse_block_size,                        \
      blocksparse_head_sliding_step);

#define CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE) \
  switch (is_block_sparse) {                                               \
    case true:                                                             \
      CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, true);     \
      break;                                                               \
    case false:                                                            \
      CALL_V2_LAUNCHER(T, CACHE_T, BLOCK_SIZE, IS_FP8_KV_CACHE, false);    \
      break;                                                               \
  }

// NOTE(woosuk): To reduce the compilation time, we omitted block sizes
// 1, 2, 4, 64, 128, 256.
#define CALL_V2_LAUNCHER_BLOCK_SIZE(T, CACHE_T, KV_DTYPE)         \
  switch (block_size) {                                           \
    case 8:                                                       \
      CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, 8, KV_DTYPE);         \
      break;                                                      \
    case 16:                                                      \
      CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, 16, KV_DTYPE);        \
      break;                                                      \
    case 32:                                                      \
      CALL_V2_LAUNCHER_SPARSITY(T, CACHE_T, 32, KV_DTYPE);        \
      break;                                                      \
    default:                                                      \
      TORCH_CHECK(false, "Unsupported block size: ", block_size); \
      break;                                                      \
  }

void paged_attention_v2(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor&
        tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,       // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,  // [num_heads]
    double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,      // [num_seqs]
    int64_t block_size, int64_t max_seq_len,
    const c10::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, double k_scale, double v_scale,
    const int64_t tp_rank, const int64_t blocksparse_local_blocks,
    const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size,
    const int64_t blocksparse_head_sliding_step) {
  const bool is_block_sparse = (blocksparse_vert_stride > 1);
  DISPATCH_BY_KV_CACHE_DTYPE(query.dtype(), kv_cache_dtype,
                             CALL_V2_LAUNCHER_BLOCK_SIZE)
}

#define LAUNCH_DATTENTION(HEAD_SIZE)   \
  if(use_reduce) { \
    VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize( \
      ((void*)vllm::dattention_kernel<scalar_t, cache_t, KV_DTYPE, BLOCK_SIZE, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>), \
      shared_mem_size);  \
    vllm::dattention_kernel<scalar_t, cache_t, KV_DTYPE, BLOCK_SIZE, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE> \
        <<<grid, block, shared_mem_size, stream>>>( \
              exp_sums_ptr, max_logits_ptr, tmp_out_ptr,\
              query_ptr, \
              layer_offset, \
              whole_block_size, max_seq_len, \
              row_ptr, \
              col_ptr, \
              seq_lens_ptr, \
              q_stride, num_kv_heads, scale,  \
              alibi_slopes_ptr, k_scale, v_scale, h_counter_array);  \
    vllm::paged_attention_v2_reduce_kernel<cache_t, HEAD_SIZE, NUM_THREADS,     \
                                         PARTITION_SIZE>                       \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, seq_lens_ptr,    \
          max_num_partitions); \
  } \
  else { \
    VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize( \
      ((void*)vllm::dattention_kernel<scalar_t, cache_t, KV_DTYPE, BLOCK_SIZE, HEAD_SIZE, NUM_THREADS>), \
      shared_mem_size);  \
    vllm::dattention_kernel<scalar_t, cache_t, KV_DTYPE, BLOCK_SIZE, HEAD_SIZE, NUM_THREADS> \
        <<<grid, block, shared_mem_size, stream>>>( \
              nullptr, nullptr, out_ptr,\
              query_ptr, \
              layer_offset, \
              whole_block_size, max_seq_len, \
              row_ptr, \
              col_ptr, \
              seq_lens_ptr, \
              q_stride, num_kv_heads, scale,  \
              alibi_slopes_ptr, k_scale, v_scale, h_counter_array);  \
   }


#if DATTN_SHIFT_PERHEAD_QKMAX
// Function to read qk_max values from a file and copy them to the constant memory array
void set_qk_max_values(const std::string& filename) {
    // Static variable to ensure initialization happens only once
    static bool is_initialized = false;

    printf("\033[33mDebug: per-head init called once\033[0m\n");

    // Check if already initialized
    if (is_initialized) {
        return;  // Skip initialization if already done
    }
    printf("[horenc] Should appear only ONE time (how many threads init qk_max)\n");

    // Create a host vector to hold qk_max values for each layer and head
    std::vector<float> qk_max_values_host(MAX_LAYERS * MAX_HEADS);

    // Open the file
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Read values from the file into the host array
    for (int i = 0; i < MAX_LAYERS * MAX_HEADS; ++i) {
        if (!(infile >> qk_max_values_host[i])) {
            std::cerr << "Error: Insufficient values in file, expecting "
                      << MAX_LAYERS * MAX_HEADS << " values." << std::endl;
            return;
        }
    }

    // Close the file
    infile.close();

    // Copy data from host memory to device constant memory
    cudaMemcpyToSymbol(qk_max_values, qk_max_values_host.data(), qk_max_values_host.size() * sizeof(float));
}

// Wrapper function to ensure set_qk_max_values is only called once
void initialize_qk_max_values(const std::string& filename) {
    std::call_once(init_flag, set_qk_max_values, filename);
}
#endif

template <typename scalar_t, typename cache_t, vllm::Fp8KVCacheDataType KV_DTYPE, 
          int BLOCK_SIZE, int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void dattention_launcher(
  torch::Tensor& output,    // [num_seqs, num_heads, head_size]
  torch::Tensor& exp_sums, 
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,     // [num_seqs, num_heads, head_size]
  bool use_reduce, 
  int64_t layer_idx,
  int64_t num_layers, 
  int64_t max_seq_len, 
  torch::Tensor & seq_lens,
  torch::Tensor & cache_row_mapping, 
  torch::Tensor & cache_col_mapping,  
  int64_t num_kv_heads,
  double  scale,
  const c10::optional<torch::Tensor>&  alibi_slopes,
  double k_scale,
  double v_scale 
) {
  int64_t num_seqs = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t head_size = query.size(2);

  int max_num_partitions = 1; 
  if(use_reduce)
	max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);

  int64_t key_block_size = (num_heads * head_size) * BLOCK_SIZE;
  int64_t layer_block_size = key_block_size * 2;  
  int64_t whole_block_size = layer_block_size * num_layers;  
  int64_t layer_offset = layer_idx * key_block_size; 

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  scalar_t* out_ptr = reinterpret_cast<scalar_t*>(output.data_ptr());
  float* exp_sums_ptr = nullptr;
  float* max_logits_ptr = nullptr; 
  scalar_t* tmp_out_ptr = nullptr;
  if(use_reduce) {
    exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
    max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
    tmp_out_ptr = reinterpret_cast<scalar_t*>(tmp_out.data_ptr());
  }
  
  scalar_t* query_ptr = reinterpret_cast<scalar_t*>(query.data_ptr());
  int* seq_lens_ptr = seq_lens.data_ptr<int>();
  int64_t * row_ptr = reinterpret_cast<int64_t*>(cache_row_mapping.data_ptr());
  int64_t * col_ptr = reinterpret_cast<int64_t*>(cache_col_mapping.data_ptr());

  int64_t q_stride = query.stride(0);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

  int logits_size;
  if(use_reduce) {
    logits_size = PARTITION_SIZE * sizeof(float);
  }
  else {
    int padded_max_seq_len =
      DIVIDE_ROUND_UP(max_seq_len, BLOCK_SIZE) * BLOCK_SIZE;
    logits_size = padded_max_seq_len * sizeof(float);
  }

  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

#if DATTN_SHIFT_PERHEAD_QKMAX
  // call once
  // qk_max_hardcoded_values_medium.txt
  // qk_max_hardcoded_values_short.txt
  // phi_hardcoded_values_short_medium_summary_240930.txt
  //initialize_qk_max_values("qk_max_hardcoded_values_medium.txt");
  //initialize_qk_max_values("qk_max_hardcoded_values_short.txt");
  //initialize_qk_max_values("phi_hardcoded_values_short_medium_summary_240930.txt"); // = best_phi_values_short_medium_summary_241006.txt
  initialize_qk_max_values("best_phi_values_short_medium_summary_241006.txt"); // 2-2 // = phi_hardcoded_values_short_medium_summary_240930.txt
  //initialize_qk_max_values("avg_phi_values_short_medium_summary_241006.txt"); // 1-2
  //initialize_qk_max_values("1_1_in_a_file_241024.txt"); // write 1_1 into a file
#endif

  dim3 grid(num_heads, num_seqs, max_num_partitions);

  fprintf(stderr, "thread_blocks %ld num_headds %ld num_seqs %ld max_num_partitions %ld max_seq_len %ld\n", num_heads*num_seqs*max_num_partitions , num_heads, num_seqs, max_num_partitions, max_seq_len);

  // each thread block will be 128 threads
  dim3 block(NUM_THREADS);


  if(use_reduce)
    fprintf(stderr, "reduced thread_blocks %ld \n", num_heads*num_seqs);

  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  uint64_t *h_counter_array = NULL;
#if DATTN_DEBUG_OVERFLOW_ROLLBACK_USING_VARIABLE // rollback counter
  uint64_t *d_counter_array;
  h_counter_array = (uint64_t*)malloc(NUM_THREADS * sizeof(uint64_t));
  // alloc memory
  cudaMalloc(&d_counter_array, NUM_THREADS * sizeof(uint64_t));
  cudaMemset(d_counter_array, 0, NUM_THREADS * sizeof(uint64_t));  // init to 0
#endif

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_DATTENTION(64);
      break;
    case 80:
      LAUNCH_DATTENTION(80);
      break;
    case 96:
      LAUNCH_DATTENTION(96);
      break;
    case 112:
      LAUNCH_DATTENTION(112);
      break;
    case 128:
      LAUNCH_DATTENTION(128);
      break;
    case 192:
      LAUNCH_DATTENTION(192);
      break;
    case 256:
      LAUNCH_DATTENTION(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head size: ", head_size);
      break;
  }  

  if(to_profile) {
    cudaDeviceSynchronize();
    end_time = std::chrono::high_resolution_clock::now();
    total += end_time - start_time; 
    if(step_index % 512 == 0) {
      fprintf(stderr, "step_index-%ld time %f\n", step_index, total);  
      total = std::chrono::duration<double, std::milli>(0); 
    }
  }

#if DATTN_DEBUG_OVERFLOW_ROLLBACK_USING_VARIABLE // rollback counter
  // TODO: I don't know when is done..... I cannot do&free it all at once
  // copy results from GPU to host
  cudaMemcpy(h_counter_array, d_counter_array, NUM_THREADS * sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // accumulate all threads' counter
  uint64_t total_count = 0;
  for (int i = 0; i < NUM_THREADS; i++) {
      printf("t[%d] count: %lu\n", i, h_counter_array[i]);
      total_count += h_counter_array[i];
  }

  // print result
  printf("Total count: %lu\n", total_count);

  // free memory
  cudaFree(d_counter_array);
  free(h_counter_array);
#endif
}

void dattention(
    torch::Tensor& output,    // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& tmp_out,    // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,     // [num_seqs, num_heads, head_size]
    bool use_reduce, 
    int64_t layer_idx,
    int64_t num_layers, 
    int64_t block_size,
    int64_t max_seq_len, 
    torch::Tensor & seq_lens,
    torch::Tensor & cache_row_mapping, 
    torch::Tensor & cache_col_mapping,  
    const std::string& kv_cache_dtype,
    int64_t num_kv_heads,
    double scale,
    const c10::optional<torch::Tensor>&  alibi_slopes,
    double k_scale,
    double v_scale 
  ) {
  assert(block_size == 16 || block_size == 32);

  if (kv_cache_dtype == "auto" && block_size == 16) {                                                 
    if (query.dtype() == at::ScalarType::Float) {               
      dattention_launcher<float, float, vllm::Fp8KVCacheDataType::kAuto, 16>( 
          output, exp_sums, max_logits, tmp_out, query, use_reduce, layer_idx, num_layers, max_seq_len,  
          seq_lens, cache_row_mapping, cache_col_mapping, 
          num_kv_heads, scale, alibi_slopes, k_scale, v_scale);
    } else if (query.dtype() == at::ScalarType::Half) {
        dattention_launcher<uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto, 16>( 
          output, exp_sums, max_logits, tmp_out, query, use_reduce, layer_idx, num_layers, max_seq_len,  
          seq_lens, cache_row_mapping, cache_col_mapping, 
          num_kv_heads, scale, alibi_slopes, k_scale, v_scale); 
    } 
  }
  else if (kv_cache_dtype == "auto" && block_size == 32) {                                                 
    if (query.dtype() == at::ScalarType::Float) {               
      dattention_launcher<float, float, vllm::Fp8KVCacheDataType::kAuto, 32>( 
          output, exp_sums, max_logits, tmp_out, query, use_reduce, layer_idx, num_layers, max_seq_len,  
          seq_lens, cache_row_mapping, cache_col_mapping, 
          num_kv_heads, scale, alibi_slopes, k_scale, v_scale);
    } else if (query.dtype() == at::ScalarType::Half) {
        dattention_launcher<uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto, 32>( 
          output, exp_sums, max_logits, tmp_out, query, use_reduce, layer_idx, num_layers, max_seq_len,  
          seq_lens, cache_row_mapping, cache_col_mapping, 
          num_kv_heads, scale, alibi_slopes, k_scale, v_scale); 
    } 
  }
  else {                     
    printf("errors for dattention_launcher: dtype: %s, block_size %ld!!\n", kv_cache_dtype.c_str(), block_size);
    exit(0);
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP