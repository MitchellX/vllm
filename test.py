import torch
from vllm import _dattn_ops as dattn

max_gpu_memory_size=8824815616
cache_block_size=8388608
region_cache_size=1241513984
print(8824815616 > 40 * 8388608)

# 创建一个 kvCacheAllocator 实例（参数按实际情况填写）
allocator = dattn.kvCacheAllocator(max_gpu_memory_size, cache_block_size, region_cache_size)

# 假设你已经通过 reserveRegion 或其他方式为某个请求分配了 GPU 缓存，
# 并且你知道对应的 gpu_cache_id、start_block 和需要的 block 数量
gpu_cache_id = 0         # 举例
start_block = 0          # 举例
need_blocks = 4          # 举例

# 调用 copyKVCache 进行 GPU->CPU 拷贝，注意输出会在标准输出中打印
result = allocator.copyKVCache(gpu_cache_id, start_block, need_blocks, "GPU2CPU")
print("copyKVCache result:", result)