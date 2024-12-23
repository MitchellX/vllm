'''
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)
  - https://github.com/vllm-project/vllm/pull/6102/commits
'''

import torch
from vllm.logger import init_logger
from typing import List, Optional, Tuple, Type
import sys

logger = init_logger(__name__)

try:
    import vllm._dattn_C # noqa: F401
except ImportError as e:
    logger.warning("Import dattn error msg: %s", e.msg)

"""
# It seems that there is no need for this function, since we will utilize the 
# same 
# cache device ptr, used for kv cache tensor
class kvCacheRegion:
    def __init__(self):
        self._ptr = torch.classes._dattn_C.kvCacheRegion()
    
    @property
    def reserved_page_num(self):
        return self._ptr.revervedPageNum
    
    @reserved_page_num.setter
    def reserved_page_num(self, value:int):
        self._ptr.reservedPageNum = value
    
    @property
    def allocated_page_num(self):
        return self._ptr.allocatedPageNum
    
    @allocated_page_num.setter
    def allocated_page_num(self, value:int):
        self._ptr.allocatedPageNum = value
"""


# cache allocator based dAttention, used to manage kv cache tensor
class kvCacheAllocator:
    def __init__(self, 
                 max_seq_length, 
                 layers_num, 
                 heads_num, 
                 head_size, 
                 block_size, 
                 dtype_size,
        ):
        self.block_size = block_size
        self._allocator = torch.classes._dattn_C.kvCacheAllocator(max_seq_length, 
                                                                  layers_num,
                                                                  heads_num,
                                                                  head_size, 
                                                                  block_size,
                                                                  dtype_size 
                                                                  )
        #self.page_size = self._allocator.getPageSize()

    
    #def reserve_cache_ptr(self, ptr:CacheDevicePtr, page_num:int = 1):
    def reserve_cache_region(self, req_id:int = 0):
        #print(f"NOOW, in reserve_cache_region, with req_id:{req_id}")
        ptr = self._allocator.reserveRegion(req_id)
        #print(f"NOOW, in reserve_cache_region, with req_id:{req_id}, ptr:{ptr}")
        # TODO: wrap the ptr to a tensor
        #return wrapDptr2Tensor()
        return ptr 

    def alloc_cpu_caches(self, cache_num:int, cache_space_per_req: int):
        #print(f"NOOW, in reserve_cpu_caches, with req_id:{req_id}")
        return self._allocator.allocCPUCaches(cache_num, cache_space_per_req)

    #def alloc_cache_ptr(self, ptr:CacheDevicePtr, page_num:int = 1, offset:int = 0):    
    #def free_cache_ptr(self, ptr:CacheDevicePtr):
    #def release_cache_ptr(self, ptr:CacheDevicePtr, page_num: int = 0, offset: int = 0):
   
    def release_cache_regions(self, free_caches: List[int]):
        self._allocator.releaseRegions(free_caches)
        return 

    def update_cache_blocks(self, 
                            immediate_allocate: bool, 
                            free_caches: List[int], 
                            req_cache_blocks:List[List[int]], 
                            to_swap_out: List[List[int]], 
                            to_swap_in: List[List[int]]):
        return self._allocator.updateCacheBlocks(immediate_allocate, free_caches, req_cache_blocks, to_swap_out, to_swap_in)
     