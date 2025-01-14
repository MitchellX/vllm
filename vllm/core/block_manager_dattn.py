'''
 Copyright (c) ByteDance Inc.
 Authors: 
  - Tongping Liu (tongping.liu@bytedance.com)

  This file will manage blocks and cache ids for both CPU and GPU memory. 
  However, the address related to each cache id will be tracked and managed by CacheEngineDattn

  Adopted from https://github.com/vllm-project/vllm/pull/6102/commits
'''
from collections import deque
from typing import Dict, List, Optional, Tuple

from vllm.core.block.utils import check_no_caching_or_swa_for_blockmgr_encdec
from vllm.core.evictor_v1 import EvictionPolicy, Evictor, make_evictor
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device, Counter
from collections import deque
from dataclasses import dataclass, field
import sys
import torch

logger = init_logger(__name__)

# This needs to re-design
class CPUCacheAllocator: 
    def __init__(self, num_blocks: int): 
        self.num_free_blocks = num_blocks
        self.total_blocks = num_blocks
        self.free_blocks =[(0, num_blocks-1)]
        self.allocated_blocks = {}

    def allocate(self, num_blocks: int):
        for i, (start, end) in enumerate(self.free_blocks):
            # Check if the range is large enough
            if end - start + 1 >= num_blocks:  
                allocated_start = start
                allocated_end = start + num_blocks - 1

                # Update free blocks
                if allocated_end < end:
                    self.free_blocks[i] = (allocated_end + 1, end)
                else:
                    self.free_blocks.pop(i)

                # Track allocated blocks
                self.allocated_blocks[allocated_start] = num_blocks
                return allocated_start

        # No sufficient free range found
        return None

    def free(self, start_block):
        """
        Free a previously allocated range of blocks.
        :param start_block: Starting block of the range to free.
        """
        if start_block not in self.allocated_blocks:
            raise ValueError("Invalid block range to free")

        num_blocks = self.allocated_blocks.pop(start_block)
        freed_range = (start_block, start_block + num_blocks - 1)

        # Merge the freed range into free blocks
        self.free_blocks.append(freed_range)
        self.free_blocks = self._merge_free_blocks()

    def _merge_free_blocks(self):
        """
        Merge contiguous ranges in the free blocks list.
        :return: A merged list of free blocks.
        """
        if not self.free_blocks:
            return []

        # Sort the free blocks by start
        self.free_blocks.sort()

        merged_blocks = [self.free_blocks[0]]
        for current_start, current_end in self.free_blocks[1:]:
            last_start, last_end = merged_blocks[-1]
            if current_start <= last_end + 1:  # Overlapping or contiguous ranges
                merged_blocks[-1] = (last_start, max(last_end, current_end))
            else:
                merged_blocks.append((current_start, current_end))

        return merged_blocks

class CacheAllocator:
    def __init__(self, name: str, num_caches: int):
        self.num_caches = num_caches
        self.type = name 
        # kv_caches: tracking the available cache ids
        self.kv_caches = deque(range(num_caches))

    def allocate(self) -> int:
        assert len(self.kv_caches) > 0, f"Please set self.num_caches to a bigger value"
        cache_id = self.kv_caches.popleft() 
        #    print(f"ERROR: self.kv_caches is 000000000 NOW", file=sys.stderr)
        #elif self.type == "cpu":
        #    print(f"ERROR checking: allocated a cpu cache:{cache_id}, remaining cache:{len(self.kv_caches)}", file=sys.stderr) 
        return cache_id

    def free(self, cache_id: int):
        self.kv_caches.appendleft(cache_id)

        #assert cache_id == self.kv_caches[0]
        #print(f"after free-{cache_id} of {self.type}, the left item:{self.kv_caches[0]} ", file=sys.stderr)
        #self.kv_caches.append(cache_id)

    def get_free_caches(self):
        return len(self.kv_caches)

class SwappedCPUCache:
    def __init__(self, start_block, blocks):
        self.start_block = start_block
        self.blocks = blocks

class BlockSpaceManagerDAttn(BlockSpaceManager):
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.03,
        sliding_window: Optional[int] = None, # Not supported
        enable_caching: bool = False, # Not supported
        vmm_frequency: int = 8, 
        num_caches: int = 0,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        # For every 16 steps, we will perform vmm updates by invoking update_cache_blocks
        self.vmm_frequency = vmm_frequency
        self.vmm_frequency_mask = vmm_frequency - 1 
        
        # Tracking the number of gpu_blocks (including self.cached_free_gpu_blocks) 
        self.num_free_gpu_blocks = num_gpu_blocks
        self.num_free_cpu_blocks = num_cpu_blocks

        num_gpu_caches = num_caches

        print(f"self.num_free_cpu_blocks-{self.num_free_cpu_blocks}, vmm_frequency-{vmm_frequency}", file=sys.stderr)
        # use to alloc cache buffer id for seq
        self.gpu_allocator = CacheAllocator("cuda", num_gpu_caches)
        self.cpu_allocator = CPUCacheAllocator(num_cpu_blocks)

        # Watermark indicates that the least amount of blocks should be free. 
        assert watermark >= 0.0
        self.watermark_blocks = 1
        #int(watermark * num_gpu_blocks)
        
        # Mapping from cache_id to the number of allocated blocks.
        # The information is more persitent across different steps
        self.allocated_gpu_blocks: Dict[int, int] = {} 
        # Pre-allocate one block for the first cache,  to support graph capture
        self.allocated_gpu_blocks[0] = 1 

        # Temporary buffer for each step. self.step() will collect these information and freed all 
        # caches of to_free_gpu_caches
        self.to_allocate_blocks: Dict[int, int] = {} 
        self.to_free_blocks: Dict[int, int] = {} 

        # Useful when admitting new requests or swapping in some requests. 
        # Then we prefer those requests that just exit.
        # Making cached_free_gpu_blocks a part of num_free_gpu_blocks
        self.cached_free_gpu_blocks: int = 0

        # to_free_gpu_caches keeps the requests that are freed in the current step
        self.to_free_gpu_caches: Dict[int, int] = {}
        self.immediate_allocate = False

        # Maintain the mapping between seq.req_id and SwappedCPUCache (cache_id, blocks)
        self.swapped_out_caches: Dict[int, SwappedCPUCache] = {}

        # number of active requests (which will be used to improve the scheduler)
        self.total_active_reqs = 0

        # Track the step information, used for periodical memory management
        self.step_index = 0
    
    def _predict_n_blocks(self, tokens: int) -> int:
        if tokens == 0:
            return 0
        
        # when tokens is 14 blocks and self.vmm_frequency is 2, then 
        # we will need to allocate a new block immediately due to the asynchronous commnication 
        return (tokens + self.vmm_frequency + self.block_size + 1) // self.block_size 

    def _get_n_blocks(self, tokens: int) -> int:
        return (tokens + self.block_size - 1) // self.block_size 

    def _check_availability(self, need_blocks) -> AllocStatus:
        # Ensure that one request should not use more than 90% or 99% of memory
        # This can avoid frequent cache eviction 
        if (self.num_total_gpu_blocks - need_blocks < self.watermark_blocks):
            return AllocStatus.NEVER
        
        #if self.num_free_gpu_blocks - need_blocks >= self.watermark_blocks:
        if self.num_free_gpu_blocks > need_blocks:
            # Make sure that we are not holding more than schedule_config.max_num_seqs
            if self.gpu_allocator.get_free_caches() > 0 or len(self.to_free_gpu_caches) > 0:
                return AllocStatus.OK
            else:
                return AllocStatus.LATER
        else:
            return AllocStatus.LATER

    # This function is invoked only in the prefill phase
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        if (self.step_index & self.vmm_frequency_mask):
            return AllocStatus.LATER
    
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        check_no_caching_or_swa_for_blockmgr_encdec(self, seq_group)

        # get_seqs will collect a list of sequence with status equalling to SequenceStatus.WAITING
        # then we will get the first sequence in this group 
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        self_num_required_blocks = self._predict_n_blocks(tokens=seq.get_len())
        cross_seq = seq_group.get_encoder_seq()
        cross_num_required_blocks = 0 
        if cross_seq:
            cross_num_required_blocks = self._predict_n_blocks(tokens = cross_seq.get_len())

        num_required_blocks = self_num_required_blocks + \
                              cross_num_required_blocks

        return self._check_availability(num_required_blocks)


    # This function is only invoked by _allocate_and_set_running (invoked by _schedule_prefills)
    # Allocate a GPU cache when admitting a new request in prefill phase.
    def allocate(self, seq_group: SequenceGroup) -> None:
        # Allocate decoder sequences
        #
        # NOTE: Here we assume that all sequences in the group have the same
        # decoder prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        
        need_blocks = self._predict_n_blocks(tokens=seq.get_len())

        self.immediate_allocate = True 
        #print(f"NNOOOOOOWWW allocate sequence-{seq.seq_id} at step_index-{self.step_index}, need_blocks:{need_blocks}, tokens:{seq.get_len()}", file=sys.stderr) 
        cache_id = self._allocate_gpu_cache(need_blocks)
        
        seq.cache_id = cache_id
        seq.data.cache_id = cache_id

    #  Allocate a new GPU cache, when the available GPU blocks are sufficient
    def _allocate_gpu_cache(self, need_blocks: int) -> Tuple[int, int]:
        cache_id = -1
        to_allocate = True
        
        # update total_active_reqs and num_free_gpu_blocks
        self.total_active_reqs +=1
        self.num_free_gpu_blocks -= need_blocks

        allocated_block_num = 0
        # Prefer to reuse the to_free_gpu_caches at first, as some pages have been allocated already. 
        if self.cached_free_gpu_blocks > 0:
            # Make it block_diff a big number for the better comparison
            block_diff = need_blocks*100

            # Find one kv_cache with the smallest difference on the number of blocks
            # The found cache can have more or less available blocks.   
            for id, num_blocks in self.to_free_gpu_caches.items():
                diff = abs(num_blocks - need_blocks)
                
                # kv_cache : cache_id, blocks 
                if diff < block_diff:
                    cache_id = id
                    block_diff = diff

                    allocated_block_num = num_blocks

                    # No need to check anymore if we already found a perfect one
                    if diff == 0:
                        break 

            # Remove this item from the to_free_gpu_caches
            del self.to_free_gpu_caches[cache_id]
            self.cached_free_gpu_blocks -= allocated_block_num
            
            #print(f"reuse cache-{cache_id}: allocated_blocks-{allocated_block_num}, need_blocks:{need_blocks}, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}", file=sys.stderr)
        else:
            # Check whether the can_allocate or can_swap_in has a bug
            if self.num_free_gpu_blocks < 0: 
                print(f"Error: self.num_free_gpu_blocks:{self.num_free_gpu_blocks}, need_blocks:{need_blocks}", file=sys.stderr)
            assert self.num_free_gpu_blocks >= 0

            cache_id = self.gpu_allocator.allocate()

        self.allocated_gpu_blocks[cache_id] = need_blocks 

        # We need to adjust the number of blocks for this cache id
        # Here, we specifically differentiate to_allocate and to_free so that 
        # we could place to_free before to_allocate in the step() function
        if allocated_block_num < need_blocks:
            self.to_allocate_blocks[cache_id] = need_blocks
        elif allocated_block_num > need_blocks: 
            self.to_free_blocks[cache_id] = need_blocks

        return cache_id

    # Invoked by _schedule_running in running phase.  
    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        
        # Only check periodically in asynchronous memory management, not each step
        if (self.step_index & self.vmm_frequency_mask):
            return True

        # Do not evict a request that have at least 16 slots to extend (at least we could do it next step)
        cache_blocks, tokens = self._get_blocks_tokens(seq_group)
        #if self.step_index > 88 and  self.step_index <= 92:
        #    print(f"step-{self.step_index} cache_id:{seq_group.request_id} real blocks:{cache_blocks}, tokens:{tokens}, freeblocks:{self.num_free_gpu_blocks}, allocate:{self._predict_n_blocks(tokens) > cache_blocks}", file=sys.stderr) 
        if self._predict_n_blocks(tokens) <= cache_blocks:
            return True


        # Simple heuristic: at least one free block for each request.
        # Since we will perform the actual allocation in the next epoch (16 steps), where 
        # each request can allocate one block successfully, then there
        # is no need to preempt. Note that self.cache_free_gpu_blocks 
        # should be included as they will be allocated first in the next epoch 
        return self.num_free_gpu_blocks >= 1
        
    # FIXME: there is no handling on num_lookahead_slots, which should be handled.  
    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int = 0,
    ) -> List[Tuple[int, int]]:

        # We only need to check periodically, not each step
        if (self.step_index & self.vmm_frequency_mask):
            return []

        """Allocate a physical token/slot for a new token."""
        cache_id = seq.cache_id

        # If the sequence is allocated, its cache_id must >= 0.
        assert cache_id >= 0

        allocated_block_num = self.allocated_gpu_blocks[cache_id]
        logical_blocks_num = self._predict_n_blocks(seq.get_len())

        #if self.step_index > 88 and  self.step_index < 92:
        #    print(f"step-{self.step_index} check block: cache_id:{cache_id}, logical_blocks_num:{logical_blocks_num}, allocated_block_num:{allocated_block_num}, real tokens:{seq.get_len()}, self.vmm_frequency:{self.vmm_frequency}, freeblocks:{self.num_free_gpu_blocks}", file=sys.stderr) 
        
        # If we need to allocate a new physical block
        if allocated_block_num < logical_blocks_num:
            if allocated_block_num != logical_blocks_num - 1: 
                print(f"append_slots cache_id:{cache_id}, logical_blocks_num:{logical_blocks_num} - allocated_block_num:{allocated_block_num}, real tokens:{seq.get_len()}", file=sys.stderr) 

            #print(f"step-{self.step_index} increase one block: cache_id:{cache_id}, allocated_block_num:{allocated_block_num}, real tokens:{seq.get_len()}, free blocks:{self.num_free_gpu_blocks} before updating", file=sys.stderr) 
            # Currently this code only supports adding one physical block in the decoding phase
            assert allocated_block_num == logical_blocks_num - 1

            self.num_free_gpu_blocks -= 1
            self.allocated_gpu_blocks[cache_id] = logical_blocks_num
            self.to_allocate_blocks[cache_id] = logical_blocks_num 
        # related to the current scheduling phase.            
        return []

    # Collect the number of physical blocks used by this sequence group 
    def _get_blocks_tokens(
            self, seq_group: SequenceGroup):
        
        cache_blocks = 0
        tokens = 0
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue

            cache_id = seq.cache_id
            cache_blocks += self.allocated_gpu_blocks[cache_id]
            tokens += seq.get_len()
        
        return cache_blocks, tokens

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        raise NotImplementedError("Forking is not supported in BlockSpaceManagerDAttn now.")

    # This is to swap_in an pre-existing block, which is slightly different 
    # from can_allocate(). 
    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        
        if (self.step_index & self.vmm_frequency_mask):
            return AllocStatus.LATER

        need_blocks = num_lookahead_slots
        req_id = None
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            if seq.is_finished():
                continue
            
            req_id = seq.seq_id
            need_blocks += self.swapped_out_caches[req_id].blocks

        # Make sure that the number of free blocks at least one block more
        #need_blocks += 1
        
        result = self._check_availability(need_blocks) 

        return result

    # A fucntion is invoked to figure out the blocks that need to be allocated. 
    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        to_swap_in_caches = []

        #print(f"SWAP IN NOW with sequence-{seq_group.request_id}, number-{seq_group.num_seqs(status=SequenceStatus.SWAPPED)} at step-{self.step_index}", file=sys.stderr)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            cpu_cache = self.swapped_out_caches[seq.seq_id] 

            need_blocks = cpu_cache.blocks
            start_block = cpu_cache.start_block

            # Free cpu cache id and update the counter
            self.cpu_allocator.free(start_block)
            self.num_free_cpu_blocks += need_blocks  
            
            # Allocate a gpu cache id, based on the need_blocks. 
            # Note that we specifically request one more block in order to accomodate vmm_frequency's memory management
            gpu_cache_id = self._allocate_gpu_cache(need_blocks + 1)

            seq.cache_id = gpu_cache_id
            seq.data.cache_id = gpu_cache_id

            # NOTE: we may not need the allocation, if gpu_cache_id 
            print(f"SWAPIN seq_id:{seq.seq_id} to cache_id:{gpu_cache_id} with tokens:{seq.get_len()}, need_blocks:{need_blocks+1}, allocated_blocks:{self.allocated_gpu_blocks[gpu_cache_id]}, free_gpu_blocks:{self.num_free_gpu_blocks}", file=sys.stderr)
            to_swap_in_caches.append([start_block, gpu_cache_id, need_blocks])
            
            # Delete this entry
            del self.swapped_out_caches[seq.seq_id]

        return to_swap_in_caches

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        cache_blocks, tokens = self._get_blocks_tokens(seq_group)
    
        return cache_blocks <= self.num_free_cpu_blocks

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
    
        to_swap_out_caches = []

        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            # Find the cache id and gpu_blocks        
            gpu_cache_id = seq.cache_id

            # Since this cache may have more blocks than its necessity, we only record the 
            # real_gpu_blocks here in order to reduce the overhead involved in copy in swapping
            need_blocks = self._get_n_blocks(seq.get_len())

            #print(f"SWAPOUT request-{seq.seq_id} with blocks-{need_blocks},  free GPU blocks:{self.num_free_gpu_blocks} at step-{self.step_index}", file=sys.stderr)

            # Free the cache related to gpu_cache_id
            self._free_cache(cache_id=gpu_cache_id)

            # Allocate the cpu cache id
            start_block = self.cpu_allocator.allocate(need_blocks)
            cpu_cache = SwappedCPUCache(start_block, need_blocks) 
            self.swapped_out_caches[seq.seq_id] = cpu_cache

            # After the swapped out, num_free_cpu_blocks should be decremented 
            self.num_free_cpu_blocks -= need_blocks
            
            print(f"SWAPOUT request-{seq.seq_id} with blocks-{need_blocks},  free GPU blocks:{self.num_free_gpu_blocks} at step-{self.step_index}", file=sys.stderr)
            
            to_swap_out_caches.append([gpu_cache_id, start_block, need_blocks]) 

        #print(f"to_swap_out_caches:{to_swap_out_caches}", file=sys.stderr)
        return to_swap_out_caches

    def _free_cache(self, cache_id: int) -> None:
        # Check whether cache_id is in the list
        if cache_id in self.to_free_gpu_caches:
            # Already freed yet, no need to do anything.
            return

        # Get blocks of this cache
        free_blocks = self.allocated_gpu_blocks[cache_id]
        #print(f"FREE gpu cache_id:{cache_id}, to_free_blocks:{free_blocks}, total free blocks before free:{self.num_free_gpu_blocks}, step:{self.step_index}", file=sys.stderr)
       
        # Note that we update self.total_active_reqs here, as free_cache() is invoked twice for every request
        self.total_active_reqs -=1
        self.allocated_gpu_blocks[cache_id] = 0

        self.to_free_gpu_caches[cache_id] = free_blocks
        self.cached_free_gpu_blocks += free_blocks
        self.num_free_gpu_blocks += free_blocks

    """
    Free a sequence. We will append the seq to to_free_gpu_caches. 
    Initially, we did this inside the memory management library. Maybe we should do it here as well. 
    """
    def free(self, seq: Sequence) -> None:

        #if seq.cache_id in self.to_free_gpu_caches:
        #    return

        #print(f"step-{self.step_index}, FREE sequence:{seq.seq_id}, cache_id:{seq.cache_id}, free_blocks:{self.num_free_gpu_blocks}", file=sys.stderr)
        self._free_cache(cache_id=seq.cache_id)
        
    def reset(self) -> None:
        # Free decoder block tables
        self.allocated_gpu_blocks.clear()
        self.num_free_gpu_blocks = self.num_total_gpu_blocks
        self.num_free_cpu_blocks = self.num_total_cpu_blocks
        
        self.to_free_gpu_caches = {}
        self.to_allocate_blocks = {}

    # A dummy function that will be never invoked
    def get_block_table(self, seq: Sequence) -> List[int]:
        # logger.warning("block table is not used in BlockSpaceManagerDAttn now.")
        return []

    def get_num_free_gpu_blocks(self) -> int:
        return self.num_free_gpu_blocks

    def get_num_free_cpu_blocks(self) -> int:
        return self.num_free_cpu_blocks

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        # logger.warning("Access all blocks in seq is not supported in BlockSpaceManagerDAttn now.")
        pass

    def get_common_computed_block_ids(self,
                                      seq_group: SequenceGroup) -> List[int]:
        # logger.warning("Common computed block ids is not supported in BlockSpaceManagerDAttn now.")
        return None  # type: ignore

    def mark_blocks_as_computed(self, seq_group: SequenceGroup, token_chunk_size: int) -> None:
        # logger.warning("Mark blocks as computed is not supported in BlockSpaceManagerDAttn now.")
        pass

    # In the end of each step's scheduling, this function is invoked to 
    # collect the information of allocation and deallocation  
    def step(self) -> Tuple[Dict[int, int], List[int], bool]:
        to_update_blocks = {}

        immediate_allocate = self.immediate_allocate
        self.immediate_allocate = False

        #print(f"in the end step-{self.step_index} with requests:{self.total_active_reqs}, allocate_blocks:{len(self.to_allocate_blocks)} now!", file=sys.stderr) 
        # We will perform virtual memory management once for every self.vmm_frequency 
        if ((self.step_index & self.vmm_frequency_mask)) and (immediate_allocate != True):
            # No need to invoke virtual memory management
            self.step_index += 1
            #print(f"step-{self.step_index}, no need to do updates, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}", file=sys.stderr) 
            return to_update_blocks, immediate_allocate

        #immediate_allocate = True
        # In the following, we place to_free_blocks in the header of to_update_blocks, which 
        # ensures that allocation can be performed without any issue. 
        to_free_blocks = 0
        # First, place all to_free_gpu_caches at first. 
        for cache_id, num_blocks in self.to_free_gpu_caches.items():
            #if self.step_index == 4684:
            #print(f"step-{self.step_index} free cache_id:{cache_id}, num_blocks:{num_blocks}", file=sys.stderr)
            # Freeing all blocks of this cache
            to_update_blocks[cache_id] = 0
            self.gpu_allocator.free(cache_id)
            to_free_blocks += num_blocks 

        # Second, place all to_free_blocks (caused by reusing a freed cache)
        for cache_id, num_blocks in self.to_free_blocks.items():
            #if self.step_index == 4684:
            #print(f"step-{self.step_index} tofree cache_id:{cache_id}, num_blocks:{num_blocks}", file=sys.stderr)
            to_update_blocks[cache_id] = num_blocks
            to_free_blocks += num_blocks

        # Third, place the caches that need to increase their blocks
        for cache_id, num_blocks in self.to_allocate_blocks.items():
            #print(f"step-{self.step_index} toallocate cache_id:{cache_id}, num_blocks:{num_blocks}", file=sys.stderr)
            to_update_blocks[cache_id] = num_blocks

        if self.to_free_gpu_caches or self.to_free_blocks or self.to_allocate_blocks:
            # Force the next epoch to wait for the memory management
            self.immediate_allocate = True
            #print(f"step-{self.step_index}, to_allocate_blocks:{len(to_update_blocks)}, freeing ({to_free_blocks} blocks), self.to_allocate_blocks:{len(self.to_allocate_blocks)}, self.to_free_gpu_caches:{len(self.to_free_gpu_caches)}, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}", file=sys.stderr)
            if self.num_free_gpu_blocks < 0:
                print(f"step-{self.step_index} ERROR: to_allocate_blocks:{len(to_update_blocks)}, freeing ({to_free_blocks} blocks), self.to_allocate_blocks:{len(self.to_allocate_blocks)}, self.to_free_gpu_caches:{len(self.to_free_gpu_caches)}, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}", file=sys.stderr)
                exit(-1)
        #else:
           #print(f"step-{self.step_index}, no need to do updates, self.num_free_gpu_blocks:{self.num_free_gpu_blocks}", file=sys.stderr) 

        # step() is invoked once after _schedule() inside Scheduler::schedule(). It is invoked once for every decode or prefill
        self.to_free_gpu_caches.clear()
        self.to_free_blocks.clear()
        self.to_allocate_blocks.clear()
        self.cached_free_gpu_blocks = 0

        # Only update the step index for decoding steps
        if immediate_allocate == False:
            self.step_index += 1  

        return to_update_blocks, immediate_allocate

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return 0