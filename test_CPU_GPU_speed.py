import torch
import argparse

def measure_transfer_speed(mode, block_size, num_blocks, iterations):
    """
    测量传输速率 (GB/s)
    :param mode: "vllm" 或 "dattn"
    :param block_size: 单个 block 的大小（字节）
    :param num_blocks: 总 block 数
    :param iterations: 迭代次数
       - vllm 模式下，总传输次数 = iterations * num_blocks（每个 block 单独传输）
       - dattn 模式下，总传输次数 = iterations（一次性传输所有 blocks）
    """
    if mode == "vllm":
        total_time_ms = 0.0
        total_bytes = block_size * num_blocks * iterations
        # 预先创建一个大小为 block_size 的 CPU 张量
        num_elements = block_size // 4  # 假设数据类型为 float32，每个元素占 4 字节
        cpu_tensor_block = torch.randn(num_elements, dtype=torch.float32)
        # 预热 GPU，不计入时间
        _ = cpu_tensor_block.to('cuda')
        torch.cuda.synchronize()
        for i in range(iterations):
            for j in range(num_blocks):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = cpu_tensor_block.to('cuda')
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                total_time_ms += elapsed_ms
        avg_time_ms = total_time_ms / (iterations * num_blocks)
        speed = total_bytes / (total_time_ms / 1000.0) / 1e9
        print(f"Mode: vllm")
        print(f"总传输字节数: {total_bytes/1e9:.2f} GB, 总耗时: {total_time_ms/1000.0:.2f} s, 平均每个 block 耗时: {avg_time_ms:.2f} ms")
        print(f"传输速率: {speed:.2f} GB/s")
    elif mode == "dattn":
        total_time_ms = 0.0
        total_bytes = block_size * num_blocks * iterations
        # 预先创建一个连续的 CPU 张量，其大小为所有 blocks 的总和
        num_elements = (block_size * num_blocks) // 4  # 假设 float32
        cpu_tensor_large = torch.randn(num_elements, dtype=torch.float32)
        # 预热 GPU
        _ = cpu_tensor_large.to('cuda')
        torch.cuda.synchronize()
        for i in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = cpu_tensor_large.to('cuda')
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            total_time_ms += elapsed_ms
        avg_time_ms = total_time_ms / iterations
        speed = total_bytes / (total_time_ms / 1000.0) / 1e9
        print(f"Mode: dattn")
        print(f"总传输字节数: {total_bytes/1e9:.2f} GB, 总耗时: {total_time_ms/1000.0:.2f} s, 平均耗时: {avg_time_ms:.2f} ms")
        print(f"传输速率: {speed:.2f} GB/s")
    else:
        raise ValueError("Unknown mode. Choose 'vllm' or 'dattn'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测量 CPU <-> GPU 传输速度 (GB/s) —— 针对 vllm 和 dattn 模式")
    parser.add_argument("--mode", type=str, default="vllm", choices=["vllm", "dattn"],
                        help="传输模式：'vllm' 为每个 block 单独传输，'dattn' 为连续整体传输")
    parser.add_argument("--block_size", type=int, default=8*1024*1024,
                        help="单个 block 大小（字节），默认 8MB")
    parser.add_argument("--num_blocks", type=int, default=237,
                        help="block 数量，默认 237")
    parser.add_argument("--iterations", type=int, default=3,
                        help="迭代次数，vllm 模式下总传输次数 = iterations * num_blocks，dattn 模式下为 iterations")
    args = parser.parse_args()

    measure_transfer_speed(args.mode, args.block_size, args.num_blocks, args.iterations)