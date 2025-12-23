#!/usr/bin/env python3
import os
import time
import sys
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
from tomesh import export

BASE_DIR = "../../levelset"
OUTPUT_DIR = "../../plys"
NUM_WORKERS = min(30, cpu_count())  # 最多30个进程

def get_script_dir():
    """获取脚本所在目录"""
    return Path(__file__).parent.resolve()

def get_processed_files(base_dir, output_dir):
    """获取已处理的文件列表 - 只有同时存在npy和ply的才算已处理"""
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有存在的npy文件编号
    npy_indices = set()
    if base_path.exists():
        for npy_file in base_path.glob("phi_*.npy"):
            name = npy_file.stem
            if name.startswith("phi_"):
                try:
                    num = int(name.split("_")[1])
                    if 0 <= num <= 599:
                        npy_indices.add(num)
                except (ValueError, IndexError):
                    pass
    
    # 获取所有存在的ply文件编号
    ply_indices = set()
    for ply_file in output_path.glob("output_*.ply"):
        name = ply_file.stem
        if name.startswith("output_"):
            try:
                num = int(name.split("_")[1])
                if 0 <= num <= 599:
                    ply_indices.add(num)
            except (ValueError, IndexError):
                pass
    
    # 只有同时存在于两个文件夹的才算已处理
    processed = npy_indices & ply_indices
    return processed

def worker(task_queue, output_dir):
    """工作进程,从队列中获取任务并处理"""
    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break
        
        index, input_path = task
        output_path = Path(output_dir) / f"output_{index:05d}.ply"
        
        try:
            print(f"[Worker {os.getpid()}] Processing phi_{index:05d}.npy")
            export(str(input_path), str(output_path))
        except Exception as e:
            # 忽略错误,继续处理下一个
            pass

def main():
    script_dir = get_script_dir()
    base_dir = script_dir / BASE_DIR
    output_dir = script_dir / OUTPUT_DIR
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建任务队列
    task_queue = Queue()
    
    # 启动工作进程池
    workers = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker, args=(task_queue, output_dir))
        p.daemon = True
        p.start()
        workers.append(p)
    
    print(f"Started {NUM_WORKERS} worker processes")
    print(f"Monitoring directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print("Press Ctrl+C to stop...")
    
    # 获取已处理的文件
    processed = get_processed_files(base_dir, output_dir)
    print(f"Already processed: {len(processed)} files")
    
    try:
        while True:
            # 扫描 levelset 目录
            if not base_dir.exists():
                time.sleep(1)
                continue
            
            # 查找所有 npy 文件
            npy_files = sorted(base_dir.glob("phi_*.npy"))
            
            for npy_file in npy_files:
                # 提取索引
                name = npy_file.stem
                if name.startswith("phi_"):
                    try:
                        index = int(name.split("_")[1])
                    except (ValueError, IndexError):
                        continue
                    
                    # 如果未处理,加入队列
                    if index not in processed:
                        task_queue.put((index, npy_file))
                        processed.add(index)
                        print(f"Queued phi_{index:05d}.npy (queue size: ~{task_queue.qsize()})")
            
            # 等待一小段时间再检查
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping workers...")
        
        # 发送结束信号
        for _ in range(NUM_WORKERS):
            task_queue.put(None)
        
        # 等待所有工作进程结束
        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        
        print(f"Total queued: {len(processed)} files")


if __name__ == "__main__":
    main()
