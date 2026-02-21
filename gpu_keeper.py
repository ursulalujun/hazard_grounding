import time
import torch
import pynvml
import multiprocessing
from datetime import datetime

# ================= 配置区域 =================
CHECK_INTERVAL = 600    # 检查间隔：600秒 (10分钟)
BURN_DURATION = 60      # 运算持续时间：60秒
UTIL_THRESHOLD = 30     # 阈值：负载低于 30% 则触发
MATRIX_SIZE = 10000     # 矩阵大小
# ===========================================

def get_gpu_utilization(gpu_id, handle):
    """获取指定GPU的利用率"""
    try:
        mem_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return mem_info.gpu
    except pynvml.NVMLError as error:
        print(f"[GPU {gpu_id}] 获取状态失败: {error}")
        return 100 

def gpu_worker(gpu_id):
    """
    单个GPU的工作进程
    """
    # 在 spawn 模式下，每个子进程都需要重新初始化 nvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    
    print(f"[启动] 监控进程已启动: GPU {gpu_id}")

    try:
        while True:
            current_util = get_gpu_utilization(gpu_id, handle)
            timestamp = datetime.now().strftime("%H:%M:%S")

            if current_util < UTIL_THRESHOLD:
                print(f"[{timestamp}] GPU {gpu_id} 负载低 ({current_util}%) -> 开始计算...")
                
                try:
                    # 这里的 torch 此时是在全新的进程中运行，是安全的
                    device = torch.device(f"cuda:{gpu_id}")
                    
                    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
                    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
                    
                    end_time = time.time() + BURN_DURATION
                    
                    while time.time() < end_time:
                        c = torch.mm(a, b)
                        torch.cuda.synchronize()
                    
                    del a, b, c
                    torch.cuda.empty_cache()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU {gpu_id} 结束计算，休眠中。")
                    
                except Exception as e:
                    print(f"[GPU {gpu_id}] 运算出错: {e}")
            else:
                print(f"[{timestamp}] GPU {gpu_id} 负载正常 ({current_util}%)，跳过。")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def main():
    # ---------------------------------------------------------
    # 关键修改：设置启动方式为 'spawn'
    # 注意：这行代码必须在创建任何 Process 之前执行
    # ---------------------------------------------------------
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass

    # 为了避免在父进程初始化 CUDA，我们使用 pynvml 来数显卡
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    
    print(f"检测到 {num_gpus} 张显卡，使用 'spawn' 模式启动监控...")

    processes = []
    
    for i in range(num_gpus):
        p = multiprocessing.Process(target=gpu_worker, args=(i,))
        p.start()
        processes.append(p)

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n正在停止...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()