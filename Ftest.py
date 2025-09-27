import torch

# 获取CUDA相关信息
torch_version = torch.__version__
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()

# 创建要保存的信息
info_lines = ["CUDA设备信息报告", "=" * 30, f"PyTorch版本: {torch_version}", f"CUDA是否可用: {cuda_available}"]

if cuda_available:
    info_lines.append(f"CUDA设备数量: {device_count}")
    for idx in range(device_count):
        device_name = torch.cuda.get_device_name(idx)
        device_properties = torch.cuda.get_device_properties(idx)
        memory_allocated = torch.cuda.memory_allocated(idx)

        info_lines.append(f"\n--- 设备 {idx} ---")
        info_lines.append(f"设备名称: {device_name}")
        info_lines.append(f"设备属性: {device_properties}")
        info_lines.append(f"已分配内存: {memory_allocated} bytes")
else:
    info_lines.append("未检测到可用的CUDA设备")

# 保存到txt文件
with open("cuda_info.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(info_lines))

print("CUDA信息已保存到 cuda_info.txt 文件中")
