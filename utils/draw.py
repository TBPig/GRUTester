import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


def group(x_arr: np.ndarray, y_arr: np.ndarray, group_num=1):
    if group_num == 1:
        return x_arr, y_arr

    # 获取数组长度
    length = len(y_arr)

    # 计算完整分组的数量
    full_groups = length // group_num
    remainder = length % group_num

    # 初始化结果列表
    grouped_y = []
    grouped_x = []

    # 处理完整的分组
    for i in range(full_groups):
        start_idx = i * group_num
        end_idx = start_idx + group_num
        # 计算平均值
        grouped_y.append(np.mean(y_arr[start_idx:end_idx]))
        # 对于x轴，取每组的第一个值
        grouped_x.append(x_arr[start_idx])

    # 处理剩余的元素（如果不均匀分组）
    if remainder > 0:
        start_idx = full_groups * group_num
        grouped_y.append(np.mean(y_arr[start_idx:]))
        grouped_x.append(x_arr[start_idx])

    return grouped_x, grouped_y


def calculate_ylim(data_dict, column, p=0.8):
    """
    计算Y轴范围，使80%的数据可见且曲线更清晰
    """
    # 收集所有指定列的数据
    all_values = []
    for data in data_dict.values():
        all_values.extend(data[column].tolist())

    # 如果没有数据，返回默认范围
    if not all_values:
        return None

    # 排序并计算80%分位数范围
    all_values = sorted(all_values)
    n = len(all_values)

    # 计算要保留的数据量（80%）
    keep_count = int(n * p)

    # 找到最佳窗口位置
    min_range = float('inf')
    best_min, best_max = all_values[0], all_values[-1]

    # 滑动窗口寻找最小范围
    for i in range(n - keep_count + 1):
        window_min = all_values[i]
        window_max = all_values[i + keep_count - 1]
        window_range = window_max - window_min

        if window_range < min_range:
            min_range = window_range
            best_min, best_max = window_min, window_max

    # 添加一些边距使曲线不贴边
    margin = (best_max - best_min) * 0.05
    return best_min - margin, best_max + margin


class Draw:
    def __init__(self):
        self.path = 'result'

    def run(self, data_set=None, index=None):
        """
        从指定文件夹中读取所有CSV文件，并绘制三幅曲线图：训练损失、测试损失、测试准确率
        """
        if data_set is None:
            data_set = input("请输入数据集名称：")
        if index is None:
            index = input("请输入模型组编号：")
            
        # 查找文件夹中的所有CSV文件
        folder_path = os.path.join(self.path, data_set, index)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not csv_files:
            print(f"在文件夹 {folder_path} 中未找到CSV文件")
            return

        # 读取所有数据
        data_dict = {}
        for csv_file in csv_files:
            # 获取文件名（不含扩展名）作为模型名称
            model_name = os.path.splitext(os.path.basename(csv_file))[0]
            data_dict[model_name] = pd.read_csv(csv_file)

        # 创建图像和子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 为每种曲线准备颜色
        colors = plt.cm.Blues(np.linspace(0.3, 1, len(data_dict)))

        # 定义绘图参数
        plot_configs = [
            {'column': 'train_loss', 'title': '训练损失曲线', 'ylabel': 'Train Loss'},
            {'column': 'test_loss', 'title': '测试损失曲线', 'ylabel': 'Test Loss'},
            {'column': 'test_acc', 'title': '测试准确率曲线', 'ylabel': 'Test Accuracy'}
        ]

        # 绘制三幅子图
        for ax, config in zip(axes, plot_configs):
            # 计算并设置Y轴范围
            ylim = calculate_ylim(data_dict, config['column'], p=0.95)
            if ylim is not None:
                ax.set_ylim(*ylim)

            # 绘制曲线
            for i, (model_name, data) in enumerate(data_dict.items()):
                ax.plot(data['epoch'], data[config['column']], label=model_name, color=colors[i])
            ax.set_title(config['title'])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(config['ylabel'])
            ax.legend()
            ax.grid(True)

        # 调整子图间距
        plt.tight_layout()

        plt.savefig(os.path.join(folder_path, 'mnist_comparison.png'))

        plt.show()
