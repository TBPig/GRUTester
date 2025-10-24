import os
import numpy as np

import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


def group(x_arr, y_arr, group_num=1):
    if group_num == 1:
        return x_arr, y_arr

    # 转换为numpy数组以提高计算效率
    y_arr = np.array(y_arr)
    x_arr = np.array(x_arr)

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


class Draw:
    path = 'result/img'
    train_group_num = 1
    test_group_num = 1

    train_y_lim = (0, 0.05)
    test_y_lim = (0, 0.1)

    is_lim = False

    def __init__(self, serial=None):
        self.serial = 0

    def draw_output(self):
        return
        plt.figure(figsize=(10, 6))

        ax1 = plt.subplot(1, 3, 1)  # 1行3列，第1个
        self.draw_train(ax1)

        ax2 = plt.subplot(1, 3, 2)  # 1行3列，第2个
        self.draw_test(ax2)

        ax3 = plt.subplot(1, 3, 3)  # 1行3列，第3个
        self.draw_cr(ax3)

        plt.tight_layout()  # 调整子图间距
        # 格式化序列号
        serial_str = str(self.serial).zfill(3)
        plt.savefig(f'{Draw.path}/No-{serial_str}.png')
        plt.close()

        # 获取所有保存的文件并按修改时间排序
        files = [os.path.join(Draw.path, f) for f in os.listdir(Draw.path) if f.endswith('.png')]
        files.sort(key=lambda x: os.path.getmtime(x))

        # 删除多余的文件，只保留最近的7个
        if len(files) > 7:
            for file_to_remove in files[:-7]:
                os.remove(file_to_remove)

    def _draw_series(self, ax, outputs, idx_attr, data_attr, group_num, y_label, title, y_lim=None):
        """
        通用绘图方法，减少重复代码
        
        :param ax: matplotlib的轴对象
        :param outputs: 输出数据列表
        :param idx_attr: 索引属性名
        :param data_attr: 数据属性名
        :param group_num: 分组数量
        :param y_label: y轴标签
        :param title: 图表标题
        :param y_lim: y轴限制（可选）
        """
        for output in outputs:
            idx_data = getattr(output, idx_attr)
            y_data = getattr(output, data_attr)
            x_data, grouped_y_data = group(idx_data, y_data, group_num)
            ax.plot(x_data, grouped_y_data, label=output.model_name, alpha=0.5)

        ax.set_xlabel('训练进度')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        if self.is_lim and y_lim is not None:
            ax.set_ylim(*y_lim)

    def draw_train(self, ax):
        """绘制训练集损失曲线"""
        self._draw_series(
            ax,
            self.outputs,
            'train_idx',
            'train_loss',
            self.train_group_num,
            '平均损失',
            '训练集损失曲线对比',
            self.train_y_lim
        )

    def draw_test(self, ax):
        """绘制测试集损失曲线"""
        self._draw_series(
            ax,
            self.outputs,
            'test_idx',
            'test_loss',
            self.test_group_num,
            '平均损失',
            '测试集损失曲线对比',
            self.test_y_lim
        )

    def draw_cr(self, ax):
        """绘制测试集正确率曲线"""
        self._draw_series(
            ax,
            self.outputs,
            'test_idx',
            'test_cr',
            self.test_group_num,
            '平均正确率',
            '测试集正确率曲线对比'
        )