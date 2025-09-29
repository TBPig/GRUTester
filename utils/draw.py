import os

import torch
import matplotlib.pyplot as plt
import json

from utils.Output import Output
from utils.SerialCounter import SerialCounter

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


def group(x_arr, y_arr, group_num=1):
    if group_num == 1:
        return x_arr, y_arr

    tmp = []
    k = group_num
    l = len(y_arr) // k * k
    for i in range(0, l, k):
        tmp.append(sum(y_arr[i:i + k]) / k)
    y_arr = tmp
    x_arr = x_arr[0:l:k]
    return x_arr, y_arr


class Draw:
    path = 'result/img'
    train_group_num = 1
    test_group_num = 1

    train_y_lim = (0, 0.05)
    test_y_lim = (0, 0.1)

    is_lim = True

    def __init__(self, serial=None):
        self.sc = SerialCounter()
        # 如果提供了serial参数，则使用该值，否则使用SerialCounter中的当前序号
        if serial is not None:
            self.serial = serial
        else:
            self.serial = self.sc.get_serial()
        self.outputs = torch.load(f'result/data/{self.serial:04d}.outs', weights_only=False)

    def draw_output(self):
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

    def draw_train(self, ax):
        for output in self.outputs:
            output.train_idx, output.train_loss = group(output.train_idx, output.train_loss, self.train_group_num)
            ax.plot(output.train_idx, output.train_loss, label=output.model_name, alpha=0.5)
        ax.set_xlabel('训练进度')
        ax.set_ylabel('平均损失')
        ax.set_title('训练集损失曲线对比')
        ax.legend()  # 显示图例
        ax.grid(True)  # 显示网格
        if self.is_lim:
            ax.set_ylim(*self.train_y_lim)

    def draw_test(self, ax):
        for output in self.outputs:
            label = f"{output.model_name}"
            test_idx, test_loss = group(output.test_idx, output.test_loss, self.test_group_num)
            ax.plot(test_idx, test_loss, label=label, alpha=0.5)

        ax.set_xlabel('训练进度')
        ax.set_ylabel('平均损失')
        ax.set_title('测试集损失曲线对比')
        ax.legend()  # 显示图例
        ax.grid(True)  # 显示网格
        if self.is_lim:
            ax.set_ylim(*self.test_y_lim)

    def draw_cr(self, ax):
        for output in self.outputs:
            label = f"{output.model_name}"
            test_idx, test_cr = group(output.test_idx, output.test_cr, self.test_group_num)
            ax.plot(test_idx, test_cr, label=label, alpha=0.5)

        ax.set_xlabel('训练进度')
        ax.set_ylabel('平均正确率')
        ax.set_title('测试集正确率曲线对比')
        ax.legend()  # 显示图例
        ax.grid(True)  # 显示网格