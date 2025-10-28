import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import pandas as pd

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QListWidget, 
                             QSplitter, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt

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
    margin = (best_max - best_min) * 0.1
    return best_min - margin, best_max + margin


def get_model_colors(num_models):
    """
    根据模型数量获取颜色映射

    参数:
    num_models: 模型数量

    返回:
    colors: 颜色映射数组
    """
    if num_models <= 4:
        # 使用蓝色渐变
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, num_models))
    elif num_models <= 8:
        # 前4个使用蓝色渐变，其余使用红色渐变
        blue_colors = plt.cm.Blues(np.linspace(0.2, 0.9, 4))
        red_colors = plt.cm.Reds(np.linspace(0.2, 0.9, num_models - 4))
        colors = np.concatenate([blue_colors, red_colors])
    else:
        # 前4个使用蓝色渐变，中间4个使用红色渐变，其余使用紫色渐变
        blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 4))
        red_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 4))
        purple_colors = plt.cm.Purples(np.linspace(0.4, 0.9, num_models - 8))
        colors = np.concatenate([blue_colors, red_colors, purple_colors])

    return colors


class Draw:
    def __init__(self, selected_options=None):
        self.path = 'result'
        self.selected_options = selected_options

    def run(self, data_set=None, index=None):
        """
        从指定文件夹中读取所有CSV文件，并绘制曲线图
        如果提供了selected_options，则只绘制选中的图表
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

        # 定义所有可能的绘图配置
        all_plot_configs = [
            {'column': 'train_loss', 'title': '训练损失曲线', 'ylabel': 'Train Loss'},
            {'column': 'test_loss', 'title': '测试损失曲线', 'ylabel': 'Test Loss'},
            {'column': 'test_acc', 'title': '测试准确率曲线', 'ylabel': 'Test Accuracy'},
            {'column': 'learning_rate', 'title': '学习率曲线', 'ylabel': 'Learning Rate'}
        ]
        
        # 根据选中的选项过滤绘图配置，如果没有指定选项则使用所有配置
        if self.selected_options is not None:
            plot_configs = [config for config in all_plot_configs if config['column'] in self.selected_options]
            
            if not plot_configs:
                print("没有选中任何图表选项")
                return
        else:
            plot_configs = all_plot_configs

        # 创建图像和子图
        fig, axes = plt.subplots(1, len(plot_configs), figsize=(6 * len(plot_configs), 5))
        
        # 如果只有一个子图，需要特殊处理
        if len(plot_configs) == 1:
            axes = [axes]
        # 如果没有子图，直接返回
        elif len(plot_configs) == 0:
            print("没有要绘制的图表")
            return

        # 为每种曲线准备颜色
        num_models = len(data_dict)
        colors = get_model_colors(num_models)

        # 绘制子图
        for ax, config in zip(axes, plot_configs):
            # 检查数据中是否包含该列，如果没有则跳过该子图
            has_column_data = any(config['column'] in data.columns for data in data_dict.values())
            if not has_column_data:
                ax.set_visible(False)
                continue

            # 计算并设置Y轴范围
            ylim = calculate_ylim(data_dict, config['column'], p=0.9)
            if ylim is not None:
                ax.set_ylim(*ylim)

            # 绘制曲线
            for i, (model_name, data) in enumerate(data_dict.items()):
                if config['column'] in data.columns:
                    ax.plot(data['epoch'], data[config['column']], label=model_name, color=colors[i])
            ax.set_title(config['title'])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(config['ylabel'])
            ax.legend()
            ax.grid(True)

        # 调整子图间距
        plt.tight_layout()

        plt.savefig(os.path.join(folder_path, 'comparison.png'))

        plt.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GRU测试结果可视化工具")
        self.setGeometry(100, 100, 1000, 600)
        
        # 图表选项
        self.plot_options = [
            {'key': 'train_loss', 'label': '训练损失曲线', 'default': True},
            {'key': 'test_loss', 'label': '测试损失曲线', 'default': True},
            {'key': 'test_acc', 'label': '测试准确率曲线', 'default': True},
            {'key': 'learning_rate', 'label': '学习率曲线', 'default': True}
        ]
        
        # 初始化界面
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 创建主布局
        main_layout = QHBoxLayout(main_widget)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧面板 - 数据集选择
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 数据集标题
        dataset_label = QLabel("数据集选择")
        dataset_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(dataset_label)

        # 数据集列表
        self.dataset_list = QListWidget()
        self.dataset_list.clicked.connect(self.on_dataset_selected)
        left_layout.addWidget(self.dataset_list)

        # 中侧面板 - 编号选择
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)

        # 编号标题
        index_label = QLabel("编号选择")
        index_label.setAlignment(Qt.AlignCenter)
        middle_layout.addWidget(index_label)

        # 编号列表
        self.index_list = QListWidget()
        self.index_list.clicked.connect(self.on_index_selected)
        middle_layout.addWidget(self.index_list)

        # 右侧面板 - 图表选项
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 图表选项标题
        options_label = QLabel("图表选项")
        options_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(options_label)
        
        # 图表选项组
        options_group = QGroupBox()
        options_group_layout = QVBoxLayout(options_group)
        
        # 创建复选框
        self.option_checkboxes = []
        for option in self.plot_options:
            checkbox = QCheckBox(option['label'])
            checkbox.setChecked(option['default'])
            options_group_layout.addWidget(checkbox)
            self.option_checkboxes.append(checkbox)
            
        right_layout.addWidget(options_group)
        
        # 操作按钮区域
        button_panel = QWidget()
        button_layout = QVBoxLayout(button_panel)

        # 绘图按钮
        self.draw_button = QPushButton("绘制图表")
        self.draw_button.clicked.connect(self.draw_charts)
        self.draw_button.setEnabled(False)
        button_layout.addWidget(self.draw_button)

        # 刷新按钮
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh_folders)
        button_layout.addWidget(refresh_button)

        right_layout.addWidget(button_panel)
        right_layout.addStretch()

        # 添加面板到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)

        # 设置分割器比例
        splitter.setSizes([250, 350, 200])

        # 初始化文件夹列表
        self.refresh_folders()
        
        # 存储选择的数据
        self.selected_dataset = None
        self.selected_index = None
        
        # 加载保存的选项
        self.load_options()

        
    def refresh_folders(self):
        """刷新文件夹列表"""
        # 清空列表
        self.dataset_list.clear()
        self.index_list.clear()
        self.selected_dataset = None
        self.selected_index = None
        self.draw_button.setEnabled(False)
        
        # 检查result目录是否存在
        result_path = "./result"
        if not os.path.exists(result_path):
            return
            
        # 获取数据集文件夹
        datasets = [f for f in os.listdir(result_path) 
                   if os.path.isdir(os.path.join(result_path, f))]
        
        # 添加到列表
        for dataset in datasets:
            self.dataset_list.addItem(dataset)
            
    def on_dataset_selected(self, index):
        """当数据集被选中时"""
        self.selected_dataset = self.dataset_list.currentItem().text()
        self.load_index_list()
        
    def load_index_list(self):
        """加载编号列表"""
        # 清空编号列表
        self.index_list.clear()
        self.selected_index = None
        self.draw_button.setEnabled(False)
        
        # 构建数据集路径
        dataset_path = os.path.join("./result", self.selected_dataset)
        
        # 检查路径是否存在
        if not os.path.exists(dataset_path):
            return
            
        # 获取编号文件夹
        indexes = [f for f in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, f))]
        
        # 添加到列表
        for index in indexes:
            self.index_list.addItem(index)
            
    def on_index_selected(self, index):
        """当编号被选中时"""
        self.selected_index = self.index_list.currentItem().text()
        self.draw_button.setEnabled(True)
        
    def draw_charts(self):
        """绘制图表"""
        if self.selected_dataset and self.selected_index:
            try:
                # 保存当前选项
                self.save_options()
                
                # 获取选中的选项
                selected_options = []
                for i, checkbox in enumerate(self.option_checkboxes):
                    if checkbox.isChecked():
                        selected_options.append(self.plot_options[i]['key'])
                
                # 使用更新后的Draw类
                drawer = Draw(selected_options)
                drawer.run(self.selected_dataset, self.selected_index)
            except Exception as e:
                print(f"绘图时出错: {e}")
                
    def save_options(self):
        """保存选项到文件"""
        options_state = []
        for i, checkbox in enumerate(self.option_checkboxes):
            options_state.append({
                'key': self.plot_options[i]['key'],
                'checked': checkbox.isChecked()
            })
            
        try:
            # 确保配置目录存在
            config_dir = './config'
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            with open('./config/gui_options.json', 'w') as f:
                json.dump(options_state, f)
        except Exception as e:
            print(f"保存选项时出错: {e}")
            
    def load_options(self):
        """从文件加载选项"""
        try:
            if os.path.exists('./config/gui_options.json'):
                with open('./config/gui_options.json', 'r') as f:
                    options_state = json.load(f)
                    
                # 应用保存的选项
                for saved_option in options_state:
                    for i, option in enumerate(self.plot_options):
                        if option['key'] == saved_option['key']:
                            self.option_checkboxes[i].setChecked(saved_option['checked'])
                            break
        except Exception as e:
            print(f"加载选项时出错: {e}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()