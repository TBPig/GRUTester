import sys
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import glob
import pandas as pd

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QListWidget, 
                             QSplitter, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox,
                             QTextEdit)
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
    # 如果best_min和best_max相等，需要添加一个小的边距以避免设置相同的Y轴限制
    if best_min == best_max:
        margin = 0.1 if best_min == 0 else abs(best_min) * 0.1
        return best_min - margin, best_max + margin
    else:
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
    def __init__(self, selected_options=None, group_options=None, data_point_percentage=0.95):
        self.path = 'result'
        self.selected_options = selected_options
        self.group_options = group_options if group_options is not None else {}
        self.data_point_percentage = data_point_percentage

    def run(self, data_set=None, index=None):
        """
        从指定文件夹中读取所有CSV文件，并绘制曲线图
        如果提供了selected_options，则只绘制选中的图表
        """
        if data_set is None:
            data_set = input("请输入数据集名称：")
        if index is None:
            index = input("请输入模型组编号：")

        # 关闭之前可能存在的图表
        plt.close('all')

        # 处理单个或多个索引
        indexes = index if isinstance(index, list) else [index]

        # 为所有索引收集数据并计算统一的y轴范围
        all_data_dicts = {}  # 存储每个索引的数据
        unified_ylim = {}    # 存储每种图表类型的统一y轴范围

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

        # 为每个索引收集数据
        for idx in indexes:
            # 查找文件夹中的所有CSV文件
            folder_path = os.path.join(self.path, data_set, idx)
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

            if not csv_files:
                print(f"在文件夹 {folder_path} 中未找到CSV文件")
                continue

            # 读取所有数据
            data_dict = {}
            for csv_file in csv_files:
                # 获取文件名（不含扩展名）作为模型名称
                model_name = os.path.splitext(os.path.basename(csv_file))[0]
                data_dict[model_name] = pd.read_csv(csv_file)
            
            all_data_dicts[idx] = data_dict

        # 为每种图表类型计算统一的y轴范围
        for config in plot_configs:
            # 收集所有索引中该类型图表的数据
            all_values = []
            for data_dict in all_data_dicts.values():
                for data in data_dict.values():
                    if config['column'] in data.columns:
                        all_values.extend(data[config['column']].tolist())
            
            # 计算统一的y轴范围
            if all_values:
                all_values = sorted(all_values)
                n = len(all_values)
                keep_count = int(n * self.data_point_percentage)
                
                min_range = float('inf')
                best_min, best_max = all_values[0], all_values[-1]
                
                for i in range(n - keep_count + 1):
                    window_min = all_values[i]
                    window_max = all_values[i + keep_count - 1]
                    window_range = window_max - window_min
                    
                    if window_range < min_range:
                        min_range = window_range
                        best_min, best_max = window_min, window_max
                
                # 如果best_min和best_max相等，需要添加一个小的边距以避免设置相同的Y轴限制
                if best_min == best_max:
                    margin = 0.1 if best_min == 0 else abs(best_min) * 0.1
                    unified_ylim[config['column']] = (best_min - margin, best_max + margin)
                else:
                    margin = (best_max - best_min) * 0.1
                    unified_ylim[config['column']] = (best_min - margin, best_max + margin)

        # 为每个索引创建图表
        for idx in indexes:
            if idx not in all_data_dicts:
                continue
                
            data_dict = all_data_dicts[idx]
            
            # 创建图像和子图
            fig, axes = plt.subplots(1, len(plot_configs), figsize=(6 * len(plot_configs), 5))
            
            # 如果只有一个子图，需要特殊处理
            if len(plot_configs) == 1:
                axes = [axes]
            # 如果没有子图，直接返回
            elif len(plot_configs) == 0:
                print("没有要绘制的图表")
                continue

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

                # 使用统一的Y轴范围
                if config['column'] in unified_ylim:
                    ax.set_ylim(*unified_ylim[config['column']])

                # 绘制曲线
                for i, (model_name, data) in enumerate(data_dict.items()):
                    if config['column'] in data.columns:
                        # 获取分组大小
                        group_size = self.group_options.get(config['column'], 1)
                        
                        # 如果需要分组，则对数据进行分组处理
                        if group_size > 1:
                            grouped_x, grouped_y = group(data['epoch'].values, data[config['column']].values, group_size)
                            ax.plot(grouped_x, grouped_y, label=model_name, color=colors[i])
                        else:
                            ax.plot(data['epoch'], data[config['column']], label=model_name, color=colors[i])
                ax.set_title(config['title'])
                ax.set_xlabel('Epoch')
                ax.set_ylabel(config['ylabel'])
                ax.legend()
                ax.grid(True)

            # 调整子图间距
            plt.tight_layout()
            
            # 保存并显示图表
            folder_path = os.path.join(self.path, data_set, idx)
            plt.savefig(os.path.join(folder_path, 'comparison.png'))
            plt.show()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GRU测试结果可视化工具")
        self.setGeometry(100, 100, 2000, 800)
        
        # 多选模式状态
        self.multiselect_mode = False
        
        # 图表选项
        self.plot_options = [
            {'key': 'train_loss', 'label': '训练损失曲线', 'default': True},
            {'key': 'test_loss', 'label': '测试损失曲线', 'default': True},
            {'key': 'test_acc', 'label': '测试准确率曲线', 'default': True},
            {'key': 'learning_rate', 'label': '学习率曲线', 'default': True}
        ]
        
        # 默认分组大小
        self.default_group_sizes = {
            'train_loss': 1,
            'test_loss': 1,
            'test_acc': 1,
            'learning_rate': 1
        }
        
        # 默认数据点百分比
        self.default_data_point_percentage = 0.95
        
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

        # 切换多选模式按钮
        self.toggle_multiselect_button = QPushButton("启用多选")
        self.toggle_multiselect_button.clicked.connect(self.toggle_multiselect)
        middle_layout.addWidget(self.toggle_multiselect_button)
        
        # 重命名按钮
        self.rename_button = QPushButton("重命名")
        self.rename_button.clicked.connect(self.rename_index)
        self.rename_button.setEnabled(False)
        middle_layout.addWidget(self.rename_button)
        
        # 删除按钮
        self.delete_button = QPushButton("删除")
        self.delete_button.clicked.connect(self.delete_index)
        self.delete_button.setEnabled(False)
        middle_layout.addWidget(self.delete_button)
        
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
        
        # 创建复选框和输入框
        self.option_checkboxes = []
        self.group_spinboxes = []
        for option in self.plot_options:
            # 创建水平布局用于放置复选框和输入框
            option_layout = QHBoxLayout()
            
            # 复选框
            checkbox = QCheckBox(option['label'])
            checkbox.setChecked(option['default'])
            self.option_checkboxes.append(checkbox)
            option_layout.addWidget(checkbox)
            
            # 标签
            group_label = QLabel("分组:")
            option_layout.addWidget(group_label)
            
            # 数值输入框
            spinbox = QSpinBox()
            spinbox.setRange(1, 1000)  # 设置范围1-1000
            spinbox.setValue(self.default_group_sizes.get(option['key'], 1))
            self.group_spinboxes.append(spinbox)
            option_layout.addWidget(spinbox)
            
            option_layout.addStretch()  # 添加弹性空间
            
            options_group_layout.addLayout(option_layout)
            
        # 添加数据点百分比设置
        percentage_layout = QHBoxLayout()
        percentage_label = QLabel("数据点百分比:")
        percentage_layout.addWidget(percentage_label)
        
        self.data_point_percentage_spinbox = QDoubleSpinBox()
        self.data_point_percentage_spinbox.setRange(0.01, 0.99)
        self.data_point_percentage_spinbox.setSingleStep(0.05)
        self.data_point_percentage_spinbox.setValue(self.default_data_point_percentage)
        self.data_point_percentage_spinbox.setDecimals(2)
        percentage_layout.addWidget(self.data_point_percentage_spinbox)
        
        options_group_layout.addLayout(percentage_layout)
        
        right_layout.addWidget(options_group)
        
        # 操作按钮区域
        button_panel = QWidget()
        button_layout = QVBoxLayout(button_panel)

        # 创建水平布局放置绘图按钮和重置按钮
        action_buttons_layout = QHBoxLayout()
        
        # 绘图按钮
        self.draw_button = QPushButton("绘制图表")
        self.draw_button.clicked.connect(self.draw_charts)
        self.draw_button.setEnabled(False)
        action_buttons_layout.addWidget(self.draw_button, 4)  # 比例为4

        # 重置分组按钮
        self.reset_group_button = QPushButton("重置分组")
        self.reset_group_button.clicked.connect(self.reset_group_sizes)
        action_buttons_layout.addWidget(self.reset_group_button, 1)  # 比例为1

        button_layout.addLayout(action_buttons_layout)

        # 刷新按钮
        refresh_button = QPushButton("刷新")
        refresh_button.clicked.connect(self.refresh_folders)
        button_layout.addWidget(refresh_button)

        right_layout.addWidget(button_panel)
        right_layout.addStretch()

        # 最右侧面板 - info.txt 编辑器
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        
        # info.txt 标题
        info_label = QLabel("实验信息编辑")
        info_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(info_label)
        
        # 文本编辑器
        self.info_text_edit = QTextEdit()
        self.info_text_edit.setPlaceholderText("选择一个编号以查看和编辑 info.txt 文件")
        info_layout.addWidget(self.info_text_edit)
        
        # 保存按钮
        self.save_info_button = QPushButton("保存信息")
        self.save_info_button.clicked.connect(self.save_info_txt)
        self.save_info_button.setEnabled(False)
        info_layout.addWidget(self.save_info_button)
        
        # 添加面板到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)
        splitter.addWidget(info_panel)

        # 设置分割器比例
        splitter.setSizes([150, 400, 200, 600])

        # 初始化文件夹列表
        self.refresh_folders()
        
        # 存储选择的数据
        self.selected_dataset = None
        self.selected_index = None
        
        # 加载保存的选项
        self.load_options()

    def closeEvent(self, event):
        """当窗口关闭时，同时关闭所有matplotlib图表窗口"""
        plt.close('all')
        event.accept()
        
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
        # 重置选中的编号
        self.selected_index = None
        self.draw_button.setEnabled(False)
        self.rename_button.setEnabled(False)
        self.delete_button.setEnabled(False)
        
    def load_index_list(self):
        """加载编号列表"""
        # 清空编号列表
        self.index_list.clear()
        self.selected_index = None
        self.draw_button.setEnabled(False)
        self.rename_button.setEnabled(False)
        
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
            
        # 重置多选模式
        self.multiselect_mode = False
        self.index_list.setSelectionMode(QListWidget.SingleSelection)
        self.toggle_multiselect_button.setText("启用多选")
        
    def on_index_selected(self, index):
        """当编号被选中时"""
        selected_items = self.index_list.selectedItems()
        if selected_items:
            # 当只选择一个项目时，加载info.txt文件
            if len(selected_items) == 1:
                self.selected_index = selected_items[0].text()
                self.load_info_txt()
                self.save_info_button.setEnabled(True)
            else:
                self.selected_index = [item.text() for item in selected_items]
                self.info_text_edit.clear()
                self.info_text_edit.setPlaceholderText("请选择单个编号以查看和编辑 info.txt 文件")
                self.save_info_button.setEnabled(False)
                
            self.draw_button.setEnabled(True)
            self.rename_button.setEnabled(len(selected_items) == 1)  # 只有当选择一个项目时才启用重命名
            self.delete_button.setEnabled(True)
        else:
            self.selected_index = None
            self.draw_button.setEnabled(False)
            self.rename_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self.save_info_button.setEnabled(False)
            self.info_text_edit.clear()
            self.info_text_edit.setPlaceholderText("选择一个编号以查看和编辑 info.txt 文件")
    
    def load_info_txt(self):
        """加载info.txt文件内容"""
        if not self.selected_dataset or not self.selected_index:
            return
            
        # 构建info.txt文件路径
        info_path = os.path.join("./result", self.selected_dataset, self.selected_index, "info.txt")
        
        # 检查文件是否存在
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.info_text_edit.setPlainText(content)
            except Exception as e:
                self.info_text_edit.setPlainText(f"无法读取文件: {str(e)}")
        else:
            # 如果文件不存在，清空文本框并提供默认提示
            self.info_text_edit.clear()
            
    def save_info_txt(self):
        """保存info.txt文件内容"""
        if not self.selected_dataset or not self.selected_index:
            return
            
        # 构建info.txt文件路径
        info_path = os.path.join("./result", self.selected_dataset, self.selected_index, "info.txt")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(info_path), exist_ok=True)
            
            # 写入内容
            content = self.info_text_edit.toPlainText()
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # 显示保存成功的消息
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "保存成功", "info.txt 文件已成功保存")
        except Exception as e:
            # 显示错误消息
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, "保存失败", f"无法保存文件: {str(e)}")

    def rename_index(self):
        """重命名选中的编号"""
        if not self.selected_index:
            return
            
        # 获取数据集路径
        dataset_path = os.path.join("./result", self.selected_dataset)
        old_index_path = os.path.join(dataset_path, self.selected_index)
        
        # 弹出输入对话框获取新名称
        from PyQt5.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "重命名", "请输入新的编号名称:", text=self.selected_index)
        
        if ok and new_name and new_name != self.selected_index:
            new_index_path = os.path.join(dataset_path, new_name)
            
            # 检查新名称是否已存在
            if os.path.exists(new_index_path):
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "重命名失败", f"编号 '{new_name}' 已存在！")
                return
                
            try:
                # 重命名文件夹
                os.rename(old_index_path, new_index_path)
                
                # 更新界面
                self.selected_index = new_name
                self.load_index_list()
                
                # 重新选择重命名后的项目
                items = self.index_list.findItems(new_name, Qt.MatchExactly)
                if items:
                    self.index_list.setCurrentItem(items[0])
                    
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "错误", f"重命名失败: {str(e)}")

    def delete_index(self):
        """删除选中的编号"""
        if not self.selected_index:
            return
            
        # 获取数据集路径
        dataset_path = os.path.join("./result", self.selected_dataset)
        index_path = os.path.join(dataset_path, self.selected_index)
        
        # 确认删除操作
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(self, "确认删除", 
                                   f"确定要删除编号 '{self.selected_index}' 及其所有数据吗？此操作不可撤销！",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                # 删除文件夹及其内容
                import shutil
                shutil.rmtree(index_path)
                
                # 更新界面
                self.load_index_list()
                self.info_text_edit.clear()
                self.save_info_button.setEnabled(False)
                
                # 显示删除成功消息
                QMessageBox.information(self, "删除成功", f"编号 '{self.selected_index}' 已成功删除")
            except Exception as e:
                QMessageBox.critical(self, "删除失败", f"删除编号时出错: {str(e)}")

    def draw_charts(self):
        """绘制图表"""
        if self.selected_dataset and self.selected_index:
            try:
                # 保存当前选项
                self.save_options()
                
                # 获取选中的选项
                selected_options = []
                group_options = {}
                for i, checkbox in enumerate(self.option_checkboxes):
                    if checkbox.isChecked():
                        option_key = self.plot_options[i]['key']
                        selected_options.append(option_key)
                        # 获取对应的分组大小
                        group_options[option_key] = self.group_spinboxes[i].value()
                
                # 获取数据点百分比
                data_point_percentage = self.data_point_percentage_spinbox.value()
                
                # 使用更新后的Draw类
                drawer = Draw(selected_options, group_options, data_point_percentage)
                drawer.run(self.selected_dataset, self.selected_index)
            except Exception as e:
                print(f"绘图时出错: {e}")
                
    def reset_group_sizes(self):
        """将所有分组大小重置为1"""
        for spinbox in self.group_spinboxes:
            spinbox.setValue(1)

    def save_options(self):
        """保存选项到文件"""
        options_state = []
        for i, (checkbox, spinbox) in enumerate(zip(self.option_checkboxes, self.group_spinboxes)):
            options_state.append({
                'key': self.plot_options[i]['key'],
                'checked': checkbox.isChecked(),
                'group_size': spinbox.value()
            })
            
        # 保存数据点百分比
        options_state.append({
            'key': 'data_point_percentage',
            'value': self.data_point_percentage_spinbox.value()
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
                    # 处理常规选项
                    for i, option in enumerate(self.plot_options):
                        if option['key'] == saved_option['key']:
                            self.option_checkboxes[i].setChecked(saved_option['checked'])
                            if 'group_size' in saved_option:
                                self.group_spinboxes[i].setValue(saved_option['group_size'])
                            break
                    # 处理数据点百分比选项
                    if saved_option['key'] == 'data_point_percentage':
                        self.data_point_percentage_spinbox.setValue(saved_option['value'])
        except Exception as e:
            print(f"加载选项时出错: {e}")

    def toggle_multiselect(self):
        """切换多选模式"""
        self.multiselect_mode = not self.multiselect_mode
        
        if self.multiselect_mode:
            self.index_list.setSelectionMode(QListWidget.MultiSelection)
            self.toggle_multiselect_button.setText("禁用多选")
        else:
            self.index_list.setSelectionMode(QListWidget.SingleSelection)
            self.toggle_multiselect_button.setText("启用多选")
            # 清除现有选择
            self.index_list.clearSelection()
            
        # 重置选择状态
        self.selected_index = None
        self.draw_button.setEnabled(False)
        self.rename_button.setEnabled(False)
        self.delete_button.setEnabled(False)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()