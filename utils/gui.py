import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QListWidget, 
                             QSplitter)
from PyQt5.QtCore import Qt
from utils.draw import Draw


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GRU测试结果可视化工具")
        self.setGeometry(100, 100, 800, 600)
        
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

        # 右侧面板 - 编号选择
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 编号标题
        index_label = QLabel("编号选择")
        index_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(index_label)

        # 编号列表
        self.index_list = QListWidget()
        self.index_list.clicked.connect(self.on_index_selected)
        right_layout.addWidget(self.index_list)

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

        # 添加面板到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # 设置分割器比例
        splitter.setSizes([300, 500])

        # 初始化文件夹列表
        self.refresh_folders()
        
        # 存储选择的数据
        self.selected_dataset = None
        self.selected_index = None

        
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
                drawer = Draw()
                drawer.run(self.selected_dataset, self.selected_index)
            except Exception as e:
                print(f"绘图时出错: {e}")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()