import torch
import sys
import argparse

def main():
    print(f"cuda:{torch.cuda.is_available()}")
    parser = argparse.ArgumentParser(description='GRU Tester')
    parser.add_argument('--test', choices=['PTB', 'MNIST', 'CopyTask'], help='选择要运行的测试类型')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数 (默认: 2)')
    parser.add_argument('--draw-only', action='store_true', help='仅绘制图表，不运行测试')
    parser.add_argument('--group', type=int, default=0, help='选择要训练的模型组 (默认: 0)')

    args = parser.parse_args()

    if args.draw_only:
        from PyQt5.QtWidgets import QApplication
        from utils.Draw import MainWindow

        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())

    from utils.Comparator.Basic import BasicComparator
    from utils.Comparator.MNIST import MNISTComparer
    from utils.Comparator.PTB import PTBComparer
    from utils.Comparator.CopyTask import CopyTaskComparer

    comparator_map = {
        'PTB': PTBComparer,
        'MNIST': MNISTComparer,
        'CopyTask': CopyTaskComparer
    }
    c: BasicComparator = comparator_map[args.test]()
    c.epochs = args.epochs
    c.choice(args.group)
    c.run()

if __name__ == "__main__":
    main()