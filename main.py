import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='GRU BasicTester')
    parser.add_argument('--test', choices=['PTB', 'MNIST'], help='选择要运行的测试类型')
    parser.add_argument('--draw-only', action='store_true', help='仅绘制图表，不运行测试')
    parser.add_argument('--find', action='store_true', help='是否进行参数搜索')
    parser.add_argument('--group', type=int, default=0, help='选择要训练的模型组 (默认: 0)')

    args = parser.parse_args()

    if args.draw_only:
        from PyQt5.QtWidgets import QApplication
        from utils.Draw import MainWindow

        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    elif args.find:
        from utils.Finder import MNISTFinder
        finder_map = {
            'MNIST': MNISTFinder,
        }
        f = finder_map[args.test]()
        f.run()
    else:
        from utils.Comparator import MNISTComparer
        from utils.Comparator import PTBComparer
        comparator_map = {
            'PTB': PTBComparer,
            'MNIST': MNISTComparer,
        }
        c = comparator_map[args.test]()
        c.choice(args.group)
        c.run()


if __name__ == "__main__":
    main()
