import torch
import sys
import argparse

from utils.Comparator.Basic import BasicComparator
from utils.Comparator.MNIST import MNISTComparer
from utils.Comparator.PTB import PTBComparer
from utils.Comparator.CopyTask import CopyTaskComparer
from utils.draw import Draw

def main():
    parser = argparse.ArgumentParser(description='GRU Tester')
    parser.add_argument('--test', choices=['PTB', 'MNIST', 'CopyTask'], 
                        default='PTB', help='选择要运行的测试类型')
    parser.add_argument('--draw-only', action='store_true', 
                        help='仅绘制图表，不运行测试')
    parser.add_argument('--epochs', type=int, default=2,
                        help='训练轮数 (默认: 2)')
    
    args = parser.parse_args()

    if args.draw_only:
        d = Draw()
        d.draw_output()
        print(1)
        return

    print(f"cuda:{torch.cuda.is_available()}")
    
    if args.test == 'PTB':
        c: BasicComparator = PTBComparer()
        c.epoch_num = args.epochs
        c.run()
        d = Draw()
        d.draw_output()
    elif args.test == 'MNIST':
        c: BasicComparator = MNISTComparer()
        c.epoch_num = args.epochs
        c.run()
        d = Draw()
        d.draw_output()
    elif args.test == 'CopyTask':
        # 对于CopyTask，需要创建相应的比较器类
        c: BasicComparator = CopyTaskComparer()
        c.epoch_num = args.epochs
        c.run()
        d = Draw()
        d.draw_output()
    else:
        print(f"未知的测试类型: {args.test}")
        sys.exit(1)

if __name__ == "__main__":
    main()