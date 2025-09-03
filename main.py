import torch

from utils.Comparator.Basic import BasicComparator
from utils.Comparator.MNIST import MNISTComparer
from utils.Comparator.PTB import PTBComparer
from utils.draw import Draw

flag = False
flag = True

if flag:
    print(f"cuda:{torch.cuda.is_available()}")
    c: BasicComparator = MNISTComparer()
    c.run()
    d = Draw()
    d.draw_output()
else:
    d = Draw()
    d.draw_output()
