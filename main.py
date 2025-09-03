import torch

from utils.Comparator.PTB import PTBComparer
from utils.draw import Draw

flag = False
flag = True

if flag:
    print(f"cuda:{torch.cuda.is_available()}")
    c = PTBComparer()
    c.run()
    d = Draw()
    d.draw_output()
else:
    d = Draw()
    d.draw_output()
