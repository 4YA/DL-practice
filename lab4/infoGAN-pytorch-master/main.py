from model import *
from trainer import Trainer
from argparse import ArgumentParser
import torch
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
fe = FrontEnd()
d = D()
q = Q()
g = G()
parser = ArgumentParser()
parser.add_argument("-m", dest="mode" ,type=int)
parser.add_argument("-i", dest="index" ,type=int)
parser.add_argument("-e", dest="epoch" ,type=int)
args = parser.parse_args()

for i in [fe, d, q, g]:
  i.cuda()
  i.apply(weights_init)

trainer = Trainer(g, fe, d, q)
if args.mode == 0:
  trainer.train()
else:
   
  dis_c = torch.FloatTensor(10, 10).cuda()
  con_c = torch.FloatTensor(10, 2).cuda()
  noise = torch.FloatTensor(10, 52).cuda()

  dis_c = Variable(dis_c)
  con_c = Variable(con_c)
  noise = Variable(noise)

  g.load_state_dict(torch.load("G_{}.h5".format(args.epoch)))
  g.eval()
  idx = args.index
  one_hot = np.zeros((10, 10))
  one_hot[range(10), idx] = 1
  fix_noise = torch.Tensor(10, 52).uniform_(-1, 1)
  c = np.linspace(-1, 1, 10).reshape(-1, 1)
 
  c1 = np.hstack([c, np.zeros_like(c)])
  c2 = np.hstack([np.zeros_like(c), c])
  noise.data.copy_(fix_noise)
  dis_c.data.copy_(torch.Tensor(one_hot))
  con_c.data.copy_(torch.from_numpy(c1))
  z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)
  x_save = g(z)
  save_image(x_save.data, './c1_{}.png'.format(idx), nrow=1)
   
  con_c.data.copy_(torch.from_numpy(c2))
  z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)
  x_save = g(z)
  save_image(x_save.data, './c2_{}.png'.format(idx), nrow=1)
