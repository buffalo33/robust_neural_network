#!/usr/bin/env python3 
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024 
batch_size = 128


def test_natural(net, test_loader):
    '''Basic testing function.'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total



######################### ResNet ###############################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,sigma=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.sigma = sigma

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x + self.sigma*torch.randn(x.size()[0],x.size()[1],x.size()[2],x.size()[3],device=device))))
        out = self.bn2(self.conv2(out + self.sigma*torch.randn(out.size()[0],out.size()[1],out.size()[2],out.size()[3],device=device)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


### ResNet
class ResNet(nn.Module):
    # If using a diffrent block than BasicBlock, you must add a argument sigma=0 to the init function of this block.
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], low_dim=10,sigma=0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.out_planes = 32
        self.sigma = sigma

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.out_planes, num_blocks[0], stride=1,sigma=self.sigma)
        self.layer2 = self._make_layer(block, self.out_planes*2, num_blocks[1], stride=2,sigma=self.sigma)
        self.layer3 = self._make_layer(block, self.out_planes*2*2, num_blocks[2], stride=2,sigma=self.sigma)
        self.layer4 = self._make_layer(block, self.out_planes*2*2*2, num_blocks[3], stride=2,sigma=self.sigma)
        self.fc1 = nn.Linear(self.out_planes*2*2*2*block.expansion, self.out_planes*2*2)
        self.fc2 = nn.Linear(self.out_planes*2*2, self.out_planes*2)
        self.fc3 = nn.Linear(self.out_planes*2, low_dim)
        # self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride,sigma=0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,sigma=self.sigma))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.div(x*255, 1, rounding_mode='trunc')/255
        out = F.relu(self.bn1(self.conv1(x + self.sigma*torch.randn(x.size()[0],x.size()[1],x.size()[2],x.size()[3],device=device))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # out = self.l2norm(out)
        out = F.log_softmax(out, dim=1)
        return out


############################ Basic ###############################
'''Basic neural network architecture (from pytorch doc).'''
class BasicNet(nn.Module):

    def __init__(self):
        super().__init__()
        planes = 64
        self.conv1 = nn.Conv2d(3, planes, 5) #32 -4
        self.pool = nn.MaxPool2d(3, 3)       #28 /3
        self.bn = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 5) #9 -4
        self.fc1 = nn.Linear(planes, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        
    def forward(self, x):
        x = self.bn(self.pool(F.relu(self.conv1(x))))
        x = self.bn(self.pool(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x



############################ EfficientNet ################################

"""
Original file is located at
    https://colab.research.google.com/drive/1zW4yQoNZyg9twfbcs_orwtmQ36u97ETl
"""

"""# Basic Layers"""

class ConvBnAct(nn.Module):
  """Layer grouping a convolution, batchnorm, and activation function"""
  def __init__(self, n_in, n_out, kernel_size=3, 
               stride=1, padding=0, groups=1, bias=False,
               bn=True, act=True):
    super().__init__()
    
    self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                          stride=stride, padding=padding,
                          groups=groups, bias=bias)
    self.bn = nn.BatchNorm2d(n_out) if bn else nn.Identity()
    self.act = nn.SiLU() if act else nn.Identity()
  
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)
    return x

class SEBlock(nn.Module):
  """Squeeze-and-excitation block"""
  def __init__(self, n_in, r=24):
    super().__init__()

    self.squeeze = nn.AdaptiveAvgPool2d(1)
    self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in//r, kernel_size=1),
                                    nn.SiLU(),
                                    nn.Conv2d(n_in//r, n_in, kernel_size=1),
                                    nn.Sigmoid())
  
  def forward(self, x):
    y = self.squeeze(x)
    y = self.excitation(y)
    return x * y

class DropSample(nn.Module):
  """Drops each sample in x with probability p during training"""
  def __init__(self, p=0):
    super().__init__()

    self.p = p
  
  def forward(self, x):
    if (not self.p) or (not self.training):
      return x
    
    batch_size = len(x)
    random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
    bit_mask = self.p<random_tensor

    x = x.div(1-self.p)
    x = x * bit_mask
    return x

class MBConvN(nn.Module):
  """MBConv with an expansion factor of N, plus squeeze-and-excitation"""
  def __init__(self, n_in, n_out, expansion_factor,
               kernel_size=3, stride=1, r=24, p=0):
    super().__init__()

    padding = (kernel_size-1)//2
    expanded = expansion_factor*n_in
    self.skip_connection = (n_in == n_out) and (stride == 1)

    self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded, kernel_size=1)
    self.depthwise = ConvBnAct(expanded, expanded, kernel_size=kernel_size, 
                               stride=stride, padding=padding, groups=expanded,)
    self.se = SEBlock(expanded, r=r)
    self.reduce_pw = ConvBnAct(expanded, n_out, kernel_size=1,
                               act=False)
    self.dropsample = DropSample(p)
  
  def forward(self, x):
    residual = x

    x = self.expand_pw(x)
    x = self.depthwise(x)
    x = self.se(x)
    x = self.reduce_pw(x)

    if self.skip_connection:
      x = self.dropsample(x)
      x = x + residual

    return x

class MBConv1(MBConvN):
  def __init__(self, n_in, n_out, kernel_size=3,
               stride=1, r=24, p=0):
    super().__init__(n_in, n_out, expansion_factor=1,
                     kernel_size=kernel_size, stride=stride,
                     r=r, p=p)

class MBConv6(MBConvN):
  def __init__(self, n_in, n_out, kernel_size=3,
               stride=1, r=24, p=0):
    super().__init__(n_in, n_out, expansion_factor=6,
                     kernel_size=kernel_size, stride=stride,
                     r=r, p=p)

"""# Scaling Functions"""

def create_stage(n_in, n_out, num_layers, layer_type=MBConv6, 
                 kernel_size=3, stride=1, r=24, p=0):
  """Creates a Sequential consisting of [num_layers] layer_type"""
  layers = [layer_type(n_in, n_out, kernel_size=kernel_size,
                       stride=stride, r=r, p=p)]
  layers += [layer_type(n_out, n_out, kernel_size=kernel_size,
                        r=r, p=p) for _ in range(num_layers-1)]
  layers = nn.Sequential(*layers)
  return layers

def scale_width(w, w_factor):
  """Scales width given a scale factor"""
  w *= w_factor
  new_w = (int(w+4) // 8) * 8
  new_w = max(8, new_w)
  if new_w < 0.9*w:
     new_w += 8
  return int(new_w)

"""# EfficientNet"""

class EfficientNet(nn.Module):
 
  """Generic EfficientNet that takes in the width and depth scale factors and scales accordingly"""
  def __init__(self, w_factor=1, d_factor=1,
               out_sz=10):
    super().__init__()

    base_widths = [(32, 16), (16, 24), (24, 40),
                   (40, 80), (80, 112), (112, 192),
                   (192, 320), (320, 1280)]
    base_depths = [1, 2, 2, 3, 3, 4, 1]

    scaled_widths = [(scale_width(w[0], w_factor), scale_width(w[1], w_factor)) 
                     for w in base_widths]
    scaled_depths = [math.ceil(d_factor*d) for d in base_depths]
    
    kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
    strides = [1, 2, 2, 2, 1, 2, 1]
    ps = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]

    self.stem = ConvBnAct(3, scaled_widths[0][0], stride=2, padding=1)
    
    stages = []
    for i in range(7):
      layer_type = MBConv1 if (i == 0) else MBConv6
      r = 4 if (i == 0) else 24
      stage = create_stage(*scaled_widths[i], scaled_depths[i],
                           layer_type, kernel_size=kernel_sizes[i], 
                           stride=strides[i], r=r, p=ps[i])
      stages.append(stage)
    self.stages = nn.Sequential(*stages)

    self.pre_head = ConvBnAct(*scaled_widths[-1], kernel_size=1)

    self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(),
                              nn.Linear(scaled_widths[-1][1], out_sz))

  def feature_extractor(self, x):
    x = torch.div(x*255, 1, rounding_mode='trunc')/255
    x = self.stem(x)
    x = self.stages(x)
    x = self.pre_head(x)
    return x

  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.head(x)
    x = F.log_softmax(x, dim=1)
    return x


class EfficientNetB0(EfficientNet):
  def __init__(self, out_sz=10):
    w_factor = 1
    d_factor = 1
    super().__init__(w_factor, d_factor, out_sz)
class EfficientNetB1(EfficientNet):
  def __init__(self, out_sz=10):
    w_factor = 1
    d_factor = 1.1
    super().__init__(w_factor, d_factor, out_sz)
class EfficientNetB2(EfficientNet):
  def __init__(self, out_sz=10):
    w_factor = 1.1
    d_factor = 1.2
    super().__init__(w_factor, d_factor, out_sz)
class EfficientNetB3(EfficientNet):
  def __init__(self, out_sz=10):
    w_factor = 1.2
    d_factor = 1.4
    super().__init__(w_factor, d_factor, out_sz)
class EfficientNetB4(EfficientNet):
  def __init__(self, out_sz=10):
    w_factor = 1.4
    d_factor = 1.8
    super().__init__(w_factor, d_factor, out_sz)
class EfficientNetB5(EfficientNet):
  def __init__(self, out_sz=10):
    w_factor = 1.6
    d_factor = 2.2
    super().__init__(w_factor, d_factor, out_sz)
class EfficientNetB6(EfficientNet):
  def __init__(self, out_sz=10):
    w_factor = 1.8
    d_factor = 2.6
    super().__init__(w_factor, d_factor, out_sz)
class EfficientNetB7(EfficientNet):
  def __init__(self, out_sz=10):
    w_factor = 2
    d_factor = 3.1
    super().__init__(w_factor, d_factor, out_sz)



class Net(ResNet):
#class Net(BasicNet):
#class Net(EfficientNetB0):
#class Net(EfficientNetB7):

  model_file="models/default_model.pth"
  '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''

  #def __init__(self,sigma=0):
  #  super().__init__(sigma=sigma)

  def save(self, model_file):
      '''Helper function, use it to save the model weights after training.'''
      torch.save(self.state_dict(), model_file)

  def load(self, model_file):
      self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

      
  def load_for_testing(self, project_dir='./'):
      '''This function will be called automatically before testing your
          project, and will load the model weights from the file
          specify in Net.model_file.
          
          You must not change the prototype of this function. You may
          add extra code in its body if you feel it is necessary, but
          beware that paths of files used in this function should be
          refered relative to the root of your project directory.
      '''        
      self.load(os.path.join(project_dir, Net.model_file))




def train_model(net, train_loader, test_loader, pth_filename, num_epochs, alpha=0, epsilon=0.07, defence='adversarial', sigma=0, odr=2):
    '''Basic training function (from pytorch doc.)'''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(net.parameters())

    LOSS = []
    ACC = []

    for epoch in range(num_epochs):  # loop over the dataset multiple times

      running_loss = 0.0
      with tqdm(train_loader) as bt:
        for i, data in enumerate(bt):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if defence == 'randomized_input':
                # Randomize inputs
                inputs += sigma*torch.randn_like(inputs, device=device)
                outputs = net(inputs)
            else:
                outputs = net(inputs)

            if (defence == 'adversarial') or (defence == 'mix'):
              #inputs.requires_grad = True

              #loss_before = criterion(outputs, labels)

              if defence == 'mix':
                  order = np.random.choice([np.inf, 2])
              else:
                  order = odr

              inputs = pgd_attack(net, inputs, labels, order, epsilon)

              outputs = net(inputs)

              loss = criterion(outputs, labels)
              #loss = alpha*loss_before + (1-alpha)*loss_after

            else:
              loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        test_nat = test_natural(net, test_loader)
        #test_atk2, _ = test_under_attack(net, test_loader, pgd_attack, order=2, epsilon=0.3, alpha=2/255, iters=10 )
        #test_atkinf, _ = test_under_attack(net, test_loader, pgd_attack, order=np.inf, epsilon=0.3, alpha=2/255, iters=10 )

        print('loss=',running_loss / (i+1), 'acc=',test_nat)#, 'acc2=', test_atk2, 'accinf', test_atkinf)
        LOSS.append(running_loss / (i+1))
        ACC.append(test_nat)
        net.save(pth_filename)
    print('Model saved in {}'.format(pth_filename))

    plt.plot(np.arange(len(LOSS)), LOSS)
    plt.savefig('loss.png')
    plt.plot(np.arange(len(ACC)), ACC)
    plt.savefig('acc.png')



def fgsm_attack(model, images, labels, epsilon=0.3) :

    images.requires_grad = True
    outputs = model(images)
    loss = nn.NLLLoss()

    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    adv_images = images + epsilon*images.grad.sign()
    images = torch.clamp(adv_images, min=0, max=1).detach_()
    model.zero_grad()

    return images


def pgd_attack(model, images, labels, order=2, eps=0.078, alpha=2/255, iters=5) :

    alpha=eps/iters
    delta = torch.zeros_like(images)
    loss_fn = nn.NLLLoss()

    delta.requires_grad_()
    for ii in range(iters):
        outputs = model(images + delta)
        loss = loss_fn(outputs, labels)
        model.zero_grad()

        loss.backward()

        if order == np.inf:
            delta.data = delta.data + alpha*torch.sign(delta.grad.data)
            delta.data = torch.clamp(images.data + delta.data, 0, 1) - images.data
            delta.data = torch.clamp(delta.data, -eps, eps)

        elif order == 2:
            grad = delta.grad.data
            norm = torch.norm(grad, dim=(1,2,3), p=2)
            norm = torch.maximum(norm, torch.ones_like(norm) * 1e-6)
            grad = torch.mul(grad, 1/norm[:,None,None,None])
            delta.data = delta.data + alpha*grad
            delta.data = torch.clamp(images.data + delta.data, 0, 1) - images.data

            norm = torch.norm(delta.data, dim=(1,2,3), p=2)
            factor = torch.minimum(eps / norm, torch.ones_like(norm))
            delta.data = torch.mul(delta.data, factor[:,None,None,None])

        delta.grad.data.zero_()

    x_adv = torch.clamp(images + delta, 0, 1)
    model.zero_grad()
    return x_adv



def test_under_attack(model, test_loader, attack, order=2, epsilon=0.3, alpha=2/255, iters=10 ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        
        init_pred = model(images).max(1, keepdim=True)[1]

        # Call Attack
        adv_images = attack(model, images, labels, order, epsilon, alpha=2/255, iters=10)

        # Re-classify the perturbed image
        final_pred = model(adv_images).max(1, keepdim=True)[1]

        for i_pred, f_pred, adv_img, lab in zip(init_pred, final_pred, adv_images, labels):
          if i_pred.item() == lab.item():
              correct += 1
          """
              # Special case for saving 0 epsilon examples
              if (epsilon == 0) and (len(adv_examples) < 5):
                  adv_ex = adv_img.squeeze().detach().cpu().numpy()
                  adv_examples.append( (i_pred.item(), f_pred.item(), adv_ex) )
          else:
              # Save some adv examples for visualization later
              if len(adv_examples) < 5:
                  adv_ex = adv_img.squeeze().detach().cpu().numpy()
                  adv_examples.append( (i_pred.item(), f_pred.item(), adv_ex) )
          """
        #elif f_pred.item() == lab.item():
        #  correct += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(valid_size)*100

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def get_train_loader(dataset, valid_size=valid_size, batch_size=batch_size):
    '''Split dataset into [train:valid] and return a DataLoader for the training part.'''

    indices = list(range(len(dataset)))
    train_sampler = torch.utils.data.SubsetRandomSampler(indices[valid_size:])
    train = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)

    return train

def get_validation_loader(dataset, valid_size=valid_size, batch_size=batch_size):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)

    return valid



def main():

    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")

    parser.add_argument('-a', '--alpha', type=float, default=0,
                        help="Alpha for defence during training")

    parser.add_argument('-p', '--epsilon', type=float, default=0.05,
                        help="Epsilon for during training")

    parser.add_argument('-d', '--defence', type=str, default="",
                      help="Use defence mecanisme")

    parser.add_argument('-s', '--sigma', type=float, default=0,
                  help="Sigma used for randomized training.")
      
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    if args.defence == 'randomized_net':
      sigma = args.sigma
      net = Net(sigma=sigma)
    else:
      net = Net()

    net.to(device)

    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transforms.ToTensor()) 
    valid_loader = get_validation_loader(cifar, valid_size, batch_size=batch_size)

    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        print("Training model")
        print(args.model_file)
        #net.load(args.model_file)

        train_transform = transforms.Compose([transforms.ToTensor()]) 
        cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=train_transform)
        train_loader = get_train_loader(cifar, valid_size, batch_size=batch_size)
        train_model(net, train_loader, valid_loader, args.model_file, args.num_epochs, alpha=args.alpha, epsilon=args.epsilon, defence=args.defence,sigma=args.sigma)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print("Testing with model from '{}'. ".format(args.model_file))

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.

    net.load(args.model_file)

    acc = test_natural(net, valid_loader)
    print("Model natural accuracy (valid): {}".format(acc))

    if args.model_file != Net.model_file:
        print("Warning: '{0}' is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              "you should rename/link '{0}' to '{1}'.".format(args.model_file, Net.model_file))


if __name__ == '__main__':
  main()

