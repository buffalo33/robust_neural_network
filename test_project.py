import os, os.path, sys
import argparse
import importlib 
import importlib.abc
import torch, torchvision
from torch.nn.modules.container import ModuleList
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import model as mo

torch.seed()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

valid_size = 1024
batch_size = 128

def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if os.path.exists(project_dir) and os.path.isdir(project_dir) and os.path.isfile(module_filename):
        print("Found valid project in '{}'.".format(project_dir))
    else:
        print("Fatal: '{}' is not a valid project directory.".format(project_dir))
        raise FileNotFoundError 

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)

    return project_module

def test_natural(net, test_loader, num_samples=1):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            images, labels = data[0].to(device), data[1].to(device)
            #total = 0
            #correct = 0
            for _ in range(num_samples):
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    print(total)
    return correct / total



def fgsm_attack(model, images, labels, epsilon=0.3, alpha=2/255, iters=20) :

    images.requires_grad = True
    outputs = model(images)
    loss = nn.NLLLoss()

    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    adv_images = images + epsilon*images.grad.sign()
    images = torch.clamp(adv_images, min=0, max=1).detach_()

    return images

def pgd_attack(model, images, labels, order=2, eps=0.078, alpha=4/255, iters=5) :
  
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


def nes_attack(model, images, labels, order=2, eps=0.078,
               alpha=2/255, iters=30, N=100, eta=100):

    delta = torch.zeros_like(images)

    for _ in range(iters):

        good_pred = torch.argmax(model(images+delta), dim=1) == labels
        d = delta[good_pred]
        img = images[good_pred] + d
        lab = labels[good_pred]
        g = torch.zeros_like(img)

        for _ in range(N):
            noise = torch.randn_like(img) 
            g += torch.mul(~(torch.argmax(model(torch.clamp(img + eta * noise, 0, 1)), dim=1)==lab)[: , None, None, None], noise)
            g -= torch.mul(~(torch.argmax(model(torch.clamp(img - eta * noise, 0, 1)), dim=1)==lab)[: , None, None, None], noise)

        g = g/(2*N*eta)

        # perform the step
        d = lp_step_norm(img=img, delta=d, p=order,
                  grad=g, alpha=alpha, eps=eps)
        delta[good_pred] = d
        
    return torch.clamp(images+delta, 0, 1) #, 2 * N * torch.ones(_shape[0])



def lp_step_norm(img, delta, p, grad, alpha, eps):

    if p == np.inf:
        delta = delta + alpha*torch.sign(grad)
        delta = torch.clamp(img + delta, 0, 1) - img
        delta = torch.clamp(delta, -eps, eps)
        return delta

    elif p == 2:
        norm = torch.norm(grad, dim=(1,2,3), p=2)
        norm = torch.maximum(norm, torch.ones_like(norm) * 1e-6)
        grad = torch.mul(grad, 1/norm[:,None,None,None])
        delta = delta + alpha*grad
        delta = torch.clamp(img + delta, 0, 1) - img

        norm = torch.norm(delta, dim=(1,2,3), p=2)
        norm = torch.maximum(norm, torch.ones_like(norm) * 1e-6)
        factor = torch.minimum(eps / norm, torch.ones_like(norm))
        delta = torch.mul(delta, factor[:,None,None,None])
        return delta


def test_under_attack(model, test_loader, attack, order=2, epsilon=0.3, alpha=2/255, iters=10, N=100 ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)
        
        init_pred = model(images).max(1, keepdim=True)[1]

        # Call Attack
        adv_images = attack(model=model, images=images, labels=labels, order=order, eps=epsilon, alpha=alpha, iters=iters)

        # Re-classify the perturbed image
        final_pred = model(adv_images).max(1, keepdim=True)[1]

        for i_pred, f_pred, adv_img, lab in zip(init_pred, final_pred, adv_images, labels):
          if f_pred.item() == lab.item():
              correct += 1
              '''
              # Special case for saving 0 epsilon examples
              if (epsilon == 0) and (len(adv_examples) < 5):
                  adv_ex = adv_img.squeeze().detach().cpu().numpy()
                  adv_examples.append( (i_pred.item(), f_pred.item(), adv_ex) )
          else:
              # Save some adv examples for visualization later
              if len(adv_examples) < 5:
                  adv_ex = adv_img.squeeze().detach().cpu().numpy()
                  adv_examples.append( (i_pred.item(), f_pred.item(), adv_ex) )
              '''
        #elif f_pred.item() == lab.item():
        #  correct += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(valid_size)

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def get_validation_loader(dataset, valid_size=valid_size, batch_size=batch_size):
    '''Split dataset into [train:valid] and return a DataLoader for the validation part.'''

    indices = list(range(len(dataset)))
    valid_sampler = torch.utils.data.SubsetRandomSampler(indices[:valid_size])
    valid = torch.utils.data.DataLoader(dataset, sampler=valid_sampler, batch_size=batch_size)
    return valid

def test_debug(model_type=""):
  #plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex", "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14})
  plt.figure(figsize=(14, 7))
  plt.grid(b=True, which='major', color='#999999', linestyle='--')
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.25)

  transform = transforms.Compose([transforms.ToTensor()])
  cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
  valid_loader = get_validation_loader(cifar, batch_size=128)

  epsilons = np.array([0, 0.04,0.08])

  net = mo.Net()
  net.to(device)
  net.load("models/"+model_type)
  accuracies = []
  for e_idx, epsilon in enumerate(epsilons):
    print("e_idx "+str(e_idx))
    if epsilon==0.:
      acc = test_natural(net=net, test_loader=valid_loader)
    else:
      print(epsilon)
      acc, adv_examples = test_under_attack(model=net, test_loader=valid_loader, attack=pgd_attack, epsilon=epsilon, alpha=2/255, iters=30 )
    print("acc")
    print(acc)
    accuracies.append(acc)

  lab = f'$\sigma$={model_type}'
  ls='-'
  plt.plot(epsilons, accuracies, label=lab, ls=ls)
  
  plt.title(r"Randomized training under attack.", fontsize=30, pad=15)
  plt.xlabel(r"$\epsilon$ of PGD attack with $l_{2}$ norm.", fontsize=25, labelpad=15)
  plt.ylabel(r"Accuracy", fontsize=25, labelpad=15)
  plt.legend(loc="best")
  plt.savefig('debug_'+model_type+'.pdf', backend='pdf')
  plt.show()
  return 0


def compare_models(models,graph_name,curves_name):
  plt.figure(figsize=(14, 7))
  plt.grid(b=True, which='major', color='#999999', linestyle='--')
  plt.minorticks_on()
  plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.25)

  transform = transforms.Compose([transforms.ToTensor()])
  cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
  valid_loader = get_validation_loader(cifar, batch_size=128)

  epsilons = np.arange(0, 0.09, 0.01)
  
  for model_idx, model_type in enumerate(models):
    net = mo.Net()
    net.to(device)
    net.load("models/"+model_type)
    accuracies = []
    for e_idx, epsilon in enumerate(epsilons):
      print("e_idx "+str(e_idx))
      if epsilon==0.:
        acc = test_natural(net=net, test_loader=valid_loader)
      else:
        print(epsilon)
        acc, adv_examples = test_under_attack(model=net, test_loader=valid_loader, attack=nes_attack, epsilon=epsilon, alpha=2/255, iters=30 )  
      print("acc")
      print(acc)
      accuracies.append(acc)

    lab = f'{curves_name[model_idx]}'
    ls='-'
    plt.plot(epsilons, accuracies, label=lab, ls=ls)
  
  plt.title(f'{graph_name}', fontsize=30, pad=15)
  plt.xlabel(r"$\epsilon$ of PGD attack with $l_{2}$ norm.", fontsize=25, labelpad=15)
  plt.ylabel(r"Accuracy", fontsize=25, labelpad=15)
  plt.legend(loc="best")
  plt.savefig(graph_name+'.pdf', backend='pdf')
  plt.show()

  plt.imshow(adv_examples[2])
  plt.savefig('testNES.pdf', backend='pdf')

  return 0



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", metavar="project-dir", nargs="?", default=os.getcwd(),
                        help="Path to the project directory to test.")
    parser.add_argument("-b", "--batch-size", type=int, default=128,
                        help="Set batch size.")
    parser.add_argument("-s", "--num-samples", type=int, default=1,
                        help="Num samples for testing (required to test randomized networks).")
    parser.add_argument("-e", "--epsilon", type=float, default=0,
                        help="Epsilon that mutiply the fgsm")
    parser.add_argument("-a", "--attack", type=str, default='PGD',
                        help="Type of attack")
    parser.add_argument("-m", "--model_type", type=str, default='default_model.pth',
                        help="Model file name.")

    args = parser.parse_args()
    project_module = load_project(args.project_dir)

    #print(args.model_type)
    #test_debug(args.model_type)
    #models = ["test_rand_0.pth","test_rand_1.pth","test_rand_3.pth","test_rand_5.pth"]
    #curves_name = ["$\sigma = 0$","$\sigma = 0.1$","$\sigma = 0.3$","$\sigma = 0.5$"]
    #compare_models(models=models,graph_name="Randomized input training under attack.",curves_name=curves_name)
    #
    #models = ["test_adv.pth","test_rand_1.pth"]
    #curves_name = ["$\sigma = 0.1$","$\epsilon = 0.03$"]
    #compare_models(models=models,graph_name="Adversarial training vs Randomized input.",curves_name=curves_name)

    #models = ["test_rand_net_0.pth","test_rand_net_1.pth","test_rand_net_3.pth"]
    #curves_name = ["$\sigma = 0$","$\sigma = 0.1$","$\sigma = 0.3$"]
    #compare_models(models=models,graph_name="Randomized network training under attack.",curves_name=curves_name)
    
    #models = ["test_rand_net_1.pth","test_rand_1.pth",]
    #curves_name = ["Network $\sigma = 0.1$","Input $\sigma = 0.1$"]
    #compare_models(models=models,graph_name="Randomized network vs. Randomized Input.",curves_name=curves_name)

    def load_net():
      net = project_module.Net()
      net.to(device)
      #net.load_for_testing(project_dir=args.project_dir)
      return net
#
    transform = transforms.Compose([transforms.ToTensor()])
    cifar = torchvision.datasets.CIFAR10('./data/', download=True, transform=transform)
    valid_loader = get_validation_loader(cifar, batch_size=args.batch_size)
#   
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "pgf.texsystem": "pdflatex", "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14})
    plt.figure(figsize=(14, 7))
    plt.grid(b=True, which='major', color='#999999', linestyle='--')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='--', alpha=0.25)
  #
    model = load_net()
    epsilons = np.arange(0, 0.4, 0.04)#[0, 0.01, 0.03, 0.05, 0.1]
    odr = 2#np.inf
    accuracies = []
#
    examples = []
    print(test_natural(model, valid_loader))
    #plt.figure(figsize=(20, 10))
    for j, mod in enumerate(["basic", "adv_inf", "adv_2", "mix"]):
        model.load(args.project_dir+'/models/'+mod+'.pth')
        accuracies = []
        for i, epsilon in enumerate(tqdm(epsilons)):
          if epsilon==0:
            acc = test_natural(model, valid_loader)
          else:
            acc, adv_examples = test_under_attack(model, valid_loader, pgd_attack, order=2, epsilon=epsilon, alpha=2/255, iters=20)
      #    
          accuracies.append(acc)
        plt.plot(epsilons, accuracies, label=f'{["basic", "adv linf", "adv l2", "mix"][j]}')
#
    #plt.plot(epsilons, accuracies[:len(epsilons)], label='Norm L2')
    #plt.plot(epsilons, accuracies[len(epsilons):], label='Norm Linf')
  #
    plt.title(r"Comparaison of basic, adversarial and mix train", fontsize=30, pad=15)
    plt.xlabel(r"$\epsilon$ of PGD attack with l2 norm", fontsize=25, labelpad=15)
    plt.ylabel(r"Accuracy", fontsize=25, labelpad=15)
    plt.legend(loc="best")
    plt.savefig('MixTrain.pdf', backend='pgf')
    plt.show()

if __name__ == "__main__":
    main()