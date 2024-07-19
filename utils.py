import numpy as np
from tqdm import tqdm
from torch.utils import data
import torch
from models import ResNet20_CIFAR10, convnet_fc, DenseNet121, LeNet5, VAE
import GPUtil as GPUtil
import os
import random
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision


def load_original_x(dataset, n=10000):
    # load original data
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)

        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
        original_x = collect_n_samples(n, train_loader, has_labels=False)
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
        original_x = collect_n_samples(n, train_loader, has_labels=False)

    elif dataset == 'gtsrb':
        x_train, y_train = np.load('/home/data/data/gtsrb/x_train.npy') / 255., np.load(
            '/home/data/data/gtsrb/y_train.npy')
        original_x = np.moveaxis(x_train, source=-1, destination=1)[:n]
    elif dataset == 'imagenet':
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        train_dataset = torchvision.datasets.ImageFolder(
            root='/home/data/data/imagenet/train',
            transform=data_transform)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
        original_x = collect_n_samples(1500, train_loader, has_labels=False)
    else:
        raise NotImplementedError

    return original_x


def similarity_projection(data_info, ref, data, eps, similarity_mode=np.inf):
    assert similarity_mode in [np.inf, 2], 'bounded norm error, only support L-inf and L-2 norm'
    # clip by eps
    diff = data - ref
    if similarity_mode == 2:
        r = np.sqrt(np.sum(diff ** 2))
        if r > eps:
            new_diff = diff / r * eps
            new_data = ref + new_diff
        else:
            new_data = data
    else:
        new_diff = np.clip(diff, - eps / 255., eps / 255.)
        new_data = ref + new_diff

    # feasibility constraint
    new_data = np.clip(new_data, 0., 1.)

    return np.float32(new_data)


def load_protect_model(dataset):
    # load protect model
    if dataset == 'cifar10':
        protect_model = ResNet20_CIFAR10(normalize=True).cuda()
        protect_model_path = './resnet20_cifar10.pth'
    elif dataset == 'gtsrb':
        protect_model = convnet_fc(normalize=True).cuda()
        protect_model_path = './convnet_fc_gtsrb.pth'
    elif dataset == 'mnist':
        protect_model = LeNet5().cuda()
        protect_model_path = './lenet_5_mnist.pth'
    elif dataset == 'imagenet':
        protect_model = DenseNet121(normalize=True).cuda()
        protect_model_path = None  # pretrained model, no need to load
    else:
        raise NotImplementedError
    if dataset != 'imagenet':
        protect_model.load_state_dict(torch.load(protect_model_path)['model'])

    protect_model.eval()  # set to eval mode
    return protect_model


def set_random_seed(seed=1234):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def pick_gpu():
    """
    Picks a GPU with the least memory load.
    :return:
    """
    try:
        gpu = GPUtil.getFirstAvailable(order='memory', maxLoad=2, maxMemory=0.8, includeNan=False,
                                       excludeID=[], excludeUUID=[])[0]
        return gpu
    except Exception as e:
        print(e)
        return "0"


def reserve_gpu(mode_or_id):
    """ Chooses a GPU.
    If None, uses the GPU with the least memory load.
    """
    if mode_or_id:
        gpu_id = mode_or_id
        os.environ["CUDA_VISIBLE_DEVICES"] = mode_or_id
    else:
        gpu_id = str(pick_gpu())
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Selecting GPU id {gpu_id}")


def collect_n_samples(n: int,
                      data_loader: data.DataLoader,
                      class_label: int = None,
                      has_labels: bool = True,
                      reduce_labels: bool = False,
                      verbose=True):
    """ Collects n samples from a data loader.
    :param n Number of samples to load. Set to 'np.inf' for all samples.
    :param data_loader The data loader to load the examples from
    :param class_label Load only examples from this target class
    :param has_labels Does the dataset have labels?
    :param reduce_labels Reduce labels.
    :param verbose Show the progress bar
    """
    x_samples, y_samples = [], []
    with tqdm(desc=f"Collecting samples: 0/{n}", total=n, disable=not verbose) as pbar:
        if has_labels:
            for (x, y) in data_loader:
                if len(x_samples) >= n:
                    break
                # Reduce soft labels.
                y_full = y.clone()
                if y.dim() > 1:
                    y = y.argmax(dim=1)

                # Compute indices of samples we want to keep.
                idx = np.arange(x.shape[0])
                if class_label:
                    idx, = np.where(y == class_label)

                if len(idx) > 0:
                    x_samples.extend(x[idx].detach().cpu().numpy())
                    if reduce_labels:
                        y_samples.extend(y[idx].detach().cpu().numpy())
                    else:
                        y_samples.extend(y_full[idx].detach().cpu().numpy())
                    # print(len(x_samples))
                    pbar.n = len(x_samples)
                    pbar.refresh()
                    pbar.set_description(f"Collecting samples: {min(len(x_samples)+1, n)}/{n}")

            if n == np.inf:
                return np.asarray(x_samples), np.asarray(y_samples)

            if len(x_samples) < n:
                print(f"[WARNING]: Could not find enough samples. (Found: {len(x_samples)}, Expected: {n})")
            return np.asarray(x_samples[:n]), np.asarray(y_samples[:n])
        else:   # No labels.
            for x,y in data_loader:
                x_samples.extend(x.detach().cpu().numpy())
                pbar.set_description(f"Collecting samples: {min(len(x_samples)+1, n)}/{n}")
                pbar.update(len(x_samples))
                if len(x_samples) >= n:
                    break

            if len(x_samples) < n:
                print(f"[WARNING]: Could not find enough samples. (Found: {len(x_samples)}, Expected: {n})")
            return np.asarray(x_samples[:n])


class DatasetInfo:
    def __init__(self, dataset):
        self.name = dataset
        self.train_batch_size = 32
        self.eval_batch_size = 256  # 64 if debug else
        self.load_path = "../../data/%s" % dataset
        # self.accept_clean_acc_degrade = 0.05
        self.accept_trapdoor_acc = 0.94
        self.data_augmentation = False
        self.accept_cosine_benign_trapdoor = -np.inf

        if dataset == "mnist":
            self.img_shape = (1, 28, 28)
            self.num_classes = 10
            self.epochs = 5  # 60  # 30  #
            self.accept_clean_acc = 0.97
            self.clip_max = 1.
            self.clip_min = 0.

            def lr_schedule(epoch):
                lr = 1e-3
                # if epoch > 20:
                #     lr *= 1e-1
                if epoch > 40:
                    lr *= 1e-1
                elif epoch > 50:
                    lr *= 1e-2
                print('Learning rate: ', lr)
                return lr

        elif dataset == "cifar10":
            self.img_shape = (3, 32, 32)
            self.num_classes = 10
            self.load_with_keras = True
            self.epochs = 200
            self.data_augmentation = True
            self.train_batch_size = 128
            self.eval_batch_size = 64
            self.accept_clean_acc = 0.82
            self.clip_max = 1.
            self.clip_min = 0.
            # self.name = "cifar10"
            self.max_step = math.ceil(50000 / self.train_batch_size)

            def lr_schedule(epoch):
                # lr = 1e-3
                # if epoch > 90:
                #     lr *= 1e-3
                # elif epoch > 80:
                #     lr *= 1e-2
                # elif epoch > 60:
                #     lr *= 1e-1
                # print('Learning rate: ', lr)
                lr = 1e-3
                if epoch > 180:
                    lr *= 0.5e-3
                elif epoch > 160:
                    lr *= 1e-3
                elif epoch > 120:
                    lr *= 1e-2
                elif epoch > 80:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr

        elif dataset == "gtsrb":
            self.img_shape = (3, 32, 32)
            self.num_classes = 43
            self.epochs = 30
            self.accept_clean_acc = 0.93
            self.clip_max = 1.
            self.clip_min = 0.

            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 20:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr


        elif dataset == "cifar100":
            self.img_shape = (32, 32, 3)
            self.num_classes = 100
            self.load_with_keras = True
            self.train_batch_size = 32
            self.epochs = 200
            self.accept_clean_acc = 0.70
            self.clip_max = 1.
            self.clip_min = 0.
            self.mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            self.std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 20:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr

        elif dataset == "youtube_face":
            self.img_shape = (224, 224, 3)
            self.num_classes = 1283
            self.epochs = 1  # 10 for all label and clean from scratch
            self.eval_batch_size = 32
            self.accept_clean_acc = 0.98

            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 5:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr

        elif dataset == "imagenet":
            # only use to keep interface consistence and get num of classes
            self.num_classes = 1000
            self.epochs = 50
            self.eval_batch_size = 4
            self.clip_max = 255.
            self.clip_min = 0.
            self.img_shape = (224, 224, 3)
            self.train_batch_size = 32
            # self.name = "imagenet"
            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 10:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr

        elif dataset == "vggface2":
            # only use to keep interface consistence and get num of classes
            self.num_classes = 2622
            self.epochs = 50
            self.eval_batch_size = 4
            self.clip_max = 255.
            self.clip_min = 0.
            self.img_shape = (224, 224, 3)
            # self.name = "imagenet"
            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 10:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr
        else:
            raise NotImplementedError

        self.lr_schedule = lr_schedule
        self.num_batch_train = 0
        self.num_batch_val = 0
        self.num_batch_test = 0