import numpy as np
import torch
import argparse
import torchvision.utils as vutils
import torch.nn.functional as F
import torch.multiprocessing
from utils import DatasetInfo, load_protect_model, similarity_projection, reserve_gpu, load_original_x, eval_model
import matplotlib.pyplot as plt
import os
torch.multiprocessing.set_sharing_strategy('file_system')
import mlconfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument("--gpu", type=str, default='1', help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--num", type=int, default=10, help="training num.")
    parser.add_argument("--eps", type=float, default=1.0, help="l2_bound.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="dataset name")
    parser.add_argument("--debug", type=bool, default=True, help="whether to debug")
    parser.add_argument("--k", type=int, default=2, help="top k classes with highest confidence to maximize self-entropy")

    return parser.parse_args()


def shannon_entropy(x):
    return (-x * x.log()).sum()


def sensitive_sample_gen(data_info, x, model, eps, similarity_constraint=True, similarity_mode=2,
                         n_iter=500, lr=1.0, gpu=True, verbose=True,  k=None):
    '''
        :param data_info:
        :param x:
        :param model:
        :param fp_id:
        :param similarity_constraint:
        :param similarity_mode:
        :param eps:
        :param n_iter:
        :param lr:
        :param gpu:
        :param early_stop:
        :param early_stop_th:
        :param verbose:
        :param num_of_class:
        :param debug:
        :param softmax:
        :param k:
        :return:
    '''
    x.requires_grad = True
    x_origin = x.detach().cpu().numpy()
    model.eval()
    optimizer = torch.optim.SGD(params=[x], lr=lr)
    # record sensitivity and confidence loss
    entropy_loss = []


    others_confidence_loss = []
    k = data_info.num_classes if k is None else k  # default k is num of classes, i.e., top-k
    best_x = None
    loss_min = torch.inf
    for i in range(n_iter):
        # Zero gradients for every iter
        optimizer.zero_grad()

        logits = torch.squeeze(model(x))
        softmax_out = torch.softmax(logits, dim=-1)
        # Shannon entropy loss: self-entropy
        values, indices = torch.topk(softmax_out, k=data_info.num_classes, dim=-1, largest=True)
        loss_shannon_entropy = shannon_entropy(softmax_out[indices[:k]])
        loss = -loss_shannon_entropy

        loss.backward()
        optimizer.step()

        x_new = x.detach().cpu().numpy()
        if similarity_constraint:
            x_new = similarity_projection(data_info, x_origin, x_new, eps, similarity_mode=similarity_mode)

        if verbose:
            if i % 100 == 0:
                print('Iteration %d, shannon entropy loss: %.4f' % (i, loss_shannon_entropy))

        if gpu:
            x.data = torch.tensor(x_new).cuda()
        else:
            x.data = torch.tensor(x_new)

        # entropy_loss.append(float(loss_shannon_entropy))
        
        # return the best x with the min loss: maximize shannon entropy
        if loss < loss_min:
            loss_min = loss
            best_x = x.detach().cpu().numpy()

    return best_x


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman')
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    args = parse_args()
    reserve_gpu(args.gpu)
    np.random.seed(1999)
    dataset_info = DatasetInfo(args.dataset)
    # load protect model
    protect_model = load_protect_model(args.dataset)
    # eval_model(protect_model, dataset_info.name)
    eps = args.eps

    # load original dataset
    x_org = load_original_x(args.dataset)

    # generate sensitive samples
    batch_size, n_iter = 1, 5000  
    k = args.k  # top k classes with highest confidence to maximize self-entropy
    assert k <= dataset_info.num_classes and k >= 2, 'k should be in [2, num_of_class]'
    ss_path = 'outputs/fingerprint/%s/k=%d/BISF_%f_%d_n=%d.npy' % (args.dataset, k, eps,  n_iter, args.num)
    ss_org_path = 'outputs/fingerprint/%s/k=%d/BISF_%f_%d_org_n=%d.npy' % (args.dataset,  k, eps, n_iter, args.num)

    os.makedirs('outputs/fingerprint/%s/k=%d' % (args.dataset, k), exist_ok=True)
    if os.path.exists(ss_path):
        sensitive_samples = np.load(ss_path)[:args.num]
        org_image = np.load(ss_org_path)[:args.num]
    else:
        sensitive_samples = []
        org_image = []
        for batch_id, x in enumerate(x_org):
            print('-------------------Generating fingerprint %d-------------------' % batch_id)
            x_ss = sensitive_sample_gen(dataset_info,
                                        torch.from_numpy(x_org[batch_id:batch_id+1]).float().cuda(),
                                        protect_model,
                                        fp_id=batch_id,
                                        gpu=True,
                                        similarity_constraint=True,
                                        n_iter=n_iter,
                                        similarity_mode=2,
                                        eps=eps,
                                        verbose=True,
                                        debug=False,
                                        k=k,
                                        )
            # x_ss = x_ss.cpu().detach().numpy()
            sensitive_samples.append(x_ss)
            org_image.append(x_org[batch_id:batch_id+1])

            if (batch_id + 1) > args.num: break

        sensitive_samples = np.concatenate(sensitive_samples, axis=0)
        org_image = np.concatenate(org_image, axis=0)
        # save fingerprints and original images all together in npy format
        np.save(ss_path, sensitive_samples)
        np.save(ss_org_path, org_image)
        # save fingerprints and original x as images
        vutils.save_image(torch.from_numpy(sensitive_samples), 'outputs/fingerprint/%s/k=%d/IBSF_%f_%d_n=%d.png' %
                          (args.dataset, k, eps, n_iter, args.num), normalize=False)
        vutils.save_image(torch.from_numpy(org_image), 'outputs/fingerprint/%s/k=%d/IBSF_%f_%d_org_n=%d.png' %
                          (args.dataset, k, eps, n_iter, args.num), normalize=False)