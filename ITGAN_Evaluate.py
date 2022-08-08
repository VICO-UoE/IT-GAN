import time
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from utils import get_dataset, get_network, DiffAugment, ParamDiffAug, epoch, get_time
import BigGAN
from copy import deepcopy

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model_eval', type=str, default='ResNet18BN', help='model')
    parser.add_argument('--exp', type=int, default=0, help='exp')
    parser.add_argument('--Epoch_evaltrain', type=int, default=200, help='epochs to train a network')
    parser.add_argument('--num_evalnet', type=int, default=3, help='train a number of networks per experiment')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_train_net', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--mode', type=str, default='', help='train mode')
    parser.add_argument('--diffaug_choice', type=str, default='Auto', help='diffaug_choice')

    # for ConvNet only
    parser.add_argument('--width_net', type=int, default=128, help='width')
    parser.add_argument('--depth_net', type=int, default=3, help='depth')
    parser.add_argument('--act', type=str, default='relu', help='act')
    parser.add_argument('--normlayer', type=str, default='instancenorm', help='normlayer')
    parser.add_argument('--pooling', type=str, default='avgpooling', help='pooling')

    args = parser.parse_args()

    # for augmentation
    param_diffaug = ParamDiffAug()

    if args.diffaug_choice == 'Auto':
        if args.dataset in ['MNIST', 'SVHN']:
            args.diffaug_choice = 'color_translation_cutout_scale_rotate'
        elif args.dataset in ['FashionMNIST', 'CIFAR10', 'CIFAR100']:
            args.diffaug_choice = 'color_translation_cutout_flip_scale_rotate'
        else:
            exit('Auto diffaug_choice is not defined for dataset: %s' % args.dataset)
    else:
        args.diffaug_choice = 'None'

    # gpu usage
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    root_path = '.'  #Todo: modify path
    data_path = os.path.join(root_path, '../data') #Todo: modify path
    print('gpu number = %d' % (torch.cuda.device_count()))

    args.dis_metric = 'ours' # gradient matching metric, 'ours' is from DC.
    args.device = device # gradient matching metric


    print('args:')
    print(args.__dict__)
    print('param_diffaug:')
    print(param_diffaug.__dict__)
    print('device: ', device)


    accs_dict = dict()
    accs_dict['real'] = []
    accs_dict['GAN'] = []
    accs_dict['GAN_Inversion'] = []
    accs_dict['EfficientGAN'] = []

    exp = args.exp
    print('\n\n\nexperiment %d'%exp)

    channel, shape_img, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, data_path)
    num_train = dst_train.__len__()
    print('dst_train length: ', num_train)


    ''' load data '''
    indices_class = [[] for c in range(num_classes)]

    images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0)
    images_all = images_all.to(device)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

    print('Total dataset images_all shape: ', images_all.shape)
    print('Total dataset images_all mean = [%.4f, %.4f, %.4f], std = [%.4f, %.4f, %.4f]' % (torch.mean(images_all[:, 0]), torch.mean(images_all[:, 1]), torch.mean(images_all[:, 2]),
                                                                              torch.std(images_all[:, 0]), torch.std(images_all[:, 1]), torch.std(images_all[:, 2])))

    def get_images(c, num):  # get random num images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:num]
        return images_all[idx_shuffle]

    weight_path = os.path.join(data_path, 'G_Pretrained_%s_exp%d.pth' % (args.dataset, exp)) #Todo: modify path
    print('use G model: ', weight_path)

    dim_z = 128
    G = BigGAN.Generator(G_ch=64, dim_z=dim_z, bottom_width=4, resolution=32,
                         G_kernel_size=3, G_attn='0', n_classes=num_classes,
                         num_G_SVs=1, num_G_SV_itrs=1,
                         G_shared=False, shared_dim=0, hier=False,
                         cross_replica=False, mybn=False,
                         G_activation=nn.ReLU(inplace=False),
                         G_lr=2e-4, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
                         BN_eps=1e-5, SN_eps=1e-08, G_mixed_precision=False, G_fp16=False,
                         G_init='N02', skip_init=False, no_optim=False,
                         G_param='SN', norm_style='bn').to(device)
    G.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    G.eval()  # Train? My conclusion is that .eval() is good for pretrained weights and mean/std, but .train() is good for random weights.
    # G.train()
    for param in G.parameters():
        param.requires_grad = False

    mean_GAN = [0.5, 0.5, 0.5]
    std_GAN = [0.5, 0.5, 0.5]

    def renormalize(img):
        return torch.cat([(((img[:, 0] * std_GAN[0] + mean_GAN[0]) - mean[0]) / std[0]).unsqueeze(1),
                          (((img[:, 1] * std_GAN[1] + mean_GAN[1]) - mean[1]) / std[1]).unsqueeze(1),
                          (((img[:, 2] * std_GAN[2] + mean_GAN[2]) - mean[2]) / std[2]).unsqueeze(1)], dim=1)


    def generate(z, lab):
        num_max = 500  # Error occurs when batch size of G is large.
        num = z.shape[0]
        if num > num_max:
            img_syn = []
            for i in range(int(np.ceil(num / num_max))):
                img_syn.append(renormalize(G(z[i * num_max: (i + 1) * num_max], lab[i * num_max: (i + 1) * num_max])))
            return torch.cat(img_syn, dim=0)
        else:
            return renormalize(G(z, lab))


    save_path = os.path.join(root_path, 'results', 'EfficientGAN') #Todo: modify path
    if not os.path.exists(save_path):
        os.mkdir(save_path)






    ''' visualize  '''
    # fpath = os.path.join(data_path, 'EfficientGAN_final_%s_ConvNet_lrz0.001_exp%d.pt' % (args.dataset, exp)) #Todo: modify path
    # print('use EfficientGAN vectors: %s' % fpath)
    # data_z = torch.load(fpath, map_location=device)
    # z_eff_all = deepcopy(data_z['z_eff_all']).detach()
    #
    # num_vis_pc = 10
    # images_real_tosave = []
    # images_syn_tosave = []
    # for c in range(min(num_classes, 10)):
    #     idx = deepcopy(indices_class[c])
    #     np.random.shuffle(idx)
    #     idx = idx[:num_vis_pc]
    #     z_vis = z_eff_all[idx].detach()
    #     lab_vis = labels_all[idx].detach()
    #     img_real_vis = images_all[idx].detach()
    #     images_real_tosave += [img_real_vis]
    #     img_syn_vis = deepcopy(renormalize(G(z_vis, lab_vis)).detach())
    #     images_syn_tosave += [img_syn_vis]
    #
    # # save_name = os.path.join(save_path, '%s_real.pdf'%(args.dataset)) #Todo: modify path
    # # save_image_tensor(torch.cat(images_real_tosave, dim=0), mean, std, save_name, num_vis_pc)
    # save_name = os.path.join(save_path, '%s_syn.pdf' % (args.dataset)) #Todo: modify path
    # save_image_tensor(torch.cat(images_syn_tosave, dim=0), mean, std, save_name, num_vis_pc)
    # print('save to %s'%save_name)



    ''' evaluate '''
    if args.mode == '':
        mode_all = ['GAN', 'GAN_Inversion', 'EfficientGAN']
    else:
        mode_all = [args.mode]

    for mode in mode_all:
        if mode == 'real':
            def load_batch(idx):
                lab = labels_all[idx]
                img = images_all[idx]
                return img.detach(), lab.detach()

        elif mode == 'GAN':
            def load_batch(idx):
                z = torch.randn(size=(idx.shape[0], dim_z), dtype=torch.float, requires_grad=False, device=device)
                lab = labels_all[idx]
                img = generate(z, lab)
                return img.detach(), lab.detach()

        elif mode == 'GAN_Inversion':
            ''' load GAN inversion z '''
            fpath = os.path.join(data_path, 'GANInversion_final_%s_ConvNet_lrz0.100_exp%d.pt' % (args.dataset, exp)) #Todo: modify path
            print('use GAN inversion vectors: %s' % fpath)
            data_z = torch.load(fpath, map_location=device)
            z_inverse_all = deepcopy(data_z['z_inverse_all']).detach()

            images_inv_all = []
            for i in range(int(np.ceil(num_train/args.batch_train_net))):
                idx = np.arange(i*args.batch_train_net, min((i+1)*args.batch_train_net, num_train))
                images_inv_all.append(generate(z_inverse_all[idx], labels_all[idx]).detach())
            images_inv_all = torch.cat(images_inv_all, dim=0)
            print('generate images_inv_all shape:', images_inv_all.shape)
            def load_batch(idx):
                # img = generate(z_inverse_all[idx], labels_all[idx])
                img = images_inv_all[idx]
                lab = labels_all[idx]
                return img.detach(), lab.detach()

        elif mode == 'EfficientGAN':
            ''' load Efficient z '''
            fpath = os.path.join(data_path, 'EfficientGAN_final_%s_ConvNet_lrz0.001_exp%d.pt' % (args.dataset, exp)) #Todo: modify path
            print('use EfficientGAN vectors: %s' % fpath)
            data_z = torch.load(fpath, map_location=device)
            z_eff_all = deepcopy(data_z['z_eff_all']).detach()

            images_eff_all = []
            for i in range(int(np.ceil(num_train/args.batch_train_net))):
                idx = np.arange(i*args.batch_train_net, min((i+1)*args.batch_train_net, num_train))
                images_eff_all.append(generate(z_eff_all[idx], labels_all[idx]).detach())
            images_eff_all = torch.cat(images_eff_all, dim=0)
            print('generate images_eff_all shape:', images_eff_all.shape)
            def load_batch(idx):
                # img = generate(z_eff_all[idx], labels_all[idx])
                img = images_eff_all[idx]
                lab = labels_all[idx]
                return img.detach(), lab.detach()
        else:
            def load_batch(idx):
                return None, None
            exit('unknown mode: %s'%mode)

        accs_test = []
        for eval_exp in range(args.num_evalnet):
            print('--------------------------------------------------------------')
            print('evaluate mode: %s' % (mode))
            print('evaluate model: %s' % (args.model_eval))
            print('args.batch_train_net: ', args.batch_train_net)
            print('args.Epoch_evaltrain: ', args.Epoch_evaltrain)
            print('args.lr_net: ', args.lr_net)
            print('labels_all: ', labels_all.shape)
            num_evaltrain = int(labels_all.shape[0])
            print('num_evaltrain: ', num_evaltrain)

            # random for test
            net = get_network(args.model_eval, channel, num_classes, args.width_net, args.depth_net, args.act, args.normlayer, args.pooling, shape_img)

            criterion = nn.CrossEntropyLoss().to(device)
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)  # no cuda version

            for ep_eval in range(args.Epoch_evaltrain + 1):
                train_begin = time.time()
                net.train()
                idx_rand = np.random.permutation(num_evaltrain)
                acc_train = []
                loss_train = []

                if ep_eval == args.Epoch_evaltrain // 2:
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net / 10, momentum=0.9, weight_decay=0.0005)  # no cuda version

                for it in range(int(np.ceil(num_evaltrain // args.batch_train_net))):
                    img, lab = load_batch(idx_rand[it * args.batch_train_net: (it + 1) * args.batch_train_net])
                    img = DiffAugment(img, args.diffaug_choice, param=param_diffaug)
                    output = net(img.float())
                    loss = criterion(output, lab)

                    optimizer_net.zero_grad()
                    loss.backward()
                    optimizer_net.step()

                    acc_train.append(np.mean(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy())))
                    loss_train.append(loss.item())

                train_end = time.time()
                time_train = train_end - train_begin

                if ep_eval % 1 == 0 or ep_eval == args.Epoch_evaltrain:
                    loss_test, acc_test, acc_separate = epoch('test', 0, testloader, net, optimizer_net, criterion, device=device, flag_print=False)
                    print('%s epoch %d/%d  time = %.1f  lr = %.4f  train_loss = %.4f  train_acc = %.4f  test_acc = %.4f' % (
                    get_time(), ep_eval, args.Epoch_evaltrain, time_train, optimizer_net.param_groups[0]['lr'], np.mean(loss_train), np.mean(acc_train), acc_test))

            accs_test.append(acc_test)
            accs_dict[mode].append(acc_test)

        print('Evaluation: train z iter = %d evaluate %d %s test acc = %.4f std = %.4f\n' % (-1, len(accs_test), args.model_eval, np.mean(accs_test), np.std(accs_test)))
        print('Evaluation: train z iter = %d evaluate %d %s all results: ' % (-1, len(accs_test), args.model_eval), accs_test)
        print('============================================================\n\n')


    print('\n\n\n')
    print('final comparison:')
    for mode in accs_dict.keys():
        print('%s evalute %d %s acc = %.2f$\pm$%.2f all = %s'%(mode, len(accs_dict[mode]), args.model_eval, np.mean(accs_dict[mode])*100, np.std(accs_dict[mode])*100, accs_dict[mode]))


if __name__ == '__main__':
    main()

