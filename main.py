import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import argparse

from utils import get_score_from_all_slices
from model import UNet, EMA
from loss import get_loss
from data import train_data_generator, val_data_generator

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
num_patients = 229
weight = 1


def test(dataloader, model, log_dir, save):
    model.eval()
    predicts = []
    targets = []

    if save:
        m = 0
        images = []
        pre_dir = log_dir + 'pre/'
        if not os.path.isdir(pre_dir):
            os.mkdir(pre_dir)

        label_dir = log_dir + 'lab/'
        if not os.path.isdir(label_dir):
            os.mkdir(label_dir)

        img_dir = log_dir + 'img/'
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)

    with torch.no_grad():
        for step, (batch_img, batch_label, batch_mask) in enumerate(dataloader):
            try:
                batch_img_slice = batch_img.to(device)
                batch_label = batch_label.to(device)
                batch_mask = batch_mask.to(device)
                batch_pre, _ = model(batch_img.to(device), batch_mask)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            if step == 0:
                predicts = batch_pre
                targets = batch_label
                if save:
                    images = batch_img_slice
            else:
                predicts = torch.cat([predicts, batch_pre], dim=0)
                targets = torch.cat([targets, batch_label], dim=0)
                if save is True:
                    images = torch.cat([images, batch_img_slice], dim=0)

        list_of_predicts = predicts.split(1, dim=0)
        for idx in range(len(list_of_predicts) // 2):
            predict = torch.cat([list_of_predicts[idx * 2], torch.flip(list_of_predicts[idx * 2 + 1], dims=[3])], dim=3)

            if idx == 0:
                predicts = predict
            else:
                predicts = torch.cat([predicts, predict], dim=0)

        if save is True:
            for i in range(targets.size(0)):
                if torch.sum(targets[i]) != 0:
                    p_mask = predicts[i]
                    zero = torch.zeros_like(p_mask)
                    one = torch.ones_like(p_mask)
                    p_mask = torch.where(p_mask > 0.5, one, zero)

                    df = pd.DataFrame(p_mask.view((224, 192)).cuda().data.cpu().numpy())
                    df.to_csv(pre_dir + 'pre_' + str(m) + '.csv', header=False, index=False)
                    df = pd.DataFrame(targets[i].view((224, 192)).cuda().data.cpu().numpy())
                    df.to_csv(label_dir + 'lab_' + str(m) + '.csv', header=False, index=False)
                    df = pd.DataFrame(images[i].view((224, 192)).cuda().data.cpu().numpy())
                    df.to_csv(img_dir + 'img_' + str(m) + '.csv', header=False, index=False)
                    m = m + 1

    return predicts, targets


def train(fold, train_patient_indexes, val_patient_indexes, args):
    n = args.epochs
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size

    log_dir = 'fold_' + str(fold) + '/'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    model = UNet().to(device)
    ema = EMA(0.999)
    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    dice_loss = get_loss().to(device)

    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=1e-3, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96, last_epoch=-1)

    train_img, train_label, train_domain, train_mask = train_data_generator(patient_indexes=train_patient_indexes, fold=fold)
    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_img).to(device), torch.FloatTensor(train_label).to(device),
                                       torch.Tensor(train_domain).to(device), torch.FloatTensor(train_mask).to(device))
    weights = [weight for i in range(train_label.shape[0])]
    sampler = Data.sampler.WeightedRandomSampler(weights, num_samples=(num_patients // 4) * 184, replacement=False)
    train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=train_batch_size, sampler=sampler)
    # train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=train_batch_size, num_workers=8, persistent_workers=True)

    val_img, val_label, val_mask = val_data_generator(patient_indexes=val_patient_indexes)
    torch_dataset = Data.TensorDataset(torch.FloatTensor(val_img).to(device), torch.FloatTensor(val_label).to(device),
                                       torch.FloatTensor(val_mask).to(device))
    val_loader = Data.DataLoader(dataset=torch_dataset, batch_size=val_batch_size, shuffle=False)

    best_dice = 0.
    correct = 0.
    num = 0.
    Loss_c_reverse = 0.
    Loss_dice = 0.
    Total_dice = 0.

    for epoch in range(n):
        model.train()
        for step, (batch_img, batch_label, batch_d, batch_mask) in enumerate(train_loader):
            l = batch_label.split(96, dim=3)[0]
            r = torch.flip(batch_label.split(96, dim=3)[1], dims=[3])
            batch_label = torch.cat([l, r], dim=0)

            batch_d = torch.cat([batch_d, batch_d], dim=0)

            batch_pre, batch_domain = model(batch_img, batch_mask)

            loss_c_reverse = F.cross_entropy(batch_domain.float(), batch_d.long())
            loss_dice = dice_loss(batch_pre, batch_label)
            total_loss = loss_c_reverse + loss_dice

            optimizer.zero_grad()
            loss = total_loss.requires_grad_()
            loss.backward()
            optimizer.step()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.update(name, param.data)

            domain_pre = torch.squeeze(torch.argmax(batch_domain, dim=1))
            correct = correct + torch.sum(domain_pre.eq(batch_d))
            num = num + train_batch_size * 2
            Loss_c_reverse = Loss_c_reverse + loss_c_reverse.item()
            Loss_dice = Loss_dice + loss_dice.item()
            Total_dice = Total_dice + total_loss.item()

        scheduler.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                backup_params.register(name, param.data)
                param.data.copy_(ema.get(name))

        predicts, targets = test(val_loader, model, log_dir, False)
        score = get_score_from_all_slices(labels=targets, predicts=predicts)

        val_dice = np.mean(score['dice'])
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), '{}checkpoint.pt'.format(log_dir))

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))

    model.load_state_dict(torch.load('{}checkpoint.pt'.format(log_dir)))
    predicts, targets = test(val_loader, model, log_dir, False)
    score = get_score_from_all_slices(labels=targets, predicts=predicts)

    df = pd.DataFrame(score)
    df.to_csv(log_dir + 'score_record.csv', header=False, index=False)

    mean_score = {}
    for key in score.keys():
        mean_score[key] = np.mean(score[key])

    return mean_score


def main(args):
    patients_indexes = np.array([i for i in range(num_patients)])
    sites = [np.arange(0, 55), np.arange(55, 89), np.arange(89, 116), np.arange(116, 128), np.arange(128, 155),
               np.arange(155, 169), np.arange(169, 180), np.arange(180, 215), np.arange(215, 229)]
    test_score = {}
    for i in range(9):
        val_patient_indexes = sites[i]
        val_patient_indexes.sort(axis=0)
        train_patient_indexes = np.delete(patients_indexes, val_patient_indexes)

        mean_score = train(fold=i + 1, train_patient_indexes=train_patient_indexes, val_patient_indexes=val_patient_indexes, args=args)
        df = pd.DataFrame(mean_score, index=[0])
        df.to_csv('fold_{}/validation_score.csv'.format(i + 1), header=False, index=False)

        for key in mean_score.keys():
            if i == 0:
                test_score[key] = [mean_score[key]]
            else:
                test_score[key] = test_score[key] + [mean_score[key]]

    for key in test_score.keys():
        print('Testing {} on all sites is: {:.4f}±{:.4f}.'
              .format(key, np.mean(np.array(test_score[key])), np.std(np.array(test_score[key]))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    main(args)