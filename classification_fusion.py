# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/05/04  9:33
# @Author  : Cao Xu
# @FileName: classification_fusion.py
"""
Description:   kidney classification fusion
"""
import os
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from classification_util import ClassificationDataset, image_transforms, prepare_model, EarlyStopping, FusionDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib
import matplotlib.pyplot as plt
import torch.multiprocessing
from collections import Counter
import warnings

matplotlib.use('AGG')
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")


def train(data_dir, num_epochs, bs, pt_dir, category_num, model_name, device, lr):
    for i in range(4):
        print('五折交叉验证 第{}次实验:'.format(i))
        test_path = [os.path.join(data_dir, 'fold4/')]
        fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/']
        valid_path = [os.path.join(data_dir, fold_list[3 - i])]
        train_path = [os.path.join(data_dir, x) for x in fold_list if x != fold_list[3 - i]]
        '随机test database'
        # fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/', 'fold4/']
        # valid_path = [data_dir + fold_list[i]]
        # fold_list.remove(fold_list[i])
        # if i == 4:
        #     test_path = [data_dir + fold_list[-1]]
        #     fold_list.remove(fold_list[-1])
        # else:
        #     test_path = [data_dir + fold_list[i]]
        #     fold_list.remove(fold_list[i])
        # train_path = [data_dir + fold_list[0], data_dir + fold_list[1], data_dir + fold_list[2]]

        train_dataset = FusionDataset(img_path=train_path, category_num=category_num, train=True,
                                      transforms=image_transforms['train'])
        valid_dataset = FusionDataset(img_path=valid_path, category_num=category_num, train=True,
                                      transforms=image_transforms['valid'])
        test_dataset = FusionDataset(img_path=test_path, category_num=category_num, train=True,
                                     transforms=image_transforms['valid'])
        train_size, valid_size, test_size = train_dataset.length, valid_dataset.length, test_dataset.length
        print('train_size:{}, valid_size:{}, test_size:{}'.format(train_size, valid_size, test_size))
        'class weight'
        label_count = Counter(train_dataset.labels)
        class_weights = torch.tensor([1 / label_count.get(0), 1 / label_count.get(1)])
        'dataloader'
        train_loader = DataLoader(train_dataset, bs, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, bs, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, bs, shuffle=False, num_workers=4)
        'model, optimizer, scheduler, warmup, loss_function '
        model, optimizer, scheduler, warmup, loss_func = prepare_model(category_num, model_name, lr, num_epochs, device,
                                                                       class_weights)
        'EarlyStopping'
        early_stopping = EarlyStopping(pt_dir, patience=200)
        best_test_acc, best_valid_acc, best_valid_recall, best_epoch = 0.0, 0.0, 0.0, 0
        history = []
        error_sample = []
        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))
            train_loss, valid_loss, num_correct = 0.0, 0.0, 0

            'train'
            model.train()
            torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
            for step, batch in enumerate(train_loader):
                inputs1 = batch[0].to(device)
                inputs2 = batch[1].to(device)
                labels = batch[2].to(device)

                outputs = model(inputs1, inputs2)
                loss_step = loss_func(outputs, labels)
                train_loss += loss_step.item()
                temp_num_correct = torch.eq(outputs.argmax(dim=1), labels).sum().float().item()
                num_correct += temp_num_correct

                optimizer.zero_grad()  # reset gradient
                loss_step.backward()
                optimizer.step()
                scheduler.step()
                warmup.step()

            train_loss /= len(train_loader)
            train_acc = num_correct / train_size
            print("Train Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, train_loss, train_acc))

            'validation'
            num_correct = 0
            valid_true, valid_pred = [], []
            model.eval()
            torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
            with torch.no_grad():
                for step, batch in enumerate(valid_loader):
                    inputs1 = batch[0].to(device)
                    inputs2 = batch[1].to(device)
                    labels = batch[2].to(device)

                    outputs = model(inputs1, inputs2)
                    # for r in range(len(torch.eq(outputs.argmax(dim=1), labels))):
                    #     if torch.eq(outputs.argmax(dim=1), labels)[r].item() is False:
                    #         error_sample.append(img_name[r] + ',' + str(labels[r].item()))
                    loss_step = loss_func(outputs, labels)
                    valid_loss += loss_step.item()
                    num_correct += torch.eq(outputs.argmax(dim=1), labels).sum().float().item()

                    valid_true.extend(labels.cpu().numpy())
                    valid_pred.extend(outputs.argmax(dim=1).cpu().numpy())

            valid_loss /= len(valid_loader)
            valid_acc = num_correct / valid_size
            print("Val Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, valid_loss, valid_acc))
            'best acc save checkpoint'
            if best_valid_acc < valid_acc:
                print('confusion_matrix:\n{}'.format(confusion_matrix(valid_true, valid_pred)))
                print(
                    'classification_report:\n{}'.format(classification_report(valid_true, valid_pred, digits=4)))
                best_valid_acc = valid_acc
                best_epoch = epoch + 1
                torch.save(model, pt_dir + 'fold' + str(i) + '-best-acc-model.pt')
            'best sensitivity save checkpoint'
            malign_recall = classification_report(valid_true, valid_pred, output_dict=True)['1'].get('recall')
            if malign_recall > best_valid_recall:
                print(
                    'best malignant recall report:\n{}'.format(classification_report(valid_true, valid_pred, digits=4)))
                best_valid_recall = malign_recall
                torch.save(model, pt_dir + 'fold' + str(i) + '-best-recall-model.pt')

            print("Epoch: {:03d}, Train Loss: {:.4f}, Acc: {:.4f}, Valid Loss: {:.4f}, Acc:{:.4f}"
                  .format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))
            print("validation best: {:.4f} at epoch {}".format(best_valid_acc, best_epoch))
            # 早停止
            early_stopping(valid_loss, model)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
            history.append([train_loss, valid_loss])

        history = np.array(history)
        plt.clf()  # 清图
        plt.plot(history)
        plt.legend(['Tr Loss', 'Val Loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.ylim(0, np.max(history))
        plt.savefig(pt_dir + 'loss_curve' + str(i) + '.png')

        'test'
        # model = torch.load(pt_dir + 'fold' + str(i) + '-best-acc-model.pt')   # best-recall-model.pt
        num_correct = 0
        test_true, test_pred = [], []
        model.eval()
        torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                inputs1 = batch[0].to(device)
                inputs2 = batch[1].to(device)
                labels = batch[2].to(device)
                img_name = batch[3]
                outputs = model(inputs1, inputs2)
                for r in range(len(torch.eq(outputs.argmax(dim=1), labels))):
                    if torch.eq(outputs.argmax(dim=1), labels)[r].item() is False:
                        error_sample.append(img_name[r] + ',' + str(labels[r].item()))
                num_correct += torch.eq(outputs.argmax(dim=1), labels).sum().float().item()

                test_true.extend(labels.cpu().numpy())
                test_pred.extend(outputs.argmax(dim=1).cpu().numpy())

            test_acc = num_correct / test_size
            print("test accuracy: {:.4f}".format(test_acc))
            print('confusion_matrix:\n{}'.format(confusion_matrix(test_true, test_pred)))
            print('classification_report:\n{}'.format(classification_report(test_true, test_pred, digits=4)))


def classification():
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'resnet50'
    data_dir = '/home/ai999/project/kidney-quality-control/单灰阶单CDFI整理-5fold/'
    category_num = 2
    bs = 20
    lr = 0.01
    num_epochs = 500
    data = 'classification-model/20250318-kidney-blood-MultiHeadAttentionResnet-classify-'
    save_path = data + str(category_num) + 'class-' + model_name + '-bs' + str(bs) + '-lr' + str(lr) + '/'
    pt_dir = 'classification_model/' + save_path
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    print('测试肾癌{}分类,{}模型, batch size等于{}下的分类结果：'.format(category_num, model_name, bs))
    train(data_dir, num_epochs, bs, pt_dir, category_num, model_name, device, lr)
    print('done')


if __name__ == '__main__':
    classification()
