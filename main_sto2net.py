#sto2net
for idx in range(1,6):
    from operator import truediv
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import itertools
    import torch
    import torchvision.transforms as transforms
    from preprocessing_dataset import MyDataset
    from py_cot import *
    import copy
    import time
    import numpy as np
    import os
    import random
    import logging
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(3407)




    def save_model(model, save_path):
        # save model
        torch.save(model.state_dict(), save_path)

    # 更新混淆矩阵
    def confusion_matrix(labels, preds, conf_matrix):
        stacked = torch.stack((labels, preds), dim=1) #沿着一维度对label序列进行连接
        for p in stacked:
            tl, pl = p.tolist() #将ndarray数组对象变为嵌套的多层的list
            tl = int(tl[0])
            pl = int(pl[0])
            conf_matrix[tl, pl] = conf_matrix[tl, pl] + 1
        return conf_matrix

    def calculate_prediction(metrix):
        """
        计算精度
        """
        label_pre = []
        current_sum = 0
        for i in range(metrix.shape[0]):
            label_total_sum = metrix.sum(axis=0)[i]  # TP+FP,模型预测为正样本总数
            current_sum += metrix[i][i] #TP,预测为正实际也为正
            pre= 0
            if label_total_sum != 0:
               pre = round(100 * metrix[i][i] / label_total_sum, 4)   #这个4是保留小数点后4位的意思。
            label_pre.append(pre)
            # print("每类精度：", label_pre)
        all_pre = round(100 * current_sum / metrix.sum(), 4)
        print("总精度：", all_pre)
        return label_pre, all_pre

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    all_preds = torch.tensor([])
    all_preds = all_preds.to(device)
    all_labels = torch.tensor([])
    all_labels = all_labels.to(device)
    best_acc_list=[]
    result =str(idx)
    train_data=MyDataset(txt=r'D:\PDR\Origin\Train&Test\ID_5fold\Train_Data'+result+'.txt', transform=transforms.ToTensor())  #训练集
    train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True, num_workers=0) #一次一起训练batch_size个样本，计算平均损失函数值，更新参数
    test_data=MyDataset(txt=r'D:\PDR\Origin\Train&Test\ID_5fold\Test_Data'+result+'.txt', transform=transforms.ToTensor())  #测试集
    test_loader = DataLoader(dataset=test_data, batch_size=8, shuffle=False, num_workers=0)
    model =pyconvhgresnet50().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)#加了权重衰减效果要好一点点(跟文档里设置的一样)
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_pred = []
    best_target = []
    epoch_losses = []

    def train(ep):

        model.train()
        epoch_loss = 0
        correct_train = 0
        for iter, data in enumerate(train_loader):
            bg,label=data
            bg, label = bg.to(device), label.to(device)
            prediction = model(bg)  # prediction.size()):[27,2]
            optimizer.zero_grad()  # 因为这里梯度是累加的，所以每次记得清零
            loss = loss_func(prediction, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            pred = prediction.data.max(1, keepdim=True)[1]  # pred.size:[27,1]
            correct_train += pred.eq(label.data.view_as(pred)).cpu().sum()

        train_acc = 100. * correct_train / len(train_data)
        epoch_loss /= len(train_data)
        print('Epoch {}, loss {:.4f},Accuracy: {}/{} ({:.04f}%)'.format(ep, epoch_loss, correct_train, len(train_data),train_acc))

        fp = open(r'D:\PDR\FR\data\total2\Result_train_'+result + '.txt', 'a+')  # 结果存入train_result文本文件
        fp.write('Epoch {}\000,  loss= {:.4f}\000,  Accuracy= {:.04f}%\n'.format(ep, epoch_loss,train_acc))
        fp.close()

        return train_acc,epoch_loss

    def test():

        global best_acc
        global best_model_wts
        global best_pred
        global best_target
        all_preds = torch.tensor([])
        all_preds = all_preds.to(device)
        all_targets = torch.tensor([])
        all_targets = all_targets.to(device)  #torch.float32
        with torch.no_grad():
            model.eval()
            correct = 0
            test_loss = 0
            pred_s = []
            test_Y_s = []
            for iter, data in enumerate(test_loader):
                test_bg, test_Y=data
                test_bg,test_Y=test_bg.to(device),test_Y.to(device)
                all_targets = torch.cat((all_targets, test_Y.view(-1,1)), dim=0)
                prediction = model(test_bg)
                loss = loss_func(prediction, test_Y)
                test_loss += loss.detach().item()
                pred = prediction.data.max(1, keepdim=True)[1]
                all_preds = torch.tensor(torch.cat((all_preds, pred) , dim=0))
                pred_s.append(pred)
                correct += pred.eq(test_Y.data.view_as(pred)).cpu().sum()

            test_loss /= len(test_data)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.04f}%)\n'.format(
                test_loss, correct, len(test_data),
                100. * correct / len(test_data)))
            fp = open(r'D:\PDR\FR\data\total2\Result_test_' +result+ '.txt', 'a+')  # 结果存入test_result文本文件
            fp.write('Epoch {}\000,  loss= {:.4f}\000,  Accuracy= {:.04f}%\n'.format(epoch, test_loss, 100. * correct / len(test_data)))
            fp.close()
            epoch_acc = 100. * correct / len(test_data)
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_pred = all_preds
                best_target = all_targets
        return epoch_acc,test_loss, best_model_wts, best_acc, best_pred, best_target

    if __name__ == "__main__":
        since = time.time()
        train_accs = []
        train_losses=[]
        test_accs = []
        test_losses=[]
        MAX_EPOCH = 1000 #(128) 随机分配里：95epoch就差不多了。
        for epoch in range(1, MAX_EPOCH):
            print(epoch)
            train_acc,train_loss = train(epoch)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            test_acc,test_loss, best_model_wts, best_acc, best_pre, best_target = test()
            test_accs.append(test_acc)
            test_losses.append(test_loss)
            all_preds = torch.cat((all_preds, best_pre), dim=0)
            all_labels = torch.cat((all_labels, best_target), dim=0)
            best_acc_list.append(best_acc)

        torch.save(model.state_dict(), r'D:\PDR\FR\data\total2\\'+result + '.pth')

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.04f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc:  {:4f}'.format(best_acc))
        conf_matrix = torch.zeros(42, 42, dtype=torch.int64)
        # attack_types = ['1', '2', '3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
        #                 '21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39',
        #                 '40','41','42']  # Mask_dataset
        conf_matrix = confusion_matrix(best_target, best_pred, conf_matrix=conf_matrix)
        conf_matrix = confusion_matrix(all_preds,all_labels, conf_matrix=conf_matrix)
        label_pre, all_pre = calculate_prediction(np.array(conf_matrix))
        # label_recall, all_recall = calculate_recall(np.array(conf_matrix))  #测召回率
        #signal_f1,all_f1 = calculate_f1(label_pre, all_pre, label_recall, all_recall)  测f1

        fp = open(r'D:\PDR\FR\data\sto2\\' + result+'.txt', 'a+')  #结果存入result4文本文件
        #fp.write('\n每类精度：{}\n总精度：{}\n'.format(label_pre, all_pre))
        #fp.write('每类召回率：{}\n总召回率：{}\n'.format(label_recall, all_recall))
        #fp.write('每类f1：{}\n总f1：{}\n'.format(signal_f1,all_f1))
        fp.write('Best test Acc: {:4f}\n\n'.format(best_acc))
        fp.close()