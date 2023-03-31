import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from net import EEGNet, DeepConvNet
from dataloader import read_bci_data
from utiles import transform_dataset, plot_comparison_result

print(os.path.dirname(os.path.abspath(__file__))+"\n")

device = torch.device("cuda", 0)

if __name__ == '__main__':
    
    # ==================== hyper-parameters ===================================
    epochs = 100
    batch_size = 64
    lr = 1e-2
    # =========================================================================

    # ==================== define parameters ==================================
    loss_EEG_ELU        = 0
    loss_EEG_ReLU       = 0
    loss_EEG_LeakyReLU  = 0

    count_train = 0

    correct_train_count   = 0

    avg_acc_train_EEG_ELU = 0
    avg_acc_train_EEG_ReLU = 0
    avg_acc_train_EEG_LeakyReLU = 0
    avg_acc_test_EEG_ELU = 0
    avg_acc_test_EEG_ReLU = 0
    avg_acc_test_EEG_LeakyReLU = 0

    acc_train_history_EEG_ELU = []
    acc_train_history_EEG_ReLU = []
    acc_train_history_EEG_LeakyReLU = []
    acc_test_history_EEG_ELU = []
    acc_test_history_EEG_ReLU = []
    acc_test_history_EEG_LeakyReLU = []

    Max_acc_train_EEG_ELU = 0
    Max_acc_train_EEG_ReLU = 0
    Max_acc_train_EEG_LeakyReLU = 0
    Max_acc_test_EEG_ELU = 0
    Max_acc_test_EEG_ReLU = 0
    Max_acc_test_EEG_LeakyReLU = 0

    # =========================================================================
    
    model_EEG_ELU = EEGNet(class_num=2, act_choose="ELU")
    model_EEG_ReLU = EEGNet(class_num=2, act_choose="ReLU")
    model_EEG_LeakyReLU = EEGNet(class_num=2, act_choose="LeakyReLU")

    
    # model_EEG_ELU = DeepConvNet(class_num=2, act_choose="ELU")
    # model_EEG_ReLU = DeepConvNet(class_num=2, act_choose="ReLU")
    # model_EEG_LeakyReLU = DeepConvNet(class_num=2, act_choose="LeakyReLU")


    optimizer_EEG_ELU = optim.Adam(model_EEG_ELU.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
    optimizer_EEG_ReLU = optim.Adam(model_EEG_ReLU.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
    optimizer_EEG_LeakyReLU = optim.Adam(model_EEG_LeakyReLU.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)

    # optimizer_EEG_ELU = optim.RMSprop(model_EEG_ELU.parameters(),lr = lr, momentum = 0.9, weight_decay=1e-3)
    # optimizer_EEG_ReLU = optim.RMSprop(model_EEG_ReLU.parameters(),lr = lr, momentum = 0.9, weight_decay=1e-3)
    # optimizer_EEG_LeakyReLU = optim.RMSprop(model_EEG_LeakyReLU.parameters(),lr = lr, momentum = 0.9, weight_decay=1e-3)

    # optimizer_EEG_ELU = optim.SGD(model_EEG_ELU.parameters(), lr=0.01, momentum=0.9)
    # optimizer_EEG_ReLU = optim.SGD(model_EEG_ReLU.parameters(), lr=0.01, momentum=0.9)
    # optimizer_EEG_LeakyReLU = optim.SGD(model_EEG_LeakyReLU.parameters(), lr=0.01, momentum=0.9)



    scheduler_EEG_ELU = lr_scheduler.StepLR(optimizer_EEG_ELU, step_size=1, gamma=0.94)
    scheduler_EEG_ReLU = lr_scheduler.StepLR(optimizer_EEG_ReLU, step_size=1, gamma=0.94)
    scheduler_EEG_LeakyReLU = lr_scheduler.StepLR(optimizer_EEG_LeakyReLU, step_size=1, gamma=0.94)

    # scheduler_EEG_ELU = lr_scheduler.MultiStepLR(optimizer_EEG_ELU, milestones=[400,500,1000], gamma=0.5)
    # scheduler_EEG_ReLU = lr_scheduler.MultiStepLR(optimizer_EEG_ReLU, milestones=[400,500,1000], gamma=0.5)
    # scheduler_EEG_LeakyReLU = lr_scheduler.MultiStepLR(optimizer_EEG_LeakyReLU, milestones=[400,500,1000], gamma=0.5)


    loss_func_EEG_ELU = nn.CrossEntropyLoss()
    loss_func_EEG_ReLU = nn.CrossEntropyLoss()
    loss_func_EEG_LeakyReLU = nn.CrossEntropyLoss()

    train_dataset, train_label, test_dataset, test_label = read_bci_data()

    train_data = transform_dataset(signal=train_dataset, label=train_label)
    test_data = transform_dataset(signal=test_dataset, label=test_label)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)
    del train_dataset, train_label, train_data, test_dataset, test_label, test_data

    model_EEG_ELU.to(device)
    model_EEG_ReLU.to(device)
    model_EEG_LeakyReLU.to(device)

    

    for epoch in range(epochs):
        model_EEG_ELU.train()
        model_EEG_ReLU.train()
        model_EEG_LeakyReLU.train()
        correct_EEG_ELU = 0
        correct_EEG_ReLU = 0
        correct_EEG_LeakyReLU = 0
        count_train = 0
        with tqdm(iterable=train_dataloader, bar_format='{desc} {percentage:3.0f}%|{bar}| {postfix}',) as pbar:
            start_time = datetime.now()
            
            for iteration, batch in enumerate(train_dataloader):
                pbar.set_description_str(f"\33[37m【Epoch {epoch + 1:03d}/{epochs}】")
                signals, targets = batch['data'], batch['label']

                with torch.no_grad():
                    signals = signals.to(device)
                    targets = targets.to(device, torch.uint8)

                optimizer_EEG_ELU.zero_grad()
                optimizer_EEG_ReLU.zero_grad()
                optimizer_EEG_LeakyReLU.zero_grad()
                
                #----------------------#
                #   Forward
                #----------------------#
                predict_EEG_ELU = model_EEG_ELU(signals)
                predict_EEG_ReLU = model_EEG_ReLU(signals)
                predict_EEG_LeakyReLU = model_EEG_LeakyReLU(signals)
                
                #----------------------#
                #   caculate loss
                #----------------------#
                loss_value_all  = 0
                loss_count      = 0
                for l in range(len(predict_EEG_ELU)):
                    loss_item = loss_func_EEG_ELU(predict_EEG_ELU[l], targets[l])
                    loss_value_all  += loss_item
                    loss_count     += 1
                loss_value_EEG_ELU = loss_value_all / loss_count

                loss_value_all  = 0
                loss_count      = 0
                for l in range(len(predict_EEG_ReLU)):
                    loss_item = loss_func_EEG_ReLU(predict_EEG_ReLU[l], targets[l])
                    loss_value_all  += loss_item
                    loss_count     += 1
                loss_value_EEG_ReLU = loss_value_all / loss_count

                loss_value_all  = 0
                loss_count      = 0
                for l in range(len(predict_EEG_LeakyReLU)):
                    loss_item = loss_func_EEG_LeakyReLU(predict_EEG_LeakyReLU[l], targets[l])
                    loss_value_all  += loss_item
                    loss_count     += 1
                loss_value_EEG_LeakyReLU = loss_value_all / loss_count

                #----------------------#
                #   Backpropagation
                #----------------------#
                loss_value_EEG_ELU.backward()
                loss_value_EEG_ReLU.backward()
                loss_value_EEG_LeakyReLU.backward()
                optimizer_EEG_ELU.step()
                optimizer_EEG_ReLU.step()
                optimizer_EEG_LeakyReLU.step()


                loss_EEG_ELU += loss_value_EEG_ELU.item()
                loss_EEG_ReLU += loss_value_EEG_ReLU.item()
                loss_EEG_LeakyReLU += loss_value_EEG_LeakyReLU.item()

                # =============================== caculate Acc. =====================================================================
                correct_EEG_ELU = correct_EEG_ELU + (torch.argmax(predict_EEG_ELU,dim=1)==targets).sum().item()
                correct_EEG_ReLU = correct_EEG_ReLU + (torch.argmax(predict_EEG_ReLU,dim=1)==targets).sum().item()
                correct_EEG_LeakyReLU = correct_EEG_LeakyReLU + (torch.argmax(predict_EEG_LeakyReLU,dim=1)==targets).sum().item()
                
                
                count_train += len(batch["data"])
                avg_acc_train_EEG_ELU = correct_EEG_ELU * 100 / count_train
                avg_acc_train_EEG_ReLU = correct_EEG_ReLU * 100 / count_train
                avg_acc_train_EEG_LeakyReLU = correct_EEG_LeakyReLU * 100 / count_train
                # ====================================================================================================================
                
                cur_time = datetime.now()
                delta_time = cur_time - start_time
                delta_time = timedelta(seconds=delta_time.seconds)
                # pbar.set_postfix_str(f"train_acc={ave_acc_train:.2f}, train_loss_EEG_ELU={loss_EEG_ELU / (iteration + 1):.3f}, train_loss_EEG_ReLU={loss_EEG_ReLU / (iteration + 1):.3f}, train_loss_EEG_LeakyReLU={loss_EEG_LeakyReLU / (iteration + 1):.3f}, {delta_time}\33[0m")
                pbar.set_postfix_str(f"train_loss_EEG_ELU={loss_EEG_ELU / (iteration + 1):.3f}, train_loss_EEG_ReLU={loss_EEG_ReLU / (iteration + 1):.3f}, train_loss_EEG_LeakyReLU={loss_EEG_LeakyReLU / (iteration + 1):.3f}, Acc_EEG_ELU={avg_acc_train_EEG_ELU:.3f}%, Acc_EEG_ReLU={avg_acc_train_EEG_ReLU :.3f}%, Acc_EEG_LeakyReLU={avg_acc_train_EEG_LeakyReLU :.3f}%, {delta_time}\33[0m")
                
                pbar.update(1)
        
        acc_train_history_EEG_ELU.append(avg_acc_train_EEG_ELU)
        acc_train_history_EEG_ReLU.append(avg_acc_train_EEG_ReLU)
        acc_train_history_EEG_LeakyReLU.append(avg_acc_train_EEG_LeakyReLU)
        if Max_acc_train_EEG_ELU < avg_acc_train_EEG_ELU:
            Max_acc_train_EEG_ELU = avg_acc_train_EEG_ELU
        if Max_acc_train_EEG_ReLU < avg_acc_train_EEG_ReLU:
            Max_acc_train_EEG_ReLU = avg_acc_train_EEG_ReLU
        if Max_acc_train_EEG_LeakyReLU < avg_acc_train_EEG_LeakyReLU:
            Max_acc_train_EEG_LeakyReLU = avg_acc_train_EEG_LeakyReLU


        model_EEG_ELU.eval()
        model_EEG_ReLU.eval()
        model_EEG_LeakyReLU.eval()
        correct_EEG_ELU = 0
        correct_EEG_ReLU = 0
        correct_EEG_LeakyReLU = 0
        with tqdm(iterable=test_dataloader, bar_format='{desc} {percentage:3.0f}%|{bar}| {postfix}',) as pbar:
            start_time = datetime.now()
            for iteration, batch in enumerate(test_dataloader):
                pbar.set_description_str(f"\33[37m【Epoch {epoch + 1:03d}/{epochs}】")
                signals, targets = batch['data'], batch['label']

                with torch.no_grad():
                    signals = signals.to(device)
                    targets = targets.to(device)

                #----------------------#
                #   Forward
                #----------------------#
                predict_EEG_ELU = model_EEG_ELU(signals)
                predict_EEG_ReLU = model_EEG_ReLU(signals)
                predict_EEG_LeakyReLU = model_EEG_LeakyReLU(signals)
                
                # ================================= caculate Acc. =====================================================================
                correct_EEG_ELU = correct_EEG_ELU + (torch.argmax(predict_EEG_ELU, dim=1)==targets).sum().item()
                correct_EEG_ReLU = correct_EEG_ReLU + (torch.argmax(predict_EEG_ReLU, dim=1)==targets).sum().item()
                correct_EEG_LeakyReLU = correct_EEG_LeakyReLU + (torch.argmax(predict_EEG_LeakyReLU, dim=1)==targets).sum().item()
                
                avg_acc_test_EEG_ELU = correct_EEG_ELU *100 / (iteration + 1)
                avg_acc_test_EEG_ReLU = correct_EEG_ReLU *100 / (iteration + 1)
                avg_acc_test_EEG_LeakyReLU = correct_EEG_LeakyReLU *100 / (iteration + 1)
                # ======================================================================================================================

                cur_time = datetime.now()
                delta_time = cur_time - start_time
                delta_time = timedelta(seconds=delta_time.seconds)
                pbar.set_postfix_str(f"Acc_EEG_ELU={avg_acc_test_EEG_ELU :.3f}%, Acc_EEG_ReLU={avg_acc_test_EEG_ReLU:.3f}%, Acc_EEG_LeakyReLU={avg_acc_test_EEG_LeakyReLU :.3f}%, {delta_time}\33[0m")
                
                pbar.update(1)
        acc_test_history_EEG_ELU.append(avg_acc_test_EEG_ELU)
        acc_test_history_EEG_ReLU.append(avg_acc_test_EEG_ReLU)
        acc_test_history_EEG_LeakyReLU.append(avg_acc_test_EEG_LeakyReLU)
        
        if Max_acc_test_EEG_ELU < avg_acc_test_EEG_ELU:
            Max_acc_test_EEG_ELU = avg_acc_test_EEG_ELU
        if Max_acc_test_EEG_ReLU < avg_acc_test_EEG_ReLU:
            Max_acc_test_EEG_ReLU = avg_acc_test_EEG_ReLU
        if Max_acc_test_EEG_LeakyReLU < avg_acc_test_EEG_LeakyReLU:
            Max_acc_test_EEG_LeakyReLU = avg_acc_test_EEG_LeakyReLU
    

    print("EEGNet with ELU Max train Acc = ", Max_acc_train_EEG_ELU)
    print("EEGNet with ELU Max test Acc = ", Max_acc_test_EEG_ELU)
    print("EEGNet with ReLU Max train Acc = ", Max_acc_train_EEG_ReLU)
    print("EEGNet with ReLU Max test Acc = ", Max_acc_test_EEG_ReLU)
    print("EEGNet with LeakyReLU Max train Acc = ", Max_acc_train_EEG_LeakyReLU)
    print("EEGNet with LeakyReLU Max test Acc = ", Max_acc_test_EEG_LeakyReLU)

    plot_comparison_result(epochs, acc_test_history_EEG_ELU, acc_test_history_EEG_ReLU, acc_test_history_EEG_LeakyReLU,
                            acc_train_history_EEG_ELU, acc_train_history_EEG_ReLU, acc_train_history_EEG_LeakyReLU, Net_name = "EEG")
    
