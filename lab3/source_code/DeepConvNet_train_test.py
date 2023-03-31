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
from net import DeepConvNet
from dataloader import read_bci_data
from utiles import transform_dataset, plot_comparison_result

print(os.path.dirname(os.path.abspath(__file__))+"\n")

device = torch.device("cuda", 0)

if __name__ == '__main__':
    
    # ==================== hyper-parameters ===================================
    epochs = 150
    batch_size = 64
    lr = 1e-2
    # =========================================================================

    # ==================== define parameters ==================================
    loss_DeepConv_ELU        = 0
    loss_DeepConv_ReLU       = 0
    loss_DeepConv_LeakyReLU  = 0

    count_train = 0

    correct_train_count   = 0

    avg_acc_train_DeepConv_ELU = 0
    avg_acc_train_DeepConv_ReLU = 0
    avg_acc_train_DeepConv_LeakyReLU = 0
    avg_acc_test_DeepConv_ELU = 0
    avg_acc_test_DeepConv_ReLU = 0
    avg_acc_test_DeepConv_LeakyReLU = 0

    acc_train_history_DeepConv_ELU = []
    acc_train_history_DeepConv_ReLU = []
    acc_train_history_DeepConv_LeakyReLU = []
    acc_test_history_DeepConv_ELU = []
    acc_test_history_DeepConv_ReLU = []
    acc_test_history_DeepConv_LeakyReLU = []

    Max_acc_train_DeepConv_ELU = 0
    Max_acc_train_DeepConv_ReLU = 0
    Max_acc_train_DeepConv_LeakyReLU = 0
    Max_acc_test_DeepConv_ELU = 0
    Max_acc_test_DeepConv_ReLU = 0
    Max_acc_test_DeepConv_LeakyReLU = 0

    # =========================================================================

    model_DeepConv_ELU = DeepConvNet(class_num=2, act_choose="ELU")
    model_DeepConv_ReLU = DeepConvNet(class_num=2, act_choose="ReLU")
    model_DeepConv_LeakyReLU = DeepConvNet(class_num=2, act_choose="LeakyReLU")


    optimizer_DeepConv_ELU = optim.Adam(model_DeepConv_ELU.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
    optimizer_DeepConv_ReLU = optim.Adam(model_DeepConv_ReLU.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)
    optimizer_DeepConv_LeakyReLU = optim.Adam(model_DeepConv_LeakyReLU.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-7)

    # optimizer_DeepConv_ELU = optim.RMSprop(model_DeepConv_ELU.parameters(),lr = lr, momentum = 0.9, weight_decay=1e-3)
    # optimizer_DeepConv_ReLU = optim.RMSprop(model_DeepConv_ReLU.parameters(),lr = lr, momentum = 0.9, weight_decay=1e-3)
    # optimizer_DeepConv_LeakyReLU = optim.RMSprop(model_DeepConv_LeakyReLU.parameters(),lr = lr, momentum = 0.9, weight_decay=1e-3)

    # optimizer_DeepConv_ELU = optim.SGD(model_DeepConv_ELU.parameters(), lr=0.01, momentum=0.9)
    # optimizer_DeepConv_ReLU = optim.SGD(model_DeepConv_ReLU.parameters(), lr=0.01, momentum=0.9)
    # optimizer_DeepConv_LeakyReLU = optim.SGD(model_DeepConv_LeakyReLU.parameters(), lr=0.01, momentum=0.9)



    scheduler_DeepConv_ELU = lr_scheduler.StepLR(optimizer_DeepConv_ELU, step_size=1, gamma=0.94)
    scheduler_DeepConv_ReLU = lr_scheduler.StepLR(optimizer_DeepConv_ReLU, step_size=1, gamma=0.94)
    scheduler_DeepConv_LeakyReLU = lr_scheduler.StepLR(optimizer_DeepConv_LeakyReLU, step_size=1, gamma=0.94)

    # scheduler_DeepConv_ELU = lr_scheduler.MultiStepLR(optimizer_DeepConv_ELU, milestones=[400,500,1000], gamma=0.5)
    # scheduler_DeepConv_ReLU = lr_scheduler.MultiStepLR(optimizer_DeepConv_ReLU, milestones=[400,500,1000], gamma=0.5)
    # scheduler_DeepConv_LeakyReLU = lr_scheduler.MultiStepLR(optimizer_DeepConv_LeakyReLU, milestones=[400,500,1000], gamma=0.5)


    loss_func_DeepConv_ELU = nn.CrossEntropyLoss()
    loss_func_DeepConv_ReLU = nn.CrossEntropyLoss()
    loss_func_DeepConv_LeakyReLU = nn.CrossEntropyLoss()

    train_dataset, train_label, test_dataset, test_label = read_bci_data()

    train_data = transform_dataset(signal=train_dataset, label=train_label)
    test_data = transform_dataset(signal=test_dataset, label=test_label)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, drop_last=False)
    del train_dataset, train_label, train_data, test_dataset, test_label, test_data

    model_DeepConv_ELU.to(device)
    model_DeepConv_ReLU.to(device)
    model_DeepConv_LeakyReLU.to(device)

    

    for epoch in range(epochs):
        model_DeepConv_ELU.train()
        model_DeepConv_ReLU.train()
        model_DeepConv_LeakyReLU.train()
        correct_DeepConv_ELU = 0
        correct_DeepConv_ReLU = 0
        correct_DeepConv_LeakyReLU = 0
        count_train = 0
        with tqdm(iterable=train_dataloader, bar_format='{desc} {percentage:3.0f}%|{bar}| {postfix}',) as pbar:
            start_time = datetime.now()
            
            for iteration, batch in enumerate(train_dataloader):
                pbar.set_description_str(f"\33[37m【Epoch {epoch + 1:03d}/{epochs}】")
                signals, targets = batch['data'], batch['label']

                with torch.no_grad():
                    signals = signals.to(device)
                    targets = targets.to(device, torch.uint8)

                optimizer_DeepConv_ELU.zero_grad()
                optimizer_DeepConv_ReLU.zero_grad()
                optimizer_DeepConv_LeakyReLU.zero_grad()
                
                #----------------------#
                #   Forward
                #----------------------#
                predict_DeepConv_ELU = model_DeepConv_ELU(signals)
                predict_DeepConv_ReLU = model_DeepConv_ReLU(signals)
                predict_DeepConv_LeakyReLU = model_DeepConv_LeakyReLU(signals)
                
                #----------------------#
                #   caculate loss
                #----------------------#
                loss_value_all  = 0
                loss_count      = 0
                for l in range(len(predict_DeepConv_ELU)):
                    loss_item = loss_func_DeepConv_ELU(predict_DeepConv_ELU[l], targets[l])
                    loss_value_all  += loss_item
                    loss_count     += 1
                loss_value_DeepConv_ELU = loss_value_all / loss_count

                loss_value_all  = 0
                loss_count      = 0
                for l in range(len(predict_DeepConv_ReLU)):
                    loss_item = loss_func_DeepConv_ReLU(predict_DeepConv_ReLU[l], targets[l])
                    loss_value_all  += loss_item
                    loss_count     += 1
                loss_value_DeepConv_ReLU = loss_value_all / loss_count

                loss_value_all  = 0
                loss_count      = 0
                for l in range(len(predict_DeepConv_LeakyReLU)):
                    loss_item = loss_func_DeepConv_LeakyReLU(predict_DeepConv_LeakyReLU[l], targets[l])
                    loss_value_all  += loss_item
                    loss_count     += 1
                loss_value_DeepConv_LeakyReLU = loss_value_all / loss_count

                #----------------------#
                #   Backpropagation
                #----------------------#
                loss_value_DeepConv_ELU.backward()
                loss_value_DeepConv_ReLU.backward()
                loss_value_DeepConv_LeakyReLU.backward()
                optimizer_DeepConv_ELU.step()
                optimizer_DeepConv_ReLU.step()
                optimizer_DeepConv_LeakyReLU.step()


                loss_DeepConv_ELU += loss_value_DeepConv_ELU.item()
                loss_DeepConv_ReLU += loss_value_DeepConv_ReLU.item()
                loss_DeepConv_LeakyReLU += loss_value_DeepConv_LeakyReLU.item()

                # =============================== caculate Acc. =====================================================================
                correct_DeepConv_ELU = correct_DeepConv_ELU + (torch.argmax(predict_DeepConv_ELU,dim=1)==targets).sum().item()
                correct_DeepConv_ReLU = correct_DeepConv_ReLU + (torch.argmax(predict_DeepConv_ReLU,dim=1)==targets).sum().item()
                correct_DeepConv_LeakyReLU = correct_DeepConv_LeakyReLU + (torch.argmax(predict_DeepConv_LeakyReLU,dim=1)==targets).sum().item()
                
                
                count_train += len(batch["data"])
                avg_acc_train_DeepConv_ELU = correct_DeepConv_ELU * 100 / count_train
                avg_acc_train_DeepConv_ReLU = correct_DeepConv_ReLU * 100 / count_train
                avg_acc_train_DeepConv_LeakyReLU = correct_DeepConv_LeakyReLU * 100 / count_train
                # ====================================================================================================================
                
                cur_time = datetime.now()
                delta_time = cur_time - start_time
                delta_time = timedelta(seconds=delta_time.seconds)
                # pbar.set_postfix_str(f"train_acc={ave_acc_train:.2f}, train_loss_DeepConv_ELU={loss_DeepConv_ELU / (iteration + 1):.3f}, train_loss_DeepConv_ReLU={loss_DeepConv_ReLU / (iteration + 1):.3f}, train_loss_DeepConv_LeakyReLU={loss_DeepConv_LeakyReLU / (iteration + 1):.3f}, {delta_time}\33[0m")
                pbar.set_postfix_str(f"train_loss_DeepConv_ELU={loss_DeepConv_ELU / (iteration + 1):.3f}, train_loss_DeepConv_ReLU={loss_DeepConv_ReLU / (iteration + 1):.3f}, train_loss_DeepConv_LeakyReLU={loss_DeepConv_LeakyReLU / (iteration + 1):.3f}, Acc_DeepConv_ELU={avg_acc_train_DeepConv_ELU:.3f}%, Acc_DeepConv_ReLU={avg_acc_train_DeepConv_ReLU :.3f}%, Acc_DeepConv_LeakyReLU={avg_acc_train_DeepConv_LeakyReLU :.3f}%, {delta_time}\33[0m")
                
                pbar.update(1)
        
        acc_train_history_DeepConv_ELU.append(avg_acc_train_DeepConv_ELU)
        acc_train_history_DeepConv_ReLU.append(avg_acc_train_DeepConv_ReLU)
        acc_train_history_DeepConv_LeakyReLU.append(avg_acc_train_DeepConv_LeakyReLU)
        if Max_acc_train_DeepConv_ELU < avg_acc_train_DeepConv_ELU:
            Max_acc_train_DeepConv_ELU = avg_acc_train_DeepConv_ELU
        if Max_acc_train_DeepConv_ReLU < avg_acc_train_DeepConv_ReLU:
            Max_acc_train_DeepConv_ReLU = avg_acc_train_DeepConv_ReLU
        if Max_acc_train_DeepConv_LeakyReLU < avg_acc_train_DeepConv_LeakyReLU:
            Max_acc_train_DeepConv_LeakyReLU = avg_acc_train_DeepConv_LeakyReLU


        model_DeepConv_ELU.eval()
        model_DeepConv_ReLU.eval()
        model_DeepConv_LeakyReLU.eval()
        correct_DeepConv_ELU = 0
        correct_DeepConv_ReLU = 0
        correct_DeepConv_LeakyReLU = 0
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
                predict_DeepConv_ELU = model_DeepConv_ELU(signals)
                predict_DeepConv_ReLU = model_DeepConv_ReLU(signals)
                predict_DeepConv_LeakyReLU = model_DeepConv_LeakyReLU(signals)
                
                # ================================= caculate Acc. =====================================================================
                correct_DeepConv_ELU = correct_DeepConv_ELU + (torch.argmax(predict_DeepConv_ELU, dim=1)==targets).sum().item()
                correct_DeepConv_ReLU = correct_DeepConv_ReLU + (torch.argmax(predict_DeepConv_ReLU, dim=1)==targets).sum().item()
                correct_DeepConv_LeakyReLU = correct_DeepConv_LeakyReLU + (torch.argmax(predict_DeepConv_LeakyReLU, dim=1)==targets).sum().item()
                
                avg_acc_test_DeepConv_ELU = correct_DeepConv_ELU *100 / (iteration + 1)
                avg_acc_test_DeepConv_ReLU = correct_DeepConv_ReLU *100 / (iteration + 1)
                avg_acc_test_DeepConv_LeakyReLU = correct_DeepConv_LeakyReLU *100 / (iteration + 1)
                # ======================================================================================================================

                cur_time = datetime.now()
                delta_time = cur_time - start_time
                delta_time = timedelta(seconds=delta_time.seconds)
                pbar.set_postfix_str(f"Acc_DeepConv_ELU={avg_acc_test_DeepConv_ELU :.3f}%, Acc_DeepConv_ReLU={avg_acc_test_DeepConv_ReLU:.3f}%, Acc_DeepConv_LeakyReLU={avg_acc_test_DeepConv_LeakyReLU :.3f}%, {delta_time}\33[0m")
                
                pbar.update(1)
        acc_test_history_DeepConv_ELU.append(avg_acc_test_DeepConv_ELU)
        acc_test_history_DeepConv_ReLU.append(avg_acc_test_DeepConv_ReLU)
        acc_test_history_DeepConv_LeakyReLU.append(avg_acc_test_DeepConv_LeakyReLU)
        
        if Max_acc_test_DeepConv_ELU < avg_acc_test_DeepConv_ELU:
            Max_acc_test_DeepConv_ELU = avg_acc_test_DeepConv_ELU
        if Max_acc_test_DeepConv_ReLU < avg_acc_test_DeepConv_ReLU:
            Max_acc_test_DeepConv_ReLU = avg_acc_test_DeepConv_ReLU
        if Max_acc_test_DeepConv_LeakyReLU < avg_acc_test_DeepConv_LeakyReLU:
            Max_acc_test_DeepConv_LeakyReLU = avg_acc_test_DeepConv_LeakyReLU
    

    print("DeepConvNet with ELU       Max train Acc = ", Max_acc_train_DeepConv_ELU)
    print("DeepConvNet with ELU       Max test  Acc = ", Max_acc_test_DeepConv_ELU)
    print("DeepConvNet with ReLU      Max train Acc = ", Max_acc_train_DeepConv_ReLU)
    print("DeepConvNet with ReLU      Max test  Acc = ", Max_acc_test_DeepConv_ReLU)
    print("DeepConvNet with LeakyReLU Max train Acc = ", Max_acc_train_DeepConv_LeakyReLU)
    print("DeepConvNet with LeakyReLU Max test  Acc = ", Max_acc_test_DeepConv_LeakyReLU)

    plot_comparison_result(epochs, acc_test_history_DeepConv_ELU, acc_test_history_DeepConv_ReLU, acc_test_history_DeepConv_LeakyReLU,
                           acc_train_history_DeepConv_ELU, acc_train_history_DeepConv_ReLU, acc_train_history_DeepConv_LeakyReLU, Net_name = "DeepConv",
                           fig_name="DeepConvNet_epoch150_batch64_adam_le_2")