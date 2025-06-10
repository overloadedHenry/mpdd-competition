import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np
import json
from dataset import MyDataset
from my_model_torch import myTestModel, MambaTrans
from sklearn.model_selection import KFold
from argparse import ArgumentParser

def train(model, train_loader, epochs, optimizer, criterion, device, learning_schedule, alpha, test_loader=None):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            audio = batch['a_feature'].to(device)
            vision = batch['v_feature'].to(device)
            text = batch['p_feature'].to(device)
            labels = batch['task_label'].to(device)

            optimizer.zero_grad()
            logits, contrast_loss = model(audio, vision, text)
            classification_loss = criterion(logits, labels)
            loss = classification_loss + alpha * contrast_loss

            # print(loss)
            
            # 计算并打印梯度范数
            loss.backward()
            
            # 计算梯度范数


            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=40.0)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            optimizer.step()
            learning_schedule.step()
        print(f"Epoch {epoch + 1}/{epochs}, Final_Loss: {loss.item():.4f}, Classification_loss: {classification_loss.item():.4f}, Contrastive_loss: {contrast_loss.item():.4f}, Gradient_norm: {total_norm:.4f}")
        if test_loader is not None:
            test_acc, test_loss, correct, total = evaluate_test(model, test_loader, criterion, device, alpha)
            print(f'test Accuracy: {test_acc:.4f}, test Loss: {test_loss:.4f}, Correct: {correct}, Total: {total}')


def evaluate_val(model, val_loader, criterion, device, alpha):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            audio = batch['a_feature'].to(device)
            vision = batch['v_feature'].to(device)
            text = batch['p_feature'].to(device)
            labels = batch['task_label'].to(device)

            logits, contrast_loss = model(audio, vision, text)
            classification_loss = criterion(logits, labels)
            loss = classification_loss + alpha * contrast_loss
            val_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            # print(pred)
            # print(labels)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = val_loss / len(val_loader)
    return accuracy, avg_loss

def evaluate_test(model, test_dataloader, criterion, device, alpha):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for  batch in test_dataloader:
            
            # print(batch)
            audio = batch['a_feature'].to(device)
            vision = batch['v_feature'].to(device)
            text = batch['p_feature'].to(device)
            labels = batch['task_label'].to(device)

            logits, contrast_loss = model(audio, vision, text)
            classification_loss = criterion(logits, labels)
            loss = classification_loss + alpha * contrast_loss
            val_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total

        avg_loss = val_loss / len(test_dataloader)
    return accuracy, avg_loss, correct, total

if __name__ == '__main__':

    torch.manual_seed(42)
    parser = ArgumentParser()
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--test_root_path', type=str, default="E:\\acmm_mpdd\MPDD-data\MPDD-Young\Testing")
    parser.add_argument('--test_json_path', type=str, default="E:\\acmm_mpdd\MPDD-data\MPDD-Young\Testing\labels\Testing_files_modified.json")
    parser.add_argument('--ps_feature_path', type=str)
    parser.add_argument('--test_ps_feature_path', type=str)
    parser.add_argument('--load_params', type=bool, default=False)
    parser.add_argument('--task_label_num', type=int, default=2)
    parser.add_argument('--model', type=str, default='mamba')
    parser.add_argument('--k_fold', type=int, default=-1)
    
    args = parser.parse_args()
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root_path = args.root_path
    json_path = args.json_path
    ps_feature_path = args.ps_feature_path
    test_ps_feature_path = args.test_ps_feature_path
    
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    with open(args.test_json_path, 'r', encoding='utf-8') as f:
        test_json_data = json.load(f)
    
    # print(len(test_json_data))
    # exit()

    mpddDataset = MyDataset(root_path, json_data, ps_feature_path, task_label_num=args.task_label_num, feature_max_len=32) 
    testDataset = MyDataset(args.test_root_path, test_json_data, test_ps_feature_path, task_label_num=args.task_label_num, feature_max_len=32)
    # print(f"Dataset Size: {len(testDataset)}")
    # exit()
    epochs = 50
    batch_size = 16
    learning_rate = 1e-4
    weight_dacay=1e-4

    if args.k_fold > 0:
    # K 折交叉验证参数
        kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)
        dataset_size = len(mpddDataset)
        indices = list(range(dataset_size))

        # 存储每折的结果
        fold_accuracies = []
        
        # K 折循环
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\nFold {fold + 1}/{args.k_fold}")
            
            # 创建训练和验证的采样器
            train_subsampler = SubsetRandomSampler(train_idx)
            val_subsampler = SubsetRandomSampler(val_idx)
            
            # 创建 DataLoader
            train_loader = DataLoader(mpddDataset, batch_size=batch_size, sampler=train_subsampler)
            val_loader = DataLoader(mpddDataset, batch_size=batch_size, sampler=val_subsampler)
            
            # 初始化模型、优化器和损失函数
            if args.model == 'transformer':
                model = myTestModel(num_class=args.task_label_num).to(device)
            elif args.model == 'mamba':
                model = MambaTrans(num_class=args.task_label_num).to(device)
            else:
                raise ValueError("Invalid model name.")

            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_dacay, )
            criterion = nn.CrossEntropyLoss()
            cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=32,eta_min=2e-6)

            
            # 训练模型
            train(model, train_loader, epochs, optimizer, criterion, device, cosine_schedule, 0)
            
            # 评估模型
            accuracy_val, val_loss = evaluate_val(model, val_loader, criterion, device, 0)
            print(f"Fold {fold + 1} Validation Accuracy: {accuracy_val:.4f}, Validation Loss: {val_loss:.4f}")
            fold_accuracies.append(accuracy_val)
        
        # 计算平均性能
        avg_accuracy = np.mean(fold_accuracies)
        print(f"\nAverage Accuracy across {args.k_fold} folds: {avg_accuracy:.4f}")
        torch.save(model, f'model_params/Trans_mamba_class{args.task_label_num}.pt')
        print("Model Parameters have been Saved!!!!!")

    else:
        train_loader = DataLoader(mpddDataset, batch_size=batch_size)
        test_loader = DataLoader(testDataset, batch_size=66)
        # 初始化模型、优化器和损失函数
        if args.model == 'transformer':
            model = myTestModel(num_class=args.task_label_num).to(device)
        elif args.model == 'mamba':
            model = MambaTrans(num_class=args.task_label_num).to(device)
        else:
            raise ValueError("Invalid model name.")

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_dacay, )
        criterion = nn.CrossEntropyLoss()
        cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=32,eta_min=2e-6)

        
        # 训练模型
        train(model, train_loader, epochs, optimizer, criterion, device, cosine_schedule, 0, test_loader=test_loader)
        torch.save(model, f'E:\\acmm_mpdd\ours\model_params/Trans_mamba_class{args.task_label_num}.pt')
        print("Model Parameters have been Saved!!!!!")

