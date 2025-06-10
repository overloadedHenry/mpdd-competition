import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MyDataset
from my_model_torch import myTestModel
from argparse import ArgumentParser
import json

def evaluate(model, val_dataloader, criterion, device, alpha):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for batch_id, batch in enumerate(val_dataloader):
            
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

        avg_loss = val_loss / len(val_dataloader)
    return accuracy, avg_loss, correct, total

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_path', type=str, default=r"E:\acmm_mpdd\MPDD-data\MPDD-Young\Testing")
    parser.add_argument('--json_path', type=str, default=r"E:\acmm_mpdd\MPDD-data\MPDD-Young\Testing\labels\Testing_files_modified.json")
    parser.add_argument('--ps_feature_path', type=str, default=r"E:\acmm_mpdd\ours\descriptions_embeddings_with_ids_bge_zh_ds.npy")
    parser.add_argument('--params_path', type=str, default=r"E:\acmm_mpdd\ours\model_params\Trans_mamba_class2.pt")
    parser.add_argument('--task_label_num', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    mpddDataset = MyDataset(args.root_path, json_data, args.ps_feature_path, task_label_num=args.task_label_num, feature_max_len=32)
    val_dataloader = DataLoader(mpddDataset, batch_size=16, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model = torch.load(args.params_path)
    model = model.to(device)

    acc, avg_loss, correct, total= evaluate(model, val_dataloader, criterion, device, alpha=0)
    print("Accuracy: {:.3f}".format(acc))
    print("Avg Loss: {:.3f}".format(avg_loss))
    print("Correct: {}, Total: {}".format(correct, total))
    


        
    

