import torch
import numpy as np
import os
import pathlib
import argparse
import json
import datetime
from sklearn.metrics import roc_auc_score
from clippyadagrad import ClippyAdagrad
from torch.optim import Adagrad, AdamW, Adam
from models.logistic_regression import LogisticRegressionModel
from models.neural_collaborative_filter import NCF
import tqdm
import sys

sys.path = ['Multitask-Recommendation-Library'] + sys.path
from models.sharedbottom import SharedBottomModel
from aliexpress import AliExpressDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='SB', choices=["LR", "SB", "NCF"])
    parser.add_argument('--optimizer', default='AD', choices=["AD", "CA"])
    args = parser.parse_args()

    path = pathlib.Path("./data/AliExpress_NL")

    train_dataset = AliExpressDataset(path / "train.csv")
    test_dataset = AliExpressDataset(path / "test.csv")

    batch_size = 2048
    embed_dim = 128
    task_num = 2

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=7, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=7, shuffle=False)

    field_dims = train_dataset.field_dims
    numerical_num = train_dataset.numerical_num
    # Calculate total input dimensions for logistic regression
    input_dim = sum(field_dims) + numerical_num

    if args.model == "LR":
        model = LogisticRegressionModel(input_dim, task_num).to(device)
    elif args.model == "SB":
        model = SharedBottomModel(field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                                  tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2).to(device)
    else:
        model = NCF(field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                              tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2).to(device)
    model.train()
    criterion = torch.nn.BCELoss()

    if args.optimizer == "AD":
        optimizer = Adam(model.parameters(), lr=1e-1)
    else:
        optimizer = ClippyAdagrad(model.parameters(), lr=1e-1)

    log_interval = 100

    total_loss = 0
    train_losses = []  # List to store training losses
    loader = tqdm.tqdm(train_data_loader, smoothing=0, mininterval=1.0)
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
            device), labels.to(device)

        if args.model == "LR":
            # One-hot encode categorical features
            batch_size = categorical_fields.size(0)
            one_hot_cat = []
            for idx, dim in enumerate(field_dims):
                one_hot = torch.zeros(batch_size, dim).to(device)
                one_hot.scatter_(1, categorical_fields[:, idx].unsqueeze(1), 1)
                one_hot_cat.append(one_hot)
            cat_one_hot = torch.cat(one_hot_cat, dim=1)

            # Concatenate numerical features
            inputs = torch.cat([cat_one_hot, numerical_fields], dim=1)
            y = model(inputs)
        else:
            y = model(categorical_fields, numerical_fields)


        loss_list = [criterion(y[i], labels[:, i].float()) for i in range(labels.size(1))]
        loss = 0
        for item in loss_list:
            loss += item
        loss /= len(loss_list)
        model.zero_grad()
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

    model.eval()

    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(test_data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
                device), labels.to(device)

            if args.model == "LR":
                # One-hot encode categorical features
                batch_size = categorical_fields.size(0)
                one_hot_cat = []
                for idx, dim in enumerate(field_dims):
                    one_hot = torch.zeros(batch_size, dim).to(device)
                    one_hot.scatter_(1, categorical_fields[:, idx].unsqueeze(1), 1)
                    one_hot_cat.append(one_hot)
                cat_one_hot = torch.cat(one_hot_cat, dim=1)

                # Concatenate numerical features
                inputs = torch.cat([cat_one_hot, numerical_fields], dim=1)
                y = model(inputs)
            else:
                y = model(categorical_fields, numerical_fields)

            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(
                    torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    print('AUC:', auc_results, 'Loss:', loss_results)

    # Save the experiment results
    results = {
        'model': args.model,
        'train_losses': train_losses,
        'test_auc': auc_results,
        'test_loss': loss_results,
    }

    # Create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_filename = os.path.join(results_dir, f'results_{args.model}_{timestamp}.json')

    # Save results to a JSON file
    with open(results_filename, 'w') as f:
        json.dump(results, f)

    print(f'Results saved to {results_filename}')