import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(filename):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def plot_training_losses(results_files):
    plt.figure(figsize=(10, 6))
    for filename in results_files:
        results = load_results(filename)
        train_losses = results['train_losses']
        model_name = results['model']
        plt.plot(train_losses, label=model_name)
    plt.xlabel('Batch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss per Batch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_losses.png')
    plt.show()

def plot_test_metrics(results_files):
    models = []
    aucs = []
    losses = []
    for filename in results_files:
        results = load_results(filename)
        model_name = results['model']
        auc = results['test_auc']
        loss = results['test_loss']
        models.append(model_name)
        aucs.append(auc)
        losses.append(loss)

    # Convert lists to arrays for easier indexing
    aucs = np.array(aucs)
    losses = np.array(losses)

    # Plot AUC for each task
    num_tasks = len(aucs[0])  # Assuming all models have the same number of tasks
    for task in range(num_tasks):
        plt.figure(figsize=(8, 6))
        for i, model in enumerate(models):
            plt.bar(i, aucs[i][task], label=model)
        plt.xticks(range(len(models)), models)
        plt.xlabel('Model')
        plt.ylabel('AUC')
        plt.title(f'Test AUC Comparison - Task {task}')
        plt.tight_layout()
        plt.savefig(f'test_auc_task_{task}.png')
        plt.show()

    # Plot Loss for each task
    for task in range(num_tasks):
        plt.figure(figsize=(8, 6))
        for i, model in enumerate(models):
            plt.bar(i, losses[i][task], label=model)
        plt.xticks(range(len(models)), models)
        plt.xlabel('Model')
        plt.ylabel('Loss')
        plt.title(f'Test Loss Comparison - Task {task}')
        plt.tight_layout()
        plt.savefig(f'test_loss_task_{task}.png')
        plt.show()

if __name__ == '__main__':
    # List all result files in the 'results' directory
    results_dir = 'results'
    results_files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.json')]

    if not results_files:
        print("No results files found in the 'results' directory.")
        exit()

    # Plot training losses
    plot_training_losses(results_files)

    # Plot test metrics
    plot_test_metrics(results_files)
