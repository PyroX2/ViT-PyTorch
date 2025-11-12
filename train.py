from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torcheval.metrics import MulticlassAUPRC, MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from typing import List, Tuple
from argparse import ArgumentParser
import os


def get_config():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory. Should contain directories train, val and test splits.")
    parser.add_argument("--batch-size", type=int, required=False, default=64, help="Batch size used for training, validation and testing")
    parser.add_argument("--num-workers", type=int, required=False, default=16, help="Number of workers")
    parser.add_argument("--patience", type=int, required=False, default=10, help="Patience used for early stopping.")
    return parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

# Tensorboard logging
now = datetime.now()
writer = SummaryWriter(f'logs/{now.strftime("%Y-%m-%d %H:%M:%S")}')

class MetricsCalculator:
    def __init__(self, num_classes):
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.f1_score = MulticlassF1Score(num_classes=num_classes)
        self.auprc = MulticlassAUPRC(num_classes=num_classes)
        self.auroc = MulticlassAUROC(num_classes=num_classes)
    
    def calculate(self, outputs: List, targets: List) -> Tuple:
        outputs = torch.tensor(outputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)

        self.accuracy.update(outputs, targets)
        accuracy = self.accuracy.compute()

        self.f1_score.update(outputs, targets)
        f1_score = self.f1_score.compute()

        self.auprc.update(outputs, targets)
        auprc = self.auprc.compute()

        self.auroc.update(outputs, targets)
        auroc = self.auroc.compute()

        self.accuracy.reset()
        self.f1_score.reset()
        self.auprc.reset()
        self.auroc.reset()

        return accuracy, f1_score, auprc, auroc
    

class EarlyStopping:
    def __init__(self, patience: int = 10, init_metric_value: float = float("inf"), task: str = "min") -> None:
        self.patience = patience
        self.best_value_recorded = init_metric_value
        self.no_improvement_count = 0

        assert task in ["min", "max"], print(f"Task value of {task} is incorrect. Choose from ['min', 'max']")
        self.task = task

    def check_early_stop(self, metric) -> bool:
        if (self.task == "min" and metric < self.best_value_recorded) or (self.task == "max" and metric > self.best_value_recorded):
            self.best_value_recorded = metric
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count == self.patience:
            return True
        else:
            return False
                

# Validation
def eval(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, criterion: torch.nn.Module):
    metrics_calculator = MetricsCalculator(num_classes=len(val_loader.dataset.classes))
    
    val_loss = 0
    outputs_list = []
    targets_list = []

    model.eval()
    with torch.no_grad():
        for input_batch, target_batch in tqdm(val_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            output = model(input_batch).logits
            loss = criterion(output, target_batch)

            val_loss += loss.detach().item()
            outputs_list.extend(output.detach().cpu().tolist())
            targets_list.extend(target_batch.detach().cpu().tolist())

    del input_batch, target_batch, output, loss

    mean_val_loss = val_loss / len(val_loader)
    val_accuracy, val_f1_score, val_auprc, val_auroc = metrics_calculator.calculate(outputs_list, targets_list)

    return mean_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc

# Training
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, early_stop_patience=10):
    early_stopping = EarlyStopping(patience=early_stop_patience)
    metrics_calculator = MetricsCalculator(num_classes=len(train_loader.dataset.classes)) # For calculating metrics
    best_val_loss = float("inf") # For saving best model based on validation loss

    for epoch in range(num_epochs):
        print(f"\n --- Training epoch: {epoch} --- ")

        # Init epoch variables
        epoch_loss = 0
        outputs_list = []
        targets_list = []

        model.train()
        for input_batch, target_batch in tqdm(train_loader):
            optimizer.zero_grad()

            # Convert to device
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            output = model(input_batch).logits # Get model output logits
            loss = criterion(output, target_batch)
            loss.backward()
            optimizer.step()

            # Update epoch variables
            epoch_loss += loss.detach().item()
            outputs_list.extend(output.detach().cpu().tolist())
            targets_list.extend(target_batch.detach().cpu().tolist())

        # Clear GPU cache
        del input_batch, target_batch, output, loss
        torch.cuda.empty_cache()

        # Get training metrics
        mean_train_loss = epoch_loss / len(train_loader)
        train_accuracy, train_f1_score, train_auprc, train_auroc = metrics_calculator.calculate(outputs_list, targets_list)

        # Get validation metrics
        print("--- Evaluating ---")
        mean_val_loss, val_accuracy, val_f1_score, val_auprc, val_auroc = eval(model, val_loader, criterion)

        # Log metrics with Tensorboard
        writer.add_scalar("train/loss", mean_train_loss, global_step=epoch)
        writer.add_scalar("train/accuracy", train_accuracy, global_step=epoch)
        writer.add_scalar("train/f1_score", train_f1_score, global_step=epoch)
        writer.add_scalar("train/auprc", train_auprc, global_step=epoch)
        writer.add_scalar("train/auroc", train_auroc, global_step=epoch)

        writer.add_scalar("val/loss", mean_val_loss, global_step=epoch)
        writer.add_scalar("val/accuracy", val_accuracy, global_step=epoch)
        writer.add_scalar("val/f1_score", val_f1_score, global_step=epoch)
        writer.add_scalar("val/auprc", val_auprc, global_step=epoch)
        writer.add_scalar("val/auroc", val_auroc, global_step=epoch)

        # Save best model
        if mean_val_loss < best_val_loss:
            torch.save(model.state_dict(), "best_model_ckpt.pth")
            best_val_loss = mean_val_loss
        
        print(f"Train loss: {mean_train_loss}, Val loss: {mean_val_loss}")

        stop_training = early_stopping.check_early_stop(mean_val_loss)
        if stop_training:
            break

    print(f"\n--- Training finished after {epoch+1} epochs ---\n")

def main():
    args = get_config()
    dataset_path = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers
    early_stop_patience = args.patience

    # Dataset preparation
    transform = v2.Compose([
        v2.Resize((384, 384)),
        v2.ToTensor()
    ])

    train_ds = ImageFolder(os.path.join(dataset_path, "train/multiclass"), transform=transform)
    val_ds = ImageFolder(os.path.join(dataset_path, "val/multiclass"), transform=transform)
    test_ds = ImageFolder(os.path.join(dataset_path, "test/multiclass"), transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)


    # Load a pre-trained ViT model and feature extractor
    model_name = 'google/vit-large-patch16-384'
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name, 
        attn_implementation='eager'
    )
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(train_ds.classes))
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train(model, train_loader, val_loader, criterion, optimizer, 100, early_stop_patience=early_stop_patience)

    torch.save(model.state_dict(), "final_model_ckpt.pth")

    print("\n--- Model testing ---")
    test_loss, test_accuracy, test_f1_score, test_auprc, test_auroc = eval(model, test_loader, criterion)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1 Score: {test_f1_score}, Test AUPRC: {test_auprc}, Test AUROC: {test_auroc}")
    writer.add_scalar("test/loss", test_loss)
    writer.add_scalar("test/accuracy", test_accuracy)
    writer.add_scalar("test/f1_score", test_f1_score)
    writer.add_scalar("test/auprc", test_auprc)
    writer.add_scalar("test/auroc", test_auroc)


if __name__ == "__main__":
    main()