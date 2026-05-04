import os
import json
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

########################################
# 0. SEED (재현성)
########################################
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################
# 1. CONFIG
########################################
EXPERIMENTS = {
    "A_freeze_pretrained": {
        "pretrained": True,
        "finetune": "freeze",
        "lr": 1e-3,
    },
    "B_partial_pretrained": {
        "pretrained": True,
        "finetune": "partial",
        "lr": 1e-4,
    },
    "C_full_pretrained": {
        "pretrained": True,
        "finetune": "full",
        "lr": 1e-5,
    },
    "D_full_scratch": {
        "pretrained": False,
        "finetune": "full",
        "lr": 1e-3,
    },
}

DATA_DIR = "./data"  # class별 폴더만 있으면 됨
NUM_CLASSES = 150
EPOCHS = 10
BATCH_SIZE = 32

SAVE_MODEL_DIR = "./models"
SAVE_RESULT_DIR = "./results"

os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_RESULT_DIR, exist_ok=True)

########################################
# 2. MODEL
########################################
def build_model(pretrained=True):
    weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)

########################################
# 3. FINETUNING
########################################
def set_finetuning(model, mode):
    if mode == "freeze":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True

    elif mode == "partial":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.layer4.parameters():
            p.requires_grad = True
        for p in model.fc.parameters():
            p.requires_grad = True

    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True

    return model

########################################
# 4. DATA (자동 분할)
########################################
def get_dataloaders():

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # transform 없이 index만 가져오기용
    base_dataset = datasets.ImageFolder(DATA_DIR)

    indices = list(range(len(base_dataset)))
    random.shuffle(indices)

    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_ds = Subset(
        datasets.ImageFolder(DATA_DIR, transform=train_tf),
        train_idx
    )

    val_ds = Subset(
        datasets.ImageFolder(DATA_DIR, transform=val_tf),
        val_idx
    )

    test_ds = Subset(
        datasets.ImageFolder(DATA_DIR, transform=val_tf),
        test_idx
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader

########################################
# 5. EVAL
########################################
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

########################################
# 6. TRAIN
########################################
def train_one(name, config, train_loader, val_loader, test_loader):

    print(f"\n=== {name} ===")

    model = build_model(config["pretrained"])
    model = set_finetuning(model, config["finetune"])

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"]
    )
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)

        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    test_acc = evaluate(model, test_loader)

    ################################
    # SAVE MODEL
    ################################
    torch.save(model.state_dict(), f"{SAVE_MODEL_DIR}/{name}.pth")

    ################################
    # SAVE METRICS
    ################################
    metrics = {
        "test_accuracy": test_acc,
        "val_accuracy": val_accs[-1]
    }

    with open(f"{SAVE_RESULT_DIR}/{name}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    ################################
    # SAVE CURVE
    ################################
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(val_accs, label="val_acc")
    plt.legend()
    plt.title(name)
    plt.savefig(f"{SAVE_RESULT_DIR}/{name}.png")
    plt.close()

    return metrics

########################################
# 7. MAIN
########################################
def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    results = {}

    for name, config in EXPERIMENTS.items():
        results[name] = train_one(name, config, train_loader, val_loader, test_loader)

    print("\nFINAL RESULTS")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()