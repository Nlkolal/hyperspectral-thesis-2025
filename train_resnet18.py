import torch
import time
import torch.nn as nn
import numpy as np
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
from torchinfo import summary
from tqdm import tqdm

def resnet18_cifar(num_classes=20, pretrained=False):
    model = resnet18(weights=None)
    
    # Replace the stem
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # remove downsampling

    # Adjust classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions
    for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res
class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)

def main():

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available()))

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory=True
    num_workers = 4
    prefetch_factor=2
    persistent_workers=True
    non_blocking = True
    print(device)


    #resnet = torchvision.models.resnet18(weights=None, num_classes=200)
    resnet = resnet18_cifar()
    resnet.to(device)
    






    #size = 224
    batch_size = 64
    #resize = 256
    train_dir = Path("tiny-imagenet-20") / "train"
    val_dir = Path("tiny-imagenet-20") / "val"

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_train = torchvision.datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize, 
        ])
    )

    dataset_val = torchvision.datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.ToTensor(),
        normalize,
        ])
    )

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=pin_memory,            
                    persistent_workers=persistent_workers,    
                    prefetch_factor=prefetch_factor   
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=pin_memory,            
                    persistent_workers=persistent_workers,    
                    prefetch_factor=prefetch_factor   
    )

    opt = torch.optim.AdamW(resnet.parameters(), lr=3e-4, weight_decay=1e-4)
    lossfn = nn.CrossEntropyLoss().to(device)

    summary(resnet)

    EPOCHS = 10

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        print(f"Epoch: {epoch}")
        resnet.train()

        total = 0
        Loss = 0
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        for i, (images, labels) in tqdm(enumerate(loader_train), total=len(loader_train)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = resnet(images)
                loss = lossfn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            Loss += loss.item()
            total += images.size(0)
            #    scaler.scale(loss).backward()


        print(f"Epoch {epoch} traning done, with loss: {Loss}")

        resnet.eval()
        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(loader_val), total=len(loader_val)):
                images = images.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                outputs = resnet(images)
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
        print(f"Epoch {epoch} validation done with top1avg: {top1.avg} top5avg:{top5.avg}")




    #model.eval()
    #    total_pred, total_gt = [], []
    #    with torch.no_grad():
    #        for xb,yb in val_dl:
    #           xb, yb = xb.to(device), yb.to(device)
    #            pred = model(xb).argmax(1).cpu().numpy().ravel()
    #            gt = yb.cpu().numpy().ravel()
    #            total_pred.append(pred); total_gt.append(gt)
    #    pred = np.concatenate(total_pred)
    #    gt = np.concatenate(total_gt)
    #    dt = time.time() - t0

if __name__ == "__main__":  
    main()