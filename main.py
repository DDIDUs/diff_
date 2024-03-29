import torch
import argparse
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from torchvision.models import resnet50

from pre_data import load_dataset,DeviceDataLoader
from models.den_model import *
from models.vgg_model import *
from models.res_model import *
from models.py_model import *
from models.cus_model import CustomModel

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, dir='./output'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dir = dir

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), '{}/checkpoint_val_{:.2f}.pt'.format(self.dir,val_loss))
        torch.save(model, '{}/loss_best.pt'.format(self.dir))
        self.val_loss_min = val_loss

def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(config, num, version):
    device = "cuda:0"
    train_model = config.train_model
    if config.dataset == "cifar100":
        number_of_classes = 100
    else:
        number_of_classes = 10

    output_dir = "{}/{}_{}/".format(version, config.train_model, num)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if train_model == 'vggnet':                                                                                                 # 학습 모델 준비
        if config.dataset == "mnist" or config.dataset == "fmnist":
            model = VGG(vgg_name="VGG16m", name=config.dataset, nc=number_of_classes)
        else:
            model = VGG(vgg_name="VGG16", name=config.dataset, nc=number_of_classes)
    elif train_model == 'resnet':
        model = ResNet50(config.dataset, nc=number_of_classes)
    elif train_model == 'densenet':
        model = DenseNet(growthRate=12, depth=121, reduction=0.5,
                        bottleneck=True, nClasses=number_of_classes, data=config.dataset)
    elif train_model == "pyramidnet":
        model = PyramidNet(dataset=config.dataset, depth=50, alpha=200, num_classes=number_of_classes, bottleneck=True)
    else:
        if config.dataset == "mnist":
            model = CustomModel(1, number_of_classes, 28)
        else:
            model = CustomModel(3, number_of_classes, 32)
    model.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if config.mode == "train":
        torch.cuda.empty_cache()
        early = EarlyStopping(patience=config.patience, dir=output_dir)
        train_dataloader, valid_dataloader = load_dataset(config)

        train_dataloader = DeviceDataLoader(train_dataloader, device)
        valid_dataloader = DeviceDataLoader(valid_dataloader, device)

        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, verbose=True)

        loss_arr = []
        best_acc = 0
        best_loss = 1000
        grad_clip = 0.01
        with open(output_dir + "_log.txt", "w") as logFile:
            for i in range(config.epochs):
                print("========== {} Epochs ==========".format(i))
                #f.write("========== {} Epochs ==========".format(i) + "\n")
                model.train()
                lrs = []
                train_losses = []
                for batch in train_dataloader:
                    x, y_ = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    output = model(x)
                    loss = loss_func(output,y_)
                    loss.backward()
                    optimizer.step()

                if i % 10 ==0:
                    loss_arr.append(loss.cpu().detach().numpy())
    
                correct = 0
                total = 0
                valid_loss = 0
                
                model.eval()
                with torch.no_grad():                                                                                                   # 모델 평가
                    for image,label in valid_dataloader:
                        x = image.to(device)
                        y = label.to(device)
                            
                        output = model.forward(x)
                        valid_loss += loss_func(output, y)
                        _,output_index = torch.max(output,1)
    
                        total += label.size(0)
                        correct += (output_index == y).sum().float()
                    logText = "Epoch {:03d}, Valid Acc: {:.2f}%, Valid loss: {:.2f}\n".format(i, 100*correct/total, valid_loss)
                    train_acc = "Accuracy against Validation Data: {:.2f}%, Valid_loss: {:.2f}".format(100*correct/total, valid_loss)
                    print(logText)
                    logFile.write(logText)
                    logFile.flush()
    
                    current_acc = (correct / total) * 100
                    if current_acc > best_acc:
                        print(" Accuracy increase from {:.2f}% to {:.2f}%. Model saved".format(best_acc, current_acc))
                        best_acc = current_acc
                        torch.save(model, '{}/acc_best.pt'.format(output_dir))
                early(valid_loss, model)
    
                if early.early_stop:
                    print("stop")
                    break
                scheduler.step()
    else:
        test_dataloader = load_dataset(config)
        model = torch.load('{}acc_best.pt'.format(output_dir)).to(device)
        model.eval()
        correct = 0
        total_cnt = 0
        
        for step, batch in enumerate(test_dataloader):
            batch[0], batch[1] = batch[0].to(device), batch[1].to(device)
            total_cnt += batch[1].size(0)
            logits = model(batch[0])
            _, predict = logits.max(1)
            correct += predict.eq(batch[1]).sum().item()
            c = (predict == batch[1]).squeeze()

        valid_acc = correct / total_cnt
        print("\nTest Acc : {}, {}".format(valid_acc,output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train test model.")
    parser.add_argument('-mode', type=str, default="train", help='Select mode')
    #parser.add_argument('-Augmentation', action="store_true", help="True when using data augmentation techniques")
    parser.add_argument('-repeat_num', type=int, default=1, help="The number of times you want to run the experiment")
    
    # Device Configuration
    parser.add_argument('-gpu', type=bool, default=True, help='Specify the gpu to use')
    
    #Model Parameters
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-epochs', type=int, default=120, help='Maximum # of training epochs')
    parser.add_argument('-patience', type=int, default=30, help='Early stop')
    parser.add_argument('-dataset', type=str, default="mnist", help='Early stop')
    parser.add_argument('-train_model', type=str, default='resnet', choices=['vggnet', 'resnet', 'densenet', 'pyramidnet', 'custom'], help= 'Select neural network model')
    
    config = parser.parse_args()

    save_dir = "./output/{}/".format(config.train_model)
    tmp = 0
    version = ""
    while True:
        version = save_dir + "_" + str(tmp)
        if not os.path.exists(version):
            os.makedirs(version)
            break
        else:
            tmp += 1

    for i in range(config.repeat_num):
        train(config, i, version)
    
    config.mode = False

    for i in range(config.repeat_num):
        train(config, i, version) 