import torch
import numpy as np
import argparse
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import itertools
import csv

def diff(pred_1, pred_2):
    lst3 = [i for i, label in enumerate(pred_1) if label!=pred_2[i]]
    return len(lst3)

def True_False(y_pred_1, y_pred_2, y_test):

    count = 0
    for j, value in enumerate(y_test):
        if (y_pred_1[j]==value)&(y_pred_2[j]!=value):
            count += 1
    
    return count

def load_model_10(path, x_test):
    model = torch.load(path)
    model.to("cuda:0")
    tmp = []
    model.eval()
    acdd = 0
    total = 0
    with torch.no_grad():
        for i, j in x_test:
            i = i.to("cuda:0")
            j = j.to("cuda:0")
            l = model.forward(i)
            _,output_index = torch.max(l,1)
            total+=j.size(0)
            acdd += (output_index == j).sum().float()
            tmp.append(l)
    features = torch.cat(tuple(tmp), dim=0)
    y_pred = features.argmax(axis=1)
    res = 100*acdd/total

    return y_pred, res.item()

dataset_load_func = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist':torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'imagenet' : torchvision.datasets.ImageFolder
    }

def load_testdata(datasets):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_dataset = dataset_load_func[datasets]("./data", train=False, download=True, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    y_test = [i for _, i in test_dataset]
    return test_dataloader, y_test


def testing(y_pred1, y_pred2, y_test):
    di = diff(y_pred1, y_pred2)
    fd = True_False(y_pred1, y_pred2, y_test)

    print('-----------------Testing Result-----------------')
    print('Diff: ', di, 'Found: ', di-fd, 'Error Detection rate: ', (di-fd)/di)

def diff_detail(y_pred_1, y_pred_2, y_test):

    count_TT = 0
    count_TF = 0
    count_FT = 0
    count_FF_Same = 0
    count_FF_Diff = 0

    for j, value in enumerate(y_test):
        if (y_pred_1[j]==value)&(y_pred_2[j]==value):
            count_TT += 1
        elif (y_pred_1[j]==value)&(y_pred_2[j]!=value):
            count_TF += 1
        elif (y_pred_1[j]!=value)&(y_pred_2[j]==value):
            count_FT += 1
        elif (y_pred_1[j]!=value)&(y_pred_2[j]!=value):
            if y_pred_1[j]==y_pred_2[j]:
                count_FF_Same += 1
            else:
                count_FF_Diff += 1   

    return count_TT,count_TF,count_FT,count_FF_Diff,count_FF_Same

def analysis(model1_name,model2_name,y_pred1, y_pred2, y_test, temp):
    di = diff(y_pred1, y_pred2)
    count_TT,count_TF,count_FT,count_FF_Diff,count_FF_Same = diff_detail(y_pred1, y_pred2, y_test)

    acc_m1, acc_m1_2 = temp
    approx_acc = 1 - (di/len(y_test))

    print('-----------------Analysis Result-----------------')
    print('acc_m1: ', acc_m1, 'acc_m1: ', acc_m1_2, 'Diff #: ', di, 'approx_acc: ', approx_acc)

    f = open('./cifar10_result.csv','a', newline='')
    wr = csv.writer(f)
    wr.writerow([model1_name,model2_name,acc_m1,acc_m1_2,count_TT,count_TF,count_FT,count_FF_Diff,count_FF_Same])
    
    f.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10', choices=[
                    'cifar10', 'cifar100', 'fmnist', 'mnist'], help='Choose dataset (default: CIFAR10)')
    parser.add_argument('--method', type=str, default='Analysis', choices=[
                    'Testing', 'Analysis'], help='Choose method (default: Analysis)')
    parser.add_argument('--best', type=str, default='acc', choices=[
                    'acc', 'loss'], help='Choose Model eval method(default: acc)')

    args = parser.parse_args()

    #model_list = ["vggnet", "resnet", "densenet", "pyramidnet", "custom"]
    model_list = ["densenet"]
    
    per_list = list(itertools.permutations([0,1,2], 2))

    data, y_test = load_testdata("cifar10")

    f = open('./cifar10_result.csv','a', newline='')
    wr = csv.writer(f)
    wr.writerow(["model1","model2","acc_m1","acc_m1_2","count_TT","count_TF","count_FT","count_FF(Diff)","count_FF(Same)"])
    f.close()

    for model in model_list:   
        print("=========== {} ===========".format(model))
        for a, b in per_list:
            model1_name = "{}_{}".format(model,a)
            model2_name = "{}_{}".format(model,b)
            y_pred1, t1 = load_model_10("./output/{}/_0/{}/{}_best.pt".format(model, model1_name, args.best),data)
            y_pred2, t2 = load_model_10("./output/{}/_0/{}/{}_best.pt".format(model, model2_name, args.best),data)
            t3 = [t1, t2]
            if args.method == 'Testing':
                testing(y_pred1, y_pred2, y_test)
            elif args.method == 'Analysis':
                analysis(model1_name,model2_name,y_pred1, y_pred2, y_test, t3)
            else:
                print('Choose the wrong method')
