import  torch
import  numpy as np
import  seaborn as sns
from    torch import optim, nn
import  matplotlib.pyplot as plt
import  visdom
from    torch.utils.data import DataLoader
from    YXSB_TRAIN import YXSB_TRAIN
from    torchvision.models import resnet18
from    sklearn import metrics
from    Utils import Flatten
import  csv
import  os
from    torchsummary import summary
from    time import *
from    torchviz import make_dot

Num_Classes = 7
batchsz = 64
epochs = 50

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = YXSB_TRAIN('YXSB_TRAIN', 224, mode='train')
val_db = YXSB_TRAIN('YXSB_TRAIN', 224, mode='val')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)

viz = visdom.Visdom()

LABELS = ['Lf_I','Lf_II','Lf_III','Lf_IV','Lf_V','Lf_VI','Lf_VII']

def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure('CNN', figsize=(10, 8))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                annot_kws={'size': 20, 'color': 'w'},
                fmt="d", )
    plt.ylabel("True Label", fontsize=18)
    plt.xlabel("Predicted Label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=18)
    plt.show()

def convert2oneHot(label, Lens):
    hot = np.zeros((len(label), Lens))
    for i in range(len(label)):
        hot[i][label[i]] = 1
    return np.array(hot)

def get_num_of_sample_for_each_class(model, loader):
    model.eval()

    num_of_sample_for_each_class = torch.zeros(Num_Classes)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        for i in range(Num_Classes):
            num_of_sample_for_each_class[i] += torch.eq(y, i).sum().float().item()

    return num_of_sample_for_each_class

def evalute(model, loader):
    model.eval()

    accuracy, precision, recall, F1 = 0, 0, 0, 0
    predict_eq_actual, actual_result, predict_result = torch.zeros(Num_Classes), torch.zeros(Num_Classes), torch.zeros(Num_Classes)
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        accuracy += torch.eq(pred, y).sum().float().item()

        for i in range(Num_Classes):
            predict_result[i] += torch.eq(pred, i).sum().float().item()
            actual_result[i]  += torch.eq(y, i).sum().float().item()
            for j in range(len(pred)):
                if (torch.eq(pred[j], i) and torch.eq(y[j], i)):
                    predict_eq_actual[i] += 1

    accuracy = accuracy / total
    precision = predict_eq_actual / predict_result
    recall = predict_eq_actual / actual_result
    F1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, F1

def Loss(model, loader):
    model.eval()
    criteon = nn.CrossEntropyLoss()
    loss = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            loss_batch = criteon(logits, y)
        loss += loss_batch
    return loss / total

def main():

    trained_model = resnet18(pretrained=True)
    model = nn.Sequential(*list(trained_model.children())[:-1],
                          Flatten(),
                          nn.Linear(512, Num_Classes)
                          ).to(device)

    Parameters = summary(model, input_size=(3, 224, 224))
    print(Parameters)

    criteon = nn.CrossEntropyLoss()

    num_of_train_sample_for_each_class = get_num_of_sample_for_each_class(model, train_loader)
    num_of_val_sample_for_each_class = get_num_of_sample_for_each_class(model, val_loader)
    print(num_of_train_sample_for_each_class)
    print(num_of_val_sample_for_each_class)
    print(num_of_train_sample_for_each_class+num_of_val_sample_for_each_class)

    x = torch.randn(1, 3, 224, 224)
    x = x.cuda()
    y = model(x)
    MyConvnetvis = make_dot(y, params = dict(list(model.named_parameters()) + [('x',x)]))
    MyConvnetvis.format = "png"
    MyConvnetvis.directory = "./ResNet18_MyConvnet_Vis"
    MyConvnetvis.view()

    best_train_accuracy, best_val_accuracy, best_epoch = 0, 0, 0
    viz.line([[0,0]], [0], win='loss', opts=dict(title='loss',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='accuracy', opts=dict(title='accuracy',legend=['train', 'valid']))

    viz.line([[0,0]], [0], win='precision_class1', opts=dict(title='precision_class1', legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='precision_class2', opts=dict(title='precision_class2', legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='precision_class3', opts=dict(title='precision_class3', legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='precision_class4', opts=dict(title='precision_class4', legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='precision_class5', opts=dict(title='precision_class5', legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='precision_class6', opts=dict(title='precision_class6', legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='precision_class7', opts=dict(title='precision_class7', legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='precision_average', opts=dict(title='precision_average', legend=['train', 'valid']))

    viz.line([[0,0]], [0], win='recall_class1', opts=dict(title='recall_class1',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='recall_class2', opts=dict(title='recall_class2',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='recall_class3', opts=dict(title='recall_class3',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='recall_class4', opts=dict(title='recall_class4',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='recall_class5', opts=dict(title='recall_class5',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='recall_class6', opts=dict(title='recall_class6',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='recall_class7', opts=dict(title='recall_class7',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='recall_average', opts=dict(title='recall_average',legend=['train', 'valid']))

    viz.line([[0,0]], [0], win='F1_class1', opts=dict(title='F1_class1',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='F1_class2', opts=dict(title='F1_class2',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='F1_class3', opts=dict(title='F1_class3',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='F1_class4', opts=dict(title='F1_class4',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='F1_class5', opts=dict(title='F1_class5',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='F1_class6', opts=dict(title='F1_class6',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='F1_class7', opts=dict(title='F1_class7',legend=['train', 'valid']))
    viz.line([[0,0]], [0], win='F1_average', opts=dict(title='F1_average',legend=['train', 'valid']))

    Evaluation_Result = torch.zeros(epochs+1,52)

    begin_time = time()

    for epoch in range(1,epochs+1):
        if epoch < 10:
            optimizer = optim.Adam(model.parameters(), lr=1.0e-5)
        else:
            optimizer = optim.Adam(model.parameters(), lr=1.0e-6)
        for step, (x,y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = Loss(model, train_loader).cpu()
        val_loss = Loss(model, val_loader).cpu()
        train_accuracy, train_precision, train_recall, train_F1 = evalute(model, train_loader)
        val_accuracy, val_precision, val_recall, val_F1 = evalute(model, val_loader)
        viz.line([[float(train_loss), float(val_loss)]], [epoch], win='loss', update='append')
        viz.line([[float(train_accuracy), float(val_accuracy)]], [epoch], win='accuracy', update='append')

        viz.line([[float(train_precision[0]), float(val_precision[0])]], [epoch], win='precision_class1', update='append')
        viz.line([[float(train_precision[1]), float(val_precision[1])]], [epoch], win='precision_class2', update='append')
        viz.line([[float(train_precision[2]), float(val_precision[2])]], [epoch], win='precision_class3', update='append')
        viz.line([[float(train_precision[3]), float(val_precision[3])]], [epoch], win='precision_class4', update='append')
        viz.line([[float(train_precision[4]), float(val_precision[4])]], [epoch], win='precision_class5', update='append')
        viz.line([[float(train_precision[5]), float(val_precision[5])]], [epoch], win='precision_class6', update='append')
        viz.line([[float(train_precision[6]), float(val_precision[6])]], [epoch], win='precision_class7', update='append')
        viz.line([[float(torch.mean(train_precision)), float(torch.mean(val_precision))]], [epoch], win='precision_average', update='append')

        viz.line([[float(train_recall[0]), float(val_recall[0])]], [epoch], win='recall_class1', update='append')
        viz.line([[float(train_recall[1]), float(val_recall[1])]], [epoch], win='recall_class2', update='append')
        viz.line([[float(train_recall[2]), float(val_recall[2])]], [epoch], win='recall_class3', update='append')
        viz.line([[float(train_recall[3]), float(val_recall[3])]], [epoch], win='recall_class4', update='append')
        viz.line([[float(train_recall[4]), float(val_recall[4])]], [epoch], win='recall_class5', update='append')
        viz.line([[float(train_recall[5]), float(val_recall[5])]], [epoch], win='recall_class6', update='append')
        viz.line([[float(train_recall[6]), float(val_recall[6])]], [epoch], win='recall_class7', update='append')
        viz.line([[float(torch.mean(train_recall)), float(torch.mean(val_recall))]], [epoch], win='recall_average', update='append')

        viz.line([[float(train_F1[0]), float(val_F1[0])]], [epoch], win='F1_class1', update='append')
        viz.line([[float(train_F1[1]), float(val_F1[1])]], [epoch], win='F1_class2', update='append')
        viz.line([[float(train_F1[2]), float(val_F1[2])]], [epoch], win='F1_class3', update='append')
        viz.line([[float(train_F1[3]), float(val_F1[3])]], [epoch], win='F1_class4', update='append')
        viz.line([[float(train_F1[4]), float(val_F1[4])]], [epoch], win='F1_class5', update='append')
        viz.line([[float(train_F1[5]), float(val_F1[5])]], [epoch], win='F1_class6', update='append')
        viz.line([[float(train_F1[6]), float(val_F1[6])]], [epoch], win='F1_class7', update='append')
        viz.line([[float(torch.mean(train_F1)), float(torch.mean(val_F1))]], [epoch], win='F1_average', update='append')

        Part1 = [train_loss, val_loss, train_accuracy, val_accuracy]
        Part2 = [train_precision[0], val_precision[0],\
                 train_precision[1], val_precision[1],\
                 train_precision[2], val_precision[2],\
                 train_precision[3], val_precision[3],\
                 train_precision[4], val_precision[4],\
                 train_precision[5], val_precision[5],\
                 train_precision[6], val_precision[6],\
                 torch.mean(train_precision), torch.mean(val_precision)]
        Part3 = [train_recall[0], val_recall[0],\
                 train_recall[1], val_recall[1],\
                 train_recall[2], val_recall[2],\
                 train_recall[3], val_recall[3],\
                 train_recall[4], val_recall[4],\
                 train_recall[5], val_recall[5],\
                 train_recall[6], val_recall[6],\
                 torch.mean(train_recall), torch.mean(val_recall)]
        Part4 = [train_F1[0], val_F1[0],\
                 train_F1[1], val_F1[1],\
                 train_F1[2], val_F1[2],\
                 train_F1[3], val_F1[3],\
                 train_F1[4], val_F1[4],\
                 train_F1[5], val_F1[5],\
                 train_F1[6], val_F1[6],\
                 torch.mean(train_F1), torch.mean(val_F1)]

        Evaluation_Result[epoch,:] = torch.tensor(Part1 + Part2 + Part3 + Part4)

        if val_accuracy > best_val_accuracy:
            best_epoch = epoch
            best_val_accuracy = val_accuracy
            best_train_accuracy = train_accuracy
            torch.save(model, './ResNet18_Best/ResNet18_Best.mdl')
        if val_accuracy == 1:
            torch.save(model, './ResNet18_Best/ResNet18_Best_'+str(epoch)+'.mdl')

        if epoch % 1 == 0:

            print("The ", epoch, " Training Loss       : ", str('%.4f' % float(train_loss)))
            print("The ", epoch, " Validation Loss     : ", str('%.4f' % float(val_loss)))
            print("The ", epoch, " Train Accuracy      : ", str('%.4f' % (train_accuracy * 100)), "%")
            print("The ", epoch, " Validation Accuracy : ", str('%.4f' % (val_accuracy * 100)), "%")
            print("The ", epoch, " Training Precision  : ", str('%.4f' % (train_precision[0] * 100)), '%', ' ',\
                  str('%.4f' % (train_precision[1] * 100)), '%', ' ', str('%.4f' % (train_precision[2] * 100)), '%',' ',\
                  str('%.4f' % (train_precision[3] * 100)), '%', ' ', str('%.4f' % (train_precision[4] * 100)), '%',' ',\
                  str('%.4f' % (train_precision[5] * 100)), '%', ' ', str('%.4f' % (train_precision[6] * 100)), '%',' ',\
                  "Average Precision: ", str('%.4f' % (torch.mean(train_precision) * 100)), '%')
            print("The ", epoch, " Validation Precision: ", str('%.4f' % (val_precision[0] * 100)), '%', ' ',\
                  str('%.4f' % (val_precision[1] * 100)), '%', ' ', str('%.4f' % (val_precision[2] * 100)), '%',' ',\
                  str('%.4f' % (val_precision[3] * 100)), '%', ' ', str('%.4f' % (val_precision[4] * 100)), '%',' ',\
                  str('%.4f' % (val_precision[5] * 100)), '%', ' ', str('%.4f' % (val_precision[6] * 100)), '%',' ',\
                  "Average Precision: ", str('%.4f' % (torch.mean(val_precision) * 100)), '%')
            print("The ", epoch, " Training Recall     : ", str('%.4f' % (train_recall[0] * 100)), '%', ' ',\
                  str('%.4f' % (train_recall[1] * 100)), '%', ' ', str('%.4f' % (train_recall[2] * 100)), '%',' ',\
                  str('%.4f' % (train_recall[3] * 100)), '%', ' ', str('%.4f' % (train_recall[4] * 100)), '%',' ',\
                  str('%.4f' % (train_recall[5] * 100)), '%', ' ', str('%.4f' % (train_recall[6] * 100)), '%',' ',\
                  "Average Recall   : ", str('%.4f' % (torch.mean(train_recall) * 100)), '%')
            print("The ", epoch, " Validation Recall   : ", str('%.4f' % (val_recall[0] * 100)), '%', ' ',\
                  str('%.4f' % (val_recall[1] * 100)), '%', ' ', str('%.4f' % (val_recall[2] * 100)), '%',' ',\
                  str('%.4f' % (val_recall[3] * 100)), '%', ' ', str('%.4f' % (val_recall[4] * 100)), '%',' ',\
                  str('%.4f' % (val_recall[5] * 100)), '%', ' ', str('%.4f' % (val_recall[6] * 100)), '%',' ',\
                  "Average Recall   : ", str('%.4f' % (torch.mean(val_recall) * 100)), '%')
            print("The ", epoch, " Training F1-Score   : ", str('%.4f' % float(train_F1[0])),\
                  str('%.4f' % float(train_F1[1])), str('%.4f' % float(train_F1[2])), str('%.4f' % float(train_F1[3])),\
                  str('%.4f' % float(train_F1[4])), str('%.4f' % float(train_F1[5])), str('%.4f' % float(train_F1[6])),\
                  "Average F1-Score: ", str('%.4f' % torch.mean(train_F1)))
            print("The ", epoch, " Validation F1-Score : ", str('%.4f' % float(val_F1[0])),\
                  str('%.4f' % float(val_F1[1])), str('%.4f' % float(val_F1[2])), str('%.4f' % float(val_F1[3])),\
                  str('%.4f' % float(val_F1[4])), str('%.4f' % float(val_F1[5])), str('%.4f' % float(val_F1[6])),\
                  "Average F1-Score: ", str('%.4f' % torch.mean(val_F1)))
            print("")

    end_time = time()
    run_time = end_time - begin_time
    print('The Running Time(s): ', run_time)
    file_handle = open('ResNet18_Runtime.txt', mode='w')
    file_handle.write(str(run_time) + ' Second')
    file_handle.close()

    with open(os.path.join('./', 'ResNet18_Evaluation_Result_For_Each_Epoch.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        for Type in Evaluation_Result:
            writer.writerow(Type.tolist())

    with open(os.path.join('./', 'ResNet18_Evaluation_Result_For_Best_Epoch('+str(best_epoch)+').csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(Evaluation_Result[best_epoch].tolist())

    print('Best_Val_Acc:', best_val_accuracy, 'Best_Epoch:', best_epoch, 'Corresponding Train_Acc:', best_train_accuracy)
    model = torch.load('./ResNet18_Best/ResNet18_Best.mdl')
    print('Loaded From ResNet18_Best!')

    Parameters = summary(model, input_size=(3, 224, 224))
    print(Parameters)

    train_accuracy, train_precision, train_recall, train_F1 = evalute(model, train_loader)
    val_accuracy, val_precision, val_recall, val_F1 = evalute(model, val_loader)

    print("The ", best_epoch, " Training Accuracy   : ", str('%.4f' % (train_accuracy * 100)), '%')
    print("The ", best_epoch, " Validation Accuracy : ", str('%.4f' % (val_accuracy * 100)), '%')
    print("The ", best_epoch, " Training Precision  : ", str('%.4f' % (train_precision[0] * 100)), '%',' ',\
          str('%.4f' % (train_precision[1] * 100)), '%',' ',str('%.4f' % (train_precision[2] * 100)), '%',' ',\
          str('%.4f' % (train_precision[3] * 100)), '%',' ',str('%.4f' % (train_precision[4] * 100)), '%',' ',\
          str('%.4f' % (train_precision[5] * 100)), '%',' ',str('%.4f' % (train_precision[6] * 100)), '%',' ',\
          "Average Precision: ", str('%.4f' % (torch.mean(train_precision) * 100)), '%')
    print("The ", best_epoch, " Validation Precision: ", str('%.4f' % (val_precision[0] * 100)), '%',' ', \
          str('%.4f' % (val_precision[1] * 100)), '%',' ',str('%.4f' % (val_precision[2] * 100)), '%',' ',\
          str('%.4f' % (val_precision[3] * 100)), '%',' ',str('%.4f' % (val_precision[4] * 100)), '%',' ',\
          str('%.4f' % (val_precision[5] * 100)), '%',' ',str('%.4f' % (val_precision[6] * 100)), '%',' ',\
          "Average Precision: ", str('%.4f' % (torch.mean(val_precision) * 100)), '%')
    print("The ", best_epoch, " Training Recall     : ", str('%.4f' % (train_recall[0] * 100)), '%',' ', \
          str('%.4f' % (train_recall[1] * 100)), '%', ' ',str('%.4f' % (train_recall[2] * 100)), '%',' ',\
          str('%.4f' % (train_recall[3] * 100)), '%', ' ',str('%.4f' % (train_recall[4] * 100)), '%',' ',\
          str('%.4f' % (train_recall[5] * 100)), '%', ' ',str('%.4f' % (train_recall[6] * 100)), '%',' ',\
          "Average Recall: ", str('%.4f' % (torch.mean(train_recall) * 100)), '%')
    print("The ", best_epoch, " Validation Recall   : ", str('%.4f' % (val_recall[0] * 100)), '%',' ', \
          str('%.4f' % (val_recall[1] * 100)), '%', ' ',str('%.4f' % (val_recall[2] * 100)), '%',' ',\
          str('%.4f' % (val_recall[3] * 100)), '%', ' ',str('%.4f' % (val_recall[4] * 100)), '%',' ',\
          str('%.4f' % (val_recall[5] * 100)), '%', ' ',str('%.4f' % (val_recall[6] * 100)), '%',' ',\
          "Average Recall: ", str('%.4f' % (torch.mean(val_recall) * 100)), '%')
    print("The ", best_epoch, " Training F1-Score   : ", str('%.4f' % float(train_F1[0])), str('%.4f' % float(train_F1[1])),\
          str('%.4f' % float(train_F1[2])), str('%.4f' % float(train_F1[3])), str('%.4f' % float(train_F1[4])),\
          str('%.4f' % float(train_F1[5])), str('%.4f' % float(train_F1[6])),\
          "Average F1-Score: ", str('%.4f' % (torch.mean(train_F1))))
    print("The ", best_epoch, " Validation F1-Score : ", str('%.4f' % float(val_F1[0])), str('%.4f' % float(val_F1[1])),\
          str('%.4f' % float(val_F1[2])), str('%.4f' % float(val_F1[3])), str('%.4f' % float(val_F1[4])),\
          str('%.4f' % float(val_F1[5])), str('%.4f' % float(val_F1[6])),\
          "Average F1-Score: ", str('%.4f' % (torch.mean(val_F1))))

    with torch.no_grad():
        i = 0
        for step, (x, y) in enumerate(val_loader):
            if i == 0:
                temp_val = model(x.cuda()).cpu()
                Valid_y = convert2oneHot(y, Num_Classes)
            else:
                temp_val_1 = model(x.cuda()).cpu()
                temp_val = np.vstack((temp_val,temp_val_1))
                Valid_y = np.vstack((Valid_y,convert2oneHot(y, Num_Classes)))
            i += 1

    valid_result = []
    for i in range(len(temp_val)):
        valid_result.append(np.argmax(temp_val[i]))
    validations = np.argmax(Valid_y, axis=1)
    show_confusion_matrix(validations, valid_result)

if __name__ == '__main__':
    main()
