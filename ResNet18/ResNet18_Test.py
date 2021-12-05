import  torch
import  numpy as np
import  seaborn as sns
from    torch import nn
import  matplotlib.pyplot as plt
import  visdom
from    torch.utils.data import DataLoader
from    YXSB_TRAIN import YXSB_TRAIN
from    YXSB_TEST_NEW import YXSB_TEST_NEW
from    torchvision.models import resnet18
from    sklearn import metrics
from    Utils import Flatten
import  csv
import  os
from    torchsummary import summary

Num_Classes = 7
batchsz = 64
lr = 1e-3
epochs = 100

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = YXSB_TRAIN('YXSB_TRAIN', 224, mode='train')
val_db = YXSB_TRAIN('YXSB_TRAIN', 224, mode='val')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)

viz = visdom.Visdom()

LABELS = ['I','II','III','IV','V','VI','VII']
def show_confusion_matrix(validations, predictions):
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure('ResNet10-LSTM', figsize=(8,10))
    plt.imshow(matrix, cmap=plt.cm.Blues,extent=(-0.5, 6.5, 6.5, -0.5))
    indices = range(len(matrix))
    plt.xticks(indices, LABELS)
    plt.yticks(indices, LABELS)
    plt.rcParams['font.size'] = 20
    plt.colorbar(fraction=0.0455, pad=0.0455)
    plt.xlabel('Predicted Label', fontsize=20)
    plt.ylabel('True Label', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for first_index in range(len(matrix)):
        for second_index in range(len(matrix[first_index])):
            plt.text(first_index, second_index, matrix[first_index][second_index],ha='center',va='center',fontsize=20)
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

def test_evalute(model, loader, part_filename, batchsz_test):
    model.eval()
    result = []; result_merge = []
    section = 0
    temp = 0
    for x, y in loader:
        if temp == section:
            x,y = x.to(device), y.to(device)
            Prob = []
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1).tolist()
                for row in range(batchsz_test):
                    Prob.append(logits[row][pred[row]].tolist())
                PROB = np.argmax(Prob)
                result.append([pred[PROB],section,section+PROB+1,PROB+2])
                print(section,section+PROB+1,pred[PROB])
            section += PROB+2
        temp += 1

    temp_type = result[0][0]
    start_point = result[0][1]
    merge_point = 1
    for i in range(1,len(result)):
        if result[i][0]==temp_type:
            merge_point += 1
            if i == len(result)-1:
               result_merge.append([temp_type, merge_point, start_point, result[i][2], result[i][2]-start_point+1])
        else:
            result_merge.append([temp_type, merge_point, start_point, result[i-1][2], result[i-1][2]-start_point+1])
            temp_type = result[i][0]
            start_point = result[i][1]
            merge_point = 1
            if i == len(result)-1:
               result_merge.append([temp_type, merge_point, start_point, result[i][2], result[i][2]-start_point+1])

    with open(os.path.join('./ResNet18_Test/Result' + part_filename + '.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        for Type in result:
            writer.writerow([Type[0],Type[1],Type[2],Type[3]])

    with open(os.path.join('./ResNet18_Test/Result_Merge' + part_filename + '.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        for Type in result_merge:
            writer.writerow([Type[0],Type[1],Type[2],Type[3],Type[4]])

    return result, result_merge

def main():

    model = torch.load('./ResNet18_Best/ResNet18_Best.mdl')
    print('Loaded From ResNet18_Best/ResNet18_Best.mdl')

    Parameters = summary(model, input_size=(3, 224, 224))
    print(Parameters)

    num_of_train_sample_for_each_class = get_num_of_sample_for_each_class(model, train_loader)
    num_of_val_sample_for_each_class = get_num_of_sample_for_each_class(model, val_loader)
    print(num_of_train_sample_for_each_class)
    print(num_of_val_sample_for_each_class)
    print(num_of_train_sample_for_each_class+num_of_val_sample_for_each_class)

    train_accuracy, train_precision, train_recall, train_F1 = evalute(model, train_loader)
    val_accuracy, val_precision, val_recall, val_F1 = evalute(model, val_loader)

    print("Training Accuracy   : ", str('%.4f' % (train_accuracy * 100)), '%')
    print("Validation Accuracy : ", str('%.4f' % (val_accuracy * 100)), '%')
    print("Training Precision  : ", str('%.4f' % (train_precision[0] * 100)), '%',' ',\
          str('%.4f' % (train_precision[1] * 100)), '%',' ',str('%.4f' % (train_precision[2] * 100)), '%',' ',\
          str('%.4f' % (train_precision[3] * 100)), '%',' ',str('%.4f' % (train_precision[4] * 100)), '%',' ',\
          str('%.4f' % (train_precision[5] * 100)), '%',' ',str('%.4f' % (train_precision[6] * 100)), '%',' ',\
          "Average Precision: ", str('%.4f' % (torch.mean(train_precision) * 100)), '%')
    print("Validation Precision: ", str('%.4f' % (val_precision[0] * 100)), '%',' ', \
          str('%.4f' % (val_precision[1] * 100)), '%',' ',str('%.4f' % (val_precision[2] * 100)), '%',' ',\
          str('%.4f' % (val_precision[3] * 100)), '%',' ',str('%.4f' % (val_precision[4] * 100)), '%',' ',\
          str('%.4f' % (val_precision[5] * 100)), '%',' ',str('%.4f' % (val_precision[6] * 100)), '%',' ',\
          "Average Precision: ", str('%.4f' % (torch.mean(val_precision) * 100)), '%')
    print("Training Recall     : ", str('%.4f' % (train_recall[0] * 100)), '%',' ', \
          str('%.4f' % (train_recall[1] * 100)), '%', ' ',str('%.4f' % (train_recall[2] * 100)), '%',' ',\
          str('%.4f' % (train_recall[3] * 100)), '%', ' ',str('%.4f' % (train_recall[4] * 100)), '%',' ',\
          str('%.4f' % (train_recall[5] * 100)), '%', ' ',str('%.4f' % (train_recall[6] * 100)), '%',' ',\
          "Average Recall: ", str('%.4f' % (torch.mean(train_recall) * 100)), '%')
    print("Validation Recall   : ", str('%.4f' % (val_recall[0] * 100)), '%',' ', \
          str('%.4f' % (val_recall[1] * 100)), '%', ' ',str('%.4f' % (val_recall[2] * 100)), '%',' ',\
          str('%.4f' % (val_recall[3] * 100)), '%', ' ',str('%.4f' % (val_recall[4] * 100)), '%',' ',\
          str('%.4f' % (val_recall[5] * 100)), '%', ' ',str('%.4f' % (val_recall[6] * 100)), '%',' ',\
          "Average Recall: ", str('%.4f' % (torch.mean(val_recall) * 100)), '%')
    print("Training F1-Score   : ", str('%.4f' % float(train_F1[0])), str('%.4f' % float(train_F1[1])),\
          str('%.4f' % float(train_F1[2])), str('%.4f' % float(train_F1[3])), str('%.4f' % float(train_F1[4])),\
          str('%.4f' % float(train_F1[5])), str('%.4f' % float(train_F1[6])),\
          "Average F1-Score: ", str('%.4f' % (torch.mean(train_F1))))
    print("Validation F1-Score : ", str('%.4f' % float(val_F1[0])), str('%.4f' % float(val_F1[1])),\
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

    Path_Name = ['YXSB_TEST_17th_1','YXSB_TEST_17th_2','YXSB_TEST_17th_3','YXSB_TEST_17th_4','YXSB_TEST_17th_5',\
                 'YXSB_TEST_17th_6','YXSB_TEST_17th_7','YXSB_TEST_17th_8','YXSB_TEST_17th_9','YXSB_TEST_17th_10',\
                 'YXSB_TEST_30th_1','YXSB_TEST_30th_2','YXSB_TEST_30th_3','YXSB_TEST_30th_4','YXSB_TEST_30th_5',\
                 'YXSB_TEST_30th_6','YXSB_TEST_30th_7','YXSB_TEST_30th_8','YXSB_TEST_30th_9','YXSB_TEST_30th_10']

    for path_name in Path_Name:
        test_db = YXSB_TEST_NEW(path_name, 224)
        test_loader = DataLoader(test_db, batch_size=int(path_name[15:]), num_workers=2)
        result, result_merge = test_evalute(model, test_loader, path_name[9:], int(path_name[15:]))
        print('Result'+path_name[9:]+': ', result)
        print('Result_Merge'+path_name[9:]+': ', result_merge)

if __name__ == '__main__':
    main()
