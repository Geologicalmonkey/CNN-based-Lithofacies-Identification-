import torch
import keras
import seaborn as sns
import matplotlib.pyplot as plt
import visdom
import csv
import os
import numpy as np
import collections
import tensorflow as tf
from   scipy.io import loadmat
from   keras.models import *
from   keras.layers import *
from   keras.optimizers import *
from   sklearn import metrics
from   sklearn import preprocessing
from   time import *
import random

print(tf.test.is_gpu_available())

Batch_size = 64
rate=[0.8, 0.2]
Max_point = 228
Chanel = 5
Num_Classes = 7
normal = True
epochs = 50
train_rootdir = '.\Data\Train\Train_Data.mat'
device = torch.device('cuda')

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

def load_train_file(data_rootdir, ratio):
    file = loadmat(data_rootdir)
    file_keys = file.keys()
    for key in file_keys:
        if 'Train_Val_Feature' in key:
            Train_Val_Feature = file[key]
        if 'Train_Val_Label' in key:
            Train_Val_Label = convert2oneHot(file[key], Num_Classes)

    file = loadmat('./random_state.mat')

    state = (file['random_state'][0][0][0], file['random_state'][0][1][0], file['random_state'][0][2][0][0],
             file['random_state'][0][3][0][0], file['random_state'][0][4][0][0])
    np.random.set_state(state)
    np.random.shuffle(Train_Val_Feature)
    np.random.set_state(state)
    np.random.shuffle(Train_Val_Label)

    all_lenght = len(Train_Val_Feature)
    samp_train = int(all_lenght * ratio[0])
    Train_x = Train_Val_Feature[:samp_train]
    Train_y = Train_Val_Label[:samp_train]
    Valid_x = Train_Val_Feature[samp_train:all_lenght]
    Valid_y = Train_Val_Label[samp_train:all_lenght]
    return Train_x, Train_y, Valid_x, Valid_y

def load_file(filepath):

    file = loadmat(filepath)
    file_keys = file.keys()
    for key in file_keys:
        if 'Feature' in key:
            Feature = file[key]
    return Feature

def load_test_file(data_rootdir):

    filename_list = []; filepath_list = []; Test_x=[]
    for rootdir, dirnames, filenames in os.walk(data_rootdir):
        filenames.sort(key=lambda x: int(x[:-4]))
        for filename in filenames:
            filename_list.append(filename)
            filepath_list.append(os.path.join(rootdir, filename))

    for i in range(len(filepath_list)):
        data=load_file(filepath_list[i])
        Test_x.append(data)

    return np.array(Test_x)

def build_model():

    def Matrix_Add_Layer(tensor):
        def matrix_add(tensor):
            return tensor[0]+tensor[1]
        return Lambda(matrix_add)(tensor)

    def ResBlk(y, ch_in, ch_out, kernel_size, strides):
        y_1C = Conv1D(ch_out, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_uniform')(y)
        y_1B = BatchNormalization()(y_1C)
        y_1F = Activation('relu')(y_1B)
        y_2C = Conv1D(ch_out, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer='he_uniform')(y_1F)
        y_2B = BatchNormalization()(y_2C)
        if ch_in != ch_out:
            y_tmp = Conv1D(ch_out, kernel_size=1, strides=strides, padding='same', kernel_initializer='he_uniform')(y)
            y_tmp = BatchNormalization()(y_tmp)
            y_add = [y_tmp, y_2B]
        else:
            y_add = [y, y_2B]
        y_3 = Matrix_Add_Layer(y_add)

        y_4 = Activation('relu')(y_3)
        return y_4

    ip = Input(shape=(Max_point, Chanel))
    x = Permute((2, 1))(ip)

    net = keras.Sequential([LSTM(256,return_sequences=True), LSTM(256,return_sequences=True), LSTM(128)])
    x = net(x)
    x = Dropout(0.3)(x)
    Num_Node_of_ResNet = [64, 128, 256, 256, 128]
    y = Conv1D(Num_Node_of_ResNet[0], 8, strides=3, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y_1 = ResBlk(y, Num_Node_of_ResNet[0], Num_Node_of_ResNet[1], 5, 3)
    y_2 = ResBlk(y_1, Num_Node_of_ResNet[1], Num_Node_of_ResNet[2], 5, 3)
    y_3 = ResBlk(y_2, Num_Node_of_ResNet[2], Num_Node_of_ResNet[3], 3, 1)
    y_4 = ResBlk(y_3, Num_Node_of_ResNet[3], Num_Node_of_ResNet[4], 3, 2)

    y_4 = GlobalAveragePooling1D()(y_4)

    y = Dropout(0.3)(y_4)
    x = concatenate([x, y])
    print('GGGG:',x.shape)

    out = Dense(Num_Classes, activation='softmax')(x)
    print('HHHH:',out.shape)

    model = Model(ip, out)
    model.summary()
    return model

Train_x, Train_y, Valid_x, Valid_y = load_train_file(train_rootdir, rate)
Temp_Train_y = np.argmax(Train_y, axis=1)
Temp_Valid_y = np.argmax(Valid_y, axis=1)
print(collections.Counter(Temp_Train_y))
print(collections.Counter(Temp_Valid_y))
print(collections.Counter(np.append(Temp_Train_y,Temp_Valid_y)))

if normal:
    Train_X = np.vstack(Train_x)
    Valid_X = np.vstack(Valid_x)

    scalar = preprocessing.StandardScaler().fit(Train_X)

    Train_X = scalar.transform(Train_X)
    Valid_X = scalar.transform(Valid_X)

    Train_x = Train_X.reshape((-1,len(Train_x[0]),len(Train_x[0][0])))
    Valid_x = Valid_X.reshape((-1,len(Valid_x[0]),len(Valid_x[0][0])))

def evalute(model, X, Y):

    predict_eq_actual, actual_result, predict_result = torch.zeros(Num_Classes), torch.zeros(Num_Classes), torch.zeros(Num_Classes)
    predict_x = model.predict(X)
    pred = np.argmax(predict_x, axis=1)
    Y_1 = np.argmax(Y, axis=1)
    for i in range(Num_Classes):
        for j in range(len(pred)):
            if pred[j] == i:
                predict_result[i] += 1
            if Y_1[j] == i:
                actual_result[i] += 1
            if (pred[j] == i) and (Y_1[j] == i):
                predict_eq_actual[i] += 1

    precision = predict_eq_actual / predict_result
    recall = predict_eq_actual / actual_result
    F1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, F1

if __name__ == '__main__':
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

    model = build_model()
    lr_1, lr_2 = 5.0e-4, 1.0e-4
    best_epoch, best_val_accuracy, best_train_accuracy = 0, 0, 0
    begin_time = time()
    for epoch in range(1, epochs+1):
        if epoch <= 10:
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_1), metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_2), metrics=['accuracy'])
        model.fit(Train_x,Train_y,batch_size=Batch_size,epochs=1,validation_split=0.0,verbose=0)
        train_scores = model.evaluate(Train_x,Train_y,verbose=0)
        valid_scores = model.evaluate(Valid_x,Valid_y,verbose=0)

        train_precision, train_recall, train_F1 = evalute(model, Train_x, Train_y)
        val_precision, val_recall, val_F1 = evalute(model, Valid_x, Valid_y)

        viz.line([[train_scores[0], valid_scores[0]]], [epoch], win='loss', update='append')
        viz.line([[train_scores[1], valid_scores[1]]], [epoch], win='accuracy', update='append')

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

        Part1 = [train_scores[0], valid_scores[0], train_scores[1], valid_scores[1]]
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
        contain_nan = (True in np.isnan(Evaluation_Result))
        if contain_nan:
            break
        if valid_scores[1] > best_val_accuracy:
            best_epoch = epoch
            best_val_accuracy = valid_scores[1]
            best_train_accuracy = train_scores[1]
            model.save('./ResNet10_Best')

        if epoch % 1 == 0:
            print("The ", epoch, " Training Loss       : ", str('%.4f' % float(train_scores[0])))
            print("The ", epoch, " Validation Loss     : ", str('%.4f' % float(valid_scores[0])))
            print("The ", epoch, " Training Accuracy   : ", str('%.4f' % (train_scores[1] * 100)), "%")
            print("The ", epoch, " Validation Accuracy : ", str('%.4f' % (valid_scores[1] * 100)), "%")
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
            print("The ", epoch, " Training Recall      : ", str('%.4f' % (train_recall[0] * 100)), '%', ' ',\
                  str('%.4f' % (train_recall[1] * 100)), '%', ' ', str('%.4f' % (train_recall[2] * 100)), '%',' ',\
                  str('%.4f' % (train_recall[3] * 100)), '%', ' ', str('%.4f' % (train_recall[4] * 100)), '%',' ',\
                  str('%.4f' % (train_recall[5] * 100)), '%', ' ', str('%.4f' % (train_recall[6] * 100)), '%',' ',\
                  "Average Recall: ", str('%.4f' % (torch.mean(train_recall) * 100)), '%')
            print("The ", epoch, " Validation Recall    : ", str('%.4f' % (val_recall[0] * 100)), '%', ' ',\
                  str('%.4f' % (val_recall[1] * 100)), '%', ' ', str('%.4f' % (val_recall[2] * 100)), '%',' ',\
                  str('%.4f' % (val_recall[3] * 100)), '%', ' ', str('%.4f' % (val_recall[4] * 100)), '%',' ',\
                  str('%.4f' % (val_recall[5] * 100)), '%', ' ', str('%.4f' % (val_recall[6] * 100)), '%',' ',\
                  "Average Recall: ", str('%.4f' % (torch.mean(val_recall) * 100)), '%')

            print("The ", epoch, " Training F1-Score    : ", str('%.4f' % float(train_F1[0])),\
                  str('%.4f' % float(train_F1[1])), str('%.4f' % float(train_F1[2])), str('%.4f' % float(train_F1[3])),\
                  str('%.4f' % float(train_F1[4])), str('%.4f' % float(train_F1[5])), str('%.4f' % float(train_F1[6])),\
                  "Average F1-Score: ", str('%.4f' % torch.mean(train_F1)))
            print("The ", epoch, " Validation F1-Score: ", str('%.4f' % float(val_F1[0])),\
                  str('%.4f' % float(val_F1[1])), str('%.4f' % float(val_F1[2])), str('%.4f' % float(val_F1[3])),\
                  str('%.4f' % float(val_F1[4])), str('%.4f' % float(val_F1[5])), str('%.4f' % float(val_F1[6])),\
                  "Average F1-Score: ", str('%.4f' % torch.mean(val_F1)))
            print("")

    end_time = time()
    run_time = end_time - begin_time
    print('The Running Time(s): ', run_time)
    file_handle = open('ResNet10_LSTM_Runtime.txt', mode='w')
    file_handle.write(str(run_time) + ' Second')
    file_handle.close()

    with open(os.path.join('./', 'ResNet10_LSTM_Evaluation_Result_For_Each_Epoch.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        for Type in Evaluation_Result:
            writer.writerow(Type.tolist())

    with open(os.path.join('./', 'ResNet10_LSTM_Evaluation_Result_For_Each_Epoch('+str(best_epoch)+').csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(Evaluation_Result[best_epoch].tolist())

    print('best_epoch:', best_epoch)
    print('best_val_acc:', str('%.4f' % (best_val_accuracy * 100)), '%')
    print('best_train_acc:', str('%.4f' % (best_train_accuracy * 100)), '%')

    model = load_model('./ResNet10_Best')

    train_scores = model.evaluate(Train_x, Train_y, verbose=0)
    valid_scores = model.evaluate(Valid_x, Valid_y, verbose=0)
    train_precision, train_recall, train_F1 = evalute(model, Train_x, Train_y)
    val_precision, val_recall, val_F1 = evalute(model, Valid_x, Valid_y)

    print("The ", best_epoch, " Training Accuracy   : ", str('%.4f' % (train_scores[1] * 100)), '%')
    print("The ", best_epoch, " Validation Accuracy : ", str('%.4f' % (valid_scores[1] * 100)), '%')
    print("The ", best_epoch, " Training Precision  : ", str('%.4f' % (train_precision[0] * 100)), '%',' ',\
          str('%.4f' % (train_precision[1] * 100)), '%',' ',str('%.4f' % (train_precision[2] * 100)), '%',' ',\
          str('%.4f' % (train_precision[3] * 100)), '%',' ',str('%.4f' % (train_precision[4] * 100)), '%',' ',\
          str('%.4f' % (train_precision[5] * 100)), '%',' ',str('%.4f' % (train_precision[6] * 100)), '%',' ',\
          "Average Precision: ", str('%.4f' % (torch.mean(train_precision) * 100)), '%')
    print("The ", best_epoch, " Validation Precision: ", str('%.4f' % (val_precision[0] * 100)), '%',' ',\
          str('%.4f' % (val_precision[1] * 100)), '%',' ',str('%.4f' % (val_precision[2] * 100)), '%',' ',\
          str('%.4f' % (val_precision[3] * 100)), '%',' ',str('%.4f' % (val_precision[4] * 100)), '%',' ',\
          str('%.4f' % (val_precision[5] * 100)), '%',' ',str('%.4f' % (val_precision[6] * 100)), '%',' ',\
          "Average Precision: ", str('%.4f' % (torch.mean(val_precision) * 100)), '%')
    print("The ", best_epoch, " Training Recall     : ", str('%.4f' % (train_recall[0] * 100)), '%',' ',\
          str('%.4f' % (train_recall[1] * 100)), '%', ' ',str('%.4f' % (train_recall[2] * 100)), '%',' ',\
          str('%.4f' % (train_recall[3] * 100)), '%', ' ',str('%.4f' % (train_recall[4] * 100)), '%',' ',\
          str('%.4f' % (train_recall[5] * 100)), '%', ' ',str('%.4f' % (train_recall[6] * 100)), '%',' ',\
          "Average Recall: ", str('%.4f' % (torch.mean(train_recall) * 100)), '%')
    print("The ", best_epoch, " Validation Recall   : ", str('%.4f' % (val_recall[0] * 100)), '%',' ',\
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

    temp = model.predict(Valid_x)
    valid_result = []
    for i in range(len(temp)):
        valid_result.append(np.argmax(temp[i]))
    validations = np.argmax(Valid_y, axis=1)
    show_confusion_matrix(validations, valid_result)
