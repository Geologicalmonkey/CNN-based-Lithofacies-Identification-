import torch
import keras
import seaborn as sns
import matplotlib.pyplot as plt
import visdom
import csv
import os
import numpy as np
from   scipy.io import loadmat
from   keras.models import *
from   keras.layers import *
from   keras.optimizers import *
from   sklearn import metrics
from   sklearn import preprocessing
import random

Batch_size = 64
rate=[0.8, 0.2]
Max_point = 228
Chanel = 5
Num_Classes = 7
normal = True
epochs = 50
train_rootdir = '.\Data\Train\Train_Data.mat'
test_rootdir  = '.\Data'

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

def test_evalute(model, Test_x, part_filename, batch_size_test):
    result = []; result_merge = []
    section = 0
    temp = 0
    for x in Test_x:
        Prob = []
        if temp == section:
            result_of_batch_size_test = model.predict(x)
            class_of_batch_size_test = []
            for i in range(len(result_of_batch_size_test)):
                class_of_batch_size_test.append(np.argmax(result_of_batch_size_test[i]))
            for row in range(batch_size_test):
                Prob.append(result_of_batch_size_test[row][class_of_batch_size_test[row]])
            PROB = np.argmax(Prob)
            result.append([class_of_batch_size_test[PROB], section, section + PROB + 1, PROB + 2])
            print(section, section + PROB + 1, class_of_batch_size_test[PROB])
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

    with open(os.path.join('./ResNet10_Test/Result' + part_filename + '.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        for Type in result:
            writer.writerow([Type[0],Type[1],Type[2],Type[3]])

    with open(os.path.join('./ResNet10_Test/Result_Merge' + part_filename + '.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        for Type in result_merge:
            writer.writerow([Type[0],Type[1],Type[2],Type[3],Type[4]])

    return result, result_merge

if __name__ == '__main__':
    model = build_model()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1.0e-5), metrics=['accuracy'])

    model = load_model('./ResNet10_Best')
    print('loaded from ResNet10_Best')

    train_scores = model.evaluate(Train_x, Train_y, verbose=0)
    valid_scores = model.evaluate(Valid_x, Valid_y, verbose=0)
    train_precision, train_recall, train_F1 = evalute(model, Train_x, Train_y)
    val_precision, val_recall, val_F1 = evalute(model, Valid_x, Valid_y)

    print("The Best Training Accuracy   : ", str('%.4f' % (train_scores[1] * 100)), '%')
    print("The Best Validation Accuracy : ", str('%.4f' % (valid_scores[1] * 100)), '%')
    print("The Best Training Precision  : ", str('%.4f' % (train_precision[0] * 100)), '%',' ',\
          str('%.4f' % (train_precision[1] * 100)), '%',' ',str('%.4f' % (train_precision[2] * 100)), '%',' ',\
          str('%.4f' % (train_precision[3] * 100)), '%',' ',str('%.4f' % (train_precision[4] * 100)), '%',' ',\
          str('%.4f' % (train_precision[5] * 100)), '%',' ',str('%.4f' % (train_precision[6] * 100)), '%',' ',\
          "Average Precision: ", str('%.4f' % (torch.mean(train_precision) * 100)), '%')
    print("The Best Validation Precision: ", str('%.4f' % (val_precision[0] * 100)), '%',' ',\
          str('%.4f' % (val_precision[1] * 100)), '%',' ',str('%.4f' % (val_precision[2] * 100)), '%',' ',\
          str('%.4f' % (val_precision[3] * 100)), '%',' ',str('%.4f' % (val_precision[4] * 100)), '%',' ',\
          str('%.4f' % (val_precision[5] * 100)), '%',' ',str('%.4f' % (val_precision[6] * 100)), '%',' ',\
          "Average Precision: ", str('%.4f' % (torch.mean(val_precision) * 100)), '%')
    print("The Best Training Recall     : ", str('%.4f' % (train_recall[0] * 100)), '%',' ',\
          str('%.4f' % (train_recall[1] * 100)), '%', ' ',str('%.4f' % (train_recall[2] * 100)), '%',' ',\
          str('%.4f' % (train_recall[3] * 100)), '%', ' ',str('%.4f' % (train_recall[4] * 100)), '%',' ',\
          str('%.4f' % (train_recall[5] * 100)), '%', ' ',str('%.4f' % (train_recall[6] * 100)), '%',' ',\
          "Average Recall: ", str('%.4f' % (torch.mean(train_recall) * 100)), '%')
    print("The Best Validation Recall   : ", str('%.4f' % (val_recall[0] * 100)), '%',' ',\
          str('%.4f' % (val_recall[1] * 100)), '%', ' ',str('%.4f' % (val_recall[2] * 100)), '%',' ',\
          str('%.4f' % (val_recall[3] * 100)), '%', ' ',str('%.4f' % (val_recall[4] * 100)), '%',' ',\
          str('%.4f' % (val_recall[5] * 100)), '%', ' ',str('%.4f' % (val_recall[6] * 100)), '%',' ',\
          "Average Recall: ", str('%.4f' % (torch.mean(val_recall) * 100)), '%')
    print("The Best Training F1-Score   : ", str('%.4f' % float(train_F1[0])), str('%.4f' % float(train_F1[1])),\
          str('%.4f' % float(train_F1[2])), str('%.4f' % float(train_F1[3])), str('%.4f' % float(train_F1[4])),\
          str('%.4f' % float(train_F1[5])), str('%.4f' % float(train_F1[6])),\
          "Average F1-Score: ", str('%.4f' % (torch.mean(train_F1))))
    print("The Best Validation F1-Score : ", str('%.4f' % float(val_F1[0])), str('%.4f' % float(val_F1[1])),\
          str('%.4f' % float(val_F1[2])), str('%.4f' % float(val_F1[3])), str('%.4f' % float(val_F1[4])),\
          str('%.4f' % float(val_F1[5])), str('%.4f' % float(val_F1[6])),\
          "Average F1-Scores: ", str('%.4f' % (torch.mean(val_F1))))

    temp = model.predict(Valid_x)
    valid_result = []
    for i in range(len(temp)):
        valid_result.append(np.argmax(temp[i]))
    validations = np.argmax(Valid_y, axis=1)
    show_confusion_matrix(validations, valid_result)

    Path_Name = ['Test_17th_1','Test_17th_2','Test_17th_3','Test_17th_4','Test_17th_5',\
                 'Test_17th_6','Test_17th_7','Test_17th_8','Test_17th_9','Test_17th_10',\
                 'TEST_30th_1','TEST_30th_2','TEST_30th_3','TEST_30th_4','TEST_30th_5',\
                 'TEST_30th_6','TEST_30th_7','TEST_30th_8','TEST_30th_9','TEST_30th_10']

    for path_name in Path_Name:
        Test_x = load_test_file('.\Data\Test'+path_name[4:])
        if normal:
            Test_X = np.vstack(np.vstack(Test_x))
            Test_X = scalar.transform(Test_X)
            Test_x  = Test_X.reshape((-1,len(Test_x[0]),len(Test_x[0][0]),len(Test_x[0][0][0])))
        result, result_merge = test_evalute(model, Test_x, path_name[4:], int(path_name[10:]))
        print('result'+path_name[4:]+': ', result)
        print('result_merge'+path_name[4:]+': ', result_merge)
