import  torch
import  os, glob
import  csv
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms
from    PIL import Image
import  matplotlib.pyplot as plt

class YXSB_TEST_NEW(Dataset):

    def __init__(self, root, resize):
        super(YXSB_TEST_NEW, self).__init__()

        self.root = root
        self.resize = resize
        name2label = {}
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            name2label[name] = int(name[12:])-1

        VALUE = sorted(name2label.values())
        self.name2label = {}
        for value in VALUE:
            for item in name2label.items():
                if item[1] == value:
                    self.name2label.update({item})

        self.images, self.labels = self.load_csv('images.csv')

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    Temp1 = img.split(os.sep)[-1]
                    Temp2 = Temp1.find('_')
                    N1 = int(Temp1[Temp2 + 1:-4])
                    Temp3 = img.split(os.sep)[-3]
                    N2 = int(Temp3[15:])
                    N3 = int(Temp1[:Temp2])
                    label = (N1 - 1) * N2 + N3 - 1
                    if int((img.split(os.sep)[-3])[15:]) == 10:
                        if label % 10 == 9:
                            temp = [img, label]
                        else:
                            writer.writerow([img, label])

                        if label % 10 == 8:
                            writer.writerow(temp)
                    else:
                        writer.writerow([img, label])

        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),
            transforms.Resize((int(self.resize), int(self.resize))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        label = torch.tensor(label)
        return img, label

def Generating_CSV(YXSB_TEST):
    import  visdom

    viz = visdom.Visdom()
    db = YXSB_TEST_NEW(YXSB_TEST, 224)

    x,y = next(iter(db))
    print('sample:', x.shape, y.shape, y)
    loader = DataLoader(db, batch_size=32, shuffle=False, num_workers=8)

    for x,y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
        plt.show()

def main():
    Path_Test = ['YXSB_TEST_17th_1','YXSB_TEST_17th_2','YXSB_TEST_17th_3','YXSB_TEST_17th_4','YXSB_TEST_17th_5',\
                 'YXSB_TEST_17th_6','YXSB_TEST_17th_7','YXSB_TEST_17th_8','YXSB_TEST_17th_9','YXSB_TEST_17th_10',\
                 'YXSB_TEST_30th_1','YXSB_TEST_30th_2','YXSB_TEST_30th_3','YXSB_TEST_30th_4','YXSB_TEST_30th_5',\
                 'YXSB_TEST_30th_6','YXSB_TEST_30th_7','YXSB_TEST_30th_8','YXSB_TEST_30th_9','YXSB_TEST_30th_10']

    for i in range(len(Path_Test)):
        Generating_CSV(Path_Test[i])

if __name__ == '__main__':
    main()
