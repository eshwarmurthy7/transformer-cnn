import sys
import os
import random
import argparse
import cv2
import numpy as np
import time
from PIL import Image
from utils import create_dir
from torch.optim.lr_scheduler import StepLR

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from tqdm import tqdm

NUM_CLASSES = 11

# TODO change for IG
num_sub_classes = None
learning_rate = 0.01
BATCH_SIZE = 64
CUDA = torch.cuda.is_available()

# TODO change for IG(180)
EPOCHS = 20

# TODO change for IG(100)
SGD_LR_DECAY_STEP = 10

# TODO change for IG
LOSS_DISPLAY_STEP = 100

# TODO change for IG(20)
SAVE_STEP = 5
EXP_NAME = "wbc_model"
# CE, CE_LABEL_SMOOTH, CE_WEIGHT, FOCAL_LOSS

loss_fun = 'CE'
create_dir(EXP_NAME)

SHOULD_APPLY_FLIPPING_AUG = True
SHOULD_APPLY_COLOR_AUG = True
SHOULD_APPLY_ROTATION_AUG = True

SHOULD_TEST = True


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        

def crop_center(img,size):
    x_center = img.shape[0] // 2
    y_center = img.shape[1] // 2
    img_crop = img[x_center - size // 2: x_center + size // 2, y_center - size // 2: y_center + size // 2]
    return img_crop


def get_mean_and_std(x):
    size = 300
    img_crop = crop_center(x, size)
    x_mean, x_std = cv2.meanStdDev(img_crop)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))
    return x_mean, x_std


def guassian_noise(img, factor=0.5):
    channel = np.random.randint(0, 3)
    row, col, ch = img.shape
    mean = 0
    var = 0.1
    sigma = var ** factor
    # print(sigma, var)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_img = hsv_img / 255
    gauss = np.random.normal(mean, sigma, (row, col))
    hsv_img[:, :, channel] = hsv_img[:, :, channel] + gauss
    hsv_img[:, :, channel] = np.clip(hsv_img[:, :, channel], 0, 1)
    hsv_img = np.uint8(hsv_img * 255)
    out_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return out_img


def adjust_gamma(img):
    gamma = 0.5 + np.random.randint(20) * 0.1
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def adjust_gamma_hsv(img, gamma):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # mid = 0.5
    # mean = np.mean(hsv[:, :, 2])
    # gamma = math.log(mid * 255) / math.log(mean)
    hsv[:, :, 2] = np.power(hsv[:, :, 2], gamma).clip(0, 255).astype(np.uint8)
    out_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out_img


def hue_change(img, factor=5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 0] = img[:, :, 0] + factor
    out_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return out_img


def add_blur(img, kernel_size=3):
    blur_img = cv2.GaussianBlur(img, (0, 0), kernel_size)
    return blur_img


def auto_adjustments_with_convert_scale_abs(img, channel=1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    alow = img[:, :, channel].min()
    ahigh = img[:, :, channel].max()
    amax = 255
    amin = 0
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    img[:, :, channel] = cv2.convertScaleAbs(img[:, :, channel], alpha=alpha, beta=beta)
    out_img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return out_img


def sharpen_img(img, flag=0, factor=9):
    if flag == 0:
        blur_img = cv2.GaussianBlur(img, (3, 3), 0)
        sharp_img = cv2.addWeighted(img, 1.5, blur_img, -0.5, 0, blur_img)
    elif flag == 1:
        blur_img = cv2.GaussianBlur(img, (3, 3), 0)
        filter_ = np.array([[-1, -1, -1], [-1, factor, -1], [-1, -1, -1]])
        sharp_img = cv2.filter2D(blur_img, -1, filter_)
    elif flag == 2:
        threshold = 220
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        sharp_img = cv2.addWeighted(img, 1.0 + 1.25, blurred, -1.25, 0)  # im1 = im + 3.0*(im - im_blurred)
        low_contrast_mask = blurred > threshold
        np.copyto(sharp_img, img, where=low_contrast_mask)
    elif flag == 3:
        threshold = 220
        blurred = cv2.GaussianBlur(img, (0, 0), 5)
        sharp_img = cv2.addWeighted(img, 1.0 + 1.25, blurred, -1.25, 0)  # im1 = im + 3.0*(im - im_blurred)
        low_contrast_mask = blurred > threshold
        np.copyto(sharp_img, img, where=low_contrast_mask)
    elif flag == 4:
        threshold = 220
        blurred = cv2.GaussianBlur(img, (5, 5), 5)
        sharp_img = cv2.addWeighted(img, 1.0 + 1.25, blurred, -1.25, 0)  # im1 = im + 3.0*(im - im_blurred)
        low_contrast_mask = blurred > threshold
        np.copyto(sharp_img, img, where=low_contrast_mask)

    return sharp_img


def color_transfer(img, factor_=1.0):
    bayer = np.random.choice([True, False])
    if bayer:
        t_mean, t_std = [207.379, 134.1994, 123.2952], [39.2082, 5.7952, 5.8252]
    else:
        t_mean, t_std = [213.6969697, 129.95757576, 125.33515152], [31.465, 3.92772727, 3.87742424]
    t_std = [x * factor_ if t_std.index(x) > 0 else x for x in t_std]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv_img_mean, hsv_img_std = get_mean_and_std(hsv_img)
    height, width, channel = hsv_img.shape
    for k in range(0, channel):
        a = hsv_img[:, :, k]
        a = ((a - hsv_img_mean[k]) * t_std[k] / hsv_img_std[k]) + t_mean[k]
        a = np.round(a)
        a = np.clip(a, 0, 255)
        hsv_img[:, :, k] = a
    out_img = cv2.cvtColor(hsv_img, cv2.COLOR_LAB2BGR)
    return out_img


class WBCDataset(Dataset):

    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

        self.rotation_angles = {
        }
        self.patch_size = 160
        self.patch_size_2 = 128

        self.classes = []
        self.class_images = []

        self.images = []
        self.labels = []

        self.weights = []

        self.init()

    def init(self):
        self.classes = [data for data in os.listdir(self.root_dir) if not data.startswith('.')]
        # print(self.classes)
        self.classes = ['ig', 'giantplatelet', 'notawbc', 'atypical-blast', 'neutrophil', 'nrbc', 'eosinophil', 'lymphocyte', 'basophil', 'monocyte', 'plateletclump']
        print(self.classes)
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            class_images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if ".jpg" in f]
            self.class_images.append(class_images)
            self.images.extend(class_images)
            self.labels.extend([idx] * len(class_images))

        for i in range(len(self.classes)):
            # self.weights.append(len(self.images) * 1.0 / len(self.class_images[i]))
            self.weights.append(max([len(class_image) for class_image in self.class_images]) / len(self.class_images[i]))

        temp = list(zip(self.images, self.labels))
        random.shuffle(temp)
        self.images, self.labels = zip(*temp)

    def get_weights(self):
        return self.weights

    def get_class_names(self):
        return self.classes

    def __len__(self):
        # print(len(self.images))
        return len(self.images)

    def __getitem__(self, idx):
        if self.mode == "train":
            class_idx = random.randint(0, len(self.classes) - 1)
            img_idx = random.randint(0, len(self.class_images[class_idx]) - 1)
            img_path = self.class_images[class_idx][img_idx]
        else:
            img_path = self.images[idx]
            class_idx = self.labels[idx]

        file_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        orig_img = img.copy()
        h, w, _ = img.shape
        if h != 2 * self.patch_size or w != 2 * self.patch_size:
            img = cv2.resize(img, (2 * self.patch_size, 2 * self.patch_size))

        if self.mode == "train":
            if SHOULD_APPLY_FLIPPING_AUG:
                flip = random.randint(0, 3)
                if flip == 1:
                    img = cv2.flip(img, 0)
                if flip == 2:
                    img = cv2.flip(img, 1)
                if flip == 3:
                    img = cv2.flip(img, -1)

            if SHOULD_APPLY_COLOR_AUG:
                if class_idx not in [1, 10]:
                    apply_advance_aug_flag = random.randint(0, 1)
                    if apply_advance_aug_flag:
                        parameters_list = [1.2, 1.5, 2, 3, 3, None, 0.9, 1.02, 3, 4, 5, -5, -6, -7, 1.3, 1.5, 1.7, 0, 1, 2,
                                           3, 4, 2, (add_blur, adjust_gamma_hsv), (color_transfer, guassian_noise),
                                           (hue_change, guassian_noise), (adjust_gamma_hsv, guassian_noise),
                                           (adjust_gamma_hsv, sharpen_img)]
                        choose = random.randint(0, 27)
                        if 0 <= choose <= 2:
                            img = color_transfer(img, parameters_list[choose])
                        elif 3 <= choose <= 4:
                            img = add_blur(img, parameters_list[choose])
                        elif choose == 5:
                            img = adjust_gamma(img)
                        elif 6 <= choose <= 7:
                            img = adjust_gamma_hsv(img, parameters_list[choose])
                        elif 8 <= choose <= 13:
                            img = hue_change(img, parameters_list[choose])
                        elif 14 <= choose <= 16:
                            img = guassian_noise(img, parameters_list[choose])
                        elif 17 <= choose <= 21:
                            img = sharpen_img(img, parameters_list[choose])
                        elif choose == 22:
                            img = auto_adjustments_with_convert_scale_abs(img, parameters_list[choose])
                        elif 23 <= choose <= 27:
                            group = parameters_list[choose]
                            for func in group:
                                if func == add_blur:
                                    img = func(img, 3)
                                if func == adjust_gamma_hsv:
                                    factor = np.random.choice([0.9, 1.02])
                                    img = func(img, factor)
                                if func == color_transfer:
                                    factor = np.random.choice([1.2, 1.5, 2])
                                    img = func(img, factor)
                                if func == guassian_noise:
                                    factor = np.random.choice([1.3, 1.5, 1.7])
                                    img = func(img, factor)
                                if func == hue_change:
                                    factor = np.random.choice([3, 4, 5, -5, -6, -7])
                                    img = func(img, factor)
                                if func == sharpen_img:
                                    factor = np.random.choice([0, 1, 2, 3, 4])
                                    img = func(img, factor)
                        else:
                            print('wrong selection')

            if SHOULD_APPLY_ROTATION_AUG:
                rotation_idx = random.randint(0, 4)
                if rotation_idx < 4:
                    angle = 45 + 90 * rotation_idx
                    M = cv2.getRotationMatrix2D((self.patch_size, self.patch_size), angle, 1.0)
                    img = cv2.warpAffine(img, M, (2 * self.patch_size, 2 * self.patch_size), flags=cv2.INTER_LINEAR)

        half_patch_size = self.patch_size // 2
        half_patch_size_2 = self.patch_size_2 // 2
        img_1 = img[h // 2 - half_patch_size:h // 2 + half_patch_size,
              w // 2 - half_patch_size:w // 2 + half_patch_size, :]

        img_2 = img[h // 2 - half_patch_size_2:h // 2 + half_patch_size_2,
              w // 2 - half_patch_size_2:w // 2 + half_patch_size_2, :]

        # 0 - 255 ---> 0 - 1
        # img = img / 255

        img = img_2.astype(np.float32)

        if self.transform:
            img = self.transform(img)

        return img, class_idx, orig_img, file_name


def make_dataloaders(data_dir):
    kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = WBCDataset(os.path.join(data_dir, "train"), "train", transform=train_transforms)
    if SHOULD_TEST:
        test_dataset = WBCDataset(os.path.join(data_dir, "test"), "test", transform=test_transforms)
    else:
        test_dataset = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    if SHOULD_TEST:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, **kwargs)
    else:
        test_loader = None

    return train_loader, test_loader, train_dataset, test_dataset


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.mobilenet = models.resnet18(pretrained=True)
        self.mobilenet.classifier = nn.Identity()
        self.fc_layer = nn.Sequential(nn.Linear(1000, 512),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(512, NUM_CLASSES)
                                      )

    def forward(self, x):
        x = self.norm(x)
        x = self.mobilenet(x)
        x = x.reshape(-1, 1000)
        x = self.fc_layer(x)
        return x


def train(data, model, criterion, optimizer):
    images, labels, _, _ = data
    if CUDA:
        images, labels = images.cuda(), labels.cuda()
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data


def test(data_loader, model, criterion, vis=False):
    loss = 0
    correct_predictions = 0
    dataset_len = 0.
    if num_sub_classes:
        confusion_matrix = torch.zeros(num_sub_classes, num_sub_classes)
    else:
        confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    if vis:
        vis_dir = os.path.join(EXP_NAME, "vis")
        if num_sub_classes:
            dir_num = num_sub_classes
        else:
            dir_num = NUM_CLASSES
        for i in range(dir_num):
            for j in range(dir_num):
                create_dir(os.path.join(vis_dir, str(i) + "_" + str(j)))

    f = open('prediction.csv', 'w')

    for batch_idx, data in tqdm(enumerate(data_loader)):
        images, labels, orig_imges, file_names = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss += criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)
        correct_predictions += predictions.eq(labels).sum()
        dataset_len += labels.size()[0]

        predictions_prob = nn.Softmax(dim=1)(outputs).max(dim=1)[0]
        for j, label in enumerate(labels):
            f.write(file_names[j] + "," + str(labels[j].detach().cpu().numpy()) + "," +
                    str(predictions[j].detach().cpu().numpy()) + "," + str(predictions_prob[j].detach().cpu().numpy()) + "\n")


        temp_idx = 0
        for t, p, orig_img, file_name in zip(labels.view(-1), predictions.view(-1), orig_imges, file_names):
            confusion_matrix[t.long(), p.long()] += 1
            if vis:
                # img_name = str(batch_idx) + "_" + str(temp_idx) + ".jpg"
                img_path = os.path.join(vis_dir, str(t.cpu().long().item()) + "_" +
                                        str(p.cpu().long().item()), file_name)
                # print (img_path)
                cv2.imwrite(img_path, orig_img.numpy())
            temp_idx += 1

    f.close()
    accuracy = correct_predictions.float().cpu().item() / dataset_len
    loss = loss.cpu().item() / len(data_loader)

    return loss, accuracy, confusion_matrix


def main(data_path, model_path, model):
    train_loader, test_loader, train_dataset, _ = make_dataloaders(data_path)

    if model_path is not None:

        if not CUDA:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(model_path))
        model.fc_layer[3] = nn.Linear(512, num_sub_classes)
        for param in list(model.parameters())[0:-4]:
            param.requires_grad = False

    if CUDA:
        model = model.cuda()

    # criterion = nn.NLLLoss()
    if loss_fun == 'CE':
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # TODO
    # optimizer = optim.AdamW(model.parameters(), lr=0.00001)
    # TODO change for IG(100)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    best_accuracy = 0
    model.train()
    if num_sub_classes:
        model.apply(set_bn_eval)
    for i in range(1, EPOCHS + 1):
        print("**** Training ****")
        print('Learning rate', optimizer.param_groups[0]['lr'])
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):

            # TODO
            # if batch_idx == 300:
            #     model.eval()
            #     with torch.no_grad():
            #         test_loss, accuracy, _ = test(test_loader, model, criterion)
            #     print("Epoch: {}, Loss: {}, Accuracy: {}".format(i, test_loss, accuracy))
            #     print('Saving intermediate weight:', accuracy)
            #     torch.save(model.state_dict(), os.path.join(EXP_NAME, 'checkpoint_intermediate.pth'))
            #     exit(0)
            train_loss += train(data, model, criterion, optimizer)
            if (batch_idx + 1) % LOSS_DISPLAY_STEP == 0:
                print("Epoch: {}, Batch: {}/{}, Loss: {}".format(i, batch_idx + 1, len(train_loader),
                                                                 train_loss / LOSS_DISPLAY_STEP))
                sys.stdout.flush()
                train_loss = 0
        scheduler.step()
        if SHOULD_TEST:
            print("**** Testing ****")
            model.eval()
            with torch.no_grad():
                test_loss, accuracy, _ = test(test_loader, model, criterion)
                print("Epoch: {}, Loss: {}, Accuracy: {}".format(i, test_loss, accuracy))
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    print('Found new best Accuracy:', accuracy)
                    torch.save(model.state_dict(), os.path.join(EXP_NAME, 'checkpoint_best.pth'))
            sys.stdout.flush()

        # TODO
        if i % SAVE_STEP == 0:
            torch.save(model.state_dict(), os.path.join(EXP_NAME, 'checkpoint_' + str(i) + '.pth'))


def test_pretrained(model_path, data_path, model):
    train_loader, test_loader, train_dataset, _ = make_dataloaders(data_path)
    if CUDA:
        model = model.cuda()
    # criterion = nn.NLLLoss()
    if loss_fun == 'CE':
        criterion = nn.CrossEntropyLoss()

    state_dict = torch.load(model_path,map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()

    # Strip the sub network starts
    # state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # backbone_dict = torch.load('/Users/adhithya/Desktop/models/wbc_resnet_18.pth', map_location=torch.device('cpu'))
    # sub_class_dict = {}
    # for key in state_dict.keys():
    #     if key in ['fc_layer.0.weight', 'fc_layer.0.bias', 'fc_layer.3.weight', 'fc_layer.3.bias']:
    #         sub_class_dict[key] = state_dict[key]
    #         continue
    #     if not torch.all(state_dict[key].eq(backbone_dict[key])):
    #         print('Found mismatch in layer', key)
    #         break
    #     state_dict[key] = backbone_dict[key]
    # torch.save(sub_class_dict, './wbc_model/ig.pth')
    # exit(0)
    # Strip the sub network ends

    # verify stripped network starts

    '''state_dict = {}
    backbone_dict = torch.load('./wbc_resnet18_adithya/checkpoint_20.pth', map_location=torch.device('cpu'))
    sub_class_dict = torch.load('./ig.pth', map_location=torch.device('cpu'))
    for key in backbone_dict.keys():
        if key in ['fc_layer.0.weight', 'fc_layer.0.bias', 'fc_layer.3.weight', 'fc_layer.3.bias']:
            state_dict[key] = sub_class_dict[key]
        else:
            state_dict[key] = backbone_dict[key]'''

    # verify stripped network ends

    with torch.no_grad():
        test_loss, accuracy, confusion_matrix = test(test_loader, model, criterion, vis=True)
    print("Loss: {}, Accuracy: {}".format(test_loss, accuracy))
    np.set_printoptions(suppress=True)

    print("Confusion Matrix")
    confusion_matrix = confusion_matrix.numpy()
    print(confusion_matrix)

    precisions = confusion_matrix.diagonal() / confusion_matrix.sum(0)
    recalls = confusion_matrix.diagonal() / confusion_matrix.sum(1)
    f1 = 2 * precisions * recalls / (precisions + recalls)
    print("Precisions = {}".format(precisions))
    print("Recalls = {}".format(recalls))
    print("F1 scores = {}".format(f1))
    total = np.sum(confusion_matrix)
    accuracies = (total - confusion_matrix.sum(0) - confusion_matrix.sum(1) + 2 * confusion_matrix.diagonal()) / total
    print("Accuracies = {}".format(accuracies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="trainval",
                        help='Final fov s3 path data')
    # '/home/avilash/Users/adithya/Dataset/wbc/wbc_data'
    # '/home/avilash/Users/adithya/Dataset/ig/wbc_data'
    # wbc_clean_exp
    parser.add_argument('--data_dir', type=str, default='/Users/adhithya/Dataset/new_wbc/wbc_data_exp6_dset_6', required=False,
                        help='Output directory')
    # ./wbc_ig_clean_3/checkpoint_best.pth'
    # ./wbc_intermediate_train/checkpoint_intermediate.pth
    # model.apply(set_bn_eval)
    parser.add_argument('--model', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--sub_class_num', type=int, default='2',
                        help='no of sub classes')

    args = parser.parse_args()
    model = ConvNet()
    if args.mode == "trainval":
        main(args.data_dir, args.model, model)
    elif args.mode == "test":
        if args.model is None:
            print("Please provide model")
        else:
            if num_sub_classes:
                model.fc_layer[3] = nn.Linear(512, num_sub_classes)
            test_pretrained(args.model, args.data_dir, model)
    else:
        print("Mode should either be trainval or test")


# ['ig', 'giantplatelet', 'notawbc', 'atypical-blast', 'neutrophil', 'nrbc', 'eosinophil', 'lymphocyte', 'basophil', 'monocyte', 'plateletclump']