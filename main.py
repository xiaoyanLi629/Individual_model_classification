import torch
import torch.utils.data as Data
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from load_X_Y_data import load_X_Y_data
from cnn_test_images import cnn_test_images
import os
import torchvision
from model import CNN0
from model import CNN1
from model import CNN2
from model import CNN3
from model import CNN4
from model import CNN5
from model import CNN6
from model import CNN7
from model import CNN8
from model import CNN9


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 5

# Mnist digits dataset
DOWNLOAD_MNIST = False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_dataset = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
train_x_image = train_dataset.train_data
train_x_image = train_x_image.reshape(train_x_image.shape[0], 1, train_x_image.shape[1], train_x_image.shape[2])
train_y = train_dataset.train_labels

test_dataset = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
test_x_image = test_dataset.train_data
test_x_image = test_x_image.reshape(test_x_image.shape[0], 1, test_x_image.shape[1], test_x_image.shape[2])
test_y = test_dataset.train_labels

train_num = train_x_image.shape[0]
test_num = test_x_image.shape[0]

# model_name = input('Please input creating model name(hit enter, a random name(1000) will be generated):')

model_name = 'test_model'
print('Constructing model:', model_name)

# setting device on GPU if available, else CPU

EPOCH = 200  # train the training data n times, to save time, we just train 1 epoch
LR = 0.0001  # learning rate
stable_factor = 0  # control model stability

print('Epoch:', EPOCH)
print('Learning rate:', LR)
print('batch size:', BATCH_SIZE)

cnn0 = CNN0()
cnn1 = CNN1()
cnn2 = CNN2()
cnn3 = CNN3()
cnn4 = CNN4()
cnn5 = CNN5()
cnn6 = CNN6()
cnn7 = CNN7()
cnn8 = CNN8()
cnn9 = CNN9()

cnn0.to(device)
cnn1.to(device)
cnn2.to(device)
cnn3.to(device)
cnn4.to(device)
cnn5.to(device)
cnn6.to(device)
cnn7.to(device)
cnn8.to(device)
cnn9.to(device)

optimizer_cnn0 = torch.optim.Adam(cnn0.parameters(), lr=LR)
optimizer_cnn1 = torch.optim.Adam(cnn1.parameters(), lr=LR)
optimizer_cnn2 = torch.optim.Adam(cnn2.parameters(), lr=LR)
optimizer_cnn3 = torch.optim.Adam(cnn3.parameters(), lr=LR)
optimizer_cnn4 = torch.optim.Adam(cnn4.parameters(), lr=LR)
optimizer_cnn5 = torch.optim.Adam(cnn5.parameters(), lr=LR)
optimizer_cnn6 = torch.optim.Adam(cnn6.parameters(), lr=LR)
optimizer_cnn7 = torch.optim.Adam(cnn7.parameters(), lr=LR)
optimizer_cnn8 = torch.optim.Adam(cnn8.parameters(), lr=LR)
optimizer_cnn9 = torch.optim.Adam(cnn9.parameters(), lr=LR)

loss_func_cnn = nn.CrossEntropyLoss()  # the target label is not one-hotted

epoch_list = []
plt.ion()

# misclassified_images = np.zeros((test_num, EPOCH))

train_cnn_accuracy_list_0 = []
train_cnn_accuracy_list_1 = []
train_cnn_accuracy_list_2 = []
train_cnn_accuracy_list_3 = []
train_cnn_accuracy_list_4 = []
train_cnn_accuracy_list_5 = []
train_cnn_accuracy_list_6 = []
train_cnn_accuracy_list_7 = []
train_cnn_accuracy_list_8 = []
train_cnn_accuracy_list_9 = []

test_cnn_accuracy_list_0 = []
test_cnn_accuracy_list_1 = []
test_cnn_accuracy_list_2 = []
test_cnn_accuracy_list_3 = []
test_cnn_accuracy_list_4 = []
test_cnn_accuracy_list_5 = []
test_cnn_accuracy_list_6 = []
test_cnn_accuracy_list_7 = []
test_cnn_accuracy_list_8 = []
test_cnn_accuracy_list_9 = []


for epoch in range(EPOCH):
    epoch_list.append(epoch)
    # for step, (x_image, x_feature, y) in enumerate(train_loader):
    for i in range(int(train_num / BATCH_SIZE)):
        index = np.random.choice(range(train_num), BATCH_SIZE, replace=False)
        x_image = train_x_image[index, :, :, :]
        y = train_y[index]
        x_image = x_image.float()
        x_image = x_image.to(device)

        output_0 = cnn0(x_image)[0]
        output_1 = cnn1(x_image)[0]
        output_2 = cnn2(x_image)[0]
        output_3 = cnn3(x_image)[0]
        output_4 = cnn4(x_image)[0]
        output_5 = cnn5(x_image)[0]
        output_6 = cnn6(x_image)[0]
        output_7 = cnn7(x_image)[0]
        output_8 = cnn8(x_image)[0]
        output_9 = cnn9(x_image)[0]

        y = y.reshape(BATCH_SIZE)

        cnn_loss_0 = loss_func_cnn(output_0, y)
        cnn_loss_1 = loss_func_cnn(output_1, y)
        cnn_loss_2 = loss_func_cnn(output_2, y)
        cnn_loss_3 = loss_func_cnn(output_3, y)
        cnn_loss_4 = loss_func_cnn(output_4, y)
        cnn_loss_5 = loss_func_cnn(output_5, y)
        cnn_loss_6 = loss_func_cnn(output_6, y)
        cnn_loss_7 = loss_func_cnn(output_7, y)
        cnn_loss_8 = loss_func_cnn(output_8, y)
        cnn_loss_9 = loss_func_cnn(output_9, y)

        optimizer_cnn0.zero_grad()
        optimizer_cnn1.zero_grad()
        optimizer_cnn2.zero_grad()
        optimizer_cnn3.zero_grad()
        optimizer_cnn4.zero_grad()
        optimizer_cnn5.zero_grad()
        optimizer_cnn6.zero_grad()
        optimizer_cnn7.zero_grad()
        optimizer_cnn8.zero_grad()
        optimizer_cnn9.zero_grad()

        cnn_loss_0.backward()
        cnn_loss_1.backward()
        cnn_loss_2.backward()
        cnn_loss_3.backward()
        cnn_loss_4.backward()
        cnn_loss_5.backward()
        cnn_loss_6.backward()
        cnn_loss_7.backward()
        cnn_loss_8.backward()
        cnn_loss_9.backward()

        optimizer_cnn0.step()
        optimizer_cnn1.step()
        optimizer_cnn2.step()
        optimizer_cnn3.step()
        optimizer_cnn4.step()
        optimizer_cnn5.step()
        optimizer_cnn6.step()
        optimizer_cnn7.step()
        optimizer_cnn8.step()
        optimizer_cnn9.step()


    cnn0.eval()
    cnn1.eval()
    cnn2.eval()
    cnn3.eval()
    cnn4.eval()
    cnn5.eval()
    cnn6.eval()
    cnn7.eval()
    cnn8.eval()
    cnn9.eval()

    cnn_train_accuracy_0, cnn_train_prediction_0, cnn_train_tot_loss_0 = cnn_test_images(train_x_image, train_y, train_num, cnn0, loss_func_cnn, device)
    cnn_train_accuracy_1, cnn_train_prediction_1, cnn_train_tot_loss_1 = cnn_test_images(train_x_image, train_y, train_num, cnn1, loss_func_cnn, device)
    cnn_train_accuracy_2, cnn_train_prediction_2, cnn_train_tot_loss_2 = cnn_test_images(train_x_image, train_y, train_num, cnn2, loss_func_cnn, device)
    cnn_train_accuracy_3, cnn_train_prediction_3, cnn_train_tot_loss_3 = cnn_test_images(train_x_image, train_y, train_num, cnn3, loss_func_cnn, device)
    cnn_train_accuracy_4, cnn_train_prediction_4, cnn_train_tot_loss_4 = cnn_test_images(train_x_image, train_y, train_num, cnn4, loss_func_cnn, device)
    cnn_train_accuracy_5, cnn_train_prediction_5, cnn_train_tot_loss_5 = cnn_test_images(train_x_image, train_y, train_num, cnn5, loss_func_cnn, device)
    cnn_train_accuracy_6, cnn_train_prediction_6, cnn_train_tot_loss_6 = cnn_test_images(train_x_image, train_y, train_num, cnn6, loss_func_cnn, device)
    cnn_train_accuracy_7, cnn_train_prediction_7, cnn_train_tot_loss_7 = cnn_test_images(train_x_image, train_y, train_num, cnn7, loss_func_cnn, device)
    cnn_train_accuracy_8, cnn_train_prediction_8, cnn_train_tot_loss_8 = cnn_test_images(train_x_image, train_y, train_num, cnn8, loss_func_cnn, device)
    cnn_train_accuracy_9, cnn_train_prediction_9, cnn_train_tot_loss_9 = cnn_test_images(train_x_image, train_y, train_num, cnn9, loss_func_cnn, device)


    cnn_test_accuracy_0, cnn_test_prediction_0, cnn_test_tot_loss_0 = cnn_test_images(test_x_image, test_y, test_num, cnn0, loss_func_cnn, device)
    cnn_test_accuracy_1, cnn_test_prediction_1, cnn_test_tot_loss_1 = cnn_test_images(test_x_image, test_y, test_num, cnn1, loss_func_cnn, device)
    cnn_test_accuracy_2, cnn_test_prediction_2, cnn_test_tot_loss_2 = cnn_test_images(test_x_image, test_y, test_num, cnn2, loss_func_cnn, device)
    cnn_test_accuracy_3, cnn_test_prediction_3, cnn_test_tot_loss_3 = cnn_test_images(test_x_image, test_y, test_num, cnn3, loss_func_cnn, device)
    cnn_test_accuracy_4, cnn_test_prediction_4, cnn_test_tot_loss_4 = cnn_test_images(test_x_image, test_y, test_num, cnn4, loss_func_cnn, device)
    cnn_test_accuracy_5, cnn_test_prediction_5, cnn_test_tot_loss_5 = cnn_test_images(test_x_image, test_y, test_num, cnn5, loss_func_cnn, device)
    cnn_test_accuracy_6, cnn_test_prediction_6, cnn_test_tot_loss_6 = cnn_test_images(test_x_image, test_y, test_num, cnn6, loss_func_cnn, device)
    cnn_test_accuracy_7, cnn_test_prediction_7, cnn_test_tot_loss_7 = cnn_test_images(test_x_image, test_y, test_num, cnn7, loss_func_cnn, device)
    cnn_test_accuracy_8, cnn_test_prediction_8, cnn_test_tot_loss_8 = cnn_test_images(test_x_image, test_y, test_num, cnn8, loss_func_cnn, device)
    cnn_test_accuracy_9, cnn_test_prediction_9, cnn_test_tot_loss_9 = cnn_test_images(test_x_image, test_y, test_num, cnn9, loss_func_cnn, device)

    train_cnn_accuracy_list_0.append(cnn_train_accuracy_0)
    train_cnn_accuracy_list_1.append(cnn_train_accuracy_1)
    train_cnn_accuracy_list_2.append(cnn_train_accuracy_2)
    train_cnn_accuracy_list_3.append(cnn_train_accuracy_3)
    train_cnn_accuracy_list_4.append(cnn_train_accuracy_4)
    train_cnn_accuracy_list_5.append(cnn_train_accuracy_5)
    train_cnn_accuracy_list_6.append(cnn_train_accuracy_6)
    train_cnn_accuracy_list_7.append(cnn_train_accuracy_7)
    train_cnn_accuracy_list_8.append(cnn_train_accuracy_8)
    train_cnn_accuracy_list_9.append(cnn_train_accuracy_9)

    test_cnn_accuracy_list_0.append(cnn_test_accuracy_0)
    test_cnn_accuracy_list_1.append(cnn_test_accuracy_1)
    test_cnn_accuracy_list_2.append(cnn_test_accuracy_2)
    test_cnn_accuracy_list_3.append(cnn_test_accuracy_3)
    test_cnn_accuracy_list_4.append(cnn_test_accuracy_4)
    test_cnn_accuracy_list_5.append(cnn_test_accuracy_5)
    test_cnn_accuracy_list_6.append(cnn_test_accuracy_6)
    test_cnn_accuracy_list_7.append(cnn_test_accuracy_7)
    test_cnn_accuracy_list_8.append(cnn_test_accuracy_8)
    test_cnn_accuracy_list_9.append(cnn_test_accuracy_9)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_0, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_0, color='y', label='Train cnn model accuracy')
        plt.title('Model 0 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_1, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_1, color='y', label='Train cnn model accuracy')
        plt.title('Model 1 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_2, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_2, color='y', label='Train cnn model accuracy')
        plt.title('Model 2 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_3, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_4, color='y', label='Train cnn model accuracy')
        plt.title('Model 3 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_4, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_4, color='y', label='Train cnn model accuracy')
        plt.title('Model 4 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_5, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_5, color='y', label='Train cnn model accuracy')
        plt.title('Model 5 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_6, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_6, color='y', label='Train cnn model accuracy')
        plt.title('Model 6 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_7, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_7, color='y', label='Train cnn model accuracy')
        plt.title('Model 7 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_8, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_8, color='y', label='Train cnn model accuracy')
        plt.title('Model 8 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch >= 0:
        plt.plot(epoch_list, train_cnn_accuracy_list_9, color='b', label='Train cnn model accuracy')
        plt.plot(epoch_list, test_cnn_accuracy_list_9, color='y', label='Train cnn model accuracy')
        plt.title('Model 9 Test images accuracy vs iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
    plt.pause(0.01)

    if epoch%10 == 0:
        print('Iteration: ', epoch)
        print('Mode 0 Training image accuracy is', cnn_train_accuracy_0)
        print('Model 0 Testing image accuracy is', cnn_test_accuracy_0)

        print('Mode 1 Training image accuracy is', cnn_train_accuracy_1)
        print('Model 1 Testing image accuracy is', cnn_test_accuracy_1)

        print('Mode 2 Training image accuracy is', cnn_train_accuracy_2)
        print('Model 2 Testing image accuracy is', cnn_test_accuracy_2)

        print('Mode 3 Training image accuracy is', cnn_train_accuracy_3)
        print('Model 3 Testing image accuracy is', cnn_test_accuracy_4)

        print('Mode 4 Training image accuracy is', cnn_train_accuracy_4)
        print('Model 4 Testing image accuracy is', cnn_test_accuracy_4)

        print('Mode 5 Training image accuracy is', cnn_train_accuracy_5)
        print('Model 5 Testing image accuracy is', cnn_test_accuracy_5)

        print('Mode 6 Training image accuracy is', cnn_train_accuracy_6)
        print('Model 6 Testing image accuracy is', cnn_test_accuracy_6)

        print('Mode 7 Training image accuracy is', cnn_train_accuracy_7)
        print('Model 7 Testing image accuracy is', cnn_test_accuracy_7)

        print('Mode 8 Training image accuracy is', cnn_train_accuracy_8)
        print('Model 8 Testing image accuracy is', cnn_test_accuracy_8)

        print('Mode 9 Training image accuracy is', cnn_train_accuracy_9)
        print('Model 9 Testing image accuracy is', cnn_test_accuracy_9)

        print('#############################################################')


plt.close()

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_0, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_0, color='y', label='Test cnn model accuracy')
plt.title('Model 0 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_1, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_1, color='y', label='Test cnn model accuracy')
plt.title('Model 1 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_2, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_2, color='y', label='Test cnn model accuracy')
plt.title('Model 2 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_3, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_3, color='y', label='Test cnn model accuracy')
plt.title('Model 3 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_4, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_4, color='y', label='Test cnn model accuracy')
plt.title('Model 4 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_5, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_5, color='y', label='Test cnn model accuracy')
plt.title('Model 5 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_6, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_6, color='y', label='Test cnn model accuracy')
plt.title('Model 6 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_7, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_7, color='y', label='Test cnn model accuracy')
plt.title('Model 7 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_8, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_8, color='y', label='Test cnn model accuracy')
plt.title('Model 8 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(np.arange(EPOCH), train_cnn_accuracy_list_9, color='b', label='Train cnn model accuracy')
plt.plot(np.arange(EPOCH), test_cnn_accuracy_list_9, color='y', label='Test cnn model accuracy')
plt.title('Model 9 Test images accuracy vs iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')


plt.legend()
plt.show()

