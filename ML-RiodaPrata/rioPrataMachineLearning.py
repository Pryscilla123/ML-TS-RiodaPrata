import torch
import numpy as np 
from sklearn import datasets
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt
import pandas as pd
import time
from torch.nn.functional import mse_loss

arq = open("amostras.data", "w")
arq.write("amostra, lr, momentum, ce1, ce2, tempo, bic, aic, after_train\n")

data_rioPrata = pd.read_csv("rioPrata.csv")

data_rioPrata = data_rioPrata.iloc[:42, :]

#print(data_rioPrata)

targets = data_rioPrata.columns
names = targets[16:47]
#print(names)

controle = data_rioPrata["Vazao01"]

controle_targets = names[1:]
#print(controle_targets)

for i in range(len(controle_targets)):
    controle = controle.append(data_rioPrata[controle_targets[i]], ignore_index=True)

#print(controle)

index_with_nan = controle.index[controle.isnull()]
controle.drop(index_with_nan, 0, inplace=True)

print(controle)

janelas = 30

all_data = np.zeros([controle.size - janelas, janelas + 1])
#print(all_data)

for i in range(len(all_data)):
    for j in range(janelas+1):
        all_data[i][j] = controle.iloc[i+j]

#print(all_data)

dif = all_data.max() - all_data.min()
all_data = (all_data - all_data.min())/dif

#print(all_data)

x = all_data[:, :-1]
y = all_data[:, -1]

x_training = x[:1150, :]
y_training = y[:1150]

training_input = torch.FloatTensor(x_training)
training_output = torch.FloatTensor(y_training)

x_test = x[1150: , :]
y_test = y[1150:]

test_input = torch.FloatTensor(x_test)
test_output = torch.FloatTensor(y_test)

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, hidden_sizeT):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_sizeT = hidden_sizeT
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_sizeT)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_sizeT, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hiddenT = self.fc2(relu)
        relu = self.relu(hiddenT)
        output = self.fc3(relu)
        output = self.sigmoid(output)
        return output

def AIC(y, y_pred, n, p):
    sse = mse_loss(y_pred.squeeze(), y, reduction='sum')
    Aic = (2 * p) - n * torch.log(sse/n)
    return Aic

def BIC(y, y_pred, n, p):
    sse = mse_loss(y_pred.squeeze(), y, reduction='sum')
    Bic = n * torch.log(sse/n) + p * math.log(n)
    return Bic

def plotcharts(errors):
    errors = np.array(errors)
    plt.figure(figsize=(12, 5))
    graf02 = plt.subplot(1, 2, 1) # nrows, ncols, index
    graf02.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Epochs')
    graf03 = plt.subplot(1, 2, 2)
    graf03.set_title('Tests')
    a = plt.plot(test_output.numpy(), 'b-', label='Real')
    plt.setp(a, markersize=10)
    a = plt.plot(y_pred.detach().numpy(), 'y--', label='Predicted')
    plt.setp(a, markersize=10)
    plt.legend(loc=0)
    plt.ylim(0.0, 0.6)
    plt.savefig("graficos/amostra{}.png".format(count))
    plt.close()

input_size = training_input.size()[1]

lr = [0.6, 0.7, 0.8, 0.9]
momentum = [0.2, 0.3, 0.4, 0.5]
hidden_size = [25, 35, 45]
hidden_sizeT = [20, 30, 35]

count = 0

for i in lr:
    for j in momentum:
        for k in hidden_size:
            for l in hidden_sizeT:
                count += 1
                model = Net(input_size, k, l)
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.SGD(model.parameters(), lr=i, momentum=j)

                model.eval()
                y_pred = model(training_input)
                before_train = criterion(y_pred.squeeze(), training_output)
                print('Amostra: {}. Test loss before training: {}'.format(count, before_train.item()))

                tempo_inicial = time.time()

                #treinamento da rede
                model.train()
                epochs = 50000
                errors = []

                for epoch in range(epochs):
                    optimizer.zero_grad()
                    y_pred = model(training_input)
                    loss = criterion(y_pred.squeeze(), training_output)
                    errors.append(loss.item())
                    #backpropagation
                    loss.backward()
                    optimizer.step()

                tempo_final = time.time() - tempo_inicial

                model.eval()
                y_pred = model(test_input)
                after_train = criterion(y_pred.squeeze(), test_output)

                aic = AIC(test_output, y_pred, len(training_input), sum(p.numel() for p in model.parameters() if p.requires_grad))
                bic = BIC(test_output, y_pred, len(training_input), sum(p.numel() for p in model.parameters() if p.requires_grad))

                #print(aic, bic)
                plotcharts(errors)

                arq.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(count, i, j, k, l, tempo_final, bic, aic, after_train.item()))


arq.close()
