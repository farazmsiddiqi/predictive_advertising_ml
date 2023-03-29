import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import normalize

# load dataset
data = pd.read_csv('KAG_conversion_data.csv')

# calculate correlation coefficients
corr_clicks = data[['Clicks', 'Total_Conversion']].corr(method='pearson').iloc[0,1]
corr_impressions = data[['Impressions', 'Total_Conversion']].corr(method='pearson').iloc[0,1]

# print the correlation coefficients
print('Correlation coefficient (Clicks, Total_Conversion):', corr_clicks)
print('Correlation coefficient (Impressions, Total_Conversion):', corr_impressions)

print(data['Clicks']) 

# plt.plot(data['Clicks'])
# plt.plot(data['Impressions'].values)
# plt.plot(data['Total_Conversion'].values)
# plt.show()


x_values = data['Impressions'].values
x_train = np.array(x_values, dtype=np.float32)
print(x_train.shape)
# x_train = x_train.reshape(-1, 1)
print(x_train.shape)
x_train = normalize([x_train])
x_train = x_train[0]
print(x_train)
x_train = np.array(x_train, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = data['Total_Conversion'].values
y_train = np.array(y_values, dtype=np.float32)
# y_train = y_train.reshape(-1, 1)
y_train = normalize([y_train])
y_train = y_train[0]
print(y_train)
y_train = np.array(y_train, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.05 
epochs = 1000

model = linearRegression(inputDim, outputDim)

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))

    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cumulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)

    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad(): # we don't need gradients in the testing phase
    predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)
plt.xlabel("Impressions (Normalized)")
plt.ylabel("Total Conversions (Normalized)")
plt.title("Impressions vs Total Conversions (Normalized)")
plt.legend(loc='best')
plt.show()