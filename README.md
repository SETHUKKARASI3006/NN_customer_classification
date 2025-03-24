# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.

### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.

### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function, optimizer, and metrics.

### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.

## PROGRAM

### Name: SETHUKKARASI C
### Register Number: 212223230201

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        #Include your code here
        self.fc1 = nn.Linear(input_size, 40)
        self.fc2 = nn.Linear(40, 15)
        self.fc3 = nn.Linear(15, 10)
        self.fc4 = nn.Linear(10, 4) # 4 Output Classes (A, B, C, D)



    def forward(self, x):
        #Include your code here
        x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x

```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size = X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    #Include your code here
    for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
      optimizer.zero_grad()
      outputs = model(x_batch)
      loss = criterion(outputs, y_batch)
      loss.backward()
      optimizer.step()

    model.history = {'loss': []}
    model.history['loss'].append(loss.item())
    if (epoch + 1) % 100 == 0:
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

![data](/data_info.png)
<br>

## OUTPUT



### Confusion Matrix

![confusion_matrix](/con_mat.png)
<br>

### Classification Report

![report](/classify.png)
<br>

### New Sample Data Prediction

![prediction](/pred.png)
<br>

## RESULT
Thus, a neural network classification model for the given dataset is developed successfully.
