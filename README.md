# Kannada-MNIST-Classification-Using-Neural-Networks

**This project was implemented by Valeriya Kostyukova, Ramazan Zhylkaidarov and Tamerlan Khussainov for ROBT407 Machine Learning course.**

We developed three machine learning models
for a classification task on the Kannada-MNIST dataset.
Kannada is a language spoken predominantly by people of
Karnataka in southwestern India. It was used to create a new
dataset similar to the MNIST with Arabic numerals but with
Kannada digits instead

![image](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/5ba78f23-1e97-49b6-a77e-32a3af2df8c9)

Kannada digits keep distinctive characters which provide
an intriguing challenge for developing robust and accurate
classification models. Motivated by the desire to extend the
reach of machine learning applications to linguistic diversity,
we chose the recognition of Kannada digits as the task for
our final project. For model performance evaluation, we used
a Multiple-layer perceptron (MLP), a Convolutional Neural
Network (CNN), and a Recurrent Neural Network (RNN).
The code was subsequently submitted to the KAGGLE
competition leaderboard.

## Methodology 
### Dataset Description and Preprocessing
The structure of the dataset is similar to the MNIST with
gray-scale images of hand-drawn digits, from zero through
nine, in the Kannada script. Every image consists of 28
pixels in both height and width, resulting in a total of 784
pixels. Each pixel is assigned a single pixel-value representing
its brightness, ranging from 0 to 255, where lower values
indicate darker shades. Each image has its corresponding label
representing the digit drawn by a user. We developed a custom
function to load the dataset downloaded from the KAGGLE
website [1] on the PC.

```python
def loadData(dataset: str = "train"):
    if dataset == "train" or dataset == "validation":
        df = pd.read_csv(train_file)
    else:
        df = pd.read_csv(test_file)
    
    data = torch.from_numpy(df.iloc[:, 1:].values).to(dtype=torch.float32) / 255.0
    targets = torch.from_numpy(df.iloc[:, 0].values).to(dtype=torch.long)  # Get labels
    
    if dataset == "train":
        return data[:48000, :], targets[:48000]
    elif dataset == "validation":
        return data[48000:, :], targets[48000:]
    else:
        return data, targets
```

The choice of whether to load the training, validation, or
test dataset is determined by the input parameter, making this
function adaptive to various stages of model development. The
pixel values of the images are extracted from the DataFrame
and transformed into PyTorch tensors. The normalization
process is implemented to scale pixel values to a standardized
range of [0, 1], ensuring consistency and aiding in model
convergence during training.

To divide the dataset for training and validation, we return
the first 48,000 samples and the remaining samples, respectively. By allocating a portion of the training data to the
validation set, practitioners can monitor the model’s behavior
on unseen data and detect potential overfitting.

The CustomDataset class uses the loadData function to
initialize the corresponding data and labels for three datasets:
train, validation, and test. These datasets are divided into
batches so that during each iteration, the DataLoader will
provide the model with batches of 100 samples, allowing
for efficient and parallelized processing of the data. Also, we
reshuffled the data at every epoch to reduce model overfitting.

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data, self.targets = loadData(dataset)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]

loaders = {
    "train": data.DataLoader(CustomDataset(dataset="train"), batch_size=100, shuffle=True, num_workers=0),
    "validation": data.DataLoader(CustomDataset(dataset="validation"), batch_size=100, shuffle=True, num_workers=0),
    "test": data.DataLoader(CustomDataset(dataset="test"), batch_size=100, shuffle=False, num_workers=0),
}

```

### Model Architectures

#### Multi-Layer Perceptron
MLP is a type of artificial
neural network characterized by multiple layers of intercon-
nected nodes, or neurons. Each layer, except the input layer,
is composed of neurons that apply an activation function to
the weighted sum of their inputs. The weights and biases
associated with each connection are learned during the training
process.

$$
s_j^{(l)} = \sum_{i=1}^{d^{(l-1)}} (w_{ij}^{(l)} x_i^{(l-1)})
$$

$$
a^{(l)} = f^{(l)}(s_j^{(l)})
$$

We calculate the linear combination of the inputs and
their weights (i.e. the weighted sum) and apply an activation
function on that calculation. In case of Kannada-
MNIST, we used the ReLU activation function for the hidden
layers and the Softmax activation function in the output layer
to provide normalized class probabilities.
For the implementation of the MLP algorithm we used
the PyTorch library which automates the process of updating
weights and gradient descent during training.

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
```

We defined an MLP class with three fully connected layers
(fc1, fc2, and fc3). Our first linear layer takes 784 (the size
of the input) nodes to 128, the second takes 128 to 64, and
the third 64 to 10 (the number of classes we want to predict). 

![image](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/6a15e09c-0b40-4e2f-9e3f-e9d465a4ccf5)

These layers have weights that are learned during
the training process. The forward function defines the forward
pass of the neural network, where the input x is passed through
each layer, and ReLU activation functions are applied after the
first and second hidden layers.

#### Convolutional Neural Networks
The second model
trained for the Kannada-MNIST task was CNN. Convolution
involves taking a small matrix of numbers, often referred to as
a kernel or filter, and running it across our image to alter the
image according to the values within the filter. CNN is a type of artificial neural network designed for processing structured
grid data, such as images. CNNs use convolutional layers,
which apply filters to identify patterns in the input data, to
function. Non-linearity is introduced by activation functions
(such as the sigmoid and ReLU), and data dimensions are
decreased by pooling layers. Flattening gets the data ready
for fully connected layers, which predict the output using
learned features. The network modifies weights during training
in order to reduce prediction errors.
In the context of Kannada-MNIST, CNNs are used at
capturing local patterns in the images of hand-written digits.
The methodology closely resembles the one employed in
MLP, with the only difference being the use of convolution
instead of matrix multiplication.

$$
s_j^{(l)} = \sum_{i=1}^{d^{(l-1)}} (w_{ij}^{(l)} * x_i^{(l-1)})
$$

$$
a^{(l)} = f^{(l)}(s_j^{(l)})
$$

The following is the code for the CNN class.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.conv2_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(8000, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2_dropout(self.conv2(x)), kernel_size=3, stride=1))
        x = x.view(-1, 8000)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

```
The first convolutional layer has a kernel size of 3x3, 1
input channel (since the image is grayscale) and 10 output
channels. This layer reduces the resolution of the images to
26x26, but it also undergoes pooling so the resolution becomes
24x24. The output from the first layer is 10x24x24, which is
then processed by the activation function, which in our case
is ReLU.
Next, the second layer does the same routine but also
includes a dropout, which is when some random values in
the input are set to zero to prevent overfitting. So after the
second layer the dimensions of the data become 20x20x20
which equals to 8000 individual pixels, which is why the first
fully-connected layer has 8000 input nodes.
After the convolutions, the data is passed to the fully-
connected layers also called the dense layers. In our case we
have 3 dense layers, which are 8000 to 1000, 1000 to 100 and
100 to 10 nodes respectively. Also, after the first dense layer
we apply ReLU, and after the second dense layer we apply
dropout and ReLU. And then finally apply the last dense layer
and return the result.

![image](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/b3e899d3-e50a-4ca6-af62-6660c92781f1)

#### Recurrent Neural Networks
The third model considered
for the Kannada-MNIST task is the RNN. 

![image](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/b0c2a13b-008d-4116-a440-619184058a29)

Unlike MLPs and CNNs, RNNs are designed to handle sequential
data and capture dependencies over time. They use the same
recurrence formula for each time step. 

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t)
$$

where h\_t is the hidden state at time step t, x\_t is the input at time t, $W_{hh}$ is the weight matrix for the hidden state transition, and $W_{xh}$ is the weight matrix of the input for the hidden state transition.

The RNN generates an output and updates its hidden state at each time step by processing the input and its hidden state from the preceding step. Serving as a kind of memory, the hidden state stores data from earlier time steps.

The unnormalized log probabilities of each possible value of the discrete output is then calculated based on the current hidden state (Eq. 6).

$$
o_t = W_{hy} \cdot h_{t}
$$

And we then apply the softmax operation as a post-processing step to obtain a vector of normalized probabilities over the output (Eq. 7).

$$
y_t = softmax(o_t)
$$

By using the same set of weights and biases at every time step, parameter sharing enables the network to recognize and identify patterns throughout the whole sequence.

The RNN class implementation is provided below. This is a many-to-one architecture as it has 28 inputs and one output. Two nn.RNN layers are utilized to create higher-level abstractions and capture more non-linearities between the data, which inherently perform the recurrent operation. 

```python
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(28, 256, 2, batch_first=True)
        self.fc3 = nn.Linear(256 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28, 28)
        h0 = torch.zeros(2, x.size(0), 256)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc3(out)
        return out
```

Because the Kannada-MNIST dataset consists of 28x28 pixel images, we can treat them as 28 sequences (rows) of 28 input features (columns). Therefore, we need to reshape the dataset from [num\_data, 28*28] to [num\_data, n\_sequencies, n\_inputs]. The hidden state, where the information would be stored, was initialized with zeros as a starting point for the recurrent calculations. The subsequent reshaping and application of the fully connected layer (fc3), which corresponds to Eq. 9, allows the model to transform the RNN's output into the desired output space. 

### Experiments 
#### Training and Validation Procedures
When the classes for the three models are ready, we can
train them. The primary objective during training is to optimize
the models’ parameters (weights and biases) to minimize the
defined loss function. The training process involves iteratively
updating the weights based on the calculated gradients through
backpropagation.

For backward propagation using the cross-entropy loss,
the gradient of the loss with respect to the output layer
is computed. The error is then backpropagated through the
network, and gradients for weights and biases are calculated.
The weights are updated using gradient descent to minimize
the loss. In our calculations we used the chain rule.

$$
\quad w_{ij}^{(l)} \leftarrow w_{ij}^{(l)} - \eta \frac{\partial L}{\partial w_{ij}^{(l)}}
$$


$$
\frac{\partial L}{\partial w_{ij}^{(l)}} = \sum_{k=1}^{d^{l+1}} \frac{\partial L}{\partial s_{k}^{(l+1)}} \cdot \frac{\partial s_{k}^{(l+1)}}{\partial x_j^{(l)}} \cdot \frac{\partial x_j^{(l)}}{\partial s_j^{(l)}} \cdot \frac{\partial s_j^{(l)}}{\partial w_{ij}^{(l)}}
$$

where i is the input and j is the output to the $l^{th}$ layer.

In the code, we used one function *trainModel()* for all models. 

```python
criterion = nn.CrossEntropyLoss()

def trainModel(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(loaders['train']):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

For the MLP and CNN models, the weight update is handled
by the Adam optimizer, and the cross-entropy loss function,
described by Eq. 10, is used to quantify the difference between
predicted and actual labels. The training loop involves iterating
through batches of the training dataset, computing the forward
pass to obtain predictions, calculating the loss, backpropagating the gradients, and updating the weights accordingly.

$$
\quad L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c) \
$$

The RNN model employs the same optimization strategy with Adam and the cross-entropy loss. However, due to the sequential nature of RNNs, the hidden states are updated recursively across time steps. The backpropagation through time (BPTT) algorithm is utilized to calculate gradients over the entire sequence, enabling the model to capture temporal dependencies.

During validation, the models are evaluated on a separate dataset to assess their generalization performance.

```python
def validateModel(model):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["validation"]:
            data, target = data.to(device), target.to(device)
            output = model(data)

            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = val_loss / len(loaders["validation"].dataset)
    accuracy = correct / len(loaders["validation"].dataset)
    return val_loss, accuracy
```

It is crucial to translate the model into evaluation mode to ensure that the model makes deterministic predictions during evaluation. We also disable computations of the gradients as we wanted to ensure that model parameters remain fixed, and we only focus on obtaining predictions. Subsequently, it iterates through the validation dataset, computing predictions and evaluating their accuracy and loss. 

In addition, to maximize prediction accuracy, the outputs generated by the MLP, CNN, and RNN models are integrated into a comprehensive group. This strategy involves summing the individual outputs element-wise and, thus, learning capabilities embedded in each model. The final class predictions are determined by applying the argmax operation to the aggregated outputs.

```python
testOutputs = testModel(mlp)
testOutputs = testOutputs + testModel(cnn)
testOutputs = testOutputs + testModel(rnn)
testOutputs = np.argmax(testOutputs, axis=1)
```

### Evaluations
#### Results and Evaluation Metrics
All three models showed a high accuracy in predicting the labels on a validation set. The high results indicate that the models have been implemented correctly which is also confirmed by the plots of decreasing loss and increasing accuracy.

![MLP_accuracy](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/26ebc8e7-3817-40c8-ae49-ce0ae70bc37c)
![CNN_accuracy](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/44bedb6f-161c-4ac6-87d5-22d198685404)
![RNN_accuracy](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/df4cccbd-be68-4f97-900b-7067c482dc40)

From the plots it could be seen that while the accuracy increases, the overall error in the training set very quickly approaches smaller values, which are close to zero. When the models have learned the weights, the error on the validation set stays relatively the same, showing a small loss. Similarly, the accuracy also stays high for all three models. 

The plots were constructed from the data given after 50 epochs to see the trends. However, from the plots it can be seen that the difference between the train and validation errors starts to increase after a certain number of epochs. This means that the generalization, especially for MLP and CNN, is getting worse and the problem of overfitting occurs. Thus, it seems to be reasonable to train the models on the optimal number of epochs which is 10, considering all three models' losses. This method of regularization and preventing overfitting is called *early stopping*.    

After training the models on 10 epochs, we got the following results:

```python
MLP:
Validation set: Average loss: 0.0012, Accuracy: 98.63%
CNN:
Validation set: Average loss: 0.0003, Accuracy: 99.31%
RNN:
Validation set: Average loss: 0.0011, Accuracy: 97.68%
```

An exceptional accuracy was shown the CNN model (99.31\%), followed by MLP (98.63\%) and RNN (97.68\%).

#### Visualizations
To see the learned representations of the models, we employed Principal Component Analysis (PCA) for visualization. The PCA visualization was performed on the embeddings (transformations of high-dimensional word representations into a lower-dimensional space) obtained from the validation set predictions of each model. The three-dimensional PCA plots were generated to show the distribution of data points in a reduced-dimensional space.

![photo1701425320](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/9e1b3295-4d19-4ee1-b73f-97dadf646a4b)
![photo1701425388](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/a3105971-c656-4c74-9a0c-0193b5e848c8)
![photo1701425455](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/a9ab7a4c-b8bd-4b61-b715-ada2b865f9ab)

Each cluster represents the examples of a certain class (digits from 0 to 9) predicted by the MLP, CNN, and RNN, respectively. The CNN exhibited clusters with increased spatial separation, meaning that it learned features contributing to better digit class recognition. In contrast, the clusters for MLP and RNN appeared closer, indicating potential similarities in their learned representations.

Additionally, we visualized 16 random digit images with their respective true and predicted labels.

![photo1701424628](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/16ea24fd-11a9-4223-86fb-a15de93e379e)

It can also be seen that most of the digits were classified correctly. The figure above is the classification result of MLP model. We got the similar results for the CNN and RNN models which can be visualized by running our code. 

### Discussion of Findings
By looking at the results, all three models have shown a good performance on the Kannada-MNIST digit classification task. However, the CNN is the favored one since CNNs are almost exclusively made for image and grid-like data classification.

The MLP, while also effective, operates on flattened representations of the images, treating them as one-dimensional vectors. This architecture lacks the inherent ability to capture spatial dependencies between pixels that are vital for image classification tasks. The RNN, designed for sequential data, might not fully exploit its strengths in this particular dataset, where spatial relationships are more critical. Because we had fairly "simple" gray-scaled images of digits, its performance was relatively high, but for more complex RGB images, this model is likely to show poor results. 

In terms of computational efficiency, CNN was the most expensive model in comparison with the MLP and RNN,  as the convolutional layers contribute significantly to the computational load. CNNs also use pooling layers to downsample the spatial dimensions of the input data, contributing to the computational cost. However, if running the code on a GPU, this problem becomes less significant as the parallel processing capabilities of GPUs excel in handling the computational demands associated with convolutional operations and large parameter sets. 

As discussed earlier, however, we decided to combine the results of the three models to get the most accurate predictions. This resulted in a private score of 96.74\% and public score of 96.90\% in the KAGGLE competition.

![KAGGLEscores](https://github.com/stleraburg/Kannada-MNIST-Classification-Using-Neural-Networks/assets/94596396/6c2140a7-6ec6-4003-8bb9-ae8544f35746)

### Conclusion
Our project on the Kannada-MNIST digit classification task explored the capabilities of three distinct neural network architectures: MLP, CNN, and RNN. Each model demonstrated high accuracy in predicting Kannada digits, with CNN emerging as the top performer. The CNN's ability to capture spatial dependencies in image data proved advantageous, resulting in exceptional accuracy. While MLP and RNN also performed well, their architectures posed limitations for image classification tasks. The combination of all three models further improved predictive accuracy. The project demonstrated the significance of selecting an appropriate model architecture for specific data types, with CNN standing out in the context of image classification. The successful implementation and evaluation of these models contribute to the broader goal of expanding machine learning applications to diverse linguistic datasets.

### References
<a id="1">[1]</a>
    Vinay Uday Prabhu, Walter Reade, Addison Howard. (2019). 
    Kannada MNIST. Kaggle. 
    [Kannada MNIST URL](https://kaggle.com/competitions/Kannada-MNIST)

