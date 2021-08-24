import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SGD_mini_batch
def mini_batch(x, y, batch_size):
    for i in np.arange(0, x.shape[0], batch_size):
        # 每次只會執行一次，執行完程式就會暫停，下一次再執行會從上一次執行完的地方開始執行
        yield (x[i:i + batch_size], y[i:i + batch_size]) 

# computing training_loss
def cost(targets, predictions):
    N = predictions.shape[0]
    loss = -np.sum(np.sum(targets*np.log(predictions))) / N
    return loss

# computing error rate
def error_rate(prediction, label):
    accuracy_list = []
    for i, (v1, v2) in enumerate(prediction):
        if np.argmax(prediction[i]) == np.argmax(label[i]):
            accuracy_list += [1]
        else:
            accuracy_list += [0]
    accurate_rate = np.sum(accuracy_list) * 1.0 / label.shape[0] 
    return 1 - accurate_rate

def run():
    a = []
    b = []
    c = []
    d = []
    for i in range(epochs + 1):
        nn.fit(x_train, y_train, batch_size)
        test_pred = nn.forward(x_test)
        training_pred = nn.forward(x_train)
        training_loss = cost(y_train, training_pred)
        training_error = error_rate(training_pred, y_train)
        testing_error = error_rate(test_pred, y_test)
        if i != 0 and i % batch_size == 0:
            a.append(i)
            b.append(training_loss)
            c.append(training_error)
            d.append(testing_error)
            print("epochs:" + str(i), "loss:" + str(training_loss),"training error rate:" + str(training_error),"testing error rate:" + str(testing_error))
    return a, b, c, d

def plot(numbers, traing_loss, training_error, testing_error):
    #Training loss
    plt.figure()
    plt.plot(numbers, traing_loss, linewidth=1.0)
    plt.xlabel("epochs")
    plt.ylabel("Average cross entropy")
    plt.title("Training loss")
    plt.show()
    #Training Error Rate
    plt.figure()
    plt.plot(numbers, training_error, linewidth=1.0)
    plt.xlabel("epochs")
    plt.ylabel("error rate")
    plt.title("Training Error Rate")
    plt.show()
    #Test Error Rate
    plt.figure()
    plt.plot(numbers, testing_error, linewidth=1.0)
    plt.xlabel("epochs")
    plt.ylabel("error rate")
    plt.title("Testing Error Rate")
    plt.show()
    
def standardize(x):
    z_value = (x - np.mean(x)) / np.std(x)
    return z_value
    
class DNN():
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, random_seed):
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.output_size = output_size
        
        # 權重參數初始化
        np.random.seed(random_seed)
        self.w1 = np.random.randn(self.input_size, self.hidden_size_1)  /np.sqrt(self.input_size) 
        self.w2 = np.random.randn(self.hidden_size_1, self.hidden_size_2) /np.sqrt(self.hidden_size_1)
        self.w3 = np.random.randn(self.hidden_size_2, self.output_size) /np.sqrt(self.hidden_size_2)

    def softmax(self,x):
        softmax_x = np.exp(x)
        for i in range(x.shape[0]):
            softmax_x[i] = softmax_x[i] / np.sum(softmax_x[i])
        return softmax_x
    
    def ReLu(self,x):
        return x * (x > 0)

    def ReLu_d(self,x):
        return 1. * (x > 0)
     
    def forward(self, input):
        # z1 = Input Layer * weight
        self.z1 = np.dot(input, self.w1)
        # l1 = activation function(z1)
        self.l1 = self.ReLu(self.z1) 
        self.z2 = np.dot(self.l1, self.w2) 
        self.l2 = self.ReLu(self.z2)
        self.z3 = np.dot(self.l2, self.w3)
        output = self.softmax(self.z3)
        return output
    
    # 建立一個反向的 NN
    def backpropagation(self, input, label, output):
        #第三層
        # loss function(cross entropy) 對 y 的微分乘上 softmax 對 z3 的偏微分，得到 loss function 對 z3 的微分
        self.grad_output = label - output  
        # 把 z3 的 input 乘上 loss function 對 z3 的微分，得到 loss function 對 w3 的微分
        self.grad_w3 = np.dot(self.l2.T, self.grad_output) 
        
        #第二層
        # loss function 對 z3 的微分再乘上 w3，就得到前一層 l2 
        self.grad_l2 = self.grad_output.dot(self.w3.T)
        # 把 l2 乘上一個代表 activation function 的常數，得到 loss function 對 z2 的微分  
        self.l2_delta = self.grad_l2 * self.ReLu_d(self.z2)
        #把  z2 的 input 乘上 loss function 對 z2 的微分，得到 loss function 對 w2 的微分
        self.grad_w2 = np.dot(self.l1.T, self.l2_delta)
        
        #第一層
        #把 loss function 對 z2 的微分乘上 w2 得到前一層 l1
        self.grad_l1 = self.l2_delta.dot(self.w2.T)
        # 把 l1 乘上一個代表 activation function 的常數，得到 loss function 對 z1 的微分
        self.l1_delta = self.grad_l1 * self.ReLu_d(self.z1)
        # 把 z1 的 input 乘上 loss function 對 z1 的微分，得到 loss function 對 w1 的微分
        self.grad_w1 = np.dot(input.T, self.l1_delta)

        self.w3 += learning_rate * self.grad_w3
        self.w2 += learning_rate * self.grad_w2
        self.w1 += learning_rate * self.grad_w1
              
    def fit(self, x_train, y_train, batch_size):
        for (mini_batch_x, mini_batch_y) in mini_batch(x_train, y_train, batch_size):
            output = self.forward(mini_batch_x)
            self.backpropagation(mini_batch_x, mini_batch_y, output)

# data_preprocessing
data = pd.read_csv('D://titanic.csv')
label = data['Survived'].astype(int)
data = data.drop(['Survived'],axis=1)
x_train = data[:800]
x_test = data[800:]
# 因為 softmax 會產生 0 和 1 的機率
y_train = np.array(pd.get_dummies(label[:800]))
y_test = np.array(pd.get_dummies(label[800:]))

# Q1. build model
#1. number of hidden layers : 2
#2. number of hidden units : 6
#3. learning rate : 0.0001
#4. number of iterations :30 
#5. mini_batch size : 50
batch_size = 50
epochs = 1500
learning_rate =0.0001

nn = DNN(6,6,6,2, random_seed = 666687)
numbers, traing_loss, training_error, testing_error = run()
plot(numbers, traing_loss, training_error, testing_error)

# Q2. build model (6,3,3,2)
#1. number of hidden layers : 2
#2. number of hidden units : 3
#3. learning rate : 0.0001
#4. number of iterations :30 
#5. mini_batch size : 50
nn = DNN(6,3,3,2, random_seed = 12345678)
numbers, traing_loss, training_error, testing_error = run()
plot(numbers, traing_loss, training_error, testing_error)

# Q3. Standardized
#1. number of hidden layers : 2
#2. number of hidden units : 3
#3. learning rate : 0.0001
#4. number of iterations :30 
#5. mini_batch size : 50
for col in ['Fare','Age']:
    print("standardized : "+col)
    standardized_col = standardize(data[col])
    temp =  data.drop([col],axis=1)
    standardized_data = pd.concat([temp,standardized_col], axis=1)
    x_train = standardized_data[:800]
    x_test = standardized_data[800:]
    nn = DNN(6,3,3,2, random_seed = 35712121)
    numbers, traing_loss, training_error, testing_error = run()
    plot(numbers, traing_loss, training_error, testing_error)

# Q4. choose feature affects the prediction performance the most
#1. number of hidden layers : 2
#2. number of hidden units : 3
#3. learning rate : 0.0001
#4. number of iterations :30 
#5. mini_batch size : 50
batch_size = 50
epochs = 2000
learning_rate =0.0001
for col in data.columns:
    print("take off : "+col)
    temp =  data.drop([col],axis=1)
    x_train = temp[:800]
    x_test = temp[800:]
    nn = DNN(5,3,3,2, random_seed = 353665)
    numbers, traing_loss, training_error, testing_error = run()
    plot(numbers, traing_loss, training_error, testing_error)
    
# Q5. dummy variables
#1. number of hidden layers : 2
#2. number of hidden units : 3
#3. learning rate : 0.0001
#4. number of iterations :30 
#5. mini_batch size : 50
batch_size = 50
epochs = 1500
learning_rate =0.0001
Pclass = pd.get_dummies(data.Pclass)
temp = data.drop(['Pclass'],axis=1)
dummy_data = pd.concat([temp,Pclass], axis=1)
x_train = dummy_data[:800]
x_test = dummy_data[800:]
nn = DNN(8,3,3,2, random_seed = 366561)
numbers, traing_loss, training_error, testing_error = run()
plot(numbers, traing_loss, training_error, testing_error)

# Q6. create a sample
nn = DNN(6,6,6,2, random_seed = 666687)
numbers, traing_loss, training_error, testing_error = run()
sample = np.array([3,1,25,2,2,10]).reshape(1,-1)
pred = nn.forward(sample)
print(pred)
sample = np.array([1,0,5,1,2,30]).reshape(1,-1)
pred = nn.forward(sample)
print(pred)
