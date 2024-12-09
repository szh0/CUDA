"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖lab5的内容，如果你需要的话，你可以将lab5对应代码copy到对应位置
"""
# from task0_autodiff import *
# from task0_operators import *
import numpy as np
import torchvision
from torchvision import transforms

def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # 提取数据和标签
    X_tr = trainset.data.numpy()
    y_tr = trainset.targets.numpy()
    X_te = testset.data.numpy()
    y_te = testset.targets.numpy()
    
    # 将图片数据从(60000, 28, 28)转换为(60000, 784)
    X_tr = X_tr.reshape(X_tr.shape[0], -1)
    X_te = X_te.reshape(X_te.shape[0], -1)

    # 归一化
    X_tr = X_tr.astype(np.float32) / 255.0
    X_te = X_te.astype(np.float32) / 255.0

    print(X_tr.shape)
    print(y_tr.shape)    

    return X_tr, y_tr, X_te, y_te

def set_structure(n, hidden_dim, k):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return [W1, W2]

def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """

    W1, W2 = weights
    Z1 = X @ W1
    A1 = np.maximum(Z1, 0)  # ReLU激活函数
    logit = A1 @ W2
    return logit

def backward(X, weights, y):
    """
    计算softmax损失函数相对于权重的梯度
    Args:
        logit: 2D numpy array of shape (batch_size, num_classes), 
               containing the logit predictions for each class.
        y: 1D numpy array of shape (batch_size,) 
           containing the true label of each example.
        A1: 2D numpy array of shape (batch_size, hidden_dim), 
             the output of the first layer after ReLU activation.
        weights: List of 2D array of layers weights, of shape [(input_dim, hidden_dim), (hidden_dim, num_classes)]

    Returns:
        grads: List of gradients with respect to the weights of the network.
    """

    W1, W2 = weights
    Z1 = X @ W1
    A1 = np.maximum(Z1, 0)  # ReLU激活函数
    logit = A1 @ W2
    
    Z_max = np.max(logit, axis=1, keepdims=True)
    exp_Z = np.exp(logit - Z_max)
    exp_sum = np.sum(exp_Z, axis=1, keepdims=True)
    log_softmax = exp_Z / exp_sum

    # 生成 one-hot 标签
    batch_size = X.shape[0]
    num_classes = W2.shape[1]

    y_one_hot = np.zeros((batch_size, num_classes))
    y_one_hot[np.arange(batch_size), y] = 1

    dZ = (log_softmax - y_one_hot) / batch_size

    dW2 = A1.T @ dZ
    dA1 = dZ @ W2.T
    dZ1 = dA1.copy()
    dZ1[Z1 <= 0] = 0
    dW1 = X.T @ dZ1
    
    grads = [dW1, dW2]
    return grads

def softmax_loss(Z, y):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """

    num_examples = Z.shape[0]
    num_classes = Z.shape[1]
    # 计算log_softmax
    Z_max = np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z - Z_max)
    log_probs = (Z - Z_max) - np.log(np.sum(exp_Z, axis=1, keepdims=True))
    
    # 计算损失
    correct_log_probs = -log_probs[np.arange(num_examples), y]
    loss = np.mean(correct_log_probs)
    return loss

def opti_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    if using_adam:
        Adam_epoch(X, y, weights, lr = lr, batch=batch, beta1=beta1, beta2=beta2)
    else:
        SGD_epoch(X, y, weights, lr = lr, batch=batch)

def SGD_epoch(X, y, weights, lr = 0.1, batch=100):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ## 请于此填写你的代码
    num_examples = X.shape[0]

    for i in range(0, num_examples, batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]
        grad = backward(X_batch, weights, y_batch) # include forward and backward
        # logit = forward(X_batch, weights)
        # grad = backward(logit, y_batch)  # 需要实现反向传播函数
        weights[0] -= lr * grad[0] / batch
        weights[1] -= lr * grad[1] / batch

def Adam_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """ 
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    ## 请于此填写你的代码

    num_examples = X.shape[0]
    t = 1  # 初始化时间步
    m = [np.zeros_like(w) for w in weights]  # 初始化一阶矩向量
    v = [np.zeros_like(w) for w in weights]  # 初始化二阶矩向量

    for i in range(0, num_examples, batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]
        grads = backward(X_batch, weights, y_batch)  # 计算梯度

        # 更新一阶矩向量和二阶矩向量
        for j, grad in enumerate(grads):
            m[j] = beta1 * m[j] + (1 - beta1) * grad
            v[j] = beta2 * v[j] + (1 - beta2) * np.square(grad)

        # 计算偏差校正后的一阶和二阶矩估计
        m_hat = [m[j] / (1 - beta1 ** t) for j in range(len(m))]
        v_hat = [v[j] / (1 - beta2 ** t) for j in range(len(v))]

        # 更新参数
        for j, (m_hat_j, v_hat_j) in enumerate(zip(m_hat, v_hat)):
            weights[j] -= lr * m_hat_j / (np.sqrt(v_hat_j) + epsilon)

        t += 1  # 更新时间步


def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    weights = set_structure(n, hidden_dim, k)
    np.random.seed(0)
    
    
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        opti_epoch(X_tr, y_tr, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
        train_loss, train_err = loss_err(forward(X_tr, weights), y_tr)
        test_loss, test_err = loss_err(forward(X_te, weights), y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist() 
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)
    ## using SGD optimizer 
    # train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=200, epochs=20, lr = 0.2, batch=100, beta1=0.9, beta2=0.999, using_adam=False)
    ## using Adam optimizer
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.02, batch=100, beta1=0.9, beta2=0.999, using_adam=True)
    