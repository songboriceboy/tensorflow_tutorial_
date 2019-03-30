
## <center>Tensorflow 基本开发步骤</center>

### 1 准备数据


```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
```


```python
x = np.linspace(-1,1,100)
y = 2*x+np.random.randn(100)*0.3
```


```python
plt.plot(x,y)
```




    [<matplotlib.lines.Line2D at 0x7f19750c6780>]




![png](1-Tensorflow%E5%9F%BA%E6%9C%AC%E5%BC%80%E5%8F%91%E6%AD%A5%E9%AA%A4_files/1-Tensorflow%E5%9F%BA%E6%9C%AC%E5%BC%80%E5%8F%91%E6%AD%A5%E9%AA%A4_4_1.png)



```python
x_test = np.linspace(-1,1,10)
y_test = 2*x_test
```


```python
plt.plot(x_test,y_test)
```




    [<matplotlib.lines.Line2D at 0x7f197503c208>]




![png](1-Tensorflow%E5%9F%BA%E6%9C%AC%E5%BC%80%E5%8F%91%E6%AD%A5%E9%AA%A4_files/1-Tensorflow%E5%9F%BA%E6%9C%AC%E5%BC%80%E5%8F%91%E6%AD%A5%E9%AA%A4_6_1.png)


### 2 搭建模型


```python
X = tf.placeholder(dtype=tf.float32,shape=None)
Y = tf.placeholder(dtype=tf.float32,shape=None)
W = tf.Variable(tf.random_normal(shape=[1]),name='weight')
b = tf.Variable(tf.zeros(shape=[1]),name='bais')
z = tf.multiply(W,X)+b
```


```python
cost = tf.reduce_mean(tf.square(Y-z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
```


```python
train_epochs = 20
display_step = 2
init = tf.global_variables_initializer()
```

### 3 迭代模型


```python
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(train_epochs):
        for (x_s,y_s) in zip(x,y):
            sess.run(optimizer,feed_dict={X:x_s,Y:y_s})
        if epoch%display_step==0:
            loss = sess.run(cost,feed_dict={X:x_test,Y:y_test})
            print('epoch: ',epoch,' loss:',loss)
    print("x=0.2, z=",sess.run(z,feed_dict={X:0.2}))
```

    epoch:  0  loss: 0.9508975
    epoch:  2  loss: 0.071659505
    epoch:  4  loss: 0.0039047264
    epoch:  6  loss: 0.00013607423
    epoch:  8  loss: 9.526675e-05
    epoch:  10  loss: 0.0001471997
    epoch:  12  loss: 0.0001647953
    epoch:  14  loss: 0.00016962932
    epoch:  16  loss: 0.00017089822
    epoch:  18  loss: 0.00017122082
    x=0.2, z= [0.38921914]
    

### 4 定义输入节点的方法

#### (1) 占位符
```python
X = tf.placeholder(dtype = tf.float32)
```

#### (2) 字典
```python
input_dict = {'x': tf.placeholder(dtype=tf.float32),'y':tf.placeholder(dtype=tf.float32)}
```

### 5 定义学习参数

#### (1) 直接定义
```python
W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.zeros([1]),name='bias')
```

#### (2) 字典定义
```python
para_dict = {'W':tf.Variable(tf.random_normal([1])),'b':tf.Variable(tf.zeros([1]))}
z = tf.multiply(x,para_dict['W'])+ para_dict['b']
```

### 6 初始化所有变量

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```
