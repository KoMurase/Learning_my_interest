
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import renom as rm
from renom.optimizer import Adam
from renom.cuda import set_cuda_active

# set True when using GPU
set_cuda_active(False)

#時系列データの準備
#時系列データから指定の長さの部分列を生成する関数を定義

# create [look_back] length array from [time-series] data.
# e.g.) ts:{1,2,3,4,5}, lb=3 => {1,2,3},{2,3,4},{3,4,5}
def create_dataset(ts, look_back=1):
    sub_seq, nxt = [], []
    for i in range(len(ts)-look_back):
        sub_seq.append(ts[i:i+look_back])
        nxt.append([ts[i+look_back]])
    return sub_seq, nxt

#今回は区間[−10,10]のSin波を50分割し、
#長さ５の部分列から次の観測を予測するようにデータを生成します。
x = np.linspace(-20, 20 , 50)
y = np.sin(x)
look_back = 5

sub_seq , nxt = create_dataset(y , look_back=look_back )

#split dataset into train and test set
def split_data(X , y ,train_ratio = 0.5):
    train_size = int(len(y) * train_ratio)
    X_train , y_train = X[:train_size], y[:train_size]
    X_test , y_test = X[train_size:], y[train_size:]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train , y_train , X_test , y_test

X_train , y_train , X_test , y_test = split_data(sub_seq,nxt)
train_size = X_train.shape[0]
test_size = X_test.shape[0]
print('train size : {} , test size : {}'.format(train_size,test_size))

#描画するための関数
def draw_pred_curve(e_num):
    pred_curve = []
    arr_now = X_test[0]
    for _ in range(test_size):
        for t in range(look_back):
            pred = model(np.array([arr_now[t]]))
        model.truncate()
        pred_curve.append(pred[0])
        arr_now = np.delete(arr_now, 0)
        arr_now = np.append(arr_now, pred)
    plt.plot(x[:train_size+look_back], y[:train_size+look_back], color='blue')
    plt.plot(x[train_size+look_back:], pred_curve, label='epoch:'+str(e_num)+'th')

#モデルの定義
model = rm.Sequential([
    rm.Lstm(2),
    rm.Dense(1)
])

#各パラメータの設定
batch_size = 5
max_epoch = 1000
period = 200
optimizer = Adam()

#Train Loop
i= 0
loss_prev = np.inf

#Learning curves
learning_curve = []
test_curve = []

plt.figure(figsize=(15,10))

#train loop
while(i < max_epoch):
    i+=1
    perm = np.random.permutation(train_size)
    train_loss = 0

    for j in range(train_size // batch_size):
        batch_x = X_train[perm[j*batch_size : (j+1)*batch_size]]
        batch_y = y_train[perm[j*batch_size : (j+1)*batch_size]]

        #Forward propagation
        l = 0
        z = 0
        for t in range(look_back):
            z = model(X_test[:,t].reshape(test_size,-1))
            l = rm.mse(z, y_test)
        model.truncate()
        test_loss = l.as_ndarray()
        test_curve.append(test_loss)

        if i % period == 0:
            print('epoch : {}, train loss : {}, test loss {}'.format(i ,train_loss,test_loss))
            draw_pred_curve(i)
            if test_loss > loss_prev * 0.99:
                print('Stop learning ')
                break
            else:
                loss_prev = deepcopy(test_loss)

# predicted curve
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left', fontsize=20)
plt.show()

plt.figure(figsize=(15,10))
plt.plot(learning_curve, color='blue', label='learning curve')
plt.plot(test_curve, color='orange', label='test curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(fontsize=30)
plt.show()
