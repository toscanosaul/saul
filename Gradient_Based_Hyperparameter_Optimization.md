
## Sketch for the Assignment 4
Observe that I do not include detail comments of the functions, because I want you to study the different methods of this notebook. However, you should include detailed commments in your solution, where you explain the functions, and their parameters and outputs.

You should plot the number of epoch of the SGD algorithm versus cross-entropy on the test
dataset.


```python
import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot, squared_norm
from scipy.misc import comb, logsumexp 
from sklearn.linear_model.logistic import _multinomial_grad_hess

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  
```

### Get data


```python
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64')
y = mnist.target
```

## Split data into traning and test datasets


```python
train_samples = 30000
```


```python
random_state = check_random_state(0)

permutation = random_state.permutation(X.shape[0])

X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))
```


```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000, random_state=1)
```

### Normalize data


```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```


```python

```


```python
Y_train = np.zeros((len(y_train), 10))
for i,j in enumerate(y_train):
    Y_train[i, int(j)] = 1
```


```python
Y_test = np.zeros((len(y_test), 10))
for i,j in enumerate(y_test):
    Y_test[i, int(j)] = 1
```

### Loss function


```python
def loss_function(X,Y, w, alpha=0):
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    w = w.reshape(n_classes, -1)
    fit_intercept = w.size == (n_classes * (n_features + 1))
    old_w = w.copy()
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w
    
```


```python
def grad_loss(X, Y, w, alpha=0):
    """
    """
    pass
```


```python
def hessian_loss(X, Y, w, alpha=0):
    """
    """
    pass
    
```

### Stochastic Gradient Descent


```python
fit_intercept = True
```


```python
def sgd(momentum=0.9, lr=0.01, batch_size=1001, alpha=0.1, maxepoch=50, eps=1e-8):
    """
    Real-time forward-mode algorithm using stochastic gradient descent with constant learning 
    rate. Observe that you should only find the optimal the learning rate (lr), and the 
    penalty parameter (alpha). 
    
    We use the SGD with momentum, which is defined here: 
    http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
    """
    pass
```
