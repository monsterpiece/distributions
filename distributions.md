
# 1. sigmoid 函数

公式

$$
f(x) = \frac{1}{1+e^{-x}}
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
middle = 1
time = 1
x=np.linspace(-5,5,1000)  #1000个x值

y=[1/(1+np.exp(middle-i)) for i in x]

plt.plot(x,y)
plt.axvline(middle,color="black")
plt.title('sigmoid function')
plt.xlabel('in/out')
plt.show()
```

# 2. Hinge 函数

公式

$$
f(a,b,x) = 
\begin{cases}
0, & \text{x<a} \\
b(x-a), & \text{x>=a} \\
\end{cases}
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np

def hingefunc(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

mu = 1
sigma = 10
x = np.arange(-10,10,0.1)
y = hingefunc(x, mu, sigma)
plt.plot(x,y)
plt.axvline(0, color='black')
plt.title('Hinge function')
plt.xlabel('in/out')
plt.ylabel('f')
#输出
plt.show()
```


# 3. Poisson Distribution

公式
$$
f(x|\lambda) = e^{-\lambda}\frac{\lambda^x}{x!}
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import math

def poissonfunc(x,_lambda):
    pdf = np.exp(-_lambda)*((_lambda ** x)/(math.factorial(np.rint(x))))
    return pdf

_lambda = 10
x = np.arange(-10,100,0.1)
y = poissonfunc(x, _lambda)
plt.plot(x,y)
plt.axvline(0, color='black')
plt.title('Poisson distribution')
plt.xlabel('in/out')
plt.ylabel('f')
#输出
plt.show()
```


# 3. Normal Distribution

公式
$$
f(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp{\lbrace -\frac{1}{2\sigma^2}(x-\mu)^2 \rbrace}
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np

def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

mu = 1
sigma = 20
x = np.arange(-10,100,0.1)
y = normfun(x, mu, sigma)
plt.plot(x,y)
plt.axvline(0, color='black')
plt.title('Normal distribution')
plt.xlabel('in/out')
plt.ylabel('f')
#输出
plt.show()
```

# 4. Student-t Distribution

公式
$$
f(x|\mu,\sigma^2,\nu) = \frac{1}{\sqrt{2\pi}\sigma}\exp{\lbrace -\frac{1}{2\sigma^2}(x-\mu)^2 \rbrace}
$$
```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np

def betafunc(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

mu = 1
sigma = 10
x = np.arange(-10,10,0.1)
y = normfun(x, mu, sigma)
plt.plot(x,y)
plt.axvline(0, color='black')
plt.title('Normal distribution')
plt.xlabel('in/out')
plt.ylabel('f')
#输出
plt.show()
```

# 6. Beta Distribution

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np

def betafunc(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

mu = 1
sigma = 10
x = np.arange(-10,10,0.1)
y = normfun(x, mu, sigma)
plt.plot(x,y)
plt.axvline(0, color='black')
plt.title('Normal distribution')
plt.xlabel('in/out')
plt.ylabel('f')
#输出
plt.show()
```



# 5. Gamma Distribution

公式
$$
f(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp{\lbrace -\frac{1}{2\sigma^2}(x-\mu)^2 \rbrace}
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np

def betafunc(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf

mu = 1
sigma = 10
x = np.arange(-10,10,0.1)
y = normfun(x, mu, sigma)
plt.plot(x,y)
plt.axvline(0, color='black')
plt.title('Normal distribution')
plt.xlabel('in/out')
plt.ylabel('f')
#输出
plt.show()
```
