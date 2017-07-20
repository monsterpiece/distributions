
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


# 4. Normal Distribution

公式
$$
f(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp{\lbrace -\frac{1}{2\sigma^2}(x-\mu)^2 \rbrace}
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.linspace(-10,10,100)
mu = 1
sigma = 20
y = stats.norm.pdf(x, loc=mu, scale=sigma)
plt.plot(x,y,'r-',label='$pdf$')
plt.axvline(0, color='black')
plt.title('Normal distribution')
plt.xlabel('in/out')
plt.ylabel('f')

plt.legend()
plt.show()
```

# 5. Student-t Distribution

公式
$$
f(x|\nu) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu \pi}\Gamma(\nu/2)}(1+t^2/\nu)^{-(\nu+1)/2}
$$
在Student t 分布中，
- 当自由度v=1时，pdf尾巴很长，亦称为Cauchy分布
- 通常取v=4，在很多问题上可以得到很好性能

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

nu = 1.0
x = np.linspace(-10,10,1000)
y1 = stats.t.pdf(x,nu)
nu = 3.0
y2 = stats.t.pdf(x,nu)
nu = 0.1
y3 = stats.t.pdf(x,nu)
plt.plot(x,y1,'b-',label='$\nu=1.0$')
plt.plot(x,y2,'r-',label='$\nu=3.0$')
plt.plot(x,y3,'g-',label='$\nu=0.1$')
plt.title('T distribution')
plt.xlabel('in/out')
plt.ylabel('f')
#输出
plt.legend()
plt.show()
```
# 6. F Distribution

公式
$$
f(x|d_1,d_2) = \frac{U_1/d_1}{U_2/d_2} \\
U_1,U_2是自由度为d_1,d_2的卡方分布
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.linspace(0,3,1000)
d1 = 1.0
d2 = 1.0
y1 = stats.f.pdf(x,d1,d2)
d1 = 5.0
d2 = 2.0
y2 = stats.f.pdf(x,d1,d2)
d1 = 2.0
d2 = 1.0
y3 = stats.f.pdf(x,d1,d2)
d1 = 100.0
d2 = 100.0
y4 = stats.f.pdf(x,d1,d2)
plt.plot(x,y1,'b-',label='$d1=1.0,d2=1.0$')
plt.plot(x,y2,'r-',label='$d1=5.0,d2=2.0$')
plt.plot(x,y3,'g-',label='$d1=2.0,d2=1.0$')
plt.plot(x,y4,'k-',label='$d1=100,d2=100$')
plt.title('F distribution')
plt.xlabel('in/out')
plt.ylabel('f')
plt.ylim(0,3.0)
plt.xlim(0,3.0)
plt.legend()
plt.show()
```


# 7. Gamma Distribution

公式
$$
g(x|a,b) = \frac{b^a}{\Gamma(a)}x^{a-1}e^{-xb} \\
其中 x > 0 \\
a\ \ is\ \ shape \\
b\ \ is\ \ rate
$$

```python {cmd:true, matplotlib:true}
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
x = np.linspace(0,8,1000)
a = 1.0
b = 1.0
y1= stats.gamma.pdf(x,a,scale=1.0/b)

a = 1.5
b = 1.0
y2= stats.gamma.pdf(x,a,scale=1.0/b)

a = 2.0
b = 1.0
y3= stats.gamma.pdf(x,a,scale= 1.0/b)

a = 1.5
b = 1.0
y4= stats.gamma.pdf(x,a,scale = 1.0/b)

a = 1.5
b = 2.0
y5= stats.gamma.pdf(x,a,scale= 1.0/b)

a = 1.5
b = 3.0
y6= stats.gamma.pdf(x,a,scale = 1.0/b)

plt1 = plt.subplot(211)
plt2 = plt.subplot(212)

plt1.plot(x,y1,"g-",label='$a=1.0,b=1.0$')
plt1.plot(x,y2,"b-",label='$a=1.5,b=1.0$')
plt1.plot(x,y3,"r-",label='$a=2.0,b=1.0$')
plt2.plot(x,y4,"g-",label='$a=1.5,b=1.0$')
plt2.plot(x,y5,"b-",label='$a=1.5,b=2.0$')
plt2.plot(x,y6,"r-",label='$a=1.5,b=3.0$')
#plt.axvline(0, color='black')
plt1.set_title('Gamma distribution')
plt1.set_ylabel('f')
plt1.legend()

plt2.set_xlabel('in/out')
plt2.set_ylabel('f')

plt2.legend()
plt.show()
```


# 8. Chi-square Distribution

公式
$$
f(x|n) = \frac{1}{2^{n/2}\Gamma(\frac{n}{2})}x^{n/2-1}e^{-x/2} \\
其中 x > 0 \\
f(x|n)是\Gamma(x|a,b)在a=n/2,b=2的特殊情况
$$

```python {cmd:true, matplotlib:true}
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
x = np.linspace(0,8,1000)
a = 2.0
y1= stats.chi2.pdf(x,a)

a = 3.
y2= stats.chi2.pdf(x,a)

a = 5.0
y3= stats.chi2.pdf(x,a)



plt.plot(x,y1,"g-",label='$a=2.0$')
plt.plot(x,y2,"b-",label='$a=3.0$')
plt.plot(x,y3,"r-",label='$a=5.0$')
plt.title('Chi-square distribution')
plt.ylabel('f')
plt.xlabel('in/out')
plt.legend()

plt.show()
```
# 9. Beta Distribution
公式
$$
Beta(\theta|a,b) = \frac{1}{B(a,b)}\theta^{a-1}(1-\theta)^{b-1} \\
B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.linspace(0,1,500)
arrays = [0.1,0.1,1.0,1.0,2.0,3.0,8.0,4.0]
colors = ['b-.','r--','k.','g-']
for index in range(0,8,2):
    a = arrays[index]
    b = arrays[index+1]
    color = colors[index/2]
    y = stats.beta.pdf(x,a,b)
    plt.plot(x,y,color,label='$a=1.0,b=1.0$')
plt.title('Beta distribution')
plt.ylim(0,3.0)
plt.xlabel('in/out')
plt.ylabel('f')
plt.legend()
#输出
plt.show()
```