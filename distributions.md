
[TOC]
# 1. Sigmoid 函数

公式

$$
f(x) = \frac{1}{1+e^{-x}}
$$

```python {id:'sigmoid',cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
x=np.linspace(-5,5,1000)  #1000个x值
scale = 2.0
middle = 1.0
y=[scale/(1+np.exp(middle-i)) for i in x]

plt.plot(x,y,label="$scale="+str(scale)+",middle="+str(middle)+"$")
plt.axvline(middle,color="black")
plt.title('sigmoid function')
plt.xlabel('in/out')
plt.legend()
plt.show()
```

# 2. Relu/Hinge 函数

```python {id:"relu",cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,2,100)

def hingefunc(x,a):
    tmp = a-x
    return np.maximum(tmp,0.0,tmp)

def relufunc(x,a):
    tmp = x-a
    return np.maximum(tmp,0.0,tmp)

plt.axvline(0, color='black')
plt.axhline(0, color='black')
a=0.5
y = hingefunc(x, a)
plt.plot(x,y,label='$hingefunc(a='+str(a)+')$')
y = relufunc(x,a)
plt.plot(x,y,label='relu function(a='+str(a)+')')
plt.title('Relu/Hinge function')
plt.xlabel('in/out')
plt.ylabel('f')
plt.ylim(-0.1)
plt.legend()
plt.show()
```


# 3. Poisson Distribution

公式
$$
f(x|\lambda) = e^{-\lambda}\frac{\lambda^x}{x!} \\
其中x \in \cal{N} \\
\lambda > 0
$$

```python {id:"poisson",cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.arange(0,10)
_lambda = 2
y = stats.poisson.pmf(x, _lambda)
plt.plot(x,y,label='$\lambda='+str(_lambda)+'$')
plt.title('Poisson distribution')
plt.xlabel('in/out')
plt.ylabel('f')
plt.legend()
plt.show()
```


# 4. Normal Distribution

公式
$$
f(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp{\lbrace -\frac{1}{2\sigma^2}(x-\mu)^2 \rbrace}
$$

```python {id:"normal",cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.linspace(0,1,100)
mu = 1
sigma = 20
y = stats.norm.pdf(x, loc=0.33, scale=0.08) + 2*stats.norm.pdf(x, loc=1, scale=0.16)
#y = np.sin(x*2*np.pi)+1.5*x
plt.plot(x,y,'r-',label='$norm pdf(\mu = '+str(mu)+', \sigma = '+str(sigma)+')$')
plt.axvline(0, color='black')
plt.title('Normal distribution')
plt.xlabel('in/out')
plt.ylabel('f')

plt.legend()
plt.show()
```
## 5. Laplace Distribution

公式
$$
f(x|\mu,b) = \frac{1}{2b}\exp(-\frac{|x-\mu|}{b} )
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.linspace(-10,10,500)
mu = 0
b = 2
y = stats.laplace.pdf(x, loc=mu, scale=b)

plt.plot(x,y,'r-',label='$\mu = '+str(mu)+', b = '+str(b)+'$')
plt.axvline(0, color='black')
plt.title('Laplace distribution')
plt.xlabel('in/out')
plt.ylabel('f')

plt.legend()
plt.show()
```
## 6. Student-t Distribution

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
x = np.linspace(-5,5,1000)
y1 = stats.t.pdf(x,nu)
nu = 3.0
y2 = stats.t.pdf(x,nu)
nu = 30.0
y3 = stats.t.pdf(x,nu)
y4 = stats.norm.pdf(x)
plt.plot(x,y1,'b-',label='$ \\nu =1.0 $')
plt.plot(x,y2,'r-',label='$ \\nu = 3.0 $')
plt.plot(x,y3,'g-',label='$ \\nu = 30.0 $')
plt.plot(x,y4,'k-.',label='$ normal distribution $')
plt.title('T distribution')
plt.xlabel('in/out')
plt.ylabel('f')

plt.legend()
plt.show()
```

## 对比上述分布

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 0.
sigma = 1
nu = 1.0
b = 1
x = np.linspace(-5,5,1000)
y1 = stats.t.pdf(x,nu)
y2 = stats.norm.pdf(x,mu,sigma)
y3 = stats.laplace.pdf(x,mu,b)
plt.plot(x,y1,'b-',label='$students\'t$')
plt.plot(x,y2,'r-',label='$normal distribution$')
plt.plot(x,y3,'g-',label='$laplace distribution$')
plt.title('T distribution')
plt.xlabel('in/out')
plt.ylabel('f')

plt.legend()
plt.show()
```

# 7. Gamma Distribution

公式
$$
g(x|a,b) = \frac{b^a}{\Gamma(a)}x^{a-1}e^{-xb} \\
其中 x > 0 \\
a\ \ 称为\ \ shape \\
b\ \ 称为\ \ rate \\
在一些公式中使用scale=1/b来替代b \\
\Gamma(a) = \int^{\infty}_{0}t^{a-1}e^{-t}\,{\rm d}t\,,\Gamma(a+1)=a\Gamma(a)(类比于阶乘，\Gamma(1) = 1) \\
Gamma分布和卡方分布、指数分布、F分布以及Beta分布都有关系
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

## 8. Exponential Distribution

公式
$$
f(x|\lambda) = \lambda e^{-\lambda x}\\
其中 x \geq 0, and\, \lambda > 0
$$

```python {cmd:true, matplotlib:true}
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
x = np.linspace(0,12,1000)
_lambda = 0.5
y1= stats.expon.pdf(x,scale=1.0/_lambda)

_lambda = 1.
y2= stats.expon.pdf(x,scale=1.0/_lambda)

_lambda = 5.0
y3= stats.expon.pdf(x,scale=1.0/_lambda)

plt.plot(x,y1,"g-",label='$\lambda=0.5$')
plt.plot(x,y2,"b-",label='$\lambda=1.0$')
plt.plot(x,y3,"r-",label='$\lambda=5.0$')
plt.title('Exponential distribution')
plt.ylabel('f')
plt.xlabel('in/out')
plt.legend()

plt.show()
```

## 9. Chi-square Distribution

公式
$$
f(x|n) = \frac{1}{2^{n/2}\Gamma(\frac{n}{2})}x^{n/2-1}e^{-x/2} \\
其中 x > 0 \\
f(x|n)是\Gamma(x|a,b)在a=n/2,b=2的特殊情况 \\
n 在卡方分布中被称为自由度
$$

```python {cmd:true, matplotlib:true}
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
x = np.linspace(0,8,1000)
n = 2.0
y1= stats.chi2.pdf(x,n)

n = 3.
y2= stats.chi2.pdf(x,n)

n = 5.0
y3= stats.chi2.pdf(x,n)

plt.plot(x,y1,"g-",label='$n=2.0$')
plt.plot(x,y2,"b-",label='$n=3.0$')
plt.plot(x,y3,"r-",label='$n=5.0$')
plt.title('Chi-square distribution')
plt.ylabel('f')
plt.xlabel('in/out')
plt.legend()

plt.show()
```

## 10. F Distribution

公式
$$
f(x|d_1,d_2) = \frac{U_1/d_1}{U_2/d_2} \\
U_1,U_2是自由度为d_1,d_2的卡方分布
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

x = np.linspace(0.001,3,1000)
d1 = 1.0
d2 = 1.0
y1 = stats.f.pdf(x,d1,d2)
d1 = 5.0
d2 = 2.0
y2 = stats.f.pdf(x,d1,d2)
d1 = 50.0
d2 = 10.0
y3 = stats.f.pdf(x,d1,d2)
d1 = 100.0
d2 = 100.0
y4 = stats.f.pdf(x,d1,d2)
plt.plot(x,y1,'b-',label='$d1=1.0,d2=1.0$')
plt.plot(x,y2,'r-',label='$d1=5.0,d2=2.0$')
plt.plot(x,y3,'g-',label='$d1=50.0,d2=10.0$')
plt.plot(x,y4,'k-',label='$d1=100,d2=100$')
plt.title('F distribution')
plt.xlabel('in/out')
plt.ylabel('f')
plt.ylim(0,2.5)
plt.xlim(0,3.0)
plt.legend()
plt.show()
```


# 11. Beta Distribution
公式
$$
Beta(\theta|a,b) = \frac{1}{B(a,b)}\theta^{a-1}(1-\theta)^{b-1} \\
B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}
$$

```python {cmd:true, matplotlib:true}
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

plt.figure(figsize=(6,21),dpi=80)

x = np.linspace(0,1,1000)
a_array = [0.1,1.0,2.0,4.0,8.0,100.0]
b_array = [0.1,1.0,2.0,4.0,8.0,100.0]
plt1 = plt.subplot(311)
plt2 = plt.subplot(312)
plt3 = plt.subplot(313)
for a in a_array:
    for b in b_array:
        if max(a,b)/min(a,b) > 10:
            continue
        y = stats.beta.pdf(x,a,b)
        if a < b:
            plt1.plot(x,y,label='$a='+str(a)+',b='+str(b)+'$')
            #plt1.axvline(float(a-1)/(a+b-2),0.0,10.0)
        elif a == b:
            plt3.plot(x,y,label='$a='+str(a)+',b='+str(b)+'$')
        else:
            plt2.plot(x,y,label='$a='+str(a)+',b='+str(b)+'$')
plt.suptitle('Beta distribution')
plt3.set_xlabel('in/out')
plt2.set_ylabel('f')
plt1.set_ylim(0.0,10)
plt2.set_ylim(0.0,10)
plt3.set_ylim(0.0,10)

plt1.grid(True)
plt2.grid(True)
plt3.grid(True)

plt1.legend()
plt2.legend()
plt3.legend()

plt.show()
```