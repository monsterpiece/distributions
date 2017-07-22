
# Penalty Function

This is the penalty function
## Function formula

公式如下：

$$
f(x) = \frac{g(x)}{g(x_{max})} \\
where\ g(x) = N(\mu_1,\sigma_1)+\frac{\sigma_2}{\sigma_1}N(\mu_2,\sigma_2) \\
\mu_1\ is\ the\ mean\ value\  of\  the\ gauss\ function\ with\ more\ sharpe\ crest.\\
x_{max}\ usually\ is\ set\ as\ \mu_1\\
as\ a\  result\  f(x) \in [0,1]
$$

## Matlab code

if you have a <font color='green'><i>matlab</i></font> rather than a <font color='red'><i>octave</i></font>, change the the "cmd" value to "matlab" in this markdown file.

```matlab {id:"Penalty Function",cmd:"octave",output:"markdown"}

function y = bi_gauss_pdf(x,mu1,sigma1,mu2,sigma2)
    y = normpdf(x,mu1,sigma1)+normpdf(x,mu2,sigma2)*sigma2/sigma1;
end

%% below for plot test
x = 0:0.01:1;
mu1 = 0.33;
sigma1 = (-mu1+0.5)/3;
mu2 = 1.0;
sigma2 = (mu2-0.5)/3;
y = bi_gauss_pdf(x,mu1,sigma1,mu2,sigma2);
y_max = bi_gauss_pdf(mu1,mu1,sigma1,mu2,sigma2);
y = y./y_max;

plot(x,y);
axis([0 1.2 0 1.2]);
set(gca,'xtick',0:0.1:1);
set(gca,'ytick',0:0.1:2);
hold on;
plot([0 1.2],[1 1],'--');
plot([1 1],[0 1.2],'k-');

grid on;

%% wonderful interactive method to end showing.
if waitforbuttonpress
    return;
end
```