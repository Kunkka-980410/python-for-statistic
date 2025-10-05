## 📐 PDF 与 CDF 的定义

✅ PDF（Probability Density Function，概率密度函数）
描述连续型随机变量在某一点附近“出现的密度”

不是概率本身，而是概率的“浓度”

用积分求区间概率：
```text
P(a ≤ X ≤ b) = ∫ from a to b of f(x) dx
或
P(a ≤ X ≤ b) = ∫[a, b] f(x) dx
```

✅ CDF（Cumulative Distribution Function，累积分布函数）
描述随机变量小于等于某个值的累计概率

是 PDF 的积分形式：
```text
F(x) = P(X ≤ x) = ∫[-∞, x] f(t) dt
```

用差值求区间概率：
```text
P(a ≤ X ≤ b) = F(b) - F(a)
```

## 🔍 PDF vs CDF 的差别

|特性	|PDF 𝑓(𝑥)|CDF 𝐹(𝑥)
|---|---|---
|表示内容|	概率密度|	累计概率
|值域	|非负实数	|[0, 1]
|用途	|求区间概率需积分|	求区间概率用差值
|图像形状	|峰形（如正态分布钟形）	|单调递增曲线
|点概率	|𝑃(𝑋=𝑥)=0（连续型）	|𝐹(𝑥)−𝐹(𝑥−)=0（无跳跃）

## 🧪 Python 实现：以正态分布为例

```python
from scipy.stats import norm

mu = 4
sigma = (1.2)**0.5

# PDF：求某点的概率密度
pdf_value = norm.pdf(5, loc=mu, scale=sigma)
print(f"PDF at x=5: {pdf_value:.4f}")

# CDF：求小于等于某点的累计概率
cdf_value = norm.cdf(5, loc=mu, scale=sigma)
print(f"CDF at x=5: {cdf_value:.4f}")
```

***🎯 用 PDF 和 CDF 求概率***

**✅ 用 PDF（积分）求区间概率**：
```python
from scipy.integrate import quad

# 积分计算 P(6 ≤ X ≤ 7)
prob_pdf, _ = quad(lambda x: norm.pdf(x, loc=mu, scale=sigma), 6, 7)
print(f"Probability from PDF: {prob_pdf:.4f}")
```
这行代码做了以下几件事：

`lambda x`: norm.pdf(x, loc=mu, scale=sigma)：定义一个匿名函数，表示正态分布的 PDF

`norm.pdf(x, loc=mu, scale=sigma)`：计算正态分布在点 x 的密度值

`quad(..., 6, 7)`：对这个密度函数在区间 `[6, 7]` 上进行积分

`prob_pdf`：就是积分结果，即概率值

`_`：是积分误差估计，这里不需要就用 `_` 忽略


**✅ 用 CDF（差值）求区间概率**：
```python
prob_cdf = norm.cdf(7, loc=mu, scale=sigma) - norm.cdf(6, loc=mu, scale=sigma)
print(f"Probability from CDF: {prob_cdf:.4f}")
```
两者结果一致，但 CDF 更快更稳定
