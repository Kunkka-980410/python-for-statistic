Gamma 分布是一种连续概率分布，常用于建模等待时间、寿命、风速等非负变量，具有两个参数：形状参数 `α` 和尺度参数 `β`。它是指数分布和卡方分布的推广。

```text
f(x; α, β) = (1 / (Γ(α) * β^α)) * x^(α - 1) * e^(-x / β),  for x > 0
```
📌 参数说明：
`α`：形状参数（shape parameter）

`β`：尺度参数（scale parameter）

`Γ(α)`：伽马函数（Gamma function），定义为 `Γ(α) = ∫₀^∞ t^(α−1) * e^(−t) dt`

`x`：自变量，必须大于 0


🧮 Python 实现 Gamma 分布
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

alpha = 2.0  # 形状参数
beta = 1.5   # 尺度参数
x = np.linspace(0, 10, 500)
y = gamma.pdf(x, a=alpha, scale=beta)

plt.plot(x, y, label=f'Gamma(α={alpha}, β={beta})')
plt.title("Gamma Distribution PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()
```

🔁 与其他分布的关系
指数分布是 Gamma 分布的特例（当 α = 1）

卡方分布是 Gamma 分布的特例（当 α = n/2, β = 2）

Poisson 分布的参数可以由 Gamma 分布建模

---
Gamma 分布中的两个核心参数：形状参数 `α（alpha）`和尺度参数 `β（beta）`。

🧠 一句话概括：
形状参数 `α`：决定分布的“形状”，比如是否偏斜、是否有峰值

尺度参数 `β`：控制分布的“拉伸程度”，单位变化的大小

🔍 形状参数 α（alpha）
决定分布的形状和偏态程度

越小 → 分布越偏斜，像指数分布

越大 → 分布越对称，趋近于正态分布

|`α` 值	|分布形状
|---|---
|`α = 1`|	指数分布（单峰偏右）
|`α < 1`|	极度偏斜，密度集中在靠近 0 的地方
|`α > 1`|	出现明显峰值，逐渐对称


🔍 尺度参数 `β（beta）`
控制分布的“宽度”或“单位大小”

越大 → 分布越宽，值越分散

越小 → 分布越窄，值越集中

比如在寿命建模中，`β` 表示“单位时间长度”，`α` 表示“事件发生次数”。

📊 举个例子
```python
from scipy.stats import gamma
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 500)
for alpha in [0.5, 1, 2, 5]:
    y = gamma.pdf(x, a=alpha, scale=2)
    plt.plot(x, y, label=f'α={alpha}, β=2')

plt.title("Gamma 分布随 α 变化的形状")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
```
你会看到：随着 α 增大，分布从偏斜变得越来越对称。

---
用一个真实的生信场景，从数据出发，推导 Gamma 分布的形状参数 α 和尺度参数 β，并理解它们的意义

🧬 生信场景：基因表达量建模
在 `RNA-seq` 或 `scRNA-seq` 分析中，某些基因的表达量（如 `TPM`、`FPKM`、`UMI counts`）往往是非负连续变量，而且呈现右偏分布。这种数据非常适合用 Gamma 分布建模。

📊 假设数据如下（某基因在 10 个样本中的表达量）：
```python
expression = [2.1, 3.5, 1.8, 4.2, 2.9, 3.1, 2.7, 3.8, 2.4, 3.3]
```
这些值是非负的、右偏的，符合 Gamma 分布的特征。
🧠 如何提炼出 α 和 β？
我们知道：

期望 𝜇=𝛼𝛽

方差 𝜎**2=𝛼𝛽**2

所以我们可以先计算样本均值和方差，然后反推 `α` 和 `β`：

```python
import numpy as np

data = np.array(expression)
mu = np.mean(data)
var = np.var(data, ddof=1)  # 样本方差

# 反推参数
beta = var / mu
alpha = mu / beta

print(f"样本均值 μ = {mu:.3f}")
print(f"样本方差 σ² = {var:.3f}")
print(f"推导出的形状参数 α = {alpha:.3f}")
print(f"推导出的尺度参数 β = {beta:.3f}")
```

📌 输出示例（根据数据）：
```代码
样本均值 μ = 3.080
样本方差 σ² = 0.558
```
推导出的形状参数 α = 17.030
推导出的尺度参数 β = 0.181

🔍 参数含义解析：
α = 17.03：形状参数较大 → 分布较对称，波动较小，表达稳定

β = 0.181：尺度较小 → 单位变化不大，表达集中在均值附近

这说明该基因在样本间表达较稳定，适合用 Gamma 分布建模。
