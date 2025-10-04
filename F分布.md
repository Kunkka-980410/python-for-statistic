## `F` 分布（F-distribution）
它是方差比较和方差分析（ANOVA）中的核心工具
```text
F = (X1 / d1) / (X2 / d2) ~ F(d1, d2)
```
📌 参数说明：
X1：自由度为 d1 的卡方分布变量（χ² 分布）

X2：自由度为 d2 的卡方分布变量

d1：分子自由度（通常是组间）

d2：分母自由度（通常是组内）

F(d1, d2)：表示 F 分布，具有两个自由度参数

📊 F 分布的特点
非负分布：定义域为 `𝐹≥0`

右偏：尾部较长，尤其在自由度较小时

自由度控制形状：

𝑑1：分子自由度（通常是组间）

𝑑2：分母自由度（通常是组内）

🧪 应用场景
|应用|	说明
|---|---
|方差分析（ANOVA）|	检验多个组均值是否显著不同
|回归模型显著性检验	|检验整体模型是否有效
|方差齐性检验|	比较两个样本方差是否相等

🧮 Python 实现 F 分布
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

d1, d2 = 5, 10  # 自由度
x = np.linspace(0, 5, 500)
y = f.pdf(x, d1, d2)

plt.plot(x, y, label=f'F-distribution (d1={d1}, d2={d2})')
plt.title("F Distribution PDF")
plt.xlabel("F value")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()
```
