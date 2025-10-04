## 🧠 T分布的定义
T分布（Student's t-distribution）是由英国统计学家戈塞特（William Sealy Gosset）提出，
为了解决小样本下对总体均值的估计问题。

它适用于总体服从正态分布但方差未知的情况，
尤其在样本量较小（通常 < 30）时比正态分布更可靠。  

📐 数学构造
设：𝑋∼𝑁(0,1)：标准正态分布

𝑌∼𝜒2(𝑛)：自由度为 𝑛的卡方分布

且 𝑋 与 𝑌 独立

📄 T 分布公式（Text 样式）
```代码
T = X / sqrt(Y / n) ~ t(n)
```
🔍 解释：
X 是标准正态分布 N(0,1)

Y 是自由度为 n 的卡方分布

T 是构造出的随机变量，服从自由度为 n 的 T 分布

**🧪 应用场景**
|应用|	说明
|---|---
|单样本 t 检验	|检验样本均值是否显著不同于某个值
|双样本 t 检验	|比较两个样本均值是否显著不同
|配对样本 t 检验	|比较同一对象前后变化
|置信区间估计|	在总体方差未知时估计均值的区间


例如，在医学实验中，样本量通常较小，使用 T 分布可以更准确地判断治疗效果是否显著。

✅ 总结一句话：
T分布是小样本统计推断的利器，适用于总体方差未知的情况，能更稳健地估计均值并进行假设检验

---
## python实现

✅ 基础实现：导入 T 分布对象
```python
from scipy.stats import t
```

📊 常用功能一览
1️⃣ 生成 `T` 分布概率密度函数（`PDF`）
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

df = 10  # 自由度
x = np.linspace(-4, 4, 500)
y = t.pdf(x, df)

plt.plot(x, y, label=f't-distribution (df={df})')
plt.title("T Distribution PDF")
plt.xlabel("t value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
```

2️⃣ 计算累积分布函数（`CDF`）
```python
t.cdf(1.96, df=10)  # 计算 t=1.96 时的累积概率
```

3️⃣ 计算分位点（反向查表）
```python
t.ppf(0.975, df=10)  # 查找右侧 2.5% 的临界值
```

4️⃣ 抽样模拟
```python
samples = t.rvs(df=10, size=1000)  # 生成 1000 个样本
```

5️⃣ 单样本 `T` 检验（对比均值）
```python
from scipy.stats import ttest_1samp

data = [2.3, 2.5, 2.8, 3.0, 2.7]
t_stat, p_value = ttest_1samp(data, popmean=2.5)
print(f"T统计量: {t_stat:.3f}, P值: {p_value:.3f}")
```

✅ 总结一句话：
用 `scipy.stats.t` 就能轻松实现 `T` 分布的所有功能，包括绘图、查表、抽样和假设检验，是 Python 统计分析的核心工具之一。

---
## 手动实现

✅ 用代码手动实现 `T` 分布采样
```python
import numpy as np
import matplotlib.pyplot as plt

def sample_t_distribution(df, size=1000):
    # Step 1: 生成标准正态分布样本 X
    X = np.random.normal(loc=0, scale=1, size=size)

    # Step 2: 生成自由度为 df 的卡方分布样本 Y
    Y = np.random.chisquare(df=df, size=size)

    # Step 3: 构造 T 分布样本 T = X / sqrt(Y / df)
    T = X / np.sqrt(Y / df)

    return T

# 示例：生成自由度为 10 的 T 分布样本
df = 10
samples = sample_t_distribution(df)

# 可视化
plt.hist(samples, bins=50, density=True, alpha=0.6, label=f'T(df={df})')
plt.title("T Distribution (手工构造)")
plt.xlabel("t value")
plt.ylabel("Density")
plt.grid(True)
plt.legend()
plt.show()
```

🔍 这段代码做了什么？
用 `np.random.normal()` 生成标准正态分布样本

用 `np.random.chisquare()` 生成卡方分布样本

按照公式手动构造 `T` 分布

用直方图展示结果

你可以对比 `scipy.stats.t.rvs(df)` 的结果，验证一致性。
