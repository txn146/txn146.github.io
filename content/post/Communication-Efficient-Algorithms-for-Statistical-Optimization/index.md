---
title: "Communication Efficient Algorithms for Statistical Optimization"
description: "Communication Efficient Algorithms for Statistical Optimization"
date: 2025-02-24T13:49:23Z
image: 
math: 
license: 
hidden: false
comments: true
draft: true
tags: 
    - stochastic optimization
    - subsampling
categories:
    - distributed learning
---

本篇为《Communication Efficient Algorithms for Statistical Optimization》一文的读书笔记，原文章2013年发表于 Journal of Machine Learning Research，作者为 Yuchen Zhang，John C. Duchi 和 Martin J. Wainwright。

## 背景说明
许多统计估计方法都基于经验风险最小化的形式，即通过最小化由数据上的损失函数的平均值定义的目标函数来估计一个感兴趣的参数。考虑到目前数据量的爆炸式增长，设计高效的算法来解决大规模问题是一个重要的课题。在集中式设置中，已经有许多解决经验风险最小化问题的程序，包括标准的凸优化方法和随机逼近与优化算法。然而，当数据集的规模变得极其庞大时，将所有数据存储在单个计算机中是不可行的，这篇论文的重点是研究一些分布式且通信高效的经验风险最小化程序。

在分布式框架下，最简单的统计估计方法为“平均混合（AVGM）”算法。其给定 $m$ 台机器和大小为 $N$ 的数据集，首先将数据集平均分配给每台机器，使得每台机器处理的子集大小为 $n = N/m$，然后让每台机器计算该子集的经验最小化器 $\theta_i$，最后将所有机器的参数估计值 $\theta_i$ 进行平均。然而，许多实际使用的估计器是有偏的，问题在于，在更一般的情况下，这种分布式方法是否仍然有效？是否能证明 AVGM 方法在计算效率上通常优于单台机器使用 $n = N/m$ 样本的简单方法？

## 问题设定

### 经验风险最小化

设 $\{f(\cdot;x), x \in X\}$ 为一组实值且凸的损失函数，每个函数都定义在一个包含凸集合 $\Theta \subset \mathbb{R}^d$ 的集合上。设 $P$ 为样本空间 $X$ 上的概率分布。假设每个函数 $x \mapsto f(\theta;x)$ 对 $P$ 可积，则总体风险函数 $F_0: \Theta \to \mathbb{R}$ 定义为：

$$
F_0(\theta) := \mathbb{E}_P[f(\theta;X)] = \int_X f(\theta;x)dP(x).
$$

目标是估计最小化总体风险的参数向量，如下所示：

$$
\theta^{*} := \arg\min_{\theta \in \Theta} F_0(\theta) = \arg\min_{\theta \in \Theta} \int_X f(\theta;x) dP(x),
$$

其中假设 $\theta^{*}$ 是唯一的。在实际应用中，总体分布 $P$ 是未知的，但我们可以访问一个从分布 $P$ 中抽样的样本集合 $S$。经验风险最小化是基于通过求解优化问题来估计 $\theta^{*}$：

$$
\hat{\theta} \in \arg\min_{\theta \in \Theta} \left\{ \frac{1}{|S|} \sum_{x \in S} f(\theta;x) \right\}.
$$

### 假设条件

在整个论文中，需要对参数空间、风险函数 $F_0$ 和损失函数 $f(\cdot;x): \Theta \to \mathbb{R}$ 施加一些正则性条件。这些条件是经典统计学中 M-估计（如 Lehmann 和 Casella, 1998；Keener, 2010）分析中的标准条件。

#### 假设 1 (参数空间)
参数空间 $\Theta \subset \mathbb{R}^d$ 是一个紧致凸集，并且有 $\theta^{*} \in \mathrm{int} \, \Theta$，$\ell_2$-半径为 $R = \max_{\theta \in \Theta} \|\theta - \theta^{*}\|_2$。第一个假设涉及参数空间与最优化参数之间的关系

此外，风险函数要求具有一定的曲率。我们通过 $F_0$ 的 Hessian 矩阵来正式化这一概念：

#### 假设 2 (局部强凸性)
总体风险是二次可微的，并且存在参数 $\lambda > 0$ 使得
$$
\nabla^2 F_0(\theta^{*}) \succeq \lambda I_{d \times d}.
$$
这里，$\nabla^2 F_0(\theta)$ 表示群体目标函数 $F_0$ 在 $\theta$ 处的 $d \times d$ Hessian 矩阵，我们使用 $\succeq$ 来表示半正定的矩阵顺序（即，$A \succeq B$ 表示 $A - B$ 是正半定的）。这个局部条件比全局强凸性条件更为宽松，只要求在 $\theta^{*}$ 处的群体风险 $F_0$ 满足条件。需要注意的是，对于任何方法来说，估计参数 $\theta^{*}$ 时都需要某种类型的曲率。

### 平均混合算法
考虑一个数据集，由 $N = mn$ 个样本组成，独立同分布（i.i.d.），遵循分布 $P$。在分布式设置中，我们将 $N$ 个样本数据均匀随机地分配给 $m$ 个处理器。（为简便起见，我们假设样本总数是 $m$ 的倍数）。对于每个 $i = 1, \dots, m$，我们让 $S_{1,i}$ 表示分配给第 $i$ 个处理器的数据集；构造上，它是从分布 $P$ 中独立抽取的 $n$ 个样本，而每个子集 $S_{1,i}$ 和 $S_{1,j}$ 对 $i \neq j$ 是独立的。此外，对于每个处理器 $i$，我们定义（局部）经验分布 $P_{1,i}$ 和经验目标函数 $F_{1,i}$：

$$
P_{1,i} := \frac{1}{|S_1|} \sum_{x \in S_{1,i}} \delta_x, \quad F_{1,i}(\theta) := \frac{1}{|S_1|} \sum_{x \in S_{1,i}} f(\theta;x).
$$

1. 对于每个 $i \in \{1, \dots, m\}$，处理器 $i$ 使用它的本地数据集 $S_{1,i}$ 来计算局部经验最小化器：
$$
\theta_{1,i} \in \arg\min_{\theta \in \Theta} \left\{ \frac{1}{|S_{1,i}|} \sum_{x \in S_{1,i}} f(\theta;x) \right\}.
$$

2. 这些 $m$ 个局部估计值然后被平均——即计算：
$$
\bar{\theta}_1 = \frac{1}{m} \sum_{i=1}^m \theta_{1,i}.
$$

### 子抽样平均混合算法
子抽样平均混合（SAVGM）算法是在第一种方法的基础上，增加了一个抽样层次，涉及固定的抽样率 $r \in [0,1]$，它由以下步骤组成：

1. 每个处理器 $i$ 从其本地数据集 $S_{1,i}$ 中均匀随机地抽取一个大小为 $[rn]$ 的子集 $S_{2,i}$，然后计算局部经验最小化器 $\theta_{2,i}$：

$$
\theta_{2,i} \in \arg\min_{\theta \in \Theta} \left\{ \frac{1}{|S_{2,i}|} \sum_{x \in S_{2,i}} f(\theta;x) \right\}.
$$

2. 在进行上述平均（步骤 2）后，SAVGM 算法计算自助平均 $\bar{\theta}_2$，然后返回加权组合：
$$
\bar{\theta}_{SAVGM} := \frac{\bar{\theta}_1 - r\bar{\theta}_2}{1 - r}.
$$

## 理论结果

### 光滑性条件
除了我们之前关于总体风险的假设外，我们还要求对经验风险函数施加一些正则性条件。最简单的方式是用函数 $ \theta \mapsto f(\theta;x) $ 来表示这些条件，我们注意到，正如假设 2 中所要求的，我们只需要这些条件在某些局部区域内成立，特别是在某个欧几里得球 $ U = \{ \theta \in \mathbb{R}^d | \|\theta -\theta^{*}\|_2 \leq \rho \} \subset \Theta $ 内。

#### 假设 3（光滑性）

假设函数 $ f $ 的一阶和二阶偏导数存在，并满足以下界限：

$$
\mathbb{E} \left[ \| \nabla f(\theta; X) \|^8 \right] \leq G^8, \quad \mathbb{E} \left[ \| \nabla^2 f(\theta; X) - \nabla^2 F_0(\theta) \|^8 \right] \leq H^8 \quad \text{对于所有} \, \theta \in U.
$$

此外，对于任意 $ x \in X $，Hessian 矩阵 $ \nabla^2 f(\theta;x) $ 是 $ L(x) $-Lipschitz 连续的，意味着：

$$
\| \nabla^2 f(\theta'; x) - \nabla^2 f(\theta; x) \|_2 \leq L(x) \| \theta' - \theta \|_2 \quad \text{对于所有} \, \theta, \theta' \in U,
$$

并且我们要求：

$$
\mathbb{E} \left[ L(X)^8 \right] \leq L^8 \quad \text{并且} \quad \mathbb{E} \left[ |L(X) - \mathbb{E}[L(X)]|^8 \right] \leq L^8 \quad \text{对于某个有限常数} L.
$$

### 平均混合的误差界

定理1: 在假设 1 到假设 3 的条件下，平均混合的均方误差上界为：

$$
\begin{equation}
\begin{aligned}
\mathbb{E} \left[ \|\bar{\theta} - \theta^{*}\|^2 \right] &\leq \frac{2}{nm} \mathbb{E} \left[ \|\nabla^2 F_0(\theta^{*})^{-1} \nabla f(\theta^{*};X)\|_2^2 \right] \\ 
&\quad + \frac{c}{\lambda^2 n^2} \left( H^2 \log d + \frac{L^2 G^2}{\lambda^2} \right) \mathbb{E} \left[ \|\nabla^2 F_0(\theta^{*})^{-1} \nabla f(\theta^{*}; X)\|_2^2 \right] \\
&\quad + O(m^{-1} n^{-2}) + O(n^{-3}).
\end{aligned}
\end{equation}
$$

定理 1 的一个稍弱推论使其更容易理解。特别是，注意到：

$$
\begin{equation}
\left\| \nabla^2 F_0(\theta^{*})^{-1} \nabla f(\theta^{*};x) \right\|_2
\overset{(i)}{\leq}
\left\| \nabla^2 F_0(\theta^{*})^{-1} \right\|_2 \left\| \nabla f(\theta^{*};x) \right\|_2
\overset{(ii)}{\leq} 
\frac{1}{\lambda} \left\| \nabla f(\theta^{*};x) \right\|_2,
\end{equation}
$$
其中步骤 (i) 依据不等式
$\| Ax \|_2 \leq \| A \|_2 \| x \|_2$（对任意矩阵 \( A \) 和向量 \( x \) 成立），而步骤 (ii) 依据假设 2。此外，假设 3 表明：$
\mathbb{E} \left[ \|\nabla f(\theta^{*};X)\|^2 \right] \leq G^2$, 综合这些部分，我们可以得出以下结论。

推论 2： 在定理 1 相同的条件下：

$$
\mathbb{E} \left[ \|\bar{\theta} -\theta^{*}\|^2 \right] \leq \frac{2G^2}{\lambda^2 nm} + \frac{c G^2}{\lambda^4 n^2} \left( H^2 \log d + \frac{L^2 G^2}{\lambda^2} \right) + O\left(m^{-1} n^{-2}\right) + O\left(n^{-3}\right)
$$

这个上界表明，主导项的衰减比例为 \( (nm)^{-1} \)，其系数与强凸性常数 \( \lambda \) 反比，并与损失梯度的上界 \( G \) 成正比。虽然易于解释，但上界 (8) 可能较松弛，因为它是基于相对较弱的不等式 (7) 推导而来的。
我们原始上界 (6) 的主导项涉及梯度 \( \nabla f(\theta^{*};X) \) 与逆 Hessian 的乘积。在许多统计设置中，包括线性回归问题，这种矩阵-向量乘积的影响通常会通过某种标准化方式来处理。
其中 $c$ 是一个数值常数。

## 负对数似然与费舍尔信息矩阵

损失函数 \( f(\cdot; x): \Theta \to \mathbb{R} \) 实际上是参数模型族 \( \{P_{\theta}\} \) 的负对数似然 \( \ell(x \mid \theta) \)。在适当的正则性条件下（例如 Lehmann 和 Casella，1998，第 6 章），我们可以定义 **费舍尔信息矩阵**：

\[
I(\theta^*) := \mathbb{E} \left[ \nabla \ell (X \mid \theta^*) \nabla \ell (X \mid \theta^*)^\top \right] = \mathbb{E} [\nabla^2 \ell (X \mid \theta^*)].
\]

回忆一下，\( N = mn \) 是可用的样本总数，我们定义邻域：

\[
B_2(\theta, t) := \{\theta' \in \mathbb{R}^d : \|\theta' - \theta\|_2 \leq t\}.
\]

在我们的假设下，Hájek-Le Cam 最小最大定理（van der Vaart, 1998, Theorem 8.11）保证对于 **基于 \( N \) 个样本的任何估计量 \( \hat{\theta}_N \)**，成立：

\[
\lim\inf_{c \to \infty} \lim\inf_{N \to \infty} \sup_{\theta \in B_2(\theta^*, c/\sqrt{N})} N \mathbb{E}_{\theta} \left[ \|\hat{\theta}_N - \theta\|_2^2 \right] \geq \text{tr}(I(\theta^*)^{-1}).
\]

## **推论 3**
**在定理 1 的条件基础上，假设损失函数 \( f(\cdot; x) \) 也是参数族 \( \{P_{\theta}, \theta \in \Theta\} \) 的负对数似然，则均方误差可以被上界如下：**

\[
\mathbb{E} \left[ \|\bar{\theta} - \theta^*\|^2_{\text{ICA}} \right] 
\leq \frac{2}{mn} \text{tr}(I(\theta^*)^{-1}) 
+ \frac{cm^2}{\lambda^2 N^2} \text{tr}(I(\theta^*)^{-1}) 
\left( H^2 \log d + \frac{L^2 G^2}{\lambda^2} \right) + O(M^{-N}),
\]

其中 \( c \) 是一个常数。

### **证明**
将负对数似然重写为 **Theorem 1** 的记号，我们有：

\[
\nabla \ell(x \mid \theta^*) = \nabla f(\theta^*;x).
\]

于是，我们需要计算 **费舍尔信息矩阵的逆**：

\[
I(\theta^*)^{-1} = \mathbb{E} \left[ I(\theta^*)^{-1} \nabla \ell (X \mid \theta^*) \nabla \ell (X \mid \theta^*)^\top I(\theta^*)^{-1} \right].
\]

\[
= \mathbb{E} \left[ \left( \nabla^2 F_0(\theta^*)^{-1} \nabla f(\theta^*; X) \right) \left( \nabla^2 F_0(\theta^*)^{-1} \nabla f(\theta^*; X) \right)^\top \right].
\]

现在，我们应用 **迹的线性性** 并使用 **\( \text{tr}(uu^\top) = \| u \|_2^2 \)**，得出结论。 \(\blacksquare\)

---

## **推论 3 的意义**
除了界中的常数 \( 2 \) 之外，**推论 3 说明了定理 1 本质上已经达到了最优结果**。然而，该推论的关键之处在于：
- **无需计算所有 \( N = mn \) 个样本的估计**。
- **我们仅计算 \( m \) 个独立估计值 \( \hat{\theta} \)**，然后对它们求均值以达到收敛保证。

通过检查证明，我们可以发现：
- **如果愿意牺牲高阶误差的更差常数，我们可以将定理 1 的主导项 \( \frac{2}{mn} \) 缩减为 \( (1 + c)/mn \)（对于任意 \( c > 0 \)）**。
- **然而，正如推论 3 所述，这种缩减是无法进一步改进的**，即便是通过常数因子优化。

---

## **与经典估计理论的联系**
- 由于独立求解的 \( m \) 个估计会减少方差 \( 1/m \)，我们的结果与 M-估计理论中的经典分布收敛结果类似。
- **对于平滑函数，M-估计收敛速度与独立均值收敛速度相同**（van der Vaart, 1998; Lehmann 和 Casella, 1998）。
- **合并多个独立估计的均值可以进一步降低方差**。

尽管如此，**有时候使用有偏估计（biased estimators）是有意义的**，但这会引入额外的分析复杂性，我们将在下一节进一步探讨。此外：
- **与经典渐近理论不同，我们的结果适用于有限样本，并提供均方误差的显式上界**。
- **我们的结果不依赖于特定模型，这使得它们适用于较为一般的采样分布**。


### 子抽样平均混合的误差界

### 随机梯度下降平均算法

## 数值模拟

## 实例分析

## 参考文献

- Zhang, Y., Duchi, J. C., & Wainwright, M. J. (2013). Communication-Efficient Algorithms for Statistical Optimization. *Journal of Machine Learning Research*, 14, 3321-3363.
