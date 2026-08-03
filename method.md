# III. METHODOLOGY

## A. Problem Definition and Interval Parameterization

本文考虑面向配电网运行成本的风电区间预测。给定长度为 \(T_{\mathrm{in}}\) 的历史序列
\(\boldsymbol{s}_i\)，区间预测器 \(f_{\boldsymbol{\theta}}\) 输出样本 \(i\) 的下、上界：

$$
(\widehat{\ell}_i,\widehat{u}_i)
=f_{\boldsymbol{\theta}}(\boldsymbol{s}_i),
\qquad
0\leq \widehat{\ell}_i\leq \widehat{u}_i\leq P^{\max}.
\tag{1}
$$

在当前实现中，区间预测器由 LSTM 编码器和两维输出头构成。若输出头的原始输出为
\((a_i,b_i)\)，则采用如下可行化映射：

$$
\widehat{\ell}_i=P^{\max}\sigma(a_i),
\qquad
\widehat{u}_i=\widehat{\ell}_i+
\left(P^{\max}-\widehat{\ell}_i\right)\sigma(b_i),
\tag{2}
$$

其中，\(\sigma(\cdot)\) 为 Sigmoid 函数。因此，式 (1) 中的边界与排序约束可由网络结构
直接满足，无需额外投影。

对真实风电功率 \(y_i\)，定义区间决策向量

$$
\boldsymbol{x}_i=
\begin{bmatrix}
x_i^-\\ x_i^+
\end{bmatrix}
=
\begin{bmatrix}
y_i-\widehat{\ell}_i\\
\widehat{u}_i-y_i
\end{bmatrix}.
\tag{3}
$$

在离线 QICNN 数据集中，区间满足
\(0\leq\ell_i\leq y_i\leq u_i\leq P^{\max}\)。端到端训练时，为避免预测初期的越界
造成负裕度，代码中采用

$$
\widetilde{\boldsymbol{x}}_i=
\begin{bmatrix}
[y_i-\widehat{\ell}_i]_+\\
[\widehat{u}_i-y_i]_+
\end{bmatrix},
\qquad [z]_+=\max(z,0).
\tag{4}
$$

真实值经缩放后作为条件变量，即
\(\boldsymbol{c}_i=y_i/s_c\)，当前实现取 \(s_c=2\)。

## B. Conditional QICNN Surrogate

为避免在每次梯度更新时重复求解 IEEE 33 节点二阶锥最优潮流，本文构造条件
QICNN \(V_{\boldsymbol{\phi}}(\boldsymbol{c},\boldsymbol{x})\)，逼近给定运行条件和
区间决策下的最优总成本。对固定条件 \(\boldsymbol{c}\)，该代理函数关于
\(\boldsymbol{x}\) 保持凸性。其总体形式为

$$
V_{\boldsymbol{\phi}}(\boldsymbol{c},\boldsymbol{x})
=V_{\mathrm{ICNN}}(\boldsymbol{c},\boldsymbol{x})
+V_{\mathrm{quad}}(\boldsymbol{c},\boldsymbol{x}).
\tag{5}
$$

条件网络 \(h_{\boldsymbol{\phi}}(\boldsymbol{c})\) 是一个超网络。它先将运行条件映射为
条件表示 \(\boldsymbol{h}=h_{\boldsymbol{\phi}}(\boldsymbol{c})\)，再通过两个参数头分别
生成 ICNN 主干参数和二次支路参数：

$$
\boldsymbol{\Theta}_{\mathrm{I}}(\boldsymbol{c})
=g_{\mathrm{I}}(\boldsymbol{h})
=\{\boldsymbol{W}_k^x,\boldsymbol{W}_k^z,\boldsymbol{b}_k,
\boldsymbol{w}^x,\boldsymbol{w}^z,b^o\}(\boldsymbol{c}),
$$

$$
\boldsymbol{\Theta}_{\mathrm{Q}}(\boldsymbol{c})
=g_{\mathrm{Q}}(\boldsymbol{h})
=\{\boldsymbol{B},\boldsymbol{e},\widetilde{\gamma}\}(\boldsymbol{c}).
$$

因此，条件变量 \(\boldsymbol{c}\) 的作用是为当前样本生成一组特定的凸函数参数，
即选择与当前运行状态对应的成本曲面；决策变量 \(\boldsymbol{x}\) 则是在该曲面上
进行评价的变量。条件网络可对 \(\boldsymbol{c}\) 采用任意非线性映射，因为所要求的
凸性仅针对固定 \(\boldsymbol{c}\) 时的 \(\boldsymbol{x}\)。

### 1) ICNN branch

令 \(\boldsymbol{z}_0\) 表示第一层隐状态，则 \(L\) 层 ICNN 写为

$$
\boldsymbol{z}_0=
\operatorname{ReLU}\!\left(
\boldsymbol{W}^{x}_0(\boldsymbol{c})\boldsymbol{x}
+\boldsymbol{b}_0(\boldsymbol{c})
\right),
\tag{6}
$$

$$
\boldsymbol{z}_k=
\operatorname{ReLU}\!\left(
\boldsymbol{W}^{z}_k(\boldsymbol{c})\boldsymbol{z}_{k-1}
+\boldsymbol{W}^{x}_k(\boldsymbol{c})\boldsymbol{x}
+\boldsymbol{b}_k(\boldsymbol{c})
\right),\quad k=1,\ldots,L-1,
\tag{7}
$$

$$
V_{\mathrm{ICNN}}
=
\boldsymbol{w}^{z}(\boldsymbol{c})^{\mathsf T}\boldsymbol{z}_{L-1}
+\boldsymbol{w}^{x}(\boldsymbol{c})^{\mathsf T}\boldsymbol{x}
+b^{o}(\boldsymbol{c}).
\tag{8}
$$

隐层到隐层的权重及输出隐状态权重由 Softplus 映射得到：

$$
\boldsymbol{W}^{z}_k=\operatorname{softplus}
(\widetilde{\boldsymbol{W}}^{z}_k)\succeq\boldsymbol{0},
\qquad
\boldsymbol{w}^{z}=\operatorname{softplus}
(\widetilde{\boldsymbol{w}}^{z})\succeq\boldsymbol{0}.
\tag{9}
$$

由于 ReLU 是凸且单调非减函数，式 (6)--(9) 保证
\(V_{\mathrm{ICNN}}\) 关于 \(\boldsymbol{x}\) 为凸函数。需要指出的是，
\(\boldsymbol{W}^{x}_k\) 和 \(\boldsymbol{w}^{x}\) 不需要非负约束。

### 2) Positive-semidefinite quadratic branch

二次支路定义为

$$
V_{\mathrm{quad}}
=\frac{\gamma(\boldsymbol{c})}{2}
\left\|
\boldsymbol{B}(\boldsymbol{c})\boldsymbol{x}
+\boldsymbol{e}(\boldsymbol{c})
\right\|_2^2,
\qquad
\gamma=\operatorname{softplus}(\widetilde{\gamma})\geq 0.
\tag{10}
$$

其关于 \(\boldsymbol{x}\) 的 Hessian 为

$$
\nabla_{\boldsymbol{x}}^2V_{\mathrm{quad}}
=\gamma\boldsymbol{B}^{\mathsf T}\boldsymbol{B}\succeq\boldsymbol{0},
\tag{11}
$$

故该支路为凸函数。

### 3) Convexity of QICNN

ICNN 主干和半正定二次支路均关于 \(\boldsymbol{x}\) 为凸函数，二者之和仍保持凸性。
因此，由式 (5)、(9) 和 (11) 可得

$$
V_{\boldsymbol{\phi}}
\!\left(\boldsymbol{c},
\rho\boldsymbol{x}_1+(1-\rho)\boldsymbol{x}_2\right)
\leq
\rho V_{\boldsymbol{\phi}}(\boldsymbol{c},\boldsymbol{x}_1)
+(1-\rho)V_{\boldsymbol{\phi}}(\boldsymbol{c},\boldsymbol{x}_2),
\quad \rho\in[0,1].
\tag{12}
$$

当前 QICNN 采用条件网络宽度 \((64,64)\)、ICNN 宽度 \((32,32,16)\)，以及秩为
4 的半正定二次支路。

### 4) Cost normalization and surrogate fitting

设 IEEE 33 节点优化产生的最优总成本为 \(J_i^\star\)，训练集成本均值和尺度分别为
\(\mu_J\) 与 \(s_J>0\)。归一化目标为

$$
\bar{J}_i=\frac{J_i^\star-\mu_J}{s_J},
\qquad
\widehat{J}_i=
s_JV_{\boldsymbol{\phi}}(\boldsymbol{c}_i,\boldsymbol{x}_i)+\mu_J.
\tag{13}
$$

由于 \(s_J>0\)，反归一化不会改变凸性。QICNN 使用 Smooth-\(L_1\) 损失离线训练：

$$
\min_{\boldsymbol{\phi}}\;
\mathcal{L}_{\mathrm{QICNN}}(\boldsymbol{\phi})
=\frac{1}{N}\sum_{i=1}^{N}
\operatorname{Huber}_{1}
\left(
V_{\boldsymbol{\phi}}(\boldsymbol{c}_i,\boldsymbol{x}_i)-\bar{J}_i
\right),
\tag{14}
$$

其中

$$
\operatorname{Huber}_{1}(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|<1,\\
|r|-\frac{1}{2}, & |r|\geq1.
\end{cases}
\tag{15}
$$

## C. Progressive Decision-Focused Training

QICNN 训练完成后固定参数 \(\boldsymbol{\phi}\)，仅更新区间预测器参数
\(\boldsymbol{\theta}\)。该过程同时考虑统计预测质量与下游运行成本。

### 1) Prediction loss

对分位数 \(\tau\in(0,1)\)，Pinball 损失定义为

$$
\rho_{\tau}(e)=\max\{\tau e,(\tau-1)e\},
\qquad e=y-\widehat{q}_{\tau}.
\tag{16}
$$

上下界对应 \(\tau_l=0.05\) 和 \(\tau_u=0.95\)，因此 90% 区间的预测损失为

$$
\mathcal{L}_{\mathrm{pred}}(\boldsymbol{\theta})
=\frac{1}{N_b}\sum_{i=1}^{N_b}
\left[
\rho_{\tau_l}(y_i-\widehat{\ell}_i)
+\rho_{\tau_u}(y_i-\widehat{u}_i)
\right].
\tag{17}
$$

### 2) Decision loss

冻结的 QICNN 将预测区间映射为可微的运行成本代理：

$$
\mathcal{L}_{\mathrm{dec}}(\boldsymbol{\theta})
=\frac{1}{N_b}\sum_{i=1}^{N_b}
V_{\boldsymbol{\phi}}
\left(
\frac{y_i}{s_c},
\widetilde{\boldsymbol{x}}_i(\boldsymbol{\theta})
\right),
\tag{18}
$$

其中 \(\widetilde{\boldsymbol{x}}_i\) 由式 (4) 给出。虽然
\(\boldsymbol{\phi}\) 被固定，梯度仍通过 QICNN 对区间端点反向传播：

$$
\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\mathrm{dec}}
=
\frac{\partial\mathcal{L}_{\mathrm{dec}}}
{\partial\widetilde{\boldsymbol{x}}}
\frac{\partial\widetilde{\boldsymbol{x}}}
{\partial(\widehat{\ell},\widehat{u})}
\frac{\partial(\widehat{\ell},\widehat{u})}
{\partial\boldsymbol{\theta}}.
\tag{19}
$$

### 3) Progressive gradient fusion

直接对两个损失作标量加权容易受到量纲及梯度幅值差异影响。为此，本文分别计算

$$
\boldsymbol{g}_{p}=
\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\mathrm{pred}},
\qquad
\boldsymbol{g}_{d}=
\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\mathrm{dec}},
\tag{20}
$$

并构造单位方向

$$
\boldsymbol{u}_{p}=
\frac{\boldsymbol{g}_{p}}{\|\boldsymbol{g}_{p}\|_2},
\qquad
\boldsymbol{u}_{d}=
\frac{\boldsymbol{g}_{d}}{\|\boldsymbol{g}_{d}\|_2}.
\tag{21}
$$

第 \(t\) 个 epoch 的预测梯度权重按 Sigmoid 型规律衰减：

$$
\alpha_t=
\left[1+\exp(t-c)\right]^{-\kappa},
\qquad \kappa\geq0,
\tag{22}
$$

其中，\(c\) 为转折 epoch，\(\kappa\) 控制转换陡峭程度。当前默认值为
\(c=150\) 和 \(\kappa=1\)。最终合并梯度为

$$
\boldsymbol{g}_t=
\sqrt{\|\boldsymbol{g}_{p}\|_2\|\boldsymbol{g}_{d}\|_2}
\frac{
\alpha_t\boldsymbol{u}_{p}+\boldsymbol{u}_{d}
}{
\|\alpha_t\boldsymbol{u}_{p}+\boldsymbol{u}_{d}\|_2
}.
\tag{23}
$$

式 (23) 使用两梯度范数的几何平均值确定更新幅值，同时使用
\(\alpha_t\boldsymbol{u}_{p}+\boldsymbol{u}_{d}\) 确定更新方向。在训练初期
\(\alpha_t\approx1\)，预测梯度与决策梯度共同稳定区间学习；越过转折点后
\(\alpha_t\rightarrow0\)，更新方向逐步转向下游决策目标。该策略实现了从
prediction-aware learning 到 decision-focused learning 的平滑过渡。

数值实现中，若某一梯度范数小于 \(\varepsilon\)，则采用另一非零梯度；若两者均近似
为零，更新置零。随后执行梯度裁剪：

$$
\overline{\boldsymbol{g}}_t=
\boldsymbol{g}_t
\min\left\{1,\frac{G_{\max}}
{\|\boldsymbol{g}_t\|_2+\varepsilon}\right\},
\qquad G_{\max}=5.
\tag{24}
$$

模型参数由 AdamW 更新：

$$
\boldsymbol{\theta}_{t+1}
=\operatorname{AdamW}
\left(\boldsymbol{\theta}_{t},\overline{\boldsymbol{g}}_t\right).
\tag{25}
$$

默认初始学习率为 \(5\times10^{-4}\)，权重衰减为 \(10^{-5}\)。验证集仅采用式
(17) 的分位数损失进行模型选择；当验证指标连续 40 个 epoch 未改善时提前停止，
并恢复最佳参数。学习率则在验证损失连续 10 个 epoch 未改善时减半。

## D. Implementation Correspondence

上述数学表达与文件夹实现的对应关系如下：

| 数学模块 | 实现位置 |
|:---|:---|
| 区间可行化映射，式 (1)--(2) | `model.py` 中的 `LSTMQuantileIntervalNetwork` |
| 区间决策变量，式 (3)--(4) | `icnn.py::interval_to_decision` 与 `exp.py::soc_decision_loss` |
| QICNN 条件参数网络与凸支路，式 (5)--(12) | `icnn.py::ConditionalSOCICNN` 中的 condition network、ICNN 与 quadratic 部分 |
| QICNN 归一化及离线拟合，式 (13)--(15) | `icnn.py` 与 `icnn_test.ipynb` |
| 分位数及决策损失，式 (16)--(19) | `exp.py::interval_quantile_loss` 和 `soc_decision_loss` |
| 渐进式梯度融合，式 (20)--(24) | `exp.py::prediction_decay` 和 `merged_gradient_step` |
| 端到端优化及早停，式 (25) | `exp.py::train_end_to_end` |

需要说明的是，当前代码中的 `ConditionalSOCICNN` 与 `soc_decision_loss` 是历史遗留
标识符，其中 `ConditionalSOCICNN.forward` 目前仍计算 SOC 范数支路。本文所定义的
QICNN 仅包含条件参数网络、ICNN 主干和半正定二次支路，不包含 SOC 范数支路；因此，
正式复现实验时还需在实现中删除或禁用该支路并重新训练检查点。两个 QICNN 参数组可由
同一个线性输出层拼接生成，并在 `_unpack` 中分别解析为
\(\boldsymbol{\Theta}_{\mathrm{I}}\) 和 \(\boldsymbol{\Theta}_{\mathrm{Q}}\)。
