# 论文阅读《LEARN TO DESIGN THE HEURISTICS FOR VEHICLE ROUTING PROBLEM》

[TOC]

## 一、做了什么

​		这篇文章提出了一个学习局部搜索的方法，通过不断迭代解决VRP问题。迭代的过程是将已有解通过destroy算子删除一些节点，然后通过repair算子以最小cost按顺序从后面插入节点。采用了Graph Attention Network集成node和edge嵌入作为encoder，再以GRU作为解码器。通过时间在VRP上有很好的效果，并且可以解决中大规模的问题（400nodes）。



##  二、怎么做

### 2.1 Embeddings

​		8维的初始点嵌入和2维的初始边嵌入，然后进行标准化，在通过Graph Attention Network将边和点进一步集成。



### 2.2 The Encoder

​		GAT只是集成节点信息，而VRP的边上有着重要信息，所以引入了EGATE通过Attention同时集成边和点的信息。

​		首先全连接层扩展原始嵌入：
$$
\begin{aligned}
\tilde{n}_{i} &=W_{n} * n_{i} \\
\tilde{e}_{i, j} &=W_{\text {edge }} * e_{i, j}
\end{aligned}
$$
​		然后计算pair<i,j>的attention权重：
$$
\begin{aligned}
h_{\text {concat }, i j} &=\operatorname{concat}\left(\tilde{n}_{i}, \tilde{n}_{j}, \tilde{e}_{i, j}\right) \\
w_{i, j} &=\text { LeakyReLU }\left(W_{L} * h_{\text {concat }, i j}\right) \\
\tilde{w}_{i, j} &=\frac{\exp \left(w_{i, j}\right)}{\sum_{j} \exp \left(w_{i, j}\right)}
\end{aligned}
$$
​		最后输出每个节点更新后的嵌入：
$$
n_{\mathrm{EGATE}, i}=\tilde{n}_{i}+\sum_{j} \tilde{w}_{i, j} \otimes \tilde{n}_{j}
$$
​		Encoder遵循GAT的屏蔽注意原则，EGATE通过选择性地对其进行掩蔽，允许在信息传播中排除某些边缘嵌入。同时EGATE也可以通过叠加为多层。最后进入mean-pooling层得到solution的嵌入

![](https://cdn.mathpix.com/snip/images/rtLaIniINH8B_u9WeltpQKCaIk9_BS0etCahnxKW-fY.original.fullsize.png)



### 2.3 The Decoder

​		启发式运算符每次都会迭代生成一个有序列表如公式7，公式7可以转化为公式8，所以采用RNN作为解码器。以solution embedding作为初始输入，然后解码采用attention机制输出概率分布。
$$
\mathcal{H}=\pi\left(\left[\eta_{1}, \eta_{2}, \ldots, \eta_{M}\right]\right)
$$

$$
\begin{aligned}
\mathcal{H} &=\pi\left(\eta_{1}\right) \times \pi\left(\eta_{2} \mid\left[\eta_{1}\right]\right) \ldots \times \pi\left(\eta_{M} \mid\left[\eta_{1}, \ldots, \eta_{M-1}\right]\right) \\
&=\pi\left(\eta_{1}\right) \prod_{m=2}^{M} \pi\left(\eta_{m} \mid\left[\eta_{1}, \ldots, \eta_{m-1}\right]\right)
\end{aligned}
$$

![](https://cdn.mathpix.com/snip/images/MP80_0WTZVZuAYpLf_ESZGkJXpt4fNc0w-iL4-nWBhk.original.fullsize.png)

### 2.4 Train the network

​		VRP成本函数是总行驶距离和车辆成本的总和，reward是时间步t的cost减去时间步t-1的cost。
$$
\begin{aligned}
\operatorname{Cost}_{V R P}^{(t)} &=\text {Distance}^{(t)}+C \times K^{(t)} \\
r^{(t)} &=\operatorname{cost}_{V R P}^{(t)}-\operatorname{Cost}_{V R P}^{(t-1)}
\end{aligned}
$$
​		value network是`is a two-layered feed-forward neural network, where the ﬁrst layer is a dense layer with ReLU activation and the second layer is a linear one. `

​		首先（1）计算Advantages as the TD error：
$$
\delta_{\mathrm{TD}}^{(t)} \leftarrow r^{(t)}+\gamma \hat{v}\left(E n c^{(t)}, \phi\right)-\hat{v}\left(E n c^{(t-1)}, \phi\right)
$$
​		（2）训练critic network：
$$
\phi \leftarrow \phi+\alpha_{\phi} \delta_{\mathrm{TD}}^{(t)} \nabla_{\phi} \hat{v}\left(E n c^{(t)}, \phi\right)
$$
​		（3） 通过 clipped surrogate objective Proximal Policy Optimization (PPO) 方法训练actor，` rt(θ) is the ratio of new policy over old policy`：
$$
L^{C L I P}(\theta)=\hat{\mathbb{E}}_{t}\left[\min \left(r_{t}(\theta) \delta_{\mathrm{TD}}^{(t)}, \operatorname{clip}\left(r_{t}(\theta), 1-\epsilon, 1+\epsilon\right) \delta_{\mathrm{TD}}^{(t)}\right)\right]
$$
​		如果满足以下条件才更新，其中`Rnd`是[0,1]之间的随机数，$T^{(t)}$是模拟退火（SA）$T^{(t)}=\alpha_T T^{(t-1)}$：
$$
\text { Distance }^{(t)}<\text { Distance }^{(t-1)}-T^{(t)} * \log (R n d)
$$
​		算法流程：

![](https://cdn.mathpix.com/snip/images/Y-wAcb6Mnu4sZvCCcO7bTHxfsxfDiqjcvfCCGgdf3SE.original.fullsize.png)



## 三、结果

​		将本文模型分别和三种启发式算法、AM模型对比结果如下：

![](https://cdn.mathpix.com/snip/images/pgQCgHshnD5lz7PlU28WYlnz5ViYr4HXVTqIY9ZRqO8.original.fullsize.png)

​		对于400nodes的中大规模的问题的效果如下：

![](https://cdn.mathpix.com/snip/images/m9kozh_VwqAHuZtBghEUMjq97uah_yUwH_-451Hert4.original.fullsize.png)



### 需要看的其他资料

1、The Graph Attention Network (GAT) ( Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017. )

2、 clipped surrogate objective Proximal Policy Optimization (PPO) method （ John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. ）