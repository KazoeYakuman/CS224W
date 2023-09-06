# CS224W

---

### Heterogeneous Graphs

$G = (V,E,R,T)$

- Nodes with node types $v_i \in V$

- Edges with relation types $(v_i,r,v_j) \in E$

- Node type $T(v_i)$

- Relation type $r\in R$

### Node Degrees

Avg. degree (Undirected graph): $\overline k = \lang k \rang = \frac 1 N \sum_{i=1}^N k_i = \frac {2E}{N}$

Avg. degree (Directed graph): $\overline {k^{in}} = \overline {k^{out}} = \frac EN$

---

### Node-Level Features

- Node degree: The number of neighboring nodes (Treat all neighbors equally)

- Node centrality: Take the node importance into account
  
  - Eigenvector centrality
  
  - Betweenness centrality
  
  - Closeness centrality

- Clustering coefficient

- Graphlets

##### Eigenvector Centrality

A node $v$ is important if surrounded by important neighboring nodes $u\in N(v)$.

$$
c_v = \frac 1 \lambda \sum_{u\in N(v)} c_u
$$

Rewrite the equation in the matrix form: 

$$
\lambda c = Ac
$$

 Where $A$ is the adjacency matrix, $c$ is the centrality vector, $\lambda$ is the largest eigenvalue of $A$. ($\lambda$ is always positive and unique by $Perron-Frobenius\  Theorem$).

##### Betweenness Centrality

A node is important if it lies on many shortest paths between other nodes.

$$
c_v = \sum_{s \ne v \ne t} \frac {\#(shortest\ paths\ between\ s\
 and\ t\ that\ contain\ v)
}{\#shortest\ paths\ between\ s \ and \ t}
$$

##### Closeness Centrality

A node is important if it has small shortest path lengths to all other nodes.

$$
c_v = \frac {1}{\sum_{u\ne v}shortest\ path\ length\ between\ u\ and\ v}
$$

##### Clustering Coefficient

Measures how connected $v$'s neighboring nodes are.

$$
e_v = \frac {\#(edges\ among\ neighboring\ nodes)}{\tbinom{k_v}{2}}
$$

Where $k_v$ is $\#(v's\ neighbors)$.

##### Graphlets: pre-specified subgraphs

Degree counts #(edges) that a node touches. Clustering coefficient counts #(triangles) that a node touches. Graphlet Degree Vector (GDV) counts #(graphlets) that a node touches.

> **Induced subgraph** is another graph, formed from a subset of vertices and all of the edges connecting the verices in the subset.

> **Graph Isomorphism**: Two graphs which contain the same number of nodes connected in the same way are said to be isomorphic. For example, a pentagons and a pentagram is isomorphic.

Graphlets: Rooted connected induced non-isomorphic subgraphs.

---

### Link-Level Features

- Distance-based feature: Shortest-path distance between two nodes

- Local neighborhood overlap

- Global neighborhood overlap

##### Local Neighborhood Overlap

Common neighboring nodes shared between two nodes $v_1$ and $v_2$.

- Common neighbors: $| N(v_1) \cap N(v_2)|$

- Jaccard's coefficient: $\frac {|N(v_1) \cap N(v_2)}{|N(v_1) \cup N(v_2)|}$

- Adamic-Adar index: $\sum_{u \in N(v_1) \cap N(v_2)} \frac {1}{\log(k_u)}$

##### Global Neighborhood Overlap

**Limitation of local neighborhood features:** Metric is always zero if the two nodes do not have any neighbors in common. However, the two nodes may still potentially be connected in the future.

**Katz index:** Count the number of walks of all lengths between a given pair of nodes.

Let $P_{uv}^{(k)}$ = #walks between two nodes, we will show  that

$$
P^{(k)} = A^k
$$

Obviously, $P_{uv}^{(1)}=A_{uv}$. Then let's compute $P_{uv}^{(2)}$.

First, compute #walks of length $1$ between each of $u$'s neighbor and $v$. Second, sum up these #walks across $u$'s neighbors.

$$
P_{uv}^{(2)} = \sum_{i}A_{ui} * P_{iv}^{(1)} = 
\sum_{i}A_{ui}*A_{iv} = A_{uv}^2
$$

By induction we have $P^{(k)} = A^k$.

**Katz index** between $v_1$ and $v_2$ is calculated as

$$
S_{v_1,v_2} = \sum_{l=1}^{\infin} \beta^l A^l_{v_1v_2} = (I-\beta A)^{-1}-I
$$

---

### Graph-Level Features

We want features that characterize the structure of an entire graph.

##### Kernel methods

Design kernels instead of feature vectors.

- Kernel $K(G,G') \in \mathbb R$ measures similarity b/w data.

- Kernel matrix $K = (K(G,G'))_{G,G'}$ must always be positive semidefinite (i.e., has positive eigenvalues)

- There exists a feature representation $\phi(\cdot)$ such that $K(G,G') = \phi(G)^T\phi(G')$.

- Once the kernel is defined, off-the-shelf ML model, such as kernel SVM, can be used to make prediction.

##### Graph Kernels

Measure similarity between two graphs.

- Graphlet Kernel

- Weisfeiler-Lehman Kernel

- Other kernels are also proposed
  
  - Random-walk kernel
  
  - Shortest-path graph kernel
  
  - And any more...

**Goal:** Design graph feature vector $\phi(G)$.

**Key Idea:** Bag-of-Words (BoW) for a graph. Both Graphlet Kernel and Weisfeiler-Lehman (WL) Kernel use _Bag-of-_* representation of gtaph.

---

##### Graph-Level Graphlet Features

**Key Idea:** Count number of different graphlets in a graph.

> The graphlets here do not need to be connected and are not rooted.

Given graph $G$, and a graphlet list $\mathcal G_k = (\mathcal g_1,\mathcal g_2, \dots, \mathcal g_{n_k})$, define the graphlet count vector $\mathcal f_G$ as

$$
(f_G)_i = \#(\mathcal g_i \subseteq G) \ \ for\ i=1,2,\dots,n_k
$$

Given two graphs, $G$ and $G'$, graphlet kernel is computed as

$$
K(G,G') = f_G^Tf_{G'}
$$

> If $G$ and $G'$ have different sizes, that will greatly skew the value.

We normalize each feature vector by

$$
h_G = \frac {f_G}{Sum(f_G)} \\ \  \\
K(G,G') = h_G^Th_{G'}
$$

**Limitations: Counting graphlets is expensive!**

- Counting size-$k$ graphlets for a graph with size $n$ by enumeration takes $n^k$

- This is unavoidable in the worst-case since subgraph isomorphism test is *NP-hard*

- If a graph's node degree is bounded by $d$, an $O(nd^{k-1})$ algorithm exists to count all the graphlets of size $k$

---

##### Weisfeiler-Lehman Kernel

**Goal:** Design an efficient graph feature descriptor $\phi(G)$

**Idea:** Use neighborhood structure to iteratively enrich node vocabulary. 

**Algorithm: Color Refinement**

Given a graph $G$ with a set of nodes $V$.

- Assign an initial color $c^{(0)}(v)$ to each node $v$.

- Iteratively refine node colors by
  
  $$
  c^{(k+1)}(v) = HASH(\{c^{(k)}(v), \{c^{(k)}(u)\}_{u \in N(v)}\})
  $$
  
  where $HASH$ maps different inputs to different colors.

- After $K$ steps of color refinement, $c^{(K)}(v)$ summarizes the structure of $K$-hop neighborhood.

After color refinement, WL kernel counts number of nodes with a given color. The WL kernel value is computed by the inner product of the color count vectors.

**Complexity**

- WL kernel is computationally efficient. The time complexity for color refinement at each step is linear in #(edges), since it involves aggregating neighboring colors.

- When computing a kernel value, only colors appeared in the two graphs need to be tracked. Thus, #(colors) is at most the total number of nodes.

- Counting colors takes linear-time w.r.t. #(nodes).

- In total, time complexity is linear in #(edges).

---

### Node Embedding

**Task: Map nodes into an embedding space**

- Similarity of embeddings between nodes indicates their similarity in the network.

- Encode network information.

- Potentially used for many downstream predictions.

##### Learning Node Embeddings

- Encoder maps from nodes to embeddings

- Define a node similarity function. (i.e., a measure of similarity in the original network)

- Decoder maps from embeddings to the similarith score.

- Optimize the parameters of the encoder so that
  
  $$
  similarity(u,v) \approx \mathbf Z_v^T \mathbf Z_u
  $$

**Encoder:** maps each node to a low-dimensional vector

$$
ENC(v) = \mathbf Z_v
$$

**Similarity function:** specifies how the relationships in vector space map to the relationships in the original network

$$
similarity(u,v) \approx \mathbf Z_v^T \mathbf Z_u
$$

##### "Shallow" Encoding

Simplest encoding approach: Encoder is just an embedding-lookup.

$$
ENC(v) = \mathbf Z \cdot v
$$

Where $\mathbf Z \in \mathbb R^{d\times |\mathcal V|}$ (matrix, each column is a node embedding [what we learn/optimize]), $v \in \mathbb I^{|\mathcal V|}$ (indicator vector, all zeros except a one in column indicating node $v$).

##### Random Walk Approaches for Node Embeddings

###### Notation

- Vector $\mathbf z_u$: The embedding of node $u$ (what we aim to find).

- Probability $P(v|\mathbf z_u)$: The (predicted) probability of visiting node $v$ on random walks starting from node $u$.

- Non-linear functions
  
  - Softmax function: Turns vector of $K$ real values (model prediction) into $K$ probabilities that sum to $1$
    
    $$
    \sigma(\mathbf z)[i] = \frac {e^{\mathbf z[i]}}{\sum_{j=1}^{K}e^{\mathbf z[j]}}
    $$
  
  - Sigmoid function: S-shaped function that turns real values into the range of $(0,1)$.
    
    $$
    S(x) = \frac {1}{1+e^{-x}}
    $$

> **Random Walk:** Given a graph and a starting point, we select a neighbor of it at random, and move to this neighbor; then we select a neighbor of this point at random, and move to it, etc. The (random) sequence of points visited this way is a random walk on the graph.

###### Random-Walk Embeddings

- Estimate probability of visiting node $v$ on a random walk starting from node $u$ using some random walk strategy $R$.

- Optimize embeddings to encode these random walk statistics.

**Advantage**

- Expressivity: Flexible stochasitic definition of node similarity that incorporates both local and higher-order neighborhood information. 
  
  > Idea: If random walk starting from node $u$ visits $v$ with high probability, $u$ and $v$ are similar (high-order multi-hop information)

- Efficiency: Do not need to consider all node pairs when training; only need to consider pairs that co-occur on random walks.

##### Unsupervised Feature Learning

**Intuition:** Find embedding of nodes in $d-$dimensional space that preserves similarity.

**Idea:** Learn node embedding such that nearby nodes are close together in the network.

###### Feature Learning as Optimization

Given $G = (V,E)$, our goal is to learn a mapping $f: u \rightarrow \mathbb R^d:f(u) = \mathbf z_u$.

**Log-likelihood objective:**

$$
max_f \sum_{u\in V} \log P(N_R(u)|\mathbf z_u)
$$

$N_R(u)$ is the neighborhood of node $u$ by strategy $R$.

Given node $u$, we want to learn feature representations that are predictive of the nodes in its random walk neighborhood $N_R(u)$.

###### Random Walk Optimization

- Run short fixed-length random walks starting from each node $u$ in the graph using some random walk strategy $R$.

- For each node $u$ collect $N_R(u)$, the multiset of nodes visited on random walks starting from $u$.

- Optimize embeddings according to: Given node $u$, predict its neighbors $N_R(u)$.

Optimize embeddings $\mathbf z_u$ to maximize the likelihood of random walk co-occurrences.

$$
\mathcal L = \sum_{u \in V} \sum_{v\in N_R(u)} - \log (P(v|\mathbf z_u))
$$

Parameterize $P(v|\mathbf z_u)$ using softmax:

$$
P(v|\mathbf z_u) = \frac {\exp(\mathbf z_u^T \mathbf z_v)}
{\sum_{n\in V} \exp (\mathbf z_u^T \mathbf z_n)}
$$

Putting it all together:

$$
\mathcal L = \sum_{u \in V} \sum_{v\in N_R(u)}-
\log(\frac{\exp(\mathbf z_u^T \mathbf z_v)}{\sum_{n\in V} \exp(\mathbf 
z_u^T \mathbf z_n)})
$$

- Sum over all nodes $u$.

- Sum over nodes $v$ seen on random walks starting from $u$.

- Predicted probability of $u$ and $v$ co-occuring on random walk.

*Optimizing random walk embeddings = Finding embeddings $\mathbb z_u$ that minimize $\mathcal L$*.

**Negative sampling**

$$
\log (\frac {\exp(\mathbf z_u^T \mathbf z_v)}{\sum_{n\in V} 
\exp(\mathbf z_u^T\mathbf z_n)}) \approx 
\log(\sigma(\mathbf z_u^T \mathbf z_v)) - \sum_{i=1}^{k} 
\log (\sigma (\mathbf z_u^T \mathbf z_{n_i})), n_i \sim P_V
$$

> **Why is the approximation valid:** Technically, this is a different objective. But Negative Sampling is a form of Noise Contrastive Estimation (NCE) which approx maximizes the log probability of softmax. New formulation corresponds to using a logistic regression (sigmoid func.) to distinguish the target node $v$ from nodes $n_i$ sampled from background distribution $P_v$.

Instead of normalizing w.r.t. all nodes, just normalize against $k$ random "negative samples" $n_i$.

Negative sampling allows for quick likelihood calculation.

Sample $k$ negative nodes $n_i$ each with probability proportional to its degree.

**Two conditions for $k$ (# negative samples):**

- Higher $k$ gives more robust estimates

- Higher $k$ corresponds to higher bias on negative events

- In practice $k$ ranges from $5$ to $20$

> **Can negative sample be any node or only the nodes not on the walk?** 
> 
> People often use any nodes (for efficiency). However, the most "correct" way is to use nodes not on the walk.

Objective function:

$$
\mathcal L = \sum_{u\in V} \sum_{v\in N_R(u)} - \log (P(v|\mathbf z_u))
$$



###### Gradient Descent

- Initialize $z_u$ at some randomized value for all nodes $u$.

- Iterate until convergence:
  
  - For all $u$, compute the derivative $\frac {\partial \mathcal L}{ \partial z_u}$.
  
  - For all $u$, make a step in reverse direction of derivative: $z_u \leftarrow z_u - \eta \frac {\partial \mathcal L}{ \partial z_u}$, where $\eta$ is learning rate.

**How should we randomly walk?**

Simplest idea: Just run fixed-length, unbiased random walks starting from each node.

The issue is that such notion of similarity is too constrained.

---

### Node2vec

**Goal:** Embed nodes with similar network neighborhoods close in the feature space.

We frame this goal as a maximum likelihood optimization problem, independent to the downstream prediction task.

**Key observation:** Flexible notion of network neighborhood $N_R(u)$ of node $u$ leads to rich node embeddings.

Develop biased $2^{nd}$ order random walk $R$ to generate network neighborhood $N_R(u)$ of node $u$.
