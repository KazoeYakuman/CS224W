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
