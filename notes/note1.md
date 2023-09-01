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
