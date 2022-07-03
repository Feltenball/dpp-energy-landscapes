# dpp-energy-landscapes
Applying DPPs to solve problems on energy landscapes.

Consider a DPP $\mathcal{X}$ over a ground set $\mathcal{Y}$ of position vectors $\mathbf{x}_i$ of dimension $D$ in a specified domain. Let the kernel of this DPP be $L$. Indeed DPPs admit a Gram decomposition $L = B^T B$, which can further be decomposed into a quality-diversity decomposition $L = q_i \phi_i^T \phi_j q_j$. Call $S_{ij} = \phi_i^T \phi_j$ the similarity matrix.

We then have, for a set $A$, $$\mathbb{P}_L(A) \propto \prod_{i \in A} (q_i^2) \det(S_A)$$

Therefore $S_{ij}$ is the underlying diversity model. 
