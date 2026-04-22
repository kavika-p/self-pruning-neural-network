# Self-Pruning Neural Network Report

## 1. Introduction
In this project, we implemented a self-pruning neural network where each weight has an associated gate. The gate determines whether a weight is active or pruned. A sparsity regularization term is added to the loss function to encourage the network to remove unnecessary connections during training.



## 2. Sparsity Loss Explanation
We used L1 regularization on the gate values. Since gate values are constrained between 0 and 1 (after applying a sigmoid function), minimizing their sum encourages many of them to approach zero. This effectively removes less important weights, resulting in a sparse network.



## 3. Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.001  | 29.52       | 3.95         |
| 0.01   | 30.73       | 21.51        |
| 0.1    | 31.10       | 49.01        |



## 4. Observations
- For small λ (0.001), very little pruning occurs and most weights remain active.
- For medium λ (0.01), the network begins to prune unnecessary weights, achieving moderate sparsity.
- For large λ (0.1), the network becomes highly sparse, removing nearly half of the weights.
- As λ increases, sparsity increases significantly.
- Interestingly, accuracy also improves slightly as λ increases. This suggests that pruning helps remove redundant connections and improves generalization.



## 5. Conclusion
The experiment demonstrates that increasing λ leads to higher sparsity in the network. The model successfully learns to prune less important weights during training.

Additionally, pruning acts as a form of regularization. By removing unnecessary connections, the model avoids overfitting and can even achieve better accuracy. This shows that a well-balanced sparsity penalty can reduce model complexity while maintaining or improving performance.