# AlphaQuant: LLM-Driven Automated Robust Feature Engineering for Quantitative Finance

## System Prompt

```plaintext
You are an expert in financial metrics and feature engineering. Generate PyTorch feature extraction code for a financial dataset.
You aim to engineer predictive features by proposing interestingly novel statistical risk-return indicators. USE CRITICAL THINKING!

You are deeply familiar with statistical time-series features from literature, and encouraged to draw inspiration from publications.
Use the knowledge from examples and inspiration from academic literature to propose new financial analysis feature implementations.

Each feature should be unique, interpretable, and useful for predicting Sharpe Ratios. Avoid redundancy with the examples provided.
When features are applied to the past log-returns of assets, obtained indicators should help predicting assets' future sharpe-ratios.
```

## User Prompt

```plaintext
Example features (PyTorch code snippets) are below. You should implement novel more advanced (better) features that are complementary.

{top_k_examples}

Generate {num_features} new feature extraction functions. Give concise names to the functions that summarize the feature extracted.
It must take log_returns (torch.Tensor) as the only input parameter that doesn't have any default value defined in the function body.
Prefer using tensor operations. Return only the PyTorch implementation of each feature without any additional markdown or comments.

Avoid extracting following features that have been previously tried but were redundant, come up with better ones. THINK OUTSIDE THE BOX!

{eliminated_list}

Below are the execution errors encountered in previous attempts. Avoid similar mistakes in future implementations:
{previous_errors}
```

## References

1. **AlphaQuant**: LLM-Driven Automated Robust Feature Engineering for Quantitative Finance. In *Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025)*, Singapore. Available at SSRN: [https://ssrn.com/abstract=5124841](https://ssrn.com/abstract=5124841).

2. **AlphaPortfolio**: Discovery of Portfolio Optimization and Allocation Methods Using LLMs. In *Proceedings of the Thirteenth International Conference on Learning Representations (ICLR 2025)*, Singapore. Available at SSRN: [https://ssrn.com/abstract=5118317](https://ssrn.com/abstract=5118317).

3. **AlphaSharpe**: LLM-Driven Discovery of Robust Risk-Adjusted Metrics. Submitted to the *European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD 2025)*, Porto. Preprint, arXiv:2502.00029. Available at SSRN: [https://ssrn.com/abstract=5111141](https://ssrn.com/abstract=5111141).

4. **AlphaLoss**: LLM-Driven Evolution of Robust, Interpretable, and Multi-Objective Portfolio Optimization Loss Functions. Submitted to the *European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD 2025)*, Porto. Available at SSRN: [https://ssrn.com/abstract=5263279](https://ssrn.com/abstract=5263279).

5. **AlphaEvolve**: A coding agent for scientific and algorithmic discovery. White paper by Alexander Novikov et al., June 16, 2025. Preprint, arXiv:2506.13131. Available at SSRN: [https://arxiv.org/abs/2506.13131](https://arxiv.org/abs/2506.13131).

6. **R&D-Agent-Quant**: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization. Preprint, arXiv:2505.15155v1. Available at: [https://arxiv.org/abs/2505.15155v1](https://arxiv.org/abs/2505.15155v1).
