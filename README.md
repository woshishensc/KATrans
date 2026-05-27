
## KATrans: A Kinematic-Aware Hybrid Framework for Trend-Consistent Wind Power Forecasting
> **Note:** This paper is currently under review.
This is the PyTorch implementation of KATrans, a kinematic-aware hybrid framework for trend-consistent wind power forecasting.
## Abstract
Accurate wind power forecasting (WPF) is essential for modern high-renewable grids. 
While transformer-based models excel at capturing long-range temporal dependencies, 
they typically treat continuous physical measurements as isolated semantic tokens, lacking kinematic awareness. 
Consequently, these purely data-driven methods tend to over-smooth high-frequency local dynamics, 
leading to severe prediction lags and directional errors during critical ramping events. 
In this paper, we propose KATrans, a novel kinematic-aware hybrid framework designed for trend-consistent WPF. 
We first reconstruct a kinematic phase space by introducing power velocity and acceleration, 
explicitly modeling the turbine’s inertial responses and ramping intensities. 
To integrate these physical priors without amplifying high-frequency noise, 
KATrans employs a filter-then-reason cascaded architecture: A Bidirectional LSTM operates as a local dynamic filter 
to smooth stochastic volatility while preserving physical continuity, and a Transformer module functions as a global 
context reasoner to capture inter-day weather patterns. Extensive experiments across three real-world datasets demonstrate 
that KATrans substantially reduces numerical errors (e.g., RMSE and MAPE) while notably improving trend capture accuracy 
during extreme fluctuations. Ultimately, this physics-informed paradigm provides a highly reliable and interpretable 
forecasting solution for secure power system management.
## Requirements
* PyTorch 1.9+
* Python 3.8+
* NumPy 1.21+
* Pandas 1.3+
* scikit-learn 1.0+
## Datasets
The framework is evaluated on three real-world wind power datasets:
- **Dataset 1:** [Name / source] <!-- 请补充具体数据集名称和链接 -->
- **Dataset 2:** [Name / source] <!-- 请补充具体数据集名称和链接 -->
- **Dataset 3:** [Name / source] <!-- 请补充具体数据集名称和链接 -->
## Results
The classification results for our proposed network and other competing architectures are as follows:
## Citation
If you find this code useful, please cite us in your paper.
> @article{KATrans,\
　 title={KATrans: A Kinematic-Aware Hybrid Framework for Trend-Consistent Wind Power Forecasting},\
　 author={Qiuping Bi, Cheng Shen, Jing Zhang, Zhongyi Li},\
　 year={2026},\
　 note={Under Review,\
}
