# Literature Review: Symbolic Regression Methods

## Introduction

Symbolic regression is the task of discovering mathematical expressions that best describe relationships in data. Unlike traditional regression methods that fit parameters to a predetermined functional form, symbolic regression searches the space of possible expressions to find the underlying mathematical structure. This document reviews major approaches to symbolic regression and positions JAXSR within this landscape.

## Traditional Approaches

### Genetic Programming (GP)

Genetic Programming, pioneered by Koza (1992) [1], was one of the earliest successful approaches to symbolic regression. GP evolves a population of expression trees through mutation, crossover, and selection operations.

**Key Works:**
- Koza, J.R. (1992). Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press. [1]
- Schmidt, M., & Lipson, H. (2009). Distilling free-form natural laws from experimental data. Science, 324(5923), 81-85. [2]

**Limitations:**
- Computationally expensive, requiring many function evaluations
- Solutions can be bloated (overly complex)
- Difficult to incorporate domain constraints
- Non-deterministic results

### ALAMO (Automated Learning of Algebraic Models for Optimization)

ALAMO, developed by Cozad et al. (2014) [3], takes a fundamentally different approach using best subset selection from a library of basis functions. This work directly inspired JAXSR.

**Key Features:**
- Predefined library of candidate basis functions
- Uses mixed-integer optimization for feature selection
- Information criteria (BIC, AIC) for model complexity control
- Adaptive sampling for iterative model improvement
- Support for physical constraints

**Key Works:**
- Cozad, A., Sahinidis, N. V., & Miller, D. C. (2014). Learning surrogate models for simulation-based optimization. AIChE Journal, 60(6), 2211-2227. [3]
- Wilson, Z. T., & Sahinidis, N. V. (2017). The ALAMO approach to machine learning. Computers & Chemical Engineering, 106, 785-795. [4]

**Limitations:**
- Commercial software (not open source)
- Relies on external MINLP solvers
- Limited to predefined basis function types

### SISSO (Sure Independence Screening and Sparsifying Operator)

SISSO, developed by Ouyang et al. (2018) [5], combines sure independence screening with compressed sensing for descriptor identification in materials science.

**Key Features:**
- Iterative feature construction through algebraic operations
- Compressed sensing for sparse solutions
- Multi-objective optimization for accuracy vs complexity
- Designed for materials science applications

**Key Works:**
- Ouyang, R., Curtarolo, S., Ahmetcik, E., Scheffler, M., & Ghiringhelli, L. M. (2018). SISSO: A compressed-sensing method for identifying the best low-dimensional descriptor in an immensity of offered candidates. Physical Review Materials, 2(8), 083802. [5]
- Bartel, C. J., et al. (2019). New tolerance factor to predict the stability of perovskite oxides and halides. Science Advances, 5(2), eaav0693. [6]

**Limitations:**
- Specialized for materials science descriptors
- Complex implementation
- Computationally expensive feature space expansion

### SyMANTIC (Symbolic regression via Mutual information-based ANalysis of TIme Complexity)

SyMANTIC, developed by Muthyala et al. (2025) [15], is a symbolic regression algorithm that combines information-theoretic feature selection with sparse regression to efficiently discover interpretable mathematical expressions.

**Key Features:**
- Mutual information-based feature selection for efficient candidate screening
- Adaptive feature expansion to explore large candidate spaces (10^5 to 10^10+)
- Recursive ℓ₀-based sparse regression for parsimonious models
- Pareto optimization for complexity vs accuracy trade-offs
- Built on PyTorch with GPU acceleration

**Key Works:**
- Muthyala, M. R., Sorourifar, F., Peng, Y., & Paulson, J. A. (2025). SyMANTIC: An Efficient Symbolic Regression Method for Interpretable and Parsimonious Model Discovery in Science and Beyond. arXiv preprint arXiv:2502.03367. [15]

**Limitations:**
- Requires PyTorch backend
- Relies on mutual information estimates which can be noisy for small datasets
- Feature expansion strategy may miss important interactions not captured by pairwise mutual information

### PySR (Python Symbolic Regression)

PySR by Cranmer (2023) [7] uses a modern multi-population genetic algorithm with simplification and combines Julia's speed with Python's ecosystem.

**Key Features:**
- Multi-population evolutionary search
- Automatic simplification
- Pareto optimization for complexity vs accuracy
- GPU support through Julia
- Active development and modern interface

**Key Works:**
- Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. arXiv preprint arXiv:2305.01582. [7]

**Limitations:**
- Requires Julia installation
- Non-deterministic results from evolutionary search
- Less interpretable search process

### AI Feynman

AI Feynman by Udrescu & Tegmark (2020) [8] uses neural networks combined with recursive decomposition strategies inspired by physics.

**Key Features:**
- Neural network-guided search
- Dimensional analysis
- Symmetry detection
- Recursive problem decomposition

**Key Works:**
- Udrescu, S. M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. Science Advances, 6(16), eaay2631. [8]
- Udrescu, S. M., et al. (2020). AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity. NeurIPS 2020. [9]

**Limitations:**
- Complex multi-stage pipeline
- Computationally expensive
- Specialized for physics problems

### Neural ODE + Symbolic Regression Pipelines

A promising hybrid approach combines neural differential equations with symbolic regression to discover governing equations from data. Cranmer et al. (2020) [16] demonstrated that training a neural ODE on observational data, then applying symbolic regression (e.g., PySR) to the learned vector field, can recover interpretable differential equations. This approach is implemented in the Diffrax library (Kidger, 2021) [17] using JAX/Equinox.

**Key Features:**
- Three-stage pipeline: neural approximation, symbolic extraction, fine-tuning
- Neural ODE learns smooth vector field from noisy data
- Symbolic regression discovers closed-form expressions from the learned field
- Gradient-based fine-tuning of constants in discovered expressions
- JAX-native implementation via Diffrax and Equinox

**Key Works:**
- Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D., & Ho, S. (2020). Discovering Symbolic Models from Deep Learning with Inductive Biases. NeurIPS 2020. [16]
- Kidger, P. (2021). On Neural Differential Equations. PhD Thesis, University of Oxford. Diffrax documentation: https://docs.kidger.site/diffrax/examples/symbolic_regression/ [17]

**Limitations:**
- Two-step process may propagate errors from neural approximation to symbolic extraction
- Requires choosing neural architecture and symbolic regression method separately
- Computationally expensive training of neural ODE

### Deep Learning Approaches

Recent work has explored using transformers and neural networks for symbolic regression.

**Key Works:**
- Biggio, L., et al. (2021). Neural Symbolic Regression that Scales. ICML 2021. [10]
- Kamienny, P. A., et al. (2022). End-to-end symbolic regression with transformers. NeurIPS 2022. [11]
- Valipour, M., et al. (2021). SymbolicGPT: A Generative Transformer Model for Symbolic Regression. arXiv. [12]

**Characteristics:**
- Learn to predict expressions from data patterns
- Fast inference after training
- Can generalize to new problems
- Require large training datasets

### Sparse Regression Methods

Classical sparse regression methods like LASSO and Elastic Net have been adapted for symbolic regression through feature engineering.

**Key Works:**
- Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamics. PNAS, 113(15), 3932-3937. (SINDy) [13]
- Champion, K., et al. (2019). Data-driven discovery of coordinates and governing equations. PNAS, 116(45), 22445-22451. [14]

## Comparison with JAXSR

### Design Philosophy

JAXSR takes an approach most similar to ALAMO but with several key differences:

| Feature | JAXSR | ALAMO | SISSO | PySR | SyMANTIC |
|---------|-------|-------|-------|------|----------|
| Open Source | Yes | No | Yes | Yes | Yes |
| Backend | JAX | GAMS | Python | Julia | PyTorch |
| GPU Support | Yes | No | Limited | Yes | Yes |
| Deterministic | Yes | Yes | Yes | No | Yes |
| Physical Constraints | Yes | Yes | Limited | Limited | Limited |
| Adaptive Sampling | Yes | Yes | No | No | No |
| Information Criteria | Yes | Yes | No | No | No |
| Pareto Optimization | Yes | No | Yes | Yes | Yes |

### JAXSR Advantages

1. **Fully Open Source**: Unlike ALAMO, JAXSR is MIT-licensed and freely available.

2. **JAX-Based Implementation**:
   - Automatic differentiation for potential gradient-based optimization
   - JIT compilation for performance
   - Native GPU/TPU support
   - Composable transformations (vmap, pmap)

3. **Multiple Selection Strategies**:
   - Greedy forward/backward selection (fast)
   - Exhaustive search (exact for small problems)
   - LASSO path screening (efficient for large libraries)
   - Easy to extend with new methods

4. **Flexible Constraint Handling**:
   - Output bounds
   - Monotonicity constraints
   - Convexity/concavity
   - Coefficient sign constraints
   - Linear constraints on coefficients
   - Fixed coefficients (known terms)

5. **Modern Python Interface**:
   - Scikit-learn compatible API
   - Method chaining for library construction
   - Easy serialization to JSON
   - Export to SymPy, LaTeX, pure NumPy

6. **Adaptive Sampling**:
   - Multiple sampling strategies (uncertainty, leverage, space-filling)
   - Iterative model improvement
   - Integration with experimental workflows

7. **Interpretability Focus**:
   - Pareto front visualization
   - Complexity scoring system
   - Expression simplification
   - Clear model summaries

### Limitations Compared to Other Methods

1. **Fixed Basis Library**: Unlike GP/PySR, JAXSR cannot discover truly novel functional forms outside the predefined library.

2. **Scalability**: Exhaustive search is limited to small problems; greedy methods may miss global optima.

3. **No Neural Network Integration**: Unlike AI Feynman or transformer methods, JAXSR doesn't leverage neural networks for search guidance.

## Future Directions

Based on the literature, several directions could enhance JAXSR:

1. **Differentiable Selection**: Using gradient-based optimization for feature selection through relaxations (e.g., L0 regularization, Gumbel-softmax).

2. **Neural-Guided Search**: Incorporating neural networks to predict promising basis functions.

3. **Active Learning Integration**: More sophisticated acquisition functions for adaptive sampling.

4. **Multi-Objective Optimization**: Explicit Pareto optimization for multiple objectives (accuracy, complexity, constraint satisfaction).

5. **Dimensional Analysis**: Automatic enforcement of dimensional consistency.

6. **Ensemble Methods**: Combining multiple models for improved predictions and uncertainty quantification.

## Conclusion

JAXSR fills an important gap in the symbolic regression landscape by providing an open-source, JAX-based implementation of ALAMO-style best subset selection. Its combination of flexible basis libraries, multiple selection strategies, constraint handling, and modern Python interface makes it well-suited for scientific and engineering applications where interpretability and domain knowledge incorporation are essential.

## References

[1] Koza, J.R. (1992). Genetic Programming: On the Programming of Computers by Means of Natural Selection. MIT Press. ISBN: 978-0-262-11170-6. https://mitpress.mit.edu/9780262527910/genetic-programming/

[2] Schmidt, M., & Lipson, H. (2009). Distilling free-form natural laws from experimental data. Science, 324(5923), 81-85. https://doi.org/10.1126/science.1165893

[3] Cozad, A., Sahinidis, N. V., & Miller, D. C. (2014). Learning surrogate models for simulation-based optimization. AIChE Journal, 60(6), 2211-2227. https://doi.org/10.1002/aic.14418

[4] Wilson, Z. T., & Sahinidis, N. V. (2017). The ALAMO approach to machine learning. Computers & Chemical Engineering, 106, 785-795. https://doi.org/10.1016/j.compchemeng.2017.02.010

[5] Ouyang, R., Curtarolo, S., Ahmetcik, E., Scheffler, M., & Ghiringhelli, L. M. (2018). SISSO: A compressed-sensing method for identifying the best low-dimensional descriptor in an immensity of offered candidates. Physical Review Materials, 2(8), 083802. https://doi.org/10.1103/physrevmaterials.2.083802

[6] Bartel, C. J., et al. (2019). New tolerance factor to predict the stability of perovskite oxides and halides. Science Advances, 5(2), eaav0693. https://doi.org/10.1126/sciadv.aav0693

[7] Cranmer, M. (2023). Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl. arXiv preprint arXiv:2305.01582. https://arxiv.org/abs/2305.01582

[8] Udrescu, S. M., & Tegmark, M. (2020). AI Feynman: A physics-inspired method for symbolic regression. Science Advances, 6(16), eaay2631. https://doi.org/10.1126/sciadv.aay2631

[9] Udrescu, S. M., et al. (2020). AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity. NeurIPS 2020. https://arxiv.org/abs/2006.10782

[10] Biggio, L., et al. (2021). Neural Symbolic Regression that Scales. ICML 2021. https://arxiv.org/abs/2106.06427

[11] Kamienny, P. A., et al. (2022). End-to-end symbolic regression with transformers. NeurIPS 2022. https://arxiv.org/abs/2204.10532

[12] Valipour, M., et al. (2021). SymbolicGPT: A Generative Transformer Model for Symbolic Regression. arXiv preprint arXiv:2106.14131. https://arxiv.org/abs/2106.14131

[13] Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamics. PNAS, 113(15), 3932-3937. https://doi.org/10.1073/pnas.1517384113

[14] Champion, K., et al. (2019). Data-driven discovery of coordinates and governing equations. PNAS, 116(45), 22445-22451. https://doi.org/10.1073/pnas.1906995116

[15] Muthyala, M. R., Sorourifar, F., Peng, Y., & Paulson, J. A. (2025). SyMANTIC: An Efficient Symbolic Regression Method for Interpretable and Parsimonious Model Discovery in Science and Beyond. arXiv preprint arXiv:2502.03367. https://arxiv.org/abs/2502.03367

[16] Cranmer, M., Sanchez-Gonzalez, A., Battaglia, P., Xu, R., Cranmer, K., Spergel, D., & Ho, S. (2020). Discovering Symbolic Models from Deep Learning with Inductive Biases. NeurIPS 2020. https://arxiv.org/abs/2006.11287

[17] Kidger, P. (2021). On Neural Differential Equations. PhD Thesis, University of Oxford. https://doi.org/10.5287/ora-r5kybrozr. Diffrax: https://docs.kidger.site/diffrax/
