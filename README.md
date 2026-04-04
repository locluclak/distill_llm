# BayesGPT2: Bayesian Distillation for GPT-2

This project implements Bayesian distillation for GPT-2 using Stochastic Gradient Langevin Dynamics (SGLD) and Stochastic Weight Averaging-Gaussian (SWAG).

## SWAG (Stochastic Weight Averaging-Gaussian)

The `SWAG` class in `src/optimizer.py` provides an implementation of the SWAG method for approximate Bayesian inference. It tracks the running mean and squared mean of model weights during the final stages of training to approximate the posterior distribution as a Gaussian.

### Features
- **Mean Tracking**: Computes the Stochastic Weight Averaging (SWA) solution.
- **Uncertainty Estimation**: Approximates the posterior covariance using a combination of diagonal variance and a low-rank matrix.
- **Efficient Sampling**: Allows sampling from the approximated posterior for Bayesian Model Averaging (BMA).

### How to Use for Distillation

A complete implementation for SWAG-based distillation is provided in `train_swag.py`.

1. **Configure SWAG**:
   In `config.yaml`, set the SWAG parameters:
   ```yaml
   swag:
     enabled: true
     max_num_models: 20
     interval: 100  # Collect weights every 100 steps
     scale: 0.5     # Scaling for posterior sampling
   ```

2. **Run Training**:
   Execute the SWAG-specific training script:
   ```bash
   python train_swag.py
   ```
   This will:
   - Perform distillation using the AdamW optimizer.
   - Collect weights periodically using `SWAGCallback`.
   - Evaluate the model using the last weights, the SWA mean weights, and a SWAG posterior sample.
   - Save the SWA mean model and the full SWAG state.

3. **Inference with SWAG**:
   You can load the SWAG state and sample weights for ensembling:
   ```python
   from src.optimizer import SWAG
   swag_model = SWAG(model)
   swag_model.load("path/to/swag_state.pt")
   
   # Sample and evaluate
   swag_model.sample(scale=0.5)
   outputs = model(**inputs)
   ```

## SGLD (Stochastic Gradient Langevin Dynamics)

The original SGLD-based distillation is available in `train.py`. It uses a custom `SGLD` optimizer to sample from the student's posterior during training.

### How to Use
```bash
python train.py
```
