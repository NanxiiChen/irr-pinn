# IRR-PINNs: Physics-Informed Neural Networks with Irreversibility Constraints


## Overview

This repository implements the IRR-PINNs (Irreversibility-constrained Physics-Informed Neural Networks) framework, a novel approach specifically designed to address physics problems involving irreversible processes. By explicitly incorporating **irreversibility constraints** into the loss function, IRR-PINNs guide the neural network to produce solutions that respect the inherent directionality of physical laws, leading to more accurate and stable predictions. 

### Irreversibility Constraint Formulation

We form the irreversibility constraint as an additional regularization term in the loss function of PINNs, which penalizes violations of the expected directionality of physical processes.

For **forward irreversible processes** (where the physical quantity should only increase):
```python
loss_irr = mean(ReLU(-dphi/dt))
```
For **backward irreversible processes** (where the physical quantity should only decrease):
```python
loss_irr = mean(ReLU(dphi/dt))
```
where:
- `dphi/dt` represents the time derivative of the physical quantity $\phi$
- `ReLU` activation function ensures that only violations of the irreversibility constraint contribute to the loss


Furthermore, this constraint can be generalized to any inequality-based governing equations. This simple yet powerful formulation significantly improves solution accuracy and stability for irreversible physical processes by enforcing the fundamental directionality constraints inherent in these systems.

## Examples

We demonstrate the effectiveness of the IRR-PINN framework through four challenging applications: corrosion modeling, ice melting simulation, fracture mechanics, and combustion dynamics. Each example showcases how irreversibility constraints lead to more physically consistent and accurate solutions compared to traditional PINN approaches.

### Corrosion

![Corrosion](./figures/corrosion2d1pit-radius.png)
![Corrosion](./figures/corrosion2d1pit-sol.png)
![Corrosion](./figures/corrosion2d1pit-error-log.png)
![Corrosion](./figures/corrosion2d1pit-irrloss-log.png)

### Ice melting

![Ice Melting](./figures/icemelting-radius.png)
![Ice Melting](./figures/icemelting-sol.png)
![Ice Melting](./figures/icemelting-error-log.png)


### Fracture

![Fracture](./figures/fracture-t-disp.png)
![Fracture](./figures/fracture-sol.png)

### Combustion

![Combustion](./figures/combustion-solution.png)
![Combustion](./figures/combustion-error-log.png)
![Combustion](./figures/combustion-irrloss-log.png)