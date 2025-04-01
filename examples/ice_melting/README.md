# Ice Melting

## Model description

Governing equations:
$$
\frac{\partial \phi}{\partial t} = M\left(
    \Delta \phi - \frac{F'(\phi)}{\epsilon^2}
\right) - \lambda \frac{\sqrt{2F(\phi)}}{\epsilon}
$$
where $\phi$ is the phase field variable, $F(\phi)=\dfrac{1}{4}(\phi^2-1)^2$.

Let domain be $\Omega = [-50, 50] \times [-50, 50] \times [-50, 50]$ and time interval be $[0, 5]$.

Initial condition:
$$
\phi(x, y, z, t=0) =\tanh\left(
    \frac{R_0 - R}{\sqrt{2}\epsilon}
\right)
$$
where $R_0 = 35$ and $R = \sqrt{x^2 + y^2 + z^2}$.

Boundary condition:
$$
\frac{\partial \phi}{\partial \mathbf{n}} = 0
$$

Other parameters:
- $\lambda = 5$ 
- $N=64$ 
- $h=\dfrac{100}{N}$ 
- $\epsilon=\dfrac{6h}{2\sqrt{2}\tanh^{-1}(0.9)}$ 
- $M = 0.1$ 


The analytical solution is given by:
$$
R(t) = R_0 - \lambda t
$$

## Results

![Ice Melting](../../figures/icemelting-radius.png)
![Ice Melting](../../figures/icemelting-sol.png)
![Ice Melting](../../figures/icemelting-error-log.png)
