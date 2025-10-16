# Models

The models in SAMBA are two expansions of a toy model, where the full toy model is given by

$$
 F(g) = \int_{-\infty}^{\infty} dx~ e^{-\frac{x^{2}}{2} - g^{2} x^{4}} = \frac{e^{\frac{1}{32 g^{2}}}}{2 \sqrt{2}g} K_{\frac{1}{4}}\left(\frac{1}{32 g^{2}} \right).
$$

The two expansions are limits taken at $g = 0$ and $g = \infty$:

$$
F_{s}^{N_s}(g) = \sum_{k=0}^{N_{s}} s_{k} g^{k},
$$

and 

$$
F_{l}^{N_{l}}(g) = \frac{1}{\sqrt{g}} \sum_{k=0}^{N_{l}} l_{k} g^{-k},
$$

with coefficients given as:

$$
s_{2k} = \frac{\sqrt{2} \Gamma{(2k + 1/2)}}{k!} (-4)^{k},~~~~~s_{2k + 1} = 0
$$

and

$$
l_{k} = \frac{\Gamma{\left(\frac{k}{2} + \frac{1}{4}\right)}}{2k!} \left(-\frac{1}{2}\right)^{k}.
$$

We also include models for the uncertainties of each expansion, given in the uninformative limit, for the small-$g$ expansion, as

$$
\sigma_{N_s}(g)= \Gamma(N_s+3) g^{N_s + 2} \bar{c},
$$

if $N_s$ is even, and

$$
\sigma_{N_s}(g)= \Gamma(N_s+2) g^{N_s+1} \bar{c},
$$

if $N_s$ is odd. For the large-$g$ limit,

$$
\sigma_{N_l}(g)=\frac{1}{\Gamma(N_l+2)} \frac{1}{g^{N_l+3/2}} \bar{d}.
$$

We also devise expressions for the informative limit, for the small-$g$ expansion, as

$$
\sigma_{N_s}(g)= \Gamma(N_s/2+1) (4g)^{N_s + 2} \bar{c}, 
$$

if $N_s$ is even, and

$$
\sigma_{N_s}(g)= \Gamma(N_s/2+1/2) (4g)^{N_s+1} \bar{c},
$$

if $N_s$ is odd. For the large-$g$ limit,

$$
\sigma_{N_l}(g)=\left(\frac{1}{4g}\right)^{N_l + 3/2} \frac{1}{\Gamma(N_l/2+3/2)} \bar{d}.
$$

:: samba.models
