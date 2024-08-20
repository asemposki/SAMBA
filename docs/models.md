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

: samba.models.Models
