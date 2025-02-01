## HMAMP: Designing Highly Potent Antimicrobial Peptides Using a Hypervolume-Driven Multi-Objective Deep Generative Model


HMAMP introduced a hypervolume-driven multi-objective generative model designed to simultaneously optimize multiple attributes of 
antimicrobial peptides (AMPs), addressing the challenge of safe clinical usage of peptide antibiotics. HMAMP combined multiple 
discriminator generative adversarial networks, reinforcement learning, gradient update strategies based on the hypervolume concept, non-
dominated sorting to achieve the Pareto front, and a rapid screening mechanism based on knee-point identification. HMAMP not only 
generated effective and diverse candidates that could balance multiple conflicting attributes but also offered an advanced rapid 
candidate screening process. The final computational simulations and experimental analyses demonstrated that the candidates possessed 
ideal physicochemical properties and functional structures, suggesting that HMAMP could significantly advance the discovery of peptide 
antibiotics and their safe clinical application.


## Requirement:

CUDA 10.1 
python 3.7 
pytorch 1.5.1 


## Steps:

1.Pre-training a prior generator：PTG.py

2.Training HMAMP framework: PolicyGradient.py

3.Genearting candidates: test.py

4.Obtaining Pareto Front: Hypervolume.py

5.Knee-based screening: knee_point.py




