# paper: 'Improving robustness of softmax corss-entropy loss via inference information'


Inference region which inspires us to involve a margin-like inference information to SCE, resulting in a novel inference-softmax cross entropy (I-SCE) loss, which is intuitively appealing and interpretable. The inference information is a guarantee to both the inter-class separability and the improved generalization to adversarial examples.


## environment：

python 3.7

tensorflow 1.13


## code:

SCE.py  

run: python SCE.py  

softmax corss-entropy loss train ,test, and adversarial attack
 

I-SCE.py  

run: python I-SCE.py  

Inference softmax corss-entropy loss train ,test, and adversarial attack

