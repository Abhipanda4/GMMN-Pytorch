# Generative Moment Matching Networks(GMMNs)
---------------------------------------------

This repo contains an implementation of Generative Moment Matching Networks(see <https://arxiv.org/pdf/1502.02761.pdf>) in pytorch(0.4)  

## Usage:
Clone the repo and change into the code directory. Since the pretrained weights are available, the outputs can be visualized using the command:  
``` python visualize.py --vis=gmmn ```

The autoencoder output can also be visualized using a different argument:   
``` python visualize.py --vis=autoencoder ```

In case the netowrks are to be trained from scratch, delete the models directory and run `python train.py`  

## Inferences:
Some conclusions drawn from this implementation are listed below:
* The network is very sensitive to network architecture and learning rate, Slight alterations in hyperparameters can affect results drastically.
* Increasing noise dimensionality(input to GMMN) adversely affects the quality of image generated.

## Note:
The implementation uses MNIST dataset and differs from the algorithm proposed in the paper in the following ways:
* The autoencoders are not trained layerwise and then fine tuned, all layers of autoencoder are trained simultaneoulsy using MSE loss and Adam optimizer.
* Batch size is same for both autoencoder and GMMN.
* No dropout was used while training autoencoder.

### References:  
<https://github.com/siddharth-agrawal/Generative-Moment-Matching-Networks>
