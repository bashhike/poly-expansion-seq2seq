Given a single variable equation in a compact form, the task is to expand it to a more standardized form (e.g. `ax**2 + bx + c`). The dataset contains a list of such expressions, some examples of the same are as follows: 
```
(7-3*z)*(-5*z-9)=15*z**2-8*z-63
-9*s**2=-9*s**2
(2-2*n)*(n-1)=-2*n**2+4*n-2
-2*k*(5*k-9)=-10*k**2+18*k
(3*cos(c)-19)*(7*cos(c)+13)=21*cos(c)**2-94*cos(c)-247
```

The task is similar to a machine translation task. Machine translation tasks have been extensively studied and solved using a seq-2-seq models using the encoder-decoder architecture. So, it makes sense to try out some seq-2-seq models in order to solve it. 
Following were the results obtained from a GRU model using the encoder-decoder structure. The model was trained on google colab using a learning rate of `0.001` with a batch size of 1536 for 10 epochs. 

1. GRU Encoder-Decoder: 0.7414

More details regarding the model can be found in the `network.txt` file. 

In order to run the code, Please install the dependencies present in the `requirements.txt` file first. The run the following files: 

```bash 
python preprocessing.py
```
This will prepare the training and testing data, as well as generate the vocabulary to be used for training and testing. 
Training and testing of the trained models can be performed using the `train.py` file. Please take a look at the command line params for more details. 

```bash 
python train.py --job test --vocab_file vocab.pkl --model_weights baseline_gru.pt
```

All the models used are present in the `models.py` directory. The pretrained directory contains weights for a pretrained GRU + Attention model which can be tested on the data using the `train.py` script. 
