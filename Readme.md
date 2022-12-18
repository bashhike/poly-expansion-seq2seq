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