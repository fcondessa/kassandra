# kassandra
Tensor prediction using convolutional decomposition.

Given two tensors (X and Y) Kassandra learns the mapping between them using a collection of filters such that the mapping between X and Y such that Y can be predicted from X.
This is done in a two step fashion:
  1) Decompose X as a finite sum of convolutions of synthesis filters (F) with their respective activations (H)
    X = \sum F_k* H_k
  2) Find a decomposition of Y that shares the same activation filters as in 1), such that Y is a finite sum of convolutions of analysis filters with the activations H
    Y = \sum G_k* H_k
    
