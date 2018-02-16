# kassandra
Tensor prediction using convolutional decomposition.

Given two training tensors (input tensor X and output tensor Y), Kassandra learns a convolutional mapping between X and Y such that X and Y share the same *sparse* convolutional decomposition.
The size of the convolution filters defines the dependencies of neighboring data for the predicition.
If the output tensor Y is a time/space shifted version of the input tensor X, this is akin to learning a time/space forecasting model.

This is done in a two step fashion:
  1) Decompose X as a finite sum of convolutions of synthesis filters (F) with their respective activations (H)
    X = \sum F_k* H_k
  2) Find a decomposition of Y that shares the same activation filters as in 1), such that Y is a finite sum of convolutions of analysis filters with the activations H
    Y = \sum G_k* H_k
    
The *draft* of the mathematical formulation can be found in the white paper folder.
