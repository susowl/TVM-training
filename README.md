# CNN training using TVM compiler
This repo conains simple example how to use TVM compiler for CNN training.

Code realizes a simple LeNet-5 training with SGD solver. The whole training pipline realiseg in single TVM graph:

            batch<-----TrainDB
              |           |
        | -->CNN          |
        |     |           |
        |    loss <--labels
        |     |
        |    grads
        |     |
        |    momentum = momentum factor*momentum + (1.0-momentum_factor)*grads
        |----network_params = network_params - LR*momentum

Training process ends after 1 epoch. 

To start process simply run:
    python lenet_tvm.py

Expected output:
TVM:0.44873267 | accuracy:0.90625 | second per iteration:1.5032396518446336
