#Program instructions:

Run 'run_program.py' to generate output.

Depending on the number of iterations the program can take a long time to run.

Sample output files are included. `50_iteration_results.txt` contains a subset of the predictions made, as well as the RMSE at each iteration and the final prediction accuracy.

For comparison, I also included `2_iteration_results.txt`, which contains the same information, and shows that the prediction gets much more accurate with the number of iterations our BackPropNetwork is allowed to make.

#Reasoning for the network architecture values chosen:

## Number of hidden nodes:
I chose a small number of hidden nodes because the task was not too computatinally complex.
As I increased the number of hidden nodes the execution time of my program increased very quickly.

## Initialization of weights:
I chose to use small initial weights because larger values can drive layer 1 nodes to saturation quickly. Increasing training time.

## Frequency of weight updates:
I updated weights for each feedForward iteration (each input data)

## Choice of learning rate:
I chose a somewhat large value for my learning rate because I wanted to speed up the training time. There is a lot of data to train on.

## Momentum value
I chose a small momentum value as to not strongly bias future weight trainings.





#Dependencies
Python 2.7.12
`pip install numpy`
