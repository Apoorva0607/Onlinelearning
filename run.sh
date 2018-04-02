#!/bin/bash

echo "========================"

echo "Implementation"

echo "========================"

echo "SIMPLE PERCEPTRON HYPERPARAMETERS"


python3.5 simpleperceptron_hyper.py 
echo
echo "SIMPLE PERCEPTRON TESTING"

python3.5 simple_perceptron.py
echo
echo "DYNAMIC PERCEPTRON HYPERPARAMETERS"

python3.5 dynamic_perceptron_hyperp.py
echo
echo "DYNAMIC PERCEPTRON TESTING"

python3.5 dynamic_perceptron.py
echo
echo "MARGIN PERCEPTRON HYPERPARAMETERS"

python3.5 margin_perceptron_hyper.py
echo
echo "MARGIN PERCEPTRON TESTING"

python3.5 margin_perceptron.py
echo
echo "AVERAGE PERCEPTRON HYPERPARAMETERS"

python3.5 average_perceptron_hyperp.py
echo
echo "AVERAGE PERCEPTRON TESTING"

python3.5 average_perceptron.py
echo
echo "AGGRESSIVE MARGIN PERCEPTRON HYPERPARAMETERS"

python3.5 aggressive_margin_perceptron_hyperp.py
echo
echo "AGGRESSIVE MARGIN PERCEPTRON TESTING"

python3.5 aggressive_margin_perceptron.py


echo
echo
