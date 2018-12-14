#Efficient Neural Network Implementation of the Universemachine 

*Authors: Chun-Hao To, Ethan Nadler*


## Requirements

We recommend using python3 and a virtual env. See instructions [here](https://cs230-stanford.github.io/project-starter-code.html).

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Given an n-body simulation, predict the stellar mass function of the same universe.

##Build dataset
```bash
python build_dataset.py --data_dir data/subdir --output_dir data/output
```

## Start 

3. __Train__ your experiment. Simply run
```
python train.py --data_dir data/experiment1.1/datavector_original_split/ --model_dir ./experiments/test/ --restore_file best
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the validation set.

4. Randomforest. simply run
