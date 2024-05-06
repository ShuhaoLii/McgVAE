## **Seeing the Forest for the Trees: Road-Level Insights Assisted Lane-Level Traffic Prediction**

This is the original pytorch implementation of **M**ulti-**c**hannel **g**raph-structured **V**ariational **A**uto**E**ncoder

model（McgVAE）


## Requirements
- python 3.8
- matplotlib
- numpy
- scipy
- pandas
- torch
- argparse


## Data Preparation

In the Datasets directory, we have stored processed data and raw, unprocessed data for three datasets. The folder naming convention is: 

dataset_name_type_node_count.

If you wish to process data from scratch or need to process your own data, you can run the following commands：

```python
cd Data_process
python generate_training_data.py
```

## Train Commands

You can set various experiment hyperparameters in the args.py file.

After configuring, use the following command for training and prediction:

```
python main.py
```

