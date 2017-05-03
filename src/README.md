## Running the scripts

Download the dataset from https://grouplens.org/datasets/movielens/. Unzip the dataset and place it in desired folder. We will be using only the ratings.csv file of the dataset. 

**usercf_20m.py**: Nearest Neighbors Collaborative Filtering

usage: usercf_20m.py [-h] [--k K] [--filename FILENAME]

    optional arguments:
      -h, --help           show this help message and exit
      --k K                Number of top similar users to use for making
                           predictions
      --filename FILENAME  Path to input file

  
-----------------------------------------------------------------------

**latent_factor_20m.py**

usage: latent_factor_20m.py [-h] [--k K] [--lr LR] [--lambdar LAMBDAR]
                            [--epochs EPOCHS] [--filename FILENAME]

Latent Factor Model

    optional arguments:
      -h, --help           show this help message and exit
      --k K                Number of latent factors
      --lr LR              Learning rate for Stochastic Gradient Descent
      --lambdar LAMBDAR    Regularization strength
      --epochs EPOCHS      Number of epochs for Stochastic Gradient 

Descent
  --filename FILENAME  Path to input file
------------------------------------------------------------------------

Need Spark to run the following scripts

**usercf_spark.py** 

usage: $SPARK_HOME/bin/spark-submit --master "url of spark master node" usercf_spark.py [-h] [--k K] [--filename FILENAME] [--master MASTER]

User based Collaborative Filtering on Spark

    optional arguments:
      -h, --help           show this help message and exit
      --k K                Number of top similar users to use for making
                           predictions
      --filename FILENAME  Path to input file
      --master MASTER      URL of spark master node

  
-----------------------------------------------------------------------
**latent_factor_spark.py**

usage: $SPARK_HOME/bin/spark-submit --master "url of spark master node" latent_factor_spark.py [-h] [--k K] [--lr LR] [--lambdar LAMBDAR]
                              [--epochs EPOCHS] [--filename FILENAME]
                              [--master MASTER]

Latent Factor Model

    optional arguments:
      -h, --help           show this help message and exit
      --k K                Number of latent factors
      --lr LR              Learning rate for Stochastic Gradient Descent
      --lambdar LAMBDAR    Regularization strength
      --epochs EPOCHS      Number of epochs for Stochastic Gradient Descent
      --filename FILENAME  Path to input file
      --master MASTER      URL of spark master node

  
