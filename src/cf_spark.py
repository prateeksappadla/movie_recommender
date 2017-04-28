from pyspark import SparkContext, SparkConf
import numpy as np
import pandas as pd

appName = "Recommender"
master = "local[2]"

conf = SparkConf().setAppName(appName).setMaster(master)
sc = SparkContext(conf=conf)

data = sc.textFile("./data/ml-latest-small/ratings.csv")

data_processed = data.map(lambda x: x.split(','))
