from pandas import read_csv
import numpy as np
def create_dataset(file):
  print("training data import")
  train_df = read_csv("datasets/"+str(file))
  train_df = train_df[["textID","text","selected_text","sentiment"]]
  train_df = train_df[train_df.isnull().sum(1)==0]
  train, validate, test = np.split(train_df.sample(frac=1), [int(.6 * len(train_df)), int(.8 * len(train_df))])
  train.to_csv("datasets/train.csv",index=False)
  validate.to_csv("datasets/val.csv",index=False)
  test.to_csv("datasets/test.csv",index=False)
  print("Created dataset !!")
  return train,test,validate