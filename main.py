print("imported all modules...")
import yaml
from pandas import read_csv
import pytorch_lightning  as pl
import argparse
import numpy as np
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from learning_modules.t5_tuning_modules import T5Tokenizer,TweetDataset,LoggingCallback,T5FineTuner
print("successfully imported all modules")
with open('config/configuration.yaml',"r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

print(config)
config["args_dict"]["learning_rate"] = float(config["args_dict"]["learning_rate"])
config["args_dict"]["adam_epsilon"] = float(config["args_dict"]["adam_epsilon"])

if __name__ == "__main__":
        # load Dataset
        print("training data import")
        train_df = read_csv("datasets/tweet_dataset.csv")
        train, validate, test = np.split(train_df.sample(frac=1), [int(.6 * len(train_df)), int(.8 * len(train_df))])
        train.to_csv("datasets/train.csv")
        validate.to_csv("datasets/val.csv")
        test.to_csv("datasets/test.csv")

        #token initializer
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

        #load dataset
        dataset = TweetDataset(tokenizer, data_dir='datasets/', type_path='val')
        print("Length of dataset is :",len(dataset))

        data = dataset[69]
        print(tokenizer.decode(data['source_ids']))
        print(tokenizer.decode(data['target_ids']))

        args = argparse.Namespace(**config["args_dict"])

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=args.output_dir, monitor="val_loss", mode="min", save_top_k=5
        )

        train_params = dict(
                accumulate_grad_batches=args.gradient_accumulation_steps,
                gpus=args.n_gpu,
                max_epochs=args.num_train_epochs,
                precision=16 if args.fp_16 else 32,
                amp_level=args.opt_level,
                gradient_clip_val=args.max_grad_norm,
                checkpoint_callback=checkpoint_callback,
                callbacks=[LoggingCallback()]
        )
        print(config["args_dict"])

        # Model Intializer
        model = T5FineTuner(args)

        #trainer
        trainer = pl.Trainer(**train_params)

        #fitmodel
        trainer.fit(model)

        dataset = TweetDataset(tokenizer, data_dir='datasets/', type_path='val')
        loader = DataLoader(dataset, batch_size=32, num_workers=4)







