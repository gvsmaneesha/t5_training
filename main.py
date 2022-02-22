print("imported all modules...")
import yaml
import pytorch_lightning  as pl
import argparse
import numpy as np
import textwrap
from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from learning_modules.t5_tuning_modules import T5Tokenizer,TweetDataset,LoggingCallback,T5FineTuner
from learning_modules.data_utils import create_dataset
print("successfully imported all modules")
with open('config/configuration.yaml',"r") as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

print(config)
config["args_dict"]["learning_rate"] = float(config["args_dict"]["learning_rate"])
config["args_dict"]["adam_epsilon"] = float(config["args_dict"]["adam_epsilon"])

if __name__ == "__main__":
        create_dataset("tweet_dataset.csv")

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

        print("Training model")
        #trainer
        trainer = pl.Trainer(**train_params)
        print("Fitting the model")
        #fitmodel
        trainer.fit(model)

        dataset = TweetDataset(tokenizer, data_dir='datasets/', type_path='val')
        loader = DataLoader(dataset, batch_size=32, num_workers=4)








