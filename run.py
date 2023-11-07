"""Train Knowledge Graph embeddings for link prediction."""

import argparse
import json
import logging
import os
import csv
import torch as T
import torch.optim
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizer
from utils.train import get_savedir, avg_both, format_metrics, count_params
from datetime import datetime
DATA_PATH = "./data/"

parser = argparse.ArgumentParser(
    description="Knowledge Graph Embedding"
)

parser.add_argument(
        "--model_dir"
        )

parser.add_argument(
    "--dataset", default="nations", choices=["diabetes","fb15k237", "nations", "yago_10"],
    help="Knowledge Graph dataset"
)

parser.add_argument(
    "--model", default="ComplEx", choices=all_models, help="Knowledge Graph embedding model"
)

parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)

parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)

parser.add_argument(
    "--optimizer", default="Adagrad",
    help="Optimizer"
)

parser.add_argument(
    "--max_epochs", default=1500, type=int, help="Maximum number of epochs to train for"
)

parser.add_argument(
    "--patience", default=20, type=int, help="Number of epochs before early stopping"
)

parser.add_argument(
    "--valid", default=1, type=float, help="Number of epochs before validation"
)

parser.add_argument(
    "--rank", default=32, type=int, help="Embedding dimension"
)

parser.add_argument(
    "--batch_size", default=100, type=int, help="Batch size"
)

parser.add_argument(
    "--neg_sample_size", default=-1, type=int, help="Negative sample size, -1 to not use negative sampling"
)

parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)

parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)

parser.add_argument(
    "--learning_rate", default=0.01, type=float, help="Learning rate"
)

parser.add_argument(
    "--gamma", default=1.0, type=float, help="Margin for distance-based losses"
)

parser.add_argument(
    "--bias", default="learn", type=str, choices=["constant", "learn", "none"], help="Bias type (none for no bias)"
)

parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)

parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)

parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--lm", action="store_true", help="Use Language model"
)
parser.add_argument(
    "--cuda_n", default=-1, type=int, help="Cuda core number"
)
parser.add_argument('--summary', action='store_true')

def train(args):
    save_dir = get_savedir(args.model, args.dataset)
    REMOVE_REL = False
    #REMOVE_REL = True if args.dataset == "diabetes" else False

    if REMOVE_REL:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        # file logger
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=os.path.join(save_dir, "train.log")
        )
        # stdout logger
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        logging.info("Saving logs in: {}".format(save_dir))

    # create dataset
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()
    
    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson)
    
    # load data
    logging.info("\t " + str(dataset.get_shape()))
    train_examples = dataset.get_examples("train")
    valid_examples = dataset.get_examples("valid")
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # create model
    model = getattr(models, args.model)(args)
    device = "cuda:"+str(args.cuda_n) if args.cuda_n >= 0 else "cpu"
    cuda1 = T.device(device)
    if not REMOVE_REL:
        total = count_params(model)
        logging.info("Total number of parameters {}".format(total))
        model.to(cuda1)
    else:
        if args.model is None:
            exit()

        save_dir = args.model_dir
        logging.info("\t Loading best model saved at {}".format(save_dir))
        model.load_state_dict(T.load(os.path.join(save_dir, "model.pt")))
        model.to(cuda1)
        model.eval()
        logging.info("finishing loading model")
        '''
        print("before trimming")
        # Validation metrics
        valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
        logging.info(format_metrics(valid_metrics, split="valid"))
        # Test metrics
        test_metrics = avg_both(*model.compute_metrics(test_examples, filters))
        logging.info(format_metrics(test_metrics, split="test"))
        '''
        print("after trimming")
        # Validation metrics
        valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, remove_rel = REMOVE_REL))
        logging.info(format_metrics(valid_metrics, split="valid"))
        # Test metrics
        test_metrics = avg_both(*model.compute_metrics(test_examples, filters, remove_rel = REMOVE_REL))
        logging.info(format_metrics(test_metrics, split="test"))
        exit()
    
    # get optimizer
    regularizer = getattr(regularizers, args.regularizer)(args.reg)
    optim_method = getattr(T.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(cuda1, model, regularizer, optim_method, args.batch_size, args.neg_sample_size,
                            bool(args.double_neg), args)
    counter = 0
    best_mrr = None
    best_epoch = None
    logging.info("\t Start training")
    for step in range(args.max_epochs):

        # Train step
        model.train()
        train_loss = optimizer.epoch(train_examples)
        logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

        # Valid step
        model.eval()
        #valid_loss = optimizer.calculate_valid_loss(valid_examples)
        #logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

        if (step + 1) % args.valid == 0:
            valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters))
            #valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, remove_rel = True))
            logging.info(format_metrics(valid_metrics, split="valid"))

            valid_mrr = valid_metrics["MRR"]
            if not best_mrr or valid_mrr > best_mrr:
                best_mrr = valid_mrr
                counter = 0
                best_epoch = step
                logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                T.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                model.to(cuda1)
            else:
                counter += 1
                if counter == args.patience:
                    logging.info("\t Early stopping")
                    break
                elif counter == args.patience // 2:
                    pass
                    #logging.info("\t Reducing learning rate")
                    #optimizer.reduce_lr()

    logging.info("\t Optimization finished")
    if not best_mrr:
        T.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
    else:
        logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
        model.load_state_dict(T.load(os.path.join(save_dir, "model.pt")))
    model.to(cuda1)
    model.eval()

    # Validation metrics
    valid_metrics = avg_both(*model.compute_metrics(valid_examples, filters, remove_rel = REMOVE_REL))
    logging.info(format_metrics(valid_metrics, split="valid"))

    # Test metrics
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, remove_rel = REMOVE_REL))
    logging.info(format_metrics(test_metrics, split="test"))
    if args.summary:
        summary_dict["best_epoch"] = best_epoch
        for key in valid_metrics.keys():
            summary_dict["Vaild "+key] = valid_metrics[key]
        for key in test_metrics.keys():
            summary_dict["Test "+key] = test_metrics[key]
    if args.model == 'sFourDE':
        model.save_embedding()
        print("done")

if __name__ == "__main__":
    T.set_printoptions(precision = 3)

    start_time = datetime.now()

    args = parser.parse_args()
    summary_dict = vars(args)
    if args.model == "AttH":
        args.multi_c = True

    train(args)
    if args.summary:
        summary_dict["Execution time"] = datetime.now() - start_time
        if os.path.exists('/content/drive/MyDrive/Colab Notebooks/Thesis/KGEmb1New/Test_summary_KGEmb.csv'):
            with open('/content/drive/MyDrive/Colab Notebooks/Thesis/KGEmb1New/Test_summary_KGEmb.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(summary_dict.values())
        else:
            with open('/content/drive/MyDrive/Colab Notebooks/Thesis/KGEmb1New/Test_summary_KGEmb.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(summary_dict.keys())
                writer.writerow(summary_dict.values())
