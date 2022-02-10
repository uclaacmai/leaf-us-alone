import argparse
import os, torch
import constants

from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train


def main():
    # Get command line arguments
    args = parse_arguments()
    hyperparameters = {"epochs": args.epochs, "batch_size": args.batch_size, "data_dir": args.datadir}

    # Create path for training summaries
    summary_path = None
    if args.logdir is not None:
        summary_path = f"{args.summaries}/{args.logdir}"
        os.makedirs(summary_path, exist_ok=True)

    # Use GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print out training information 
    print("\n----------------- Summary Information --------------------")
    print(f"\nSummary path: {summary_path}")
    print(f"   Data path: {constants.DATA_DIR}\n")
    print(f"Epochs: {args.epochs}")
    print(f"    Evaluate: {constants.N_EVAL}")
    print(f"  Save Model: {constants.SAVE_INTERVAL} \n")

    print(f"Training imgs: {constants.TRAIN_NUM * constants.IMG_TYPES}")
    print(f" Testing imgs: {constants.TEST_NUM * constants.IMG_TYPES}")
    print(f"   Batch size: {args.batch_size}\n")
    print("----------------------------------------------------------")


    # Initalize dataset and model. Then train the model!
    csv_path = args.datadir +'/train.csv'
    train_dataset = StartingDataset(csv_path)
    val_dataset = StartingDataset(csv_path, training_set = False)

    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
        summary_path=summary_path,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--n_eval", type=int, default=constants.N_EVAL)
    parser.add_argument("--logdir", type=str, default=constants.LOG_DIR)
    parser.add_argument("--datadir", type=str, default=constants.DATA_DIR)
    parser.add_argument("--summaries", type=str, default=constants.SUMMARIES_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    main()
