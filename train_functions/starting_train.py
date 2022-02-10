import torch
import constants
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard

"""
Trains and evaluates a model.

Args:
    train_dataset:   PyTorch dataset containing training data.
    val_dataset:     PyTorch dataset containing validation data.
    model:           PyTorch model to be trained.
    hyperparameters: Dictionary containing hyperparameters.
    n_eval:          Interval at which we evaluate our model.
    summary_path:    Path where Tensorboard summaries are located.
"""

global device
device = None 

def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path
):

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Initialize summary writer (for logging)
    tb_summary = None
    if summary_path is not None:
        tb_summary = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            input_data, label_data = batch
            pred = model(input_data)
            loss = loss_fn(pred, label_data)    # Prediction an label data should be the exact same shape.
            pred = pred.argmax(axis=1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"\n    Train Loss: {loss.item()}")

            # Periodically evaluate our model + log to Tensorboard
            if (step + 1) % n_eval == 0:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard

                train_accuracy = compute_accuracy(pred, label_data)

                if tb_summary:
                    tb_summary.add_scalar('Loss (Training)', loss, epoch)
                    tb_summary.add_scalar('Accuracy (Training)', train_accuracy, epoch)

                print(f"    Train Accu: {train_accuracy}")

                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                valid_loss, valid_accuracy = evaluate(val_loader, model, loss_fn)

                if tb_summary:
                    tb_summary.add_scalar('Loss (Validation)', valid_loss, epoch)
                    tb_summary.add_scalar('Accuracy (Validation)', valid_accuracy, epoch)

                print(f"    Valid Loss: {valid_loss}")
                print(f"    Valid Accu: {valid_accuracy}")

                model.train()

            step += 1

        print()
        if (epoch + 1) % constants.SAVE_INTERVAL:
            print("Saving model...")
            # Still need to implement this!

        print()


"""
Computes the accuracy of a model's predictions.

Example input:
    outputs: [0.7, 0.9, 0.3, 0.2]
    labels:  [1, 1, 0, 1]

Example output:
    0.75
"""

def compute_accuracy(outputs, labels):
    #print(outputs)
    #print(labels)
    outputs = torch.round(outputs.float())
    n_correct = (outputs == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


"""
Computes the loss and accuracy of a model on the validation dataset.

"""

def evaluate(val_loader, model, loss_fn):

    model.eval()

    loss, correct, count = 0, 0, 0
    for batch in val_loader:
        input_data, label_data = batch
        input_data = input_data.to(device)
        label_data = label_data.to(device)

        pred = model(input_data)
        loss += loss_fn(pred, label_data).mean().item()
        correct += (torch.argmax(pred, dim=1) == label_data).sum().item()
        count += len(label_data)

    return loss, correct/count
