import torch
import torchvision
from torch import nn
from torchvision import transforms
import pathlib
from pathlib import Path

import utils
import engine
from utils import accuracy_fn
import data_setup
import model_builder

# Main code
def main():

    # Write requirements for deployment
    utils.write_requirements("deployment_gr")

    # Device agnostic code 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the path to your image folder
    current_dir = pathlib.Path().resolve()
    data_dir = current_dir / "flower_images"
    train_dir = current_dir / Path("train")
    test_dir = current_dir / Path("test")

    # Get data libraries
    utils.get_data_libraries(data_dir,
                             train_dir,
                             test_dir,
                             clean_data = True,
                             low_image_treshold = 200,
                             split_train_ratio = 0.8,
                             split_experimental_ratio = 0.2)

    # Get pretrained efficientnet model
    model, transform = model_builder.create_effnet_b3_model(train_dir=train_dir,
                                         seed=42)

    # Hyperparameters
    BATCH_SIZE = 32
    NUM_WORKERS = 1
    LEARNING_RATE = 0.001

    # Creating train and test dataloder, getting class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir,
                                                                                   test_dir,
                                                                                   transform,
                                                                                   BATCH_SIZE,
                                                                                   NUM_WORKERS)
    

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    # Train the model
    results = engine.train(model = model,
                          train_dataloader = train_dataloader,
                          test_dataloader = test_dataloader,
                          optimizer = optimizer,
                          loss_fn = loss_fn,
                          epochs = 5,
                          device = device)


if __name__ == "__main__":
    main()
