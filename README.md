# leaf us alone 🌿

*Kaggle's Cassava Leaf Disease Classification Project using ResNet-18 - UCLA ACM AI, Projects*

This project was conducted as part of UCLA's ACM AI Projects committee, during Winter '22.

For more references, kindly check out the following resources: 

* [Project Skeleton Code (Repo)](https://github.com/uclaacmai/projects-skeleton-code)
* [Project Skeleton Notebook (Kaggle)](https://www.kaggle.com/advitdeepak/leaf-us-alone)
* [Cassava Leaf Disease Challenge (Kaggle)](https://www.kaggle.com/c/cassava-leaf-disease-classification)

> Quick Statistics: utilized data augmentation (rotate, flip, blur), achieved accuracy of: 95% 

## Running the Code Locally 

1. Create and activate a new Conda environment.

2. Install PyTorch, PIL, Pandas, TorchVision, and TensorBoard.

3. Download the Cassava Leaf dataset from [Kaggle Cassava Data](https://www.kaggle.com/c/cassava-leaf-disease-classification/data)

3. Clone this repository and run `python main.py` 

4. Modifications can be made by changing `constants.py`

### Running the Code on Kaggle

1. Navigate to the [code tab of the Kaggle competition](https://www.kaggle.com/c/cassava-leaf-disease-classification/code). Click on the "New Notebook" button to create a new notebook. The dataset should be automatically loaded in the `/kaggle/input` folder.

2. To use the GPU, click the three dots in the top-right corner and select Accelerator > GPU.

3. To access your code, run the following command (replacing the URL):

   ```
   !git clone "https://github.com/uclaacmai/leaf-us-alone"
   ```

   This should clone this repository into the `/kaggle/working` folder.

4. Change directories into your repository by running the command:

   ```
   cd leaf-us-alone
   ```

5. You should now be able to import your code normally. For instance, the following code will import the starting code:

   ```python
   import constants
   from datasets.StartingDataset import StartingDataset
   from networks.StartingNetwork import StartingNetwork
   from train_functions.starting_train import starting_train
   ```

6. If you want your code to run without keeping the tab open, you can click on "Save version" and commit your code. Make sure to save any outputs (e.g. log files) to the `/kaggle/output`, and you should be able to access them in the future.


## Further Resources 

To learn more about ACM AI, feel free to check out our [LinkTree](https://linktr.ee/acm_ai_ucla)! 
