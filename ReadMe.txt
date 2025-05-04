# Natural Language Processing (NLP) - Sentiment Analysis with LSTM

This project implements a sentiment analysis model using an LSTM (Long Short-Term Memory) network for text classification. The model uses GloVe embeddings to convert words into numerical vectors and performs binary classification (positive/negative sentiment). This project is designed for educational purposes and as a base for more advanced NLP applications.

---

## 📋 Requirements

### Libraries & Tools
- Python 3.x
- PyTorch
- Pandas
- NumPy
- Matplotlib
- GloVe pre-trained embeddings
- scikit-learn
- TensorBoard

🧑‍💻 How to Run
Run the Sentiment Analysis Model
To train and test the sentiment analysis model, use the following steps:

Prepare the Dataset
Ensure you have training_data.csv, test_data.csv, and tokens2index.json in the project directory. The tokens2index.json file should map words to indices.

Download GloVe Pre-trained Embedding
Download the GloVe embedding from GloVe website.
Extract the file and specify the path to glove.6B.200d.txt in the glove_file variable in main.py.

Setup the Environment
Install the necessary dependencies by running:

bash
pip install -r requirements.txt
Run the Model
To train the model, run the following command:

bash
python main.py
The script will:

Load the training and testing data.

Initialize the LSTM model.

Train the model using binary cross-entropy loss.

Save checkpoints periodically during training.

Evaluate the model on the test dataset.

If you want to run inference only, change the mode variable in the main.py file to 'test'.
