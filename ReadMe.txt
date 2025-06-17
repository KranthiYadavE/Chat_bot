How to Launch and Use EmotiveChat

1. Install Required Dependencies

First, make sure all the necessary libraries are installed:


pip install torch transformers scikit-learn gradio

You should have the following Python libraries available:
torch
transformers
scikit-learn
gradio
matplotlib
numpy

2. Using Pretrained Model (Recommended)

Since a pre-trained model is already provided (emotion_classifier.pth and tokenizer files): in model1 folder "/HCI_Final/model1/"

You do NOT need to retrain the DistilBERT model unless you want to.

You can directly launch the chatbot app using the provided model.


3. Open the app.py file.

Update the model paths if needed:

In app.py, make sure the file paths (around lines 15–18) point correctly to your local trained model files under the model1/ folder:

python
MODEL_PATH = "model1/model_state.pth"
TOKENIZER_CONFIG = "model1/tokenizer_config"
SPECIAL_TOKENS_MAP = "model1/special_tokens_map"
VOCAB_PATH = "model1/vocab"


4. Then, run the app:

"python app.py"

you will automatically directed to the emotive chat interface app if not you will get the local host line copy and past in the chorme bar. 

 then the app is launched , you can test.





""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
3. Optional: Training the Model (If Needed)

If you want to retrain the DistilBERT model:

Open HCI_BERT_final.ipynb in Google Colab.

Important:

Go to Runtime > Change runtime type > Select "GPU (T4)"

This reduces training time and allows faster execution.

Run all the notebook cells sequentially (Runtime > Run all).

After training is complete (approximately 90 minutes), the trained model files will be automatically saved locally:

emotion_classifier.pth

emotion_tokenizer/ (folder with tokenizer configs)

You can now use these files directly in the app by updating app.py paths as mentioned.
