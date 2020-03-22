# Transcription Start Sites Analysis
This is an extension of https://github.com/timlai4/deepTSSscrubbeR using the same input pipelines, but different models and in Python.

## Logit (Torch)
The code used to train the logistic regression is [here](https://github.com/timlai4/TSSPred/blob/master/logit_train.py). 
To load and use the pretrained model, use [this](https://github.com/timlai4/TSSPred/blob/master/logit_test.py). 
The inputs should be adjusted accordingly. However, any adjustments require the saved numpy arrays from the R pipelines in https://github.com/timlai4/deepTSSscrubbeR.
