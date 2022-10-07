# Which Indian Dance?

India is a land of diversities and we have an array of different dance forms that are popular in different parts of our country. This project aims to develop a deep learning model to identify a dance form from an image. The model can identify 8 major dance forms (Mohiniyattam, Odissi, Bharatanatyam, Kathakali, Kuchipudi, Sattriya Nritya, Kathak and Manipuri Raas Leela) with an accuracy of 67% as of now.

The model is a fine tuned version of the pretrained convolutional neural network, VGG16. The fine tuning was done with a relatively small dataset of 364 images from [this dataset](https://www.kaggle.com/datasets/somnath796/indian-dance-form-recognition). The model is deployed with Streamlit and Hugging face spaces. You can access it [here](https://huggingface.co/spaces/viswajith/indian_dance).

For more detailed discussion on the model building process. Please refer to [this notebook](https://github.com/esviswajith95/indian_dance/blob/main/notebooks/Indian_dance_classifer.ipynb).
