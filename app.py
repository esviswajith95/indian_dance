#import essential libraries
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st

#load model file
model_file = "dance.keras"
model = keras.models.load_model(model_file)

#path to sample images f0lder
example_folder_path = "sample_images/"

def get_upload():
    '''
    Grab the uploaded file from widget
    '''
    uploaded_file = st.file_uploader("Upload an image(.jpg or .png)", type=['png', 'jpg', 'jpeg'])
    return uploaded_file


def load_image(image_file):
    '''
    Load as an image object from file
    '''
    img = Image.open(image_file)
    return img


def prepare_image(img):
    '''
    Convert an image object to tensor of shape (1, 180, 180, 3)
    '''
    img = img.resize((180, 180))

    img_array = tf.keras.utils.img_to_array(
                    img, data_format=None, dtype=None
                    )
    return tf.expand_dims(img_array, axis=0)

def get_img_list(path):
    '''
    Get a list of image file names from path
    '''
    file_list = []
    for file in os.listdir(path):
        if file.endswith(( 'jpg', 'jpeg', 'png')):
            file_list.append(file)
    
    return file_list


def predict(image_tensor):
    ''' 
    Generate predicions
    '''
    labels = ['Mohiniyattam', 'Odissi', 'Bharatanatyam', 'Kathakali', 'Kuchipudi', 'Sattriya Nritya', 'Kathak', 'Manipuri Raas Leela']
    
    probs = model.predict(image_tensor)
    label_index = probs.argmax(axis=-1)[0]
    label = sorted(labels)[label_index]
    
    return label, probs

def show_probs(probs):
    '''
    Display probabilities
    '''
    labels = ['Mohiniyattam', 'Odissi', 'Bharatanatyam', 'Kathakali', 'Kuchipudi', 'Sattriya Nritya', 'Kathak', 'Manipuri Raas Leela']
    df = pd.DataFrame(columns=["dance", "probability"])
    sum_probs = probs.sum()
    for i, label in enumerate(sorted(labels)):
        df.loc[i] = [label, probs[-1][i] * 100 / sum_probs]
    
    display_container.bar_chart(df, x = "dance", y = "probability")

def display_new_prediction(file):
    '''
    Display a new prediction given the image file
    '''
    if file is not None:
        img = load_image(file)
        img = img.resize((250, 250))
        display_container.image(img)

        tensor = prepare_image(img)
        label, probs = predict(tensor)

        display_container.write(f"This is most likely an image of {label.capitalize()}.")
        display_container.write()
        display_container.write("Here are the probabilities")
        show_probs(probs)
    else:
        display_container.write("Please upload a file")


description = "The art and cultural heritage of India is unmatched in its diversity. It is often the case that an artform in one part of the country is completely unknown to people from another part of the country. \n \n This app can distinguish between 8 major indian dance forms - Mohiniyattam, Odissi, Bharatanatyam, Kathakali, Kuchipudi, Sattriya Nritya, Kathak' and Manipuri Raas Leela. \n \n Next time you are confused about an indian dance form, snap a picture and upload here." 		
		
# get the file names of example images
example_img_names = get_img_list(example_folder_path)

#container for title and uploader
with st.container() as title_container:
    st.title("Which indian dance?")
    st.caption(description)
    st.write("Check out the [code](https://github.com/esviswajith95/indian_dance))")
    uploaded_file = get_upload()
    upload_predict = st.button("Which dance?", on_click=display_new_prediction, args = (uploaded_file,))    

#side bar to display examples
with st.sidebar:
    st.title("Examples")
    for i, f in enumerate(example_img_names):
        image_path = example_folder_path + f
        img = Image.open(image_path)
        img = img.resize((180, 180))
        st.image(img)
        st.button("Try this", key = i, on_click=display_new_prediction, args = (example_folder_path+example_img_names[i],))

#container to display the current image and prediction
display_container = st.container()

#initialise empty container
display_container.empty()


