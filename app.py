#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
from streamlit_option_menu import option_menu

import splitfolders
from PIL import Image
import numpy as np
import tensorflow as tf
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.initializers import RandomNormal
from keras.models import Model
from keras import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.utils import plot_model
import numpy
from PIL import Image, ImageOps
import os
from tensorflow.keras.preprocessing.image import load_img
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers import ELU, LeakyReLU
from keras.utils import plot_model
from keras import models, layers
from tensorflow.keras import backend as K
import pywt
from numpy import asarray
from tensorflow.keras.utils import CustomObjectScope
@st.cache(allow_output_mutation=False)
# @st.experimental_memo
class DWT(layers.Layer):
    """
    Discrete Wavelet transform - tensorflow - keras
    inputs:
        name - wavelet name ( from pywavelet library)
        concat - 1 - merge transform output to one channel
               - 0 - split to 4 channels ( 1 img in -> 4 smaller img out)
    """

    def __init__(self, wavelet_name='haar', concat=1, **kwargs):
        super().__init__()
        # self._name = self.name + "_" + name
        # get filter coeffs from 3rd party lib
        wavelet = pywt.Wavelet(wavelet_name)
        self.dec_len = wavelet.dec_len
        self.concat = concat
        # decomposition filter low pass and hight pass coeffs
        db2_lpf = wavelet.dec_lo
        db2_hpf = wavelet.dec_hi

        # covert filters into tensors and reshape for convolution math
        db2_lpf = tf.constant(db2_lpf[::-1])
        self.db2_lpf = tf.reshape(db2_lpf, (1, wavelet.dec_len, 1, 1))

        db2_hpf = tf.constant(db2_hpf[::-1])
        self.db2_hpf = tf.reshape(db2_hpf, (1, wavelet.dec_len, 1, 1))

        self.conv_type = "VALID"
        self.border_padd = "SYMMETRIC"
        self.wavelet_name = wavelet_name
        self.concat = concat

    def build(self, input_shape):
        # filter dims should be bigger if input is not gray scale
        if input_shape[-1] != 1:
            # self.db2_lpf = tf.repeat(self.db2_lpf, input_shape[-1], axis=-1)
            self.db2_lpf = tf.keras.backend.repeat_elements(self.db2_lpf, input_shape[-1], axis=-1)
            # self.db2_hpf = tf.repeat(self.db2_hpf, input_shape[-1], axis=-1)
            self.db2_hpf = tf.keras.backend.repeat_elements(self.db2_hpf, input_shape[-1], axis=-1)

    def call(self, inputs, training=None, mask=None):

        # border padding symatric add coulums
        inputs_pad = tf.pad(inputs, [[0, 0], [0, 0], [self.dec_len-1, self.dec_len-1], [0, 0]], self.border_padd)

        # approximation conv only rows
        a = tf.nn.conv2d(
            inputs_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # details conv only rows
        d = tf.nn.conv2d(
            inputs_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ds - down sample
        a_ds = a[:, :, 1:a.shape[2]:2, :]
        d_ds = d[:, :, 1:d.shape[2]:2, :]

        # border padding symatric add rows
        a_ds_pad = tf.pad(a_ds, [[0, 0], [self.dec_len-1, self.dec_len-1], [0, 0], [0, 0]], self.border_padd)
        d_ds_pad = tf.pad(d_ds, [[0, 0], [self.dec_len-1, self.dec_len-1], [0, 0], [0, 0]], self.border_padd)

        # convolution is done on the rows so we need to
        # transpose the matrix in order to convolve the colums
        a_ds_pad = tf.transpose(a_ds_pad, perm=[0, 2, 1, 3])
        d_ds_pad = tf.transpose(d_ds_pad, perm=[0, 2, 1, 3])

        # aa approximation approximation
        aa = tf.nn.conv2d(
            a_ds_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ad approximation details
        ad = tf.nn.conv2d(
            a_ds_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # ad details aproximation
        da = tf.nn.conv2d(
            d_ds_pad, self.db2_lpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )
        # dd details details
        dd = tf.nn.conv2d(
            d_ds_pad, self.db2_hpf, padding=self.conv_type, strides=[1, 1, 1, 1],
        )

        # transpose back the matrix
        aa = tf.transpose(aa, perm=[0, 2, 1, 3])
        ad = tf.transpose(ad, perm=[0, 2, 1, 3])
        da = tf.transpose(da, perm=[0, 2, 1, 3])
        dd = tf.transpose(dd, perm=[0, 2, 1, 3])

        # down sample
        ll = aa[:, 1:aa.shape[1]:2, :, :]
        lh = ad[:, 1:ad.shape[1]:2, :, :]
        hl = da[:, 1:da.shape[1]:2, :, :]
        hh = dd[:, 1:dd.shape[1]:2, :, :]

        # concate all outputs ionto tensor
        if self.concat == 0:
            x = tf.concat([ll, lh, hl, hh], axis=-1)
        elif self.concat == 2:
            x = ll
        else:
            x = tf.concat([tf.concat([ll, lh], axis=1), tf.concat([hl, hh], axis=1)], axis=2)
        return x
    def get_config(self):
        config = super(DWT, self).get_config()
        config.update({'wavelet_name': self.wavelet_name, 'concat': self.concat})
        return config

tf.keras.utils.get_custom_objects().update({'DWT': DWT})
# with CustomObjectScope({'DWT': DWT}):


# import wget

# # Replace the MODEL_LINK with your Google Drive model link
# import gdown
# url1 = "https://drive.google.com/uc?id=1Lx9rVKdBtKVC2Iu0jyBCm2V4ol0FJ9iw"
# output1 = "lesion_model_000296.h5"
# if not os.path.exists("lesion_model_000296.h5"):
#     gdown.download(url1, output1, quiet=False)



fmodel = tf.keras.models.load_model("lesion_model_000172.h5")
bmodel = tf.keras.models.load_model("background_model_000172.h5")
opt = Adam(learning_rate=0.00008, beta_1=0.5)
fmodel.compile(loss=['binary_crossentropy'],optimizer=opt)
bmodel.compile(loss=['binary_crossentropy'],optimizer=opt)

def preprocess_image(image):
    image = np.array(image)
    image = (image.astype('float32')) /255
    image = np.expand_dims(image, axis=0)
    #     image = tf.reshape(image,[1,256,256,3])
    return image

def predict(image, model):
    image = preprocess_image(image)
    image = tf.image.resize(image,(256,256))
    image = tf.reshape(image,[1,256,256,3])
    fmask = fmodel.predict(image)
    bmask = bmodel.predict(image)
    
    fmask = np.around(fmask)
    bmask = np.around(bmask)
    mask = np.logical_and(fmask,bmask)
    mask = np.squeeze(mask, axis=0)
    mask = (mask > 0.5).astype(np.uint8)*255 

    
    return np.reshape(mask,[256,256,1])

# def main():
#     # Set the app title and description
#     st.title("Breast tumour classification and segmentation")
#     st.markdown("This app uses a deep learning model to perform breast lesion classification and segmentation.")

#     # Load the model
# #     model = load_model()

#     # Create a file uploader
#     uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

#     # Check if an image is uploaded
#     if uploaded_file is not None:
#         # Read the image and display it
# #         image = Image.open(uploaded_file)
#         image = load_img(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         # Make a prediction and display the mask
#         mask = predict(image, fmodel)
#         st.image(mask, caption='Segmentated Lesion', use_column_width=True)

# if __name__ == '__main__':
#     main()
def apply_mask(image, mask, color=(255, 0, 0), alpha=0.2):
    # Convert the image and mask to float32 if necessary
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    image = tf.image.resize(image,(256,256))
    # Expand dimensions of the mask to match the image
    # mask = tf.expand_dims(mask)
    
    # Apply the mask to the image
    masked_image = image * (1 - alpha * mask) + alpha * mask * color
    
    # Clip values to be in the range [0, 255]
    # masked_image = masked_image/255.0
    masked_image = (masked_image - tf.math.reduce_min(masked_image))/(tf.math.reduce_max(masked_image)-tf.math.reduce_min(masked_image))
    # masked_image = tf.clip_by_value(masked_image, 0, 1)
    
    # Convert the masked image to uint8
    # masked_image = tf.cast(masked_image, tf.uint8)
    
    return masked_image



# def apply_mask(image, mask, color=(255, 0, 0), alpha=0.5):
#     image_array = np.array(image)
    
#     # Make a copy of the image array to avoid modifying the original image
#     masked_image = image_array.copy()
#     masked_image = tf.image.resize(masked_image,(256,256))
#     # image = tf.reshape(image,[1,256,256,3])
#     # masked_image = image.copy()
#     for i in range(3):  # Loop over RGB channels
#         masked_image[:, :, i] = np.where(mask == 1,masked_image[:, :, i] * (1 - alpha) + alpha * color[i], masked_image[:, :, i])
#     return masked_image

with st.sidebar:
    choose = option_menu('App Gallery',['About','Breast Ultrasound Images','AI-Predict'],
                         icons=['house','image','question-diamond-fill'],
                         menu_icon='prescription2',default_index=0,
                         styles={
                             'container':{'padding':"5!important","background-color":"#fafafa"},
                             'icon':{"color":"orange","font-size":"25px"},
                             "nav-link":{"font-size":"16px","text-align":"left","margin":"0px","--hover-color":"#eee"},
                             "nav-link-selected": {"background-color":"#02ab21"},
                         })

if choose=='About':
    st.write("<h2>Breast lesion classification and segmentation from ultrasound images<h2>",unsafe_allow_html=True)
    st.write("DSP Research Laboratory")

# elif choose=='Monkeypox images':
#     st.write("<div align='center'><h3>Monkeypox Images<h3></div>",unsafe_allow_html=True)
#     col1,col2,col3=st.columns(3)
#     available_images=[]
#     with col1:
#         for i in range(2):
#             rand1 = random.randint(0,20)
#             if rand1 not in available_images:
#                 img1=Image.open(monkey_glob[rand1])
#                 st.image(img1)
#                 available_images.append(rand1)
#     with col2:
#         for k in range(2):
#             random2=random.randint(20,40)
#             if random2 not in available_images:
#                 img2=Image.open(monkey_glob[random2])
#                 st.image(img2)
#                 available_images.append(random2)
#     with col3:
#         for p in range(2):
#             random3=random.randint(40,60)
#             if random3 not in available_images:
#                 img3=Image.open(monkey_glob[random3])
#                 st.image(img3)
#                 available_images.append(random3)


# generate and plot augmented images

elif choose=='AI-Predict':
    # model=load_model('monkey_pox1.h5')
    image_paths1=[('b_0.png','sample1')]
    image_path2 =[('b_5.png','sample2')]
    image_path3 =[('b_10.png','sample3')]
    image_path4=[('b_12.png','sample4')]

    st.title("Breast tumour classification and segmentation")
    st.markdown("This app uses a deep learning model to perform breast lesion classification and segmentation.")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    # Check if an image is uploaded
    if uploaded_file is not None:
        # Read the image and display it
        image = load_img(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Make a prediction and display the mask
        mask = predict(image, fmodel)
        masked_image = apply_mask(image, mask)
        masked_image = np.array(masked_image)
        print(masked_image.shape)
        st.image(masked_image, caption='Segmentated Lesion', use_column_width=True)


    # st.title("Image Classification")
    # class_names = ['Monkeypox', 'Others']
    # uploaded_file = st.file_uploader("",type=['jpg','jpeg','png'])
    # if st.button('Predict'):
    #     if uploaded_file is not None:
    #         img=load_img(uploaded_file,target_size=(224,224))
    #         img=img_to_array(img)
    #         img=np.expand_dims(img,axis=0)
    #         img=img/255.0
    #         pred= model.predict(img)
    #         arg_max=np.argmax(pred)
    #         pred_int = pred[arg_max][0]
    #         if pred>0.5:
    #             st.write(f"The model is {round(pred_int*100,2)}% confident that the image shows NO signs of Monkeypox")
    #         else:
    #             st.write(f"The model is {round((1-pred_int)*100,2)}% confident that the image shows signs of Monkeypox")

    # col1,col2,col3,col4=st.columns(4)
    # with col1:
    #     for path, label in image_paths1:
    #         image = Image.open ( path )
    #         st.image ( image, caption=label )
    # with col2:
    #     for path,label in image_path2:
    #         image=Image.open(path)
    #         st.image(image,caption=label)
    # with col3:
    #     for path,label in image_path3:
    #         image=Image.open(path)
    #         st.image(image,caption=label)
    # with col4:
    #     for path,label in image_path4:
    #         image=Image.open(path)
    #         st.image(image,caption=label)

import gc

gc.collect()





