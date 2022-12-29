
## CAPTCHA Generator

#### This code generates and saves a set of CAPTCHAs as PNG files in base64 format, using a convolutional neural network (CNN) trained on the MNIST dataset.

### Requirements

 1. Python 
 2.  TensorFlow 
 3.  NumPy

## Usage

#### To generate and save the CAPTCHAs, run the following command:

    python generate_captchas.py

*This will generate and save num_captchas CAPTCHAs as PNG files in base64 format, where num_captchas is a parameter that can be adjusted in the code. The CAPTCHAs and their corresponding answers will be saved in the captchas and captcha_answers lists, respectively.

## Output

The output of the code will be a set of num_captchas PNG images saved in base64 format, along with the corresponding answers in the captcha_answers list.
Notes

> The CNN is trained on the MNIST dataset for 10 epochs, and the   
> trained model is used to generate the CAPTCHAs. The CAPTCHAs are   
> generated by randomly sampling 28x28 grayscale images from a uniform  
> distribution between 0 and 9. The base64 strings can be decoded and   
> saved as PNG files using the base64 module in Python.

