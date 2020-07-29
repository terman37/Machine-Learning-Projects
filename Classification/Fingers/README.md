# Finger recognition

### Project objectives:

Build a model able to count fingers as well as distinguish between left and right hand

### How solution will be used:

Target is to be able to take a picture showing hand (upload/from webcam) and display finger count shown from (each) hand

Next step would be to identify finger count on videos

- Train a model from finger dataset (21600 images / found on [kaggle](https://www.kaggle.com/koryakinp/fingers))

- Make an API to predict value from new picture

- Have a front end to get the picture / call API / display result

### Problem frame:

Supervised learning - images are labelled with number of fingers - L/R hand

2 classifiers:

- Binary classification for L/R hand
- Multi label Classification for #fingers (from 0 to 5) 

Approach:

- build a model from classic classifiers
- build a model from ANN / CNN

### Performance target:

Target >98% acccuracy (needs to check feasability)

### Data informations:

DataSet size: 177MB

21600 images of left and right hands fingers. 

- Training set: 18000 images 

- Test set: 3600 images 

All images are 128 by 128 pixels. 

Images are centered by the center of mass 

Noise pattern on the background

Labels are in 2 last characters of a file name. 

- L/R indicates left/right hand; 
- 0,1,2,3,4,5 indicates number of fingers.