# E2E
Eye-gaze change with latent space editing and masking based on JoJoGAN
<img width="581" alt="image" src="https://github.com/user-attachments/assets/27cecff5-35cd-47fe-a838-33eb8db0ab86" />

# Introduction
Many StyleGAN  have Eye-gaze problems which follows styles eye not contents eye. Many previous works are focus on modifing models However, this is heavy task and different as Style changes.
So in our Project, We suggest "E2E method" : Modifiy latent space and apply to style GAN. This is simple and easy-to-use to modify eye and find content-like eyes

# IDEA
1. Using eye tracking dataset, encode with GAN and find SVM normal vector to classify
2. With encoded content vector, interpolate using normal vector
3. then apply JoJoGAN generator
4. Done!

# How to use
1. Download JoJoGAN (https://github.com/mchong6/JoJoGAN)
2. get Our Main.py and put into JoJoGAN Folder
3. Define image and style
4. Control with alphas then changed images will goes to /outputs folder

#Acknowledgments
This code borrows from JOJOGAN by mchong6, StyleGAN2 by rosalinity, e4e.


