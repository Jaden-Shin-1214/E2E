# E2E
Eye-gaze change with latent space editing and masking based on JoJoGAN
![Suji_multi_interp](https://github.com/user-attachments/assets/3dc9d4af-bbad-42d5-893a-345f7aaa67a4)

# Introduction
![Suji](https://github.com/user-attachments/assets/6b650c58-4775-4e13-a97d-e5b1077b2bcc)

Many StyleGAN  have Eye-gaze problems which follows styles eye not contents eye. Many previous works are focus on modifing models However, this is heavy task and different as Style changes.
So in our Project, We suggest "E2E method" : Modifiy latent space and apply to style GAN. This is simple and easy-to-use to modify eye and find content-like eyes

# IDEA
1. Using eye tracking dataset, encode with GAN and find SVM normal vector to classify
2. With encoded content vector, interpolate using normal vector
3. then apply JoJoGAN generator
4. Done!

![Suji_final_interp](https://github.com/user-attachments/assets/8ef36158-d79f-403e-b25c-4c1129173bbc)


# How to use
1. Download JoJoGAN (https://github.com/mchong6/JoJoGAN) and checkpoints
2. get Our Main.py and put into JoJoGAN Folder
3. Generate eye_dataset folder and put two npy files into folder (This is Normal vector for eye gaze)
4. Define image and style
5. Control with alphas then changed images will goes to /outputs folder

#Acknowledgments
This code borrows from JOJOGAN by mchong6, StyleGAN2 by rosalinity, e4e.


