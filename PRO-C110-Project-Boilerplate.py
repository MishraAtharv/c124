# To Capture Frame
import cv2

# To process image array
import numpy as np

# import the tensorflow modules and load the model
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
    # Reading / Requesting a Frame from the Camera 
    status, frame = camera.read()

    # if we were successfully able to read the frame
    if status:
        # Flip the frame
        frame = cv2.flip(frame, 1)
        
        # 1. Resizing the image
        img = cv2.resize(frame, (224, 224))

        # 2. Converting the image into Numpy array and increase dimension
        test_image = np.array(img, dtype=np.float32)
        test_image = np.expand_dims(test_image, axis=0)

        # 3. Normalizing the image
        normalized_image = test_image / 255.0

        # get predictions from the model
        predictions = model.predict(normalized_image)

        # Assuming the model outputs a single class probability, change this based on your model
        predicted_class_prob = predictions[0][0]

        # Displaying the frames captured
        cv2.imshow('feed', frame)
        
        # Printing the prediction result
        print("Predicted Probability:", predicted_class_prob)

        # Waiting for 1ms
        code = cv2.waitKey(1)
        
        # If space key is pressed, break the loop
        if code == 32:
            break

# Release the camera from the application software
camera.release()

# Close the open window
cv2.destroyAllWindows()
