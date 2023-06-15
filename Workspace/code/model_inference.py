import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
import cv2

if __name__ == '__main__':
    model = load_model('model')
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _ , frame = cap.read()
        frame = frame[50:500, 50:500,:]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120,120))
        
        yhat = model.predict(np.expand_dims(resized/255,0)) # type: ignore
        sample_coords = yhat[1][0]
        print(yhat[0])
        print(yhat[1])
        if yhat[0] > 0.999: 
            # Controls the main rectangle
            cv2.rectangle(frame, 
                        tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                                (255,0,0), 2)
            # Controls the label rectangle
            cv2.rectangle(frame, 
                        tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                        [0,-30])),
                        tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                        [80,0])), 
                                (255,0,0), -1)
            
            # Controls the text rendered
            cv2.putText(frame, 'human', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                                [0,-5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow('Human Track', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()