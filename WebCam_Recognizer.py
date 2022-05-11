import numpy as np

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

import cv2
font = cv2.FONT_HERSHEY_SIMPLEX

from keras.models import load_model
model = load_model('traffic_classifier.h5')

cap=cv2.VideoCapture(0)
while cap.isOpened():
        _,imgOrg = cap.read()
        
        # Process image
        img = imgOrg
        img = cv2.resize(img,(30,30))
        img = np.expand_dims(img, axis=0)
        img = np.array(img)
        
        cv2.putText(imgOrg, "CLASS: ", (20,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
        
        # Prediction
        pred = np.argmax(model.predict([img]), axis=-1)[0]
        
        cv2.putText(imgOrg, str(pred+1)+" "+classes[pred+1], (120,35), font, 0.75, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrg)
        
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()