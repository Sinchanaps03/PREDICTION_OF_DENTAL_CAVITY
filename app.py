from flask import Flask, render_template, url_for, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
import shutil
import sqlite3
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score






# Load the trained model
model = load_model('ResNet50_model.keras')

# Load class names from the pickle file
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

def predict_image(image):
    img = load_img(image, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    print(predicted_class_index)
    predicted_class = class_names[predicted_class_index]
    print("predicted_class:", predicted_class)
    prediction1 = prediction.tolist()
    print(prediction1[0][predicted_class_index] * 100)
    return predicted_class, prediction1[0][predicted_class_index] * 100

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result)==0:
            return render_template('index.html',msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('home.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')



@app.route('/userlog.html', methods=['GET'])
def indexBt():
      return render_template('userlog.html')

@app.route('/graph.html', methods=['GET', 'POST'])
def graph():
    
    images = ['http://127.0.0.1:5000/static/Accu_plt.png',
              
              'http://127.0.0.1:5000/static/loss_plt.png',
              'http://127.0.0.1:5000/static/f1_graph.jpg',
              'http://127.0.0.1:5000/static/confusion_matrix.jpg']
    content=['Accuracy Graph',
            'Loss Graph',
            'F1-Score Graph',
            'Confusion Matrix Graph']

            
    
        
    return render_template('graph.html',images=images,content=content)
    



@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        import os
        try:
            if 'filename' not in request.files:
                return render_template('userlog.html', msg='No file part in the request')

            file = request.files['filename']

            if file.filename == '':
                return render_template('userlog.html', msg='No file selected for uploading')

            dirPath = "static/images"
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            fileList = os.listdir(dirPath)
            for fileName in fileList:
                os.remove(os.path.join(dirPath, fileName))
            fileName = file.filename
            dst = "static/images"

            file.save(os.path.join(dst, fileName))
            if not os.path.exists("test"):
                os.makedirs("test")
            shutil.copy(os.path.join(dst, fileName), os.path.join("test", fileName))
            image_path = os.path.join("test", fileName)
            image = cv2.imread(image_path)
            if image is None:
                return render_template('userlog.html', msg='Error loading image for processing. Please try another image.')
            # #color conversion
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('static/gray.jpg', gray_image)
            # #apply the Canny edge detection
            edges = cv2.Canny(image, 100, 200)
            cv2.imwrite('static/edges.jpg', edges)
            # #apply thresholding to segment the image
            retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
            cv2.imwrite('static/threshold.jpg', threshold2)
            # # create the sharpening kernel
            kernel_sharpening = np.array([[-1,-1,-1],
                                         [-1, 9,-1],
                                        [-1,-1,-1]])

            # # apply the sharpening kernel to the image
            sharpened =cv2.filter2D(image, -1, kernel_sharpening)

            # save the sharpened image
            cv2.imwrite('static/sharpened.jpg', sharpened)

            predicted_class, accuracy = predict_image("test/"+fileName)
            print("Predicted class:", predicted_class)
            print("Accuracy is:", accuracy)

            f = open('acc.txt', 'w')
            f.write(str(accuracy))
            f.close()


            str_label = predicted_class
            tre = "Treatment recommendations based on prediction"
            tre1 = ["Consult a dentist", "Maintain oral hygiene"]
            rec = "Recommendations"
            Rec1 = ["Brush twice daily", "Use fluoride toothpaste"]
            fl = "Follow-up"
            Fl1 = ["Regular check-ups"]
            detection = "Cavity detected" if predicted_class != "no cavity" else "No cavity detected"
            spread = "Monitor for spread" if predicted_class in ["moderate cavity", "severe cavity"] else "No spread risk"


            # Generate the testing graph for accuracy comparison
            # A=float( predicted_class =="mild cavity")
            # B=float(predicted_class =="moderate cavity")
            # C=float(predicted_class =="no cavity")
            # D=float(predicted_class =="severe  cavity")

            # dic={'Mild':A,'Modarate':B,'Normal':C,'Severe':D}
            # algm = list(dic.keys()) 
            # accu = list(dic.values()) 
            # fig = plt.figure(figsize = (5, 5))  
            # plt.bar(algm, accu, color ='maroon', width = 0.3)  
            # plt.xlabel("Comparision") 
            # plt.ylabel("Accuracy Level") 
            # plt.title("Accuracy Comparision between \n Dental cavity detection")
            # plt.savefig('static/matrix.png')
                        


           


           
           
            # f = open('acc.txt', 'r')
            # accuracy = f.read()
            # f.close()
            # print(accuracy)

           


            
            
            return render_template('results.html', status=str_label,status2=f'accuracy is {accuracy}',Treatment=tre,Treatment1=tre1,Recommendation=rec,Recommendation1=Rec1,FollowUp=fl,FollowUp1=Fl1,detection=detection,spread=spread,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg",ImageDisplay5="http://127.0.0.1:5000/static/matrix.png")
        except Exception as e:
            print(f"Error processing image: {e}")
            return render_template('userlog.html', msg=f'Error processing image: {e}')
    return render_template('userlog.html')




@app.route('/logout')
def logout():
    return render_template('index.html')

@app.route('/userupdate', methods=['GET', 'POST'])
def userupdate():
    if request.method == 'POST':
        oldname = request.form['oldname']
        oldpassword = request.form['oldpassword']
        newname = request.form['newname']
        newpassword = request.form['newpassword']

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        # Verify old credentials
        cursor.execute("SELECT * FROM user WHERE name=? AND password=?", (oldname, oldpassword))
        user = cursor.fetchone()

        if user:
            # Update username and password
            cursor.execute("UPDATE user SET name=?, password=? WHERE name=? AND password=?", (newname, newpassword, oldname, oldpassword))
            connection.commit()
            connection.close()
            return render_template('userupdate.html', msg='Username and password updated successfully.')
        else:
            connection.close()
            return render_template('userupdate.html', msg='Incorrect current username or password.')

    return render_template('userupdate.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
