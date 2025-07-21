from flask  import Flask,render_template,request,jsonify, redirect, url_for,flash,session,send_file
import psycopg2
import psycopg2.extras
import re
import h5py
import numpy as np
import cv2
from cv2 import dnn
from PIL import Image
import io

from werkzeug.security import generate_password_hash,check_password_hash

app=Flask(__name__)
app.secret_key="bibek-21"

DB_HOST = 'localhost'
DB_NAME = 'hackdb'
DB_USER = 'postgres'
DB_PASS = 'bibek20'

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)


############################################################################
###### COLORISATION ######

def load_model(h5_file_path):
    with h5py.File(h5_file_path, 'r') as hf:
        content = hf['prototxt'][()].decode('utf-8')
        data = hf['caffemodel'][:]
        kernel = hf['hull_pts'][:]
    
    # Save temporary files 
    with open('temp_data.caffemodel', 'wb') as f:
        f.write(data.tobytes())
    with open('temp_content.prototxt', 'w') as f:
        f.write(content)

    # Load the model
    net = dnn.readNetFromCaffe('temp_content.prototxt', 'temp_data.caffemodel')
    
    # Load kernel for colorization
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return net

# Load the model once when the Flask app starts
model = load_model(r"C:\Users\bdala\Desktop\SIH Hackathon\App_model\colorization_model.h5")

# Preprocess the image
def preprocess_image(file):
    # Open image and convert to grayscale
    image = Image.open(file).convert('RGB')
    img = np.array(image)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    return L, lab_img, img.shape

# Post-process and generate the colorized image
def postprocess_image(L, lab_img, original_shape):
    model.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = model.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (original_shape[1], original_shape[0]))
    
    L_original = cv2.split(lab_img)[0]
    colorized = np.concatenate((L_original[:, :, np.newaxis], ab_channel), axis=2)
    
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    return colorized

#############################################################################
#### APP ####

@app.route('/',methods=['GET', 'POST'])
def home():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if request.method == 'POST':
        fullname = request.form['Full_Name']
        email = request.form['email']
        information = request.form['More_information']
        
        cursor.execute('SELECT * FROM usercont WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
                flash('Email already exists!', 'error')
        else:
            cursor.execute("INSERT INTO usercont (fullname,email,information) VALUES (%s,%s,%s)",(fullname,email,information))
            conn.commit()
            flash('''Thanks for getting in touch! Your submission has been received and I will get back to you soon.''')

    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if request.method == 'POST':
        username = request.form['username']
        lpassword = request.form['lpassword']

        #Check wheather the accoubt existis or not
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        #Fetch one record and return result
        account = cursor.fetchone()

        if account:
            password_rs = account['password']
            # If accouunt exist in users table in out database
            if check_password_hash(password_rs,lpassword):
                session['loggedin'] = True
                session['username'] = account['username']
                return redirect(url_for('dashboard'))
            else:
                flash("Incorrect username/password")
        else:
            # Acount don't exist then 
            flash("Incorrect username/password")

    return render_template('Login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    if request.method == 'POST':
        usname = request.form['usname']
        spassword = request.form['password']
        confirm_password = request.form['cpassword']

        _hashed_password = generate_password_hash(spassword)
    
        # Check if account existis using MySQL
        cursor.execute("SELECT * FROM users WHERE username = %s", (usname,)) 
        account = cursor.fetchone()

        #If account exists show error and validation checks
        if account:
            flash("Account already exists")
        elif not usname or not spassword:
            flash("Please fill out the form")
        elif not re.match(r'[A-Za-z0-9]+$',usname):
            flash("User must contain only character and numbers")
        elif spassword != confirm_password:
            flash('Passwords do not match!', 'error')
        else:
            #Acount doesn't exist and the data is valid then insert the new data account into users db
            cursor.execute("INSERT INTO users (username,password) VALUES (%s,%s)",(usname,_hashed_password))
            conn.commit()
            flash("You have successfully registered")

    return render_template('Signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('colorisation.html')
# Route to handle the image upload and prediction


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Preprocess the image
    L, lab_img, original_shape = preprocess_image(file)
    
    # Get the colorized image
    colorized_image = postprocess_image(L, lab_img, original_shape)

    # Convert the result to a PIL image for returning
    result = Image.fromarray(colorized_image)
    output_io = io.BytesIO()
    result.save(output_io, format='PNG')
    output_io.seek(0)

    return send_file(output_io, mimetype='image/png')
    

if __name__ == '__main__':
    app.run(debug=True,port=5000)