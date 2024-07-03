from flask import Flask, request,render_template, redirect,session
print("Imported Flask")
from flask_sqlalchemy import SQLAlchemy
import bcrypt

import datetime
from Levenshtein import distance as levenshtein
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pandas as pd
import numpy as np
import cv2
print("Imported cv2")
import easyocr
import re
import matplotlib.pyplot as plt



max_consumption = {"protein": 50, "carbohydrate": 300, "total fat": 80, "energy": 2000, "total sugar": 50, "sodium": 2300}

print("Imported easyocr")

reader = easyocr.Reader(['en'], gpu = False)

print("Reader created")

def read_label(img_path):

    # Resize image
    def opencv_resize(image, ratio):
        width = int(image.shape[1] * ratio)
        height = int(image.shape[0] * ratio)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # Display grey scale image
    def plot_gray(image):
        plt.figure(figsize=(16,10))
        return plt.imshow(image, cmap='Greys_r')

    # Display RGB colour image
    def plot_rgb(image):
        plt.figure(figsize=(16,10))
        return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # We will use approxPolyDP for approximating more primitive contour shape consisting of as few points as possible
    # Approximate the contour by a more primitive polygon shape
    def approximate_contour(contour):
        peri = cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, 0.032 * peri, True)

    def get_rectangular_contours(contours):
        # Approximates provided contours and returns only those which have 4 vertices
        res = []
        for contour in contours:
            hull = cv2.convexHull(contour)
            peri = cv2.arcLength(hull, closed=True)
            approx = cv2.approxPolyDP(hull, 0.04 * peri, closed=True)
            if len(approx) == 4:
                res.append(approx)
        return res

    # Find 4 points of table
    def get_table_contour(contours):    
        # loop over the contours
        for c in contours:
            approx = approximate_contour(c)
            # if our approximated contour has four points, we can assume it is table's rectangle
            if len(approx) == 4:
                return approx
            
    # Convert 4 points into lines / rect      
    def contour_to_rect(contour):
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")
        # top-left point has the smallest sum
        # bottom-right has the largest sum
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # compute the difference between the points:
        # the top-right will have the minumum difference 
        # the bottom-left will have the maximum difference
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect / resize_ratio

    # Original table with warpped perspective
    def warp_perspective(img, rect):
        # unpack rectangle points: top left, top right, bottom right, bottom left
        (tl, tr, br, bl) = rect
        # compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        # compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        # destination points which will be used to map the screen to a "scanned" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        # warp the perspective to grab the screen
        return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    def plot_gray(image):
        plt.figure(figsize=(5, 5))
        return plt.imshow(image, cmap='Greys_r')

    def plot_rgb(image):
        plt.figure(figsize=(5,5))
        return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def find_amounts(text):
        amounts = re.findall(r'\d+\.\d{2}\b', text)
        floats = [float(amount) for amount in amounts]
        unique = list(dict.fromkeys(floats))
        return unique

    # Plot EasyOCR output
    def plot_extractions(img, results):
        fig, ax = plt.subplots()

        ax.imshow(img, cmap='Greys_r')
        for bbox, text, conf in results:
            a, b, c, d = bbox
            rect = patches.Rectangle((a[0], a[1]), c[0]-a[0], c[1]-a[1], linewidth=1, edgecolor='r', facecolor='none', label=text)
            ax.add_patch(rect)
        pos = ax.get_position()
        ax.axis('off')
        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                fancybox=True, shadow=True, ncol=5)
        plt.show()


    # Cell by Cell Extraction of output
    def extract_text_by_cell(results, columns, thresh=0.5):
        res_cols = []
        res_rows = [0]
        drop_val = []
        for i in range(len(results)):
            (a, b, c, d), text, conf = results[i]
            x1, x2, = a[0], c[0]
            flag = True
            for col in range(len(columns)-1):
                # if x1 > columns[col] and x2 < columns[col+1]:
                if x2 < columns[col+1]:
                    flag = False
                    res_cols.append(col)
                    break
            if flag:
                drop_val.append(i)
                res_cols.append(-1)
        #np.delete(results, drop_val)
        extracted_text = []
        row = 0
        prev_bbox, prev_text, prev_conf = results[0]
        prev_a, prev_b, prev_c, prev_d = prev_bbox
        prev_y = prev_a[1] # horizontal
        prev_height = abs(prev_a[1]-prev_c[1])
        for i in range(1, len(results)):
            if res_cols[i] == -1:
                res_rows.append(-1)
                continue
            bbox, text, conf = results[i]
            a, b, c, d = bbox
            x1, x2 = a[0], c[0]
            y = b[1]
            if abs(y-prev_y) < thresh * prev_height:
                res_rows.append(row)
            else:
                row+=1
                res_rows.append(row)
            prev_bbox, prev_text = bbox, text
            prev_y = y
            prev_height = abs(a[1]-c[1])

        prev_col = 0
        prev_row = res_rows[0]
        prev_bbox, prev_text, prev_conf = results[0]
        cell_text = ""
        row_text = []
        for i in range(len(results)):
            bbox, text, conf = results[i]
            row = res_rows[i]
            col = res_cols[i]
            if row == prev_row:
                if col == prev_col:
                    cell_text += text+" "
                else:
                    while col-prev_col != 1 and col>prev_col:
                        row_text.append(cell_text)
                        cell_text = ""
                        prev_col+=1
                    row_text.append(cell_text)
                    cell_text = text+" "
            else:
                while col-prev_col != 1 and col>prev_col:
                    row_text.append(cell_text)
                    cell_text = ""
                    prev_col+=1
                row_text.append(cell_text)
                cell_text = text+" "
                extracted_text.append(row_text)
                cell_text = text+" "
                row_text = []
            prev_col = col
            prev_row = row
            prev_bbox, prev_text, prev_conf = bbox, text, conf
        return extracted_text




    image = cv2.imread(img_path)
    #Downscale image.
    #Finding table contour is more efficient on a small image
    resize_ratio = 500 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plot_gray(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # plot_gray(blurred)
    edged = cv2.Canny(blurred, 50, 125, apertureSize=3)
    # plot_gray(edged)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # quad approximation for all contours
    contours = get_rectangular_contours(contours)
    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3)
    # plot_rgb(image_with_contours)
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    image_with_largest_contours = cv2.drawContours(image.copy(), largest_contours, -1, (0,255,0), 3)
    # plot_rgb(image_with_largest_contours)
    table_contour = get_table_contour(largest_contours)
    scanned = warp_perspective(original.copy(), contour_to_rect(table_contour))
    # plt.figure(figsize=(5, 4))
    # plt.imshow(scanned)
    blur = cv2.GaussianBlur(scanned, (0,0), sigmaX=33, sigmaY=33)
    divided = cv2.divide(scanned, blur, scale=255)
    # plt.figure(figsize=(16,10))
    # plot_gray(divided)
    gray = cv2.cvtColor(divided, cv2.COLOR_BGR2GRAY)
    # plot_gray(gray)
    kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 70))
    dilated_vert = cv2.dilate(gray, kernel_vert)
    # plot_gray(dilated_vert)
    h, w = dilated_vert.shape
    columns = [0]

    row = dilated_vert[h//2]
    j = 0
    while j < w:
        px = row[j]
        if px < 127:
            line_centre = 0
            centre_count = 0
            count = 0
            while j < w and count < w//100+1:
                px = row[j]
                if px < 127:
                    line_centre += j
                    centre_count += 1
                else:
                    count+=1
                j+=1
            line_centre /= centre_count   
            columns.append(line_centre)
        j+=1
    columns.append(w)
    # print("column coordinates:",columns)
    results = reader.readtext(gray)
    extracted_text = extract_text_by_cell(results, columns, 0.3)
    table = pd.DataFrame(extracted_text).fillna("-")
    table
    # Find the word closest to "protein"
    # For all the words in the table, find the word that is closest to "protein"
    # We will use the Levenshtein distance to find the closest word
    # The Levenshtein distance is a string metric for measuring the difference between two sequences.
    # Informally, the Levenshtein distance between two words is the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into the other.

    import Levenshtein as lev

    def find_closest_word(word, words):
        # Return index of the closest word to the given word
        min_distance = 100
        closest_word = ""
        i = 0
        ret = 0
        for w in words:
            w = w.lower()
            if "(" in w:
                w = w.split("(")[0].strip()
            # print(w, word)
            distance = lev.distance(word, w)
            if distance < min_distance:
                min_distance = distance
                closest_word = w
                ret = i
            i+=1
        return closest_word, ret

    # Search for the word "protein" in the table column 0

    protein = "sodium"

    words = table[0].values

    words

    closest_word = find_closest_word(protein, words)

    # print("Closest word to protein:", closest_word)

    # # Find the 100g protein value
    # # Find the row that contains the closest word to "protein" and extract the value in the next column

    # Find the row that contains the closest word to "protein"
    row_index = closest_word[1]

    # Extract the value in the next column
    protein_value = table.iloc[row_index, 1]

    # print("100g protein value:", protei/n_value)
    # Obtain the Carbohydrate, Added Sugars, Total Sugars, Saturated Fat, Trans Fat, Sodium values.
    # Find the closest word to each of these words and extract the value in the next column

    words = ["energy", "protein", "carbohydrate", "added sugars", "total sugars", "total fat", "sodium"]

    values = {}


    for word in words:
        closest_word = find_closest_word(word, table[0].values)
        row_index = closest_word[1]
        value = table.iloc[row_index, 1]
        value = value.strip()
        try:
            value = value.split(" ")[0]
        except:
            pass
        if value=="" or value=="-":
            value = "0"
            # print(word, value)
        elif value[-1].isalpha():
            value = value[:-1]
            # print(word, value)

        elif value[-1] == "9":
            value = value[:-2]
        elif value[-1] == "8":
            value = value[:-2]
        elif value[-1] == "7":
            value = value[:-2]
        elif value[-1] == "1":
            value = value[:-2]
        if value[-1].isalpha():
            value = value[:-2]
        # print(word, value)
        try:
            values[word] = float(value)
        except:
            values[word] = 0
        


    values
    if values["energy"] == 0:
        # Find for kcal, if found assign it as energy in values
        word = "kcal"
        words = table[1].values
        words = [str(w).lower().strip().split(" ")[-1] for w in words]
        # print(words)
        closest_word = find_closest_word(word, words)
        # print("Closest word to kcal:", closest_word)
        row_index = closest_word[1]
        # print(row_index)
        value = table.iloc[row_index, 1]
        # print("energy", value)
        value = value.strip()
        if value=="" or value=="-":
            value = "0"
            # print("energy", value)
        elif value[-1].isalpha():
            value = value[:-4]
        try:
            values["energy"] = float(value)
        except:
            values["energy"] = 0

    if values["energy"] == 0:
        # Find for kcal, if found assign it as energy in values
        word = "kcal"
        words = table[0].values
        words = [str(w).lower().strip().split(" ")[-1] for w in words]
        # print(words)
        closest_word = find_closest_word(word, words)
        # print("Closest word to kcal:", closest_word)
        row_index = closest_word[1]
        # print(row_index)
        value = table.iloc[row_index, 1]
        # print("energy", value)
        value = value.strip()
        if value=="" or value=="-":
            value = "0"
            # print("energy", value)
        elif value[-1].isalpha():
            value = value[:-4]
        try:
            values["energy"] = float(value)
        except:
            values["energy"] = 0
    return values



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'


class Food(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    protein = db.Column(db.Float)
    carbs = db.Column(db.Float)
    fats = db.Column(db.Float)
    calories = db.Column(db.Float)
    sugar = db.Column(db.Float)
    sodium = db.Column(db.Float)

    def __init__(self,name,protein,carbs,fats,calories,sugar,sodium):
        self.name = name
        self.protein = protein
        self.carbs = carbs
        self.fats = fats
        self.calories = calories
        self.sugar = sugar
        self.sodium = sodium

with app.app_context():
    db.create_all()


            

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    day1_date = db.Column(db.Date)
    day1_protein = db.Column(db.Float)
    day1_carbs = db.Column(db.Float)
    day1_fats = db.Column(db.Float)
    day1_calories = db.Column(db.Float)
    day1_sugar = db.Column(db.Float)
    day1_sodium = db.Column(db.Float)
    day2_date = db.Column(db.Date)
    day2_protein = db.Column(db.Float)
    day2_carbs = db.Column(db.Float)
    day2_fats = db.Column(db.Float)
    day2_calories = db.Column(db.Float)
    day2_sugar = db.Column(db.Float)
    day2_sodium = db.Column(db.Float)
    day3_date = db.Column(db.Date)
    day3_protein = db.Column(db.Float)
    day3_carbs = db.Column(db.Float)
    day3_fats = db.Column(db.Float)
    day3_calories = db.Column(db.Float)
    day3_sugar = db.Column(db.Float)
    day3_sodium = db.Column(db.Float)
    day4_date = db.Column(db.Date)
    day4_protein = db.Column(db.Float)
    day4_carbs = db.Column(db.Float)
    day4_fats = db.Column(db.Float)
    day4_calories = db.Column(db.Float)
    day4_sugar = db.Column(db.Float)
    day4_sodium = db.Column(db.Float)
    day5_date = db.Column(db.Date)
    day5_protein = db.Column(db.Float)
    day5_carbs = db.Column(db.Float)
    day5_fats = db.Column(db.Float)
    day5_calories = db.Column(db.Float)
    day5_sugar = db.Column(db.Float)
    day5_sodium = db.Column(db.Float)
    day6_date = db.Column(db.Date)
    day6_protein = db.Column(db.Float)
    day6_carbs = db.Column(db.Float)
    day6_fats = db.Column(db.Float)
    day6_calories = db.Column(db.Float)
    day6_sugar = db.Column(db.Float)
    day6_sodium = db.Column(db.Float)
    day7_date = db.Column(db.Date)
    day7_protein = db.Column(db.Float)
    day7_carbs = db.Column(db.Float)
    day7_fats = db.Column(db.Float)
    day7_calories = db.Column(db.Float)
    day7_sugar = db.Column(db.Float)
    day7_sodium = db.Column(db.Float)



    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        self.day1_date = None
        self.day1_protein = 0
        self.day1_carbs = 0
        self.day1_fats = 0
        self.day1_calories = 0
        self.day1_sugar = 0
        self.day1_sodium = 0
        self.day2_date = None
        self.day2_protein = 0
        self.day2_carbs = 0
        self.day2_fats = 0
        self.day2_calories = 0
        self.day2_sugar = 0
        self.day2_sodium = 0
        self.day3_date = None
        self.day3_protein = 0
        self.day3_carbs = 0
        self.day3_fats = 0
        self.day3_calories = 0
        self.day3_sugar = 0
        self.day3_sodium = 0
        self.day4_date = None
        self.day4_protein = 0
        self.day4_carbs = 0
        self.day4_fats = 0
        self.day4_calories = 0
        self.day4_sugar = 0
        self.day4_sodium = 0
        self.day5_date = None
        self.day5_protein = 0
        self.day5_carbs = 0
        self.day5_fats = 0
        self.day5_calories = 0
        self.day5_sugar = 0
        self.day5_sodium = 0
        self.day6_date = None
        self.day6_protein = 0
        self.day6_carbs = 0
        self.day6_fats = 0
        self.day6_calories = 0
        self.day6_sugar = 0
        self.day6_sodium = 0
        self.day7_date = None
        self.day7_protein = 0
        self.day7_carbs = 0
        self.day7_fats = 0
        self.day7_calories = 0
        self.day7_sugar = 0
        self.day7_sodium = 0

    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')



    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html',error='Invalid user')
    return render_template('login.html')


@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        # If today's date is not set, set it
        if user.day1_date == None:
            user.day1_date = datetime.datetime.now().date()
            # Set the other days to today's date - 1
        if user.day2_date == None:
            user.day2_date = datetime.datetime.now().date() - datetime.timedelta(days=1)
            user.day3_date = datetime.datetime.now().date() - datetime.timedelta(days=2)
            user.day4_date = datetime.datetime.now().date() - datetime.timedelta(days=3)
            user.day5_date = datetime.datetime.now().date() - datetime.timedelta(days=4)
            user.day6_date = datetime.datetime.now().date() - datetime.timedelta(days=5)
            user.day7_date = datetime.datetime.now().date() - datetime.timedelta(days=6)
            db.session.commit()
        # If today's date is not equal to day_date, shift all days
        if user.day1_date != datetime.datetime.now().date():
            user.day7_date = user.day6_date
            user.day6_date = user.day5_date
            user.day5_date = user.day4_date
            user.day4_date = user.day3_date
            user.day3_date = user.day2_date
            user.day2_date = user.day1_date
            user.day1_date = datetime.datetime.now().date()
            user.day7_protein = user.day6_protein
            user.day6_protein = user.day5_protein
            user.day5_protein = user.day4_protein
            user.day4_protein = user.day3_protein
            user.day3_protein = user.day2_protein
            user.day2_protein = user.day1_protein
            user.day1_protein = 0
            user.day7_carbs = user.day6_carbs
            user.day6_carbs = user.day5_carbs
            user.day5_carbs = user.day4_carbs
            user.day4_carbs = user.day3_carbs
            user.day3_carbs = user.day2_carbs
            user.day2_carbs = user.day1_carbs
            user.day1_carbs = 0
            user.day7_fats = user.day6_fats
            user.day6_fats = user.day5_fats
            user.day5_fats = user.day4_fats
            user.day4_fats = user.day3_fats
            user.day3_fats = user.day2_fats
            user.day2_fats = user.day1_fats
            user.day1_fats = 0
            user.day7_calories = user.day6_calories
            user.day6_calories = user.day5_calories
            user.day5_calories = user.day4_calories
            user.day4_calories = user.day3_calories
            user.day3_calories = user.day2_calories
            user.day2_calories = user.day1_calories
            user.day1_calories = 0
            user.day7_sugar = user.day6_sugar
            user.day6_sugar = user.day5_sugar
            user.day5_sugar = user.day4_sugar
            user.day4_sugar = user.day3_sugar
            user.day3_sugar = user.day2_sugar
            user.day2_sugar = user.day1_sugar
            user.day1_sugar = 0
            user.day7_sodium = user.day6_sodium
            user.day6_sodium = user.day5_sodium
            user.day5_sodium = user.day4_sodium
            user.day4_sodium = user.day3_sodium
            user.day3_sodium = user.day2_sodium
            user.day2_sodium = user.day1_sodium
            user.day1_sodium = 0
            db.session.commit()
                
            

        return render_template('dashboard.html',user=user, max_consumption=max_consumption)
    
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')

@app.route('/add_food',methods=['POST', 'GET'])
# Render the template of add_food
def add_food():
        
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        foods = Food.query.all()
        if request.method == 'POST':
            quantity = int(request.form['quantity'])
            foodsel = request.form['foodsel']
            food = Food.query.filter_by(name=foodsel).first()
            user.day1_protein += food.protein * quantity
            user.day1_carbs += food.carbs * quantity
            user.day1_fats += food.fats * quantity
            user.day1_calories += food.calories * quantity
            user.day1_sugar += food.sugar * quantity
            user.day1_sodium += food.sodium * quantity
            db.session.commit()
        return render_template('add_food.html',foods=foods)
    return redirect('/login')
        
@app.route('/reset_today')
def reset_today():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        user.day1_protein = 0
        user.day1_carbs = 0
        user.day1_fats = 0
        user.day1_calories = 0
        user.day1_sugar = 0
        user.day1_sodium = 0
        db.session.commit()
        return redirect('/dashboard')
    return redirect('/login')

def strtoint(x):
    if x == '':
        return 0
    return int(x)

@app.route('/add_custom_food',methods=['POST', 'GET'])
def add_custom_food():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        if request.method == 'POST':
            name = request.form['name']
            protein = request.form['protein']
            carbs = request.form['carbs']
            fats = request.form['fats']
            calories = request.form['calories']
            sugar = request.form['sugar']
            sodium = request.form['sodium']
            new_food = Food(name=name,protein=protein,carbs=carbs,fats=fats,calories=calories,sugar=sugar,sodium=sodium)
            db.session.add(new_food)
            db.session.commit()
            return redirect('/dashboard')
        return render_template('add_custom_food.html')
    return redirect('/login')

@app.route('/view_custom_food')
def view_custom_food():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        foods = Food.query.all()
        return render_template('view_custom_food.html',foods=foods)
    return redirect('/login')

@app.route('/delete_custom_food/<int:id>')
def delete_custom_food(id):
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        food = Food.query.filter_by(id=id).first()
        db.session.delete(food)
        db.session.commit()
        return redirect('/view_custom_food')
    return redirect('/login')

@app.route('/scan_food',methods=['POST', 'GET'])
def scan_food():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        if request.method == 'POST':
            # Obtain the image, servings and total_servings
            image = request.files['image']
            servings = request.form['servings']
            total_servings = request.form['total_servings']
            # Get the label
            # Get the absolute path
            image.save(image.filename)

            print("image.filename: ", image.filename)
            label = read_label(image.filename)
            print("label: ", label)
            # Get the food
            # label = label:  {'energy': 572.0, 'protein': 8.0, 'carbohydrate': 47.0, 'added sugars': 0.0, 'total sugars': 0.0, 'total fat': 37.0, 'sodium': 984.0}
            label_proteins = label['protein']
            label_carbs = label['carbohydrate']
            label_fats = label['total fat']
            label_calories = label['energy']
            label_sugar = label['total sugars']
            label_sodium = label['sodium']

            
            # Add the food to the user
            user.day1_protein += label_proteins * int(servings) / int(total_servings)
            user.day1_carbs += label_carbs * int(servings) / int(total_servings)
            user.day1_fats += label_fats * int(servings) / int(total_servings)
            user.day1_calories += label_calories * int(servings) / int(total_servings)
            user.day1_sugar += label_sugar * int(servings) / int(total_servings)
            user.day1_sodium += label_sodium * int(servings) / int(total_servings)
            db.session.commit()
            return redirect('/dashboard')
        return render_template('scan_food.html')
    return redirect('/login')


if __name__ == '__main__':
    app.run(debug=True)



