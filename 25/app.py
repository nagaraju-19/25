from flask import Flask, render_template, request, redirect, session
import sqlite3
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "agrodetect"

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load AI Model
model = tf.keras.models.load_model("model/plant_model.h5")

classes = [
"Tomato Leaf Spot",
"Potato Early Blight",
"Grape Black Rot"
]


def predict_image(img_path):

    img = Image.open(img_path).resize((224,224))
    img = np.array(img)/255
    img = img.reshape(1,224,224,3)

    prediction = model.predict(img)

    index = np.argmax(prediction)

    return classes[index]


def get_solution(disease):

    conn = sqlite3.connect("database/plant_disease.db")
    cursor = conn.cursor()

    cursor.execute("SELECT cure,pesticide,amount FROM diseases WHERE disease=?", (disease,))
    result = cursor.fetchone()

    conn.close()

    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login",methods=["GET","POST"])
def login():

    if request.method=="POST":

        email=request.form["email"]
        password=request.form["password"]

        conn=sqlite3.connect("database/plant_disease.db")
        cursor=conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email=? AND password=?",(email,password))

        user=cursor.fetchone()

        conn.close()

        if user:
            session["user"]=email
            return redirect("/dashboard")

    return render_template("login.html")


@app.route("/register",methods=["GET","POST"])
def register():

    if request.method=="POST":

        name=request.form["name"]
        email=request.form["email"]
        password=request.form["password"]

        conn=sqlite3.connect("database/plant_disease.db")
        cursor=conn.cursor()

        cursor.execute("INSERT INTO users(name,email,password) VALUES(?,?,?)",(name,email,password))

        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


@app.route("/dashboard",methods=["GET","POST"])
def dashboard():

    if request.method=="POST":

        file=request.files["image"]

        path=os.path.join(app.config["UPLOAD_FOLDER"],file.filename)

        file.save(path)

        disease=predict_image(path)

        cure,pesticide,amount=get_solution(disease)

        return render_template(
        "dashboard.html",
        disease=disease,
        cure=cure,
        pesticide=pesticide,
        amount=amount)

    return render_template("dashboard.html")


if __name__=="__main__":
    app.run(debug=True)