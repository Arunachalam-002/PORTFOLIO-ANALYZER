from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required
from bson.objectid import ObjectId
from db import mongo
from models import User

auth = Blueprint('auth', __name__)

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        full_name = request.form['full_name']
        password = request.form['password']
        phone = request.form['phone']
        goal = request.form['goal']

        if mongo.db.users.find_one({"email": email}):
            flash("Email already registered.")
            return redirect(url_for('auth.signup'))

        hashed = generate_password_hash(password)
        mongo.db.users.insert_one({
            "email": email,
            "full_name": full_name,
            "password": hashed,
            "phone": phone,
            "goal": goal
        })
        return redirect(url_for('auth.login'))

    return render_template("signup.html")

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        user_doc = mongo.db.users.find_one({"email": email})

        if user_doc and check_password_hash(user_doc["password"], request.form['password']):
            user = User(user_doc)
            login_user(user)
            return redirect(url_for('home'))
        flash("Invalid credentials.")
    return render_template("login.html")

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
