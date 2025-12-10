from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import os
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# --- Database Configuration (SQLite) ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Vector Database Setup (ChromaDB) ---
try:
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_collection(name="email_collection", embedding_function=sentence_transformer_ef)
    print("Connected to ChromaDB.")
except Exception as e:
    print(f"Warning: Could not connect to ChromaDB. Ensure init_vectordb.py has been run. Error: {e}")
    collection = None

# --- Helper Functions ---
def auto_label(text):
    """Heuristic to label emails based on keywords."""
    text = text.lower()
    financial_keywords = ['budget', 'invoice', 'purchase', 'financial', 'report', 'quarterly', 'bank', 'money', 'expense', 'cost', 'payment', 'transaction', 'audit', 'billing']
    urgent_keywords = ['urgent', 'immediate', 'emergency', 'deadline', 'breach', 'asap', 'critical', 'alert', 'warning', 'high priority', 'immediate action']
    hr_keywords = ['hr', 'policies', 'performance', 'review', 'insurance', 'promotion', 'holiday', 'leave', 'benefits', 'hiring', 'salary', 'recruitment', 'onboarding', 'resignation', 'interview']
    
    for word in urgent_keywords:
        if word in text: return 'Urgent'
    for word in financial_keywords:
        if word in text: return 'Financial'
    for word in hr_keywords:
        if word in text: return 'HR'
    return 'General'

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    prediction = None
    email_text = ""
    
    if request.method == 'POST':
        email_text = request.form['email']
        
        # 1. Try Heuristic First
        heuristic_pred = auto_label(email_text)
        
        if heuristic_pred != 'General':
            prediction = heuristic_pred
            print(f"Heuristic Prediction: {prediction}")
        elif collection:
            # 2. Vector Search (Semantic Classification)
            try:
                results = collection.query(
                    query_texts=[email_text],
                    n_results=5
                )
                
                # Get categories of nearest neighbors
                metadatas = results['metadatas'][0]
                categories = [m['category'] for m in metadatas]
                
                # Majority Vote
                if categories:
                    prediction = max(set(categories), key=categories.count)
                    print(f"Vector DB Prediction: {prediction} (Neighbors: {categories})")
                else:
                    prediction = "General"
            except Exception as e:
                print(f"Vector DB Error: {e}")
                prediction = "Error"
        else:
            prediction = "System Not Initialized"
            
    return render_template('index.html', prediction=prediction, email_text=email_text, username=current_user.username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user:
            flash('Username already exists.', 'error')
        else:
            new_user = User(username=username, password=generate_password_hash(password, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please login.', 'success')
            return redirect(url_for('login'))
            
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # Create tables if they don't exist
    app.run(debug=True)
