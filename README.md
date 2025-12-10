# How to Run the Email Classification Project

Follow these steps to run the project in VS Code.

## 1. Prerequisites
Ensure you have Python installed. You can check by running:
```bash
python --version
```

## 2. Setup Environment
Open a new terminal in VS Code (`Ctrl + \``) and install the required libraries:

```bash
pip install flask pandas numpy scikit-learn nltk chromadb sentence-transformers flask-sqlalchemy flask-login
```

## 3. First Time Setup (One-Time Only)
Run this command **only once** to create the database and index your emails:
```bash
python init_vectordb.py
```

## 4. How to Run (Daily)
Every time you open VS Code to work on the project, just run:
```bash
python app.py
```

## 5. Access the App
Open your web browser and go to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

## 6. Usage
1.  **Register**: Click "Create Account" on the login page to sign up.
2.  **Login**: Use your new username and password.
3.  **Classify**: Paste an email content and click "Analyze".

## Troubleshooting
- If you see "ModuleNotFoundError", make sure you ran the `pip install` command in Step 2.
- If the app says "Could not connect to ChromaDB", make sure you ran `python init_vectordb.py` successfully.
