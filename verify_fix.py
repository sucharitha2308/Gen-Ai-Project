import pickle
import os

def auto_label(text):
    """Heuristic to label emails based on keywords (synced with app.py)."""
    text = text.lower()
    
    financial_keywords = ['budget', 'invoice', 'purchase', 'financial', 'report', 'quarterly', 'bank', 'money', 'expense', 'cost', 'payment', 'transaction', 'audit', 'billing']
    urgent_keywords = ['urgent', 'immediate', 'emergency', 'deadline', 'breach', 'asap', 'critical', 'alert', 'warning', 'high priority', 'immediate action']
    hr_keywords = ['hr', 'policies', 'performance', 'review', 'insurance', 'promotion', 'holiday', 'leave', 'benefits', 'hiring', 'salary', 'recruitment', 'onboarding', 'resignation', 'interview']
    
    for word in urgent_keywords:
        if word in text:
            return 'Urgent'
    for word in financial_keywords:
        if word in text:
            return 'Financial'
    for word in hr_keywords:
        if word in text:
            return 'HR'
            
    return 'General'

def verify():
    print("Loading model and vectorizer...")
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        print("Model files not found!")
        return

    test_cases = [
        ("This is an urgent deadline for the project.", "Urgent"),
        ("Please process this invoice for payment.", "Financial"),
        ("I would like to discuss my performance review.", "HR"),
        ("Let's grab lunch tomorrow.", "General"),
        ("Immediate action required on the server breach.", "Urgent"),
        ("The quarterly financial report is attached.", "Financial"),
        ("New employee onboarding starts next week.", "HR"),
        ("Can you send me the meeting notes?", "General")
    ]

    print("\nRunning Verification Tests:")
    print("-" * 60)
    print(f"{'Test Input':<40} | {'Expected':<10} | {'Predicted':<10} | {'Result'}")
    print("-" * 60)

    for text, expected in test_cases:
        # 1. Try Heuristic
        prediction = auto_label(text)
        
        # 2. Fallback to Model if General
        if prediction == 'General':
             text_vectorized = vectorizer.transform([text])
             model_pred = model.predict(text_vectorized)[0]
             # Note: logic in app.py prefers heuristic if not General. 
             # If heuristic is General, it uses model.
             # But here, if heuristic says General, we use model.
             # If heuristic says something else, we use that.
             prediction = model_pred

        status = "PASS" if prediction == expected else "FAIL"
        print(f"{text[:37]+'...':<40} | {expected:<10} | {prediction:<10} | {status}")

if __name__ == "__main__":
    verify()
