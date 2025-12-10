import pandas as pd
import os

def parse_raw_message(raw_message):
    lines = raw_message.split('\n')
    body = []
    reading_body = False
    for line in lines:
        if reading_body:
            body.append(line)
        elif line.strip() == "":
            reading_body = True
    return "\n".join(body).strip()

def auto_label(text):
    text = text.lower()
    
    financial_keywords = ['budget', 'invoice', 'purchase', 'financial', 'report', 'quarterly', 'bank', 'money', 'expense', 'cost']
    urgent_keywords = ['urgent', 'immediate', 'emergency', 'deadline', 'breach', 'asap', 'critical', 'alert', 'warning']
    hr_keywords = ['hr', 'policies', 'performance', 'review', 'insurance', 'promotion', 'holiday', 'leave', 'benefits', 'hiring', 'salary']
    
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

def check():
    csv_path = 'emails.csv'
    if not os.path.exists(csv_path):
        print("emails.csv not found")
        return

    print("Loading first 5000 rows...")
    df = pd.read_csv(csv_path, nrows=5000)
    
    print("Parsing...")
    df['parsed_content'] = df['message'].apply(parse_raw_message)
    
    print("Labeling...")
    df['category'] = df['parsed_content'].apply(auto_label)
    
    print("Distribution:")
    print(df['category'].value_counts())

if __name__ == "__main__":
    check()
