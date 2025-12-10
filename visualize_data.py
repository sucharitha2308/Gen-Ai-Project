import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
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

def visualize():
    print("Loading data...")
    try:
        df = pd.read_csv('emails.csv', nrows=2000) # Use subset for visualization speed
    except FileNotFoundError:
        print("emails.csv not found")
        return

    print("Parsing and Labeling...")
    df['parsed_content'] = df['message'].apply(parse_raw_message)
    df['category'] = df['parsed_content'].apply(auto_label)

    print("Vectorizing...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    matrix = tfidf.fit_transform(df['parsed_content'])

    print("Running PCA...")
    pca = PCA(n_components=3)
    components = pca.fit_transform(matrix.toarray())

    viz_df = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
    viz_df['Category'] = df['category']

    print("Generating Pairplot...")
    sns.set_theme(style="ticks")
    pairplot = sns.pairplot(viz_df, hue='Category', palette='bright')
    
    output_file = 'pairplot.png'
    pairplot.savefig(output_file)
    print(f"Pairplot saved to {output_file}")

if __name__ == "__main__":
    visualize()
