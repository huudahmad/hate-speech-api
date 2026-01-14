import pandas as pd
import re
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
print("Downloading data")
df = pd.read_csv(url)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Cleaning text data")
df['cleaned_text'] = df['tweet'].apply(clean_text)

#only need to determine between clean and hate speech, so combine offensive with clean for this purpose
df['label'] = df['class'].apply(lambda x: 1 if x == 0 else 0)


df = df[['cleaned_text', 'label']]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)