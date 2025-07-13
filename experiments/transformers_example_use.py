from transformers import AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import pipeline
from scipy.special import softmax

import numpy as np

text = "AMDs stock took a hit today as investors panic-sold amidst fears of declining chip demand! With the tech sector showing weakness, are we witnessing a bearish reversal? #TechSelloff #Chips #MarketPanic"

# twitter-roberta model: example of usage
# - based on the BERT transformer encoder architecture

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

model = AutoModelForSequenceClassification.from_pretrained(MODEL)


encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

ranking = np.argsort(scores)[::-1]
resdf = []
output_scores_raw = {}
output_scores_sorted = {"positive": 0, "neutral": 0, "negative": 0}
for i in range(scores.shape[0]):
    label = config.id2label[ranking[i]]
    score = np.round(float(scores[ranking[i]]), 4)
    output_scores_raw[label] = score
    
output_scores_sorted["positive"] = output_scores_raw["positive"]
output_scores_sorted["neutral"] = output_scores_raw["neutral"]
output_scores_sorted["negative"] = output_scores_raw["negative"]   

resdf.append(output_scores_sorted) 

print(output_scores_raw)
print(output_scores_sorted)


# finbert model: example of usage
# based on the BERT transformer encoder architecture, fine-tuned for financial sentiment analysis
# doesn't really work well on exremes data points

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
score = nlp(text)

print(score)