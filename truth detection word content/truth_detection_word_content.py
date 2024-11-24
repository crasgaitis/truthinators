#using DistilRoberta from hugging face
#sota sentiment analysis model -> change last layer, to binary classification (truth/lie)
#https://huggingface.co/distilbert/distilroberta-base/blob/main/README.md

#truth = 0, lie = 1 

#pip installs needed to run this code: 
#   pip install transformers datasets torch
from transformers import pipeline, Trainer, TrainingArguments
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch
from datasets import load_dataset

#using the og model: 
'''unmasker = pipeline('fill-mask', model='distilroberta-base')
#huh = unmasker("The woman worked as a <mask>.")
print(huh)'''


dataset = load_dataset('csv', data_files={'train': 'dataset.csv'}, column_names=['text', 'label'])
#print(dataset['train'][0]) #yay it works

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_text(data):
    return tokenizer(data['text'], padding="max_length", truncation=True)

encoded_dataset = dataset.map(tokenize_text, batched=True)

encoded_dataset = encoded_dataset['train'].train_test_split(test_size=0.2)
encoded_dataset['val'] = encoded_dataset['test'].train_test_split(test_size=0.5)['train']
encoded_dataset['test'] = encoded_dataset['test'].train_test_split(test_size=0.5)['test']

print(encoded_dataset)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
#last layer modified to output truth/lie here: 
model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(model.config.hidden_size, 2))

training_details = TrainingArguments(
    output_dir='./model_outputs',   #idk
    num_train_epochs=100, #HYPERPARAM TUNE THESE A LITTLE, and add in other hyperparms we want
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=16, 
    warmup_steps=500, 
    weight_decay=0.01
)

trainer = Trainer(
    model=model, 
    args=training_details, 
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['val'], 
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained('./deception_detection_model')
tokenizer.save_pretrained('./deception_detection_model')


test_results = trainer.evaluate(eval_dataset=encoded_dataset['test'])
print(test_results)