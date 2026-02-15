import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and prepare data
df = pd.read_csv('dataset_en_clean.csv')

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Number of unique queues: {df['queue'].nunique()}")
print(f"Queues: {df['queue'].unique()}")

# Check class distribution
print("\nClass distribution:")
print(df['queue'].value_counts())

# Prepare labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['queue'])
num_classes = len(label_encoder.classes_)
print(f"\nNumber of classes: {num_classes}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, range(num_classes)))}")

# Split data with stratification
X_train, X_temp, y_train, y_temp = train_test_split(
    df['clean_text'].values, 
    df['label'].values, 
    test_size=0.3, 
    random_state=42,
    stratify=df['label'].values
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, 
    y_temp, 
    test_size=0.5, 
    random_state=42,
    stratify=y_temp
)

print(f"\nTrain size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

# Compute class weights for handling imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\nClass weights: {class_weights}")

# Custom Dataset class
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=384):  # Increased max_len
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Initialize tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Create datasets with longer max_len for better context
max_len = 384
batch_size = 16

train_dataset = TicketDataset(X_train, y_train, tokenizer, max_len)
val_dataset = TicketDataset(X_val, y_val, tokenizer, max_len)
test_dataset = TicketDataset(X_test, y_test, tokenizer, max_len)

# Create weighted sampler for training data
sample_weights = [class_weights[y].item() for y in y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    output_attentions=False,
    output_hidden_states=False
)
model = model.to(device)

# Training configuration - adjusted hyperparameters
epochs = 8  # Increased epochs
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Added weight decay
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps
)

# Training function with class weights
def train_epoch(model, data_loader, optimizer, scheduler, device, class_weights):
    model.train()
    losses = []
    correct_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Apply class weights to loss
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation function
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), all_predictions, all_labels

# Training loop
print("\nStarting training...")
best_accuracy = 0

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 50)
    
    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
    print(f'Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.4f}')
    
    val_acc, val_loss, _, _ = eval_model(model, val_loader, device)
    print(f'Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.4f}')
    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'best_model_improved.pt')
        print("Model saved!")
    
    print()

# Load best model and evaluate on test set
model.load_state_dict(torch.load('best_model_improved.pt', map_location=device))
test_acc, test_loss, test_preds, test_labels = eval_model(model, test_loader, device)
print(f'\nTest accuracy: {test_acc:.4f}')

# Detailed classification report
print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_, zero_division=0))

# Prediction function with probability threshold
def predict_ticket(text, model, tokenizer, label_encoder, device, max_len=384, threshold=0.3):
    model.eval()
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        max_prob, prediction = torch.max(probabilities, dim=1)
    
    # Check if confidence is above threshold
    if max_prob.item() < threshold:
        print(f"\nLow confidence prediction ({max_prob.item():.4f} < {threshold})")
        print("Getting top 3 predictions instead:")
        top_probs, top_indices = torch.topk(probabilities[0], 3)
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            dept = label_encoder.inverse_transform([idx.cpu().numpy()])[0]
            print(f"  {i+1}. {dept}: {prob.item():.4f}")
        return None, max_prob.item()
    
    predicted_label = label_encoder.inverse_transform(prediction.cpu().numpy())[0]
    confidence = max_prob.item()
    
    print(f"\nPredicted department: {predicted_label} (confidence: {confidence:.4f})")
    return predicted_label, confidence

# Test with example tickets
print("\n" + "="*60)
print("TESTING WITH EXAMPLE TICKETS")
print("="*60)

test_tickets = [
    "I cannot log into my account, it says password incorrect",
    "I want to return this product I bought yesterday, it's defective",
    "My credit card was charged twice for the same subscription",
    "How much does your enterprise plan cost for 50 users?",
    "The website is down and showing 502 error",
    "When is the next maintenance window for the servers?",
    "I need help with my employee benefits"
]

for ticket in test_tickets:
    print(f"\nTicket: {ticket}")
    predict_ticket(ticket, model, tokenizer, label_encoder, device)
    print("-"*40)