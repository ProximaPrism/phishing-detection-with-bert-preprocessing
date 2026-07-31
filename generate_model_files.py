import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from email.utils import parseaddr

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import keras
import keras_hub
import tensorflow as tf

import joblib

# =======================
# EDA
# =======================
df = pd.read_csv('data/AVN_Corpus.csv')

df["subject"] = df["subject"].fillna("No Subject")
df["sender"] = df["sender"].fillna("Unknown Sender")
df["receiver"] = df["receiver"].fillna("Unknown Receiver")

df["is_date_invalid"] = df["date"].isna().astype(np.float32) # 1 if invalid, 0 if valid

df['date'] = pd.to_datetime(df['date'], errors="coerce", utc=True)

df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

df = df.drop(['date', 'month', 'year'], axis=1)

def parse_sender_receiver(raw):
    name, email = parseaddr(raw)
    name = name.strip().replace('\\"', '"').strip('"').strip()
    email = email.lower().strip()
    return name, email

def extract_email_components(email):
    if '@' not in email:
        return "", ""
    username, domain = email.split('@', 1)
    return username, domain

df[["sender_displayname", "sender_email"]] = df['sender'].apply(parse_sender_receiver).apply(pd.Series)
df[["receiver_displayname", "receiver_email"]] = df["receiver"].apply(parse_sender_receiver).apply(pd.Series)

df[["sender_username", "sender_domain"]] = df["sender_email"].apply(extract_email_components).apply(pd.Series)
df[["receiver_username", "receiver_domain"]] = df["receiver_email"].apply(extract_email_components).apply(pd.Series)

df["is_sender_displayname_missing"] = (df["sender_displayname"] == "").astype(int)
df["is_sender_email_missing"] = (df["sender_email"].str.contains("@", na=False) == False).astype(int)

df["sender_email_digit_count"] = df["sender_email"].str.count(r"\d")
df["sender_email_has_hyphens"] = df["sender_email"].str.contains("-").astype(int)

df['sender_username_length'] = df['sender_username'].str.len()
df['receiver_username_length'] = df['receiver_username'].str.len()

df['sender_domain_length'] = df['sender_domain'].str.len()
df['receiver_domain_length'] = df['receiver_domain'].str.len()

# drop the sender and receiver column, as well as the other temporary features
df = df.drop(['sender', 'sender_displayname', 'sender_email', 'sender_username', 'sender_domain'], axis=1)
df = df.drop(['receiver', 'receiver_displayname', 'receiver_email', 'receiver_username', 'receiver_domain'], axis=1)

df = df.drop(['receiver_username_length', 'receiver_domain_length'], axis=1)

# =======================
# Data Cleaning
# =======================
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

df = df.drop(['hour', 'day_of_week'], axis=1)

# handle any null values
df[['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']] = df[['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos']].fillna(0)

df["does_body_contains_urls"] = df["urls"]
df = df.drop('urls', axis=1)

# =======================
# preprocessing
# =======================
X = df.drop(['label'], axis=1)
y = df['label']

X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_validate, X_test, y_validate, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=42, stratify=y_remain)

X_train_text = X_train[["subject", "body"]]
X_test_text = X_test[["subject", "body"]]
X_validate_text = X_validate[["subject", "body"]]

X_train = X_train.drop(['subject', 'body'], axis=1)
X_test = X_test.drop(['subject', 'body'], axis=1)
X_validate = X_validate.drop(['subject', 'body'], axis=1)

# save the feature order that is used
feature_order = X_train.columns.tolist()
joblib.dump(feature_order, './app/models/feature_order.pkl')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validate_scaled = scaler.transform(X_validate)

# save the scaler
joblib.dump(scaler, './app/models/scaler.pkl') # use regular .transform() for inference

X_train_text = (
    X_train_text["subject"] +
    " [SEP] " +
    X_train_text["body"]
)

X_validate_text = (
    X_validate_text["subject"] +
    " [SEP] " +
    X_validate_text["body"]
)

X_test_text = (
    X_test_text["subject"] +
    " [SEP] " +
    X_test_text["body"]
)

# gpu optimization
keras.mixed_precision.set_global_policy("mixed_float16")

preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
    "bert_base_en_uncased",
    sequence_length=512,
)

train_tokens = preprocessor(tf.constant(X_train_text.tolist(), dtype=tf.string))
validate_tokens = preprocessor(tf.constant(X_validate_text.tolist(), dtype=tf.string))
test_tokens = preprocessor(tf.constant(X_test_text.tolist(), dtype=tf.string))

backbone = keras_hub.models.BertBackbone.from_preset(
    "bert_base_en_uncased",
)

backbone.trainable = False

token_input = keras.Input(
    shape=(512,),
    dtype="int32",
    name="token_ids"
)

segment_input = keras.Input(
    shape=(512,),
    dtype="int32",
    name="segment_ids"
)

mask_input = keras.Input(
    shape=(512,),
    dtype="int32",
    name="padding_mask"
)

outputs = backbone({
    "token_ids": token_input,
    "segment_ids": segment_input,
    "padding_mask": mask_input
})

embedding = keras.Model(
    inputs=[
        token_input,
        segment_input,
        mask_input
    ],
    outputs=outputs["pooled_output"]
)

#%%
#train_token_dataset = (
#    tf.data.Dataset
#    .from_tensor_slices(train_tokens)
#    .batch(64)
#    .prefetch(tf.data.AUTOTUNE)
#)
#
#test_token_dataset = (
#    tf.data.Dataset
#    .from_tensor_slices(test_tokens)
#    .batch(64)
#    .prefetch(tf.data.AUTOTUNE)
#)
#
#validate_token_dataset = (
#    tf.data.Dataset
#    .from_tensor_slices(validate_tokens)
#    .batch(64)
#    .prefetch(tf.data.AUTOTUNE)
#)

#X_train_embeddings = embedding.predict(train_token_dataset)
#X_test_embeddings = embedding.predict(test_token_dataset)
#X_validate_embeddings = embedding.predict(validate_token_dataset)

#np.save("./embeddings/X_train_embeddings.npy", X_train_embeddings)
#np.save("./embeddings/X_test_embeddings.npy", X_test_embeddings)
#np.save("./embeddings/X_validate_embeddings.npy", X_validate_embeddings)

X_train_embeddings = np.load("./embeddings/X_train_embeddings.npy")
X_test_embeddings = np.load("./embeddings/X_test_embeddings.npy")
X_validate_embeddings = np.load("./embeddings/X_validate_embeddings.npy")

print(X_train_embeddings.shape)
print(X_test_embeddings.shape)
print(X_validate_embeddings.shape)

X_train_classifier = np.concatenate(
    [
        X_train_embeddings,
        X_train_scaled
    ],
    axis=1
)
# =======================
# Model Building
# =======================
classifier = keras.Sequential([
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(2, activation='softmax'), # generates 2 classes, with probability [indexes: 0 => normal, 1 => phishing]
])

classifier.compile(
    optimizer=keras.optimizers.Adam(2e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =======================
# Model Training
# =======================
classifier.fit(
    X_train_classifier,
    y_train,
    validation_data=(
        np.concatenate(
            [X_validate_embeddings, X_validate_scaled],
            axis=1
        ),
        y_validate
    ),
    epochs=20,
    batch_size=32
)
#%%
X_test_classifier = np.concatenate(
    [
        X_test_embeddings,
        X_test_scaled
    ],
    axis=1
)

# =======================
# Evaluation
# =======================
test_loss, test_accuracy = classifier.evaluate(
    X_test_classifier,
    y_test
)

print(test_accuracy)

y_pred_probs = classifier.predict(
    X_test_classifier
)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "normal",
            "phishing"
        ]
    )
)

cm = confusion_matrix(y_test, y_pred)

print(cm)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["normal", "phishing"],
    yticklabels=["normal", "phishing"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

probs = classifier.predict(
    X_test_classifier
)

errors_df = pd.DataFrame({
    "text": X_test_text,
    "actual": y_test,
    "predicted": y_pred,
    "phishing_probability": probs[:,1]
})

errors_df = errors_df[
    errors_df.actual != errors_df.predicted
]

errors_df.sort_values(
    "phishing_probability"
)

classifier.save("./app/models/classifier.keras")