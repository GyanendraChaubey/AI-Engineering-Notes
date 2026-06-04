# Generative AI Engineer (Part 1) — Interview 17

**Q: Start building a simple classification model with feature engineering and an API wrapper (pseudo code is acceptable).**

- I will outline the steps to build a simple classification model, perform basic feature engineering, train the model, save it, and then create an API wrapper using FastAPI.
- This approach is practical and commonly used in industry for rapid prototyping and deployment of ML models.

**Steps:**
- Load a dataset (for example, from CSV).
- Perform basic feature engineering (e.g., text cleaning, encoding).
- Train a simple classifier (e.g., Logistic Regression or SetFit for text).
- Save the trained model and any encoders.
- Build a FastAPI wrapper to load the model and serve predictions.

**Sample Pseudo Code:**

```python
# 1. Import necessary libraries
import pandas as pd # For data handling
from sklearn.model_selection import train_test_split # For splitting data
from sklearn.feature_extraction.text import TfidfVectorizer # For text feature extraction
from sklearn.linear_model import LogisticRegression # For classification
import pickle # For saving/loading models
from fastapi import FastAPI, Request # For API
import uvicorn # For running the API

# 2. Load dataset
df = pd.read_csv('data.csv') # Load data from CSV

# 3. Basic feature engineering
df['text_clean'] = df['text'].str.lower().str.strip() # Lowercase and strip text

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(df['text_clean'], df['label'], test_size=0.2) # Split data

# 5. Vectorize text
vectorizer = TfidfVectorizer() # Initialize vectorizer
X_train_vec = vectorizer.fit_transform(X_train) # Fit and transform train data
X_test_vec = vectorizer.transform(X_test) # Transform test data

# 6. Train model
clf = LogisticRegression() # Initialize classifier
clf.fit(X_train_vec, y_train) # Train classifier

# 7. Save model and vectorizer
pickle.dump(clf, open('model.pkl', 'wb')) # Save model
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb')) # Save vectorizer

# 8. Build FastAPI app
app = FastAPI() # Initialize FastAPI

# 9. Load model and vectorizer at startup
model = pickle.load(open('model.pkl', 'rb')) # Load model
vectorizer = pickle.load(open('vectorizer.pkl', 'rb')) # Load vectorizer

# 10. Define prediction endpoint
@app.post("/predict")
async def predict(request: Request):
 data = await request.json() # Get JSON data
 text = data['text'] # Extract text
 text_vec = vectorizer.transform([text]) # Vectorize input
 pred = model.predict(text_vec) # Predict label
 return {"prediction": pred[0]} # Return prediction

# 11. Run API (if running as script)
if __name__ == "__main__":
 uvicorn.run(app, host="0.0.0.0", port=8000) # Run FastAPI app
```

**Industry Note:**
- In my current projects (like UIDS), I use similar pipelines but with more advanced models (e.g., SetFit, CatBoost) and deploy using AWS SageMaker, with CI/CD managed via Azure DevOps or Jenkins.
- For production, I also implement robust input validation, logging, and monitoring as seen in my UIDS project, where model loading, input parsing, and prediction logic are modular and support multiple content types.

---

**Q: How do you split a time series dataset for training and testing, ensuring the split follows the order of the date column?**

- For time series data, you should not use random splitting because it can cause data leakage from the future into the past.
- Instead, you must split the data chronologically, so that all training data comes before the testing data in time.
- This approach ensures your model is trained only on past data and tested on future data, which matches real-world scenarios.

**Steps:**
- Sort the dataset by the date column in ascending order.
- Decide the split ratio (e.g., 80% train, 20% test).
- Use the first 80% of rows as training data and the last 20% as testing data.

**Example Code:**
```python
import pandas as pd # For data handling
# Assume df is your DataFrame with 'date' column

df = df.sort_values('date') # Sort by date in ascending order

split_index = int(len(df) * 0.8) # Calculate split index for 80% train

train_df = df.iloc[:split_index] # Select first 80% as training data

test_df = df.iloc[split_index:] # Select last 20% as testing data
```

- This method ensures there is no data leakage and the model is evaluated on unseen, future data only.
- In my projects, especially for intent classification and review analysis, I always use chronological splits for time-dependent data to maintain the integrity of the evaluation.

---

**Q: Is there a direct, ready-to-use function in sklearn for time series train-test splitting, or do you always use manual chronological splitting?**

- For time series data, I usually follow the manual chronological split approach: sort by date and split by index, as discussed.
- scikit-learn’s `train_test_split` is not suitable for time series because it shuffles data by default, which can cause data leakage.
- However, scikit-learn does provide a utility called `TimeSeriesSplit` in `sklearn.model_selection` for cross-validation on time series data.
 - `TimeSeriesSplit` is mainly used for cross-validation, not for a single train-test split.
 - For a single split, manual slicing is still the most common and reliable approach in industry.
- In my projects, especially for intent classification and review analysis, I use manual chronological splits for the initial train-test division, and `TimeSeriesSplit` for cross-validation if needed.

**Example:**
```python
from sklearn.model_selection import TimeSeriesSplit # Import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5) # Create a time series splitter

for train_index, test_index in tscv.split(df):
 train, test = df.iloc[train_index], df.iloc[test_index] # Get train and test splits
```
- But for a single 80/20 split, I stick to manual slicing after sorting by date.

---

**Q: How do you perform rolling or expanding window splits for time series data, such as training on Jan–Mar and testing on April, then training on Apr–Jun and testing on July?**

- For this scenario, we use a technique called "rolling window" or "expanding window" cross-validation, which is essential for time series model validation.
- scikit-learn provides `TimeSeriesSplit` for this purpose, which allows you to split your data into sequential train/test sets, just as described (e.g., train on Jan–Mar, test on April, etc.).
- This method helps simulate real-world forecasting, where you always train on past data and test on the next unseen period.

**Key Steps:**
- Sort your data by date.
- Use `TimeSeriesSplit` from `sklearn.model_selection` to generate train/test indices for each fold.
- For each split, the training set contains all data up to a certain point, and the test set contains the next chunk.

**Example Code:**
```python
from sklearn.model_selection import TimeSeriesSplit # Import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=4) # Define number of splits

for train_index, test_index in tscv.split(df):
 train = df.iloc[train_index] # Get training data for this split
 test = df.iloc[test_index] # Get testing data for this split
 # You can now train your model on 'train' and evaluate on 'test'
```
- This will automatically create splits like:
 - Split 1: Train = Jan–Mar, Test = April
 - Split 2: Train = Jan–Apr, Test = May
 - Split 3: Train = Jan–May, Test = June
 - etc.

- In my UIDS project, I used similar logic for robust model validation, ensuring that each fold only uses past data to predict the future, which is critical for time-dependent tasks.

---
