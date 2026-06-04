# Generative AI Engineer (Part 1) — Interview 31

**Q: Can you walk me through the steps of your XGBoost implementation for next purchase date prediction?**

- Sure, here’s how I implemented XGBoost for next purchase date prediction:

- **1. Data Collection & Preparation**
 - Gathered historical transaction data for each customer and product.
 - Cleaned the data, handled missing values, and removed duplicates.
 - Engineered features such as:
 - Days since last purchase
 - Purchase frequency
 - Average purchase interval
 - Recency, total purchases, and customer/product-level aggregates

- **2. Feature Engineering**
 - Created lag features to capture purchase patterns over time.
 - Encoded categorical variables (like product/category) using label encoding or one-hot encoding.
 - Normalized/standardized numerical features if needed.

- **3. Train-Test Split**
 - Split the data into training and test sets based on time (to avoid data leakage).

- **4. Model Training**
 - Used XGBoost Regressor for predicting the number of days until the next purchase.
 - Tuned hyperparameters (like learning rate, max depth, n_estimators) using cross-validation.

- **5. Model Evaluation**
 - Evaluated model performance using Mean Absolute Percentage Error (MAPE).
 - Achieved a MAPE of 13.6%, which was considered good for this business case.

- **6. Deployment**
 - Saved the trained model and integrated it into the marketing workflow.
 - Passed predictions to the Salesforce Marketing Cloud team for campaign targeting.

- **7. Monitoring & Iteration**
 - Monitored model performance and retrained periodically with new data.

This approach helped the business run targeted campaigns and improve customer engagement by predicting when each customer was likely to make their next purchase.

---

**Q: Why do you have to standardize features before training the XGBoost model?**

- Standardization is not strictly required for tree-based models like XGBoost, because they are not sensitive to the scale of input features.
- However, in some cases, standardizing or normalizing features can still be helpful:
 - It can make feature importance more interpretable, especially when comparing across features with different units or scales.
 - If the pipeline includes other models (like linear regression or neural networks), standardization ensures consistency and prevents issues with those algorithms.
 - It can help with convergence and stability if you use XGBoost with certain regularization or objective functions, or if you later switch to models that are sensitive to feature scale.
- In my project, I standardized features mainly for consistency across the pipeline and to keep the data ready for any future model experimentation that might require it.

---

**Q: What is normalization of a feature?**

- Normalization is a data preprocessing technique used to scale numerical feature values into a specific range, usually [0, 1].
- The main goal is to ensure that all features contribute equally to the model and to avoid dominance by features with larger values.
- The most common normalization method is Min-Max scaling:
 - For each value, subtract the minimum value of the feature and divide by the range (max - min).
 - Formula: 
 \( x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} \)
- After normalization, all feature values will be between 0 and 1.
- This is especially useful when features have different units or scales, and it helps some algorithms (like neural networks or k-means) perform better and converge faster.

---

**Q: How do you handle categorical features in preprocessing?**

- For categorical features, normalization is not applicable because they are not numerical.
- Instead, we use encoding techniques to convert categorical values into a numerical format that machine learning models can understand.
- The most common methods are:
 - **Label Encoding**: Assigns each unique category a different integer value. Useful for ordinal categories.
 - **One-Hot Encoding**: Creates a new binary column for each category, marking 1 if the row belongs to that category and 0 otherwise. Useful for nominal (non-ordered) categories.
- In my projects, like the UIDS intent classification system, I used label encoding for intent labels and sometimes one-hot encoding for other categorical features.
- This ensures that the model can process categorical data correctly and avoids introducing unintended ordinal relationships where they don’t exist.

---

**Q: What is label encoding?**

- Label encoding is a technique used to convert categorical values into numerical values.
- Each unique category in a feature is assigned a unique integer value.
- For example, if you have a feature "color" with values ["red", "green", "blue"], label encoding might assign: "red" = 0, "green" = 1, "blue" = 2.
- This allows machine learning models to process categorical data as numbers.
- In my UIDS intent classification project, I used label encoding to transform intent labels into integers before training the model, as shown in my code:
 - `le = LabelEncoder()`
 - `le.fit(train_df["l_intent"])`
 - `train_df["label"] = le.transform(train_df["l_intent"])`
- Label encoding is simple and efficient, especially for target labels or ordinal features.

---

**Q: What actually happens with the data when you associate numbers with different classes using label encoding?**

- When you use label encoding, each unique category in a categorical feature is replaced by a unique integer.
- The original categorical column (like "color" with values "red", "green", "blue") is transformed into a numerical column (like 0, 1, 2).
- This allows machine learning models to process the data, since most algorithms require numerical input.
- The mapping between category and integer is consistent, so the model can learn patterns based on these encoded values.
- In my UIDS project, for example, I used label encoding to convert intent labels into integers before training and prediction. This made it possible to train classification models and later decode predictions back to the original class names using the same encoder.
- The encoded data is now ready for model training, and the model treats each integer as a distinct class label.

---

**Q: What is the impact of using label encoding on nominal categorical features?**

- The main impact is that the model may learn false relationships between categories that do not exist.
- For nominal features (like "red", "green", "blue"), label encoding introduces artificial order and distance, which can confuse models that use mathematical operations on feature values.
- This can lead to:
 - Incorrect model assumptions about similarity or ranking between categories.
 - Reduced model accuracy and poor generalization.
 - Potential bias in predictions, as the model may treat some categories as "closer" or "further" apart.
- In practice, this is why one-hot encoding is preferred for nominal features, as it avoids introducing any unintended order or distance.
- In my experience, using the right encoding method for categorical features is critical for building robust and accurate models, especially in enterprise AI systems where data quality and model interpretability are important.

---

**Q: How can you avoid the problem of false relationships caused by label encoding on nominal categorical features?**

- Yes, to avoid introducing false relationships between nominal categories, I use one-hot encoding instead of label encoding.
- One-hot encoding creates a separate binary column for each category, so there is no artificial order or distance between them.
- For example, for the "color" feature with values "red", "green", "blue", one-hot encoding will create three columns: "color_red", "color_green", "color_blue", each with values 0 or 1.
- This ensures that the model treats each category as independent and avoids misleading the model with unintended ordinal information.
- In my projects, I always use one-hot encoding for nominal features and reserve label encoding for target variables or ordinal features where order matters.
- This approach improves model accuracy and interpretability, especially in enterprise AI systems where data quality is critical.

---

**Q: What is one-hot encoding?**

- One-hot encoding is a technique to convert categorical variables into a set of binary columns.
- For each unique category in the feature, a new column is created.
- Each row will have a value of 1 in the column corresponding to its category, and 0 in all other columns.
- For example, if the "color" feature has values "red", "green", "blue":
 - "red" becomes [1, 0, 0]
 - "green" becomes [0, 1, 0]
 - "blue" becomes [0, 0, 1]
- This method avoids introducing any artificial order or distance between categories, so the model treats each category as independent.
- In my projects, I use one-hot encoding for nominal features to ensure the model does not learn false relationships, which is important for model accuracy and interpretability.

---

**Q: What are some performance measures you can use to report model results to the business?**

- There are several key performance metrics commonly used to evaluate and report model performance to business stakeholders:
 - **Accuracy**: The proportion of correct predictions out of total predictions. It gives a general sense of how often the model is right.
 - **Balanced Accuracy**: Useful when classes are imbalanced; it averages recall obtained on each class.
 - **Precision**: The proportion of true positive predictions out of all positive predictions made by the model. It shows how reliable positive predictions are.
 - **Recall**: The proportion of true positive predictions out of all actual positives. It measures how well the model captures all relevant cases.
 - **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two, especially useful for imbalanced datasets.
 - **Macro, Micro, and Weighted F1 Scores**: These provide different ways to average F1 scores across classes, depending on the business need (e.g., treating all classes equally or weighting by class frequency).
- In my UIDS project, I used all these metrics to evaluate intent classification models and reported them to business teams for transparency and decision-making.
- These metrics help the business understand not just overall accuracy, but also how well the model performs on each class and in different scenarios, supporting better model selection and improvement.

---

**Q: What are the main performance metrics for classification models to report to the business?**

- For classification models, I typically report the following key performance metrics to the business:
 - **Accuracy**: Shows the overall percentage of correct predictions.
 - **Balanced Accuracy**: Useful when classes are imbalanced; it averages recall across all classes.
 - **Precision**: Indicates how many of the predicted positives are actually correct.
 - **Recall**: Measures how many actual positives were correctly identified.
 - **F1 Score**: Harmonic mean of precision and recall, providing a balanced measure, especially for imbalanced datasets.
 - **F1 Score (Micro, Macro, Weighted)**: 
 - Micro: Aggregates contributions of all classes to compute the average metric.
 - Macro: Calculates metric independently for each class, then takes the average.
 - Weighted: Takes class imbalance into account by weighting each class’s metric.
- In my UIDS project, I used all these metrics to evaluate and compare model performance, and shared them with business stakeholders for transparency and informed decision-making.

**interviewee**: Uh, in classifications.

---

**Q: What are some performance measures for classification models to report to the business?**

- For classification models, I typically report the following performance metrics to the business:
 - **Accuracy**: Shows the overall percentage of correct predictions.
 - **Balanced Accuracy**: Useful when classes are imbalanced; it averages recall across all classes.
 - **Precision**: Indicates how many predicted positives are actually correct.
 - **Recall**: Measures how many actual positives the model correctly identified.
 - **F1 Score (Micro, Macro, Weighted)**: Combines precision and recall; micro averages over all samples, macro averages over classes, and weighted considers class imbalance.
- These metrics provide a clear and comprehensive view of model performance, helping business stakeholders understand both overall effectiveness and how well the model handles each class.
- In my UIDS project, I used all these metrics to evaluate and report the intent classification system’s performance, ensuring transparency and actionable insights for business decisions.

---

**Q: Can you explain accuracy, balanced accuracy, precision, recall, and F1 score with examples?**

- **Accuracy**: This is the percentage of correct predictions out of all predictions.
 - *Example*: If a model predicts 90 out of 100 samples correctly, the accuracy is 90%.
- **Balanced Accuracy**: This is the average of recall obtained on each class, useful when classes are imbalanced.
 - *Example*: If you have two classes, and the model correctly predicts 80% of class A and 60% of class B, the balanced accuracy is (80% + 60%) / 2 = 70%.
- **Precision**: This is the proportion of true positive predictions out of all positive predictions made by the model.
 - *Example*: If the model predicts 20 samples as positive, and 15 of them are actually positive, precision is 15/20 = 75%.
- **Recall**: This is the proportion of true positive predictions out of all actual positives.
 - *Example*: If there are 30 actual positive samples, and the model correctly predicts 15 of them, recall is 15/30 = 50%.
- **F1 Score**: This is the harmonic mean of precision and recall, balancing both metrics.
 - *Example*: If precision is 75% and recall is 50%, F1 score = 2 * (0.75 * 0.5) / (0.75 + 0.5) ≈ 0.6 or 60%.

- In my UIDS project, I used all these metrics to evaluate the intent classification model and reported them to the business for a clear understanding of model performance.

---


**Q: What does it mean when you say Falcon is an LLM? What entities are getting trained or stored on your machine?**

- When I say Falcon-7B is an LLM (Large Language Model), I mean it is a deep learning model with billions of parameters, pre-trained on a large corpus of text data to understand and generate human language.
- In the context of fine-tuning Falcon-7B for intent classification:
 - The model’s parameters (weights and biases) are further updated using our labeled intent dataset, so it learns to map user queries to specific intent classes.
 - During training, the model learns patterns and relationships between input queries and their corresponding intents.
 - The main entities being trained and stored are:
 - **Model Weights**: The numerical parameters of the neural network, which are updated during fine-tuning.
 - **Tokenizer/Vocabulary**: The mapping from words or subwords to numerical tokens, which is required for processing text input.
 - **Configuration Files**: Metadata about the model architecture, training settings, and hyperparameters.
 - **Label Encoder**: A mapping from intent labels (like “paper jammed inside printer”) to numerical class indices.
 - After training, these entities (model weights, tokenizer, config, and label encoder) are saved to disk or cloud storage (like S3) and loaded during inference to predict intents for new queries.
- In summary, the Falcon-7B LLM is a neural network whose parameters are fine-tuned and stored, along with supporting files, to enable accurate intent classification in production.

---

**Q: What are deep learning models?**

- Deep learning models are a type of machine learning model based on artificial neural networks with multiple layers.
- These models are designed to automatically learn complex patterns and representations from large amounts of data.
- They consist of interconnected layers of artificial neurons, where each layer transforms the input data into more abstract features.
- Examples include models for image recognition (like CNNs), sequence modeling (like RNNs and Transformers), and large language models (LLMs) such as Falcon-7B, GPT, and BERT.
- In practice, deep learning models are used for tasks like natural language understanding, speech recognition, image classification, and more.
- In my projects, I have used deep learning models for intent classification, document retrieval, and generative AI tasks, leveraging their ability to handle complex, high-dimensional data and deliver high accuracy in real-world enterprise scenarios.

---

**Q: How to clean and convert the 'Amount' column to numeric values in a DataFrame with mixed formats and missing values?**

- The 'Amount' column contains dollar signs, mixed formats, and missing values ("NA").
- To analyze or process this data, you need to:
 - Remove the dollar sign and commas.
 - Convert valid entries to float.
 - Handle missing or invalid values (e.g., set as NaN).
- This is a common data cleaning step before any aggregation or analysis.

**🔑 Key Steps**:
- Read the data into a pandas DataFrame.
- Use `str.replace()` to remove '$' and ','.
- Use `pd.to_numeric()` with `errors='coerce'` to convert to float and handle invalid entries as NaN.

**💻 Code**:
```python
import pandas as pd # Import pandas for data manipulation

# Create the DataFrame from the given data
data = {
 'UserId': [1, 2, 3, 4, 5, 6], # User IDs
 'Amount': ['$20.00', '$34.23', 'NA', '$0876', '$0.01', '$9'] # Amounts as strings
}
df = pd.DataFrame(data) # Create DataFrame

# Clean the 'Amount' column
df['Amount'] = (
 df['Amount']
 .str.replace('$', '', regex=False) # Remove dollar sign
 .str.replace(',', '', regex=False) # Remove commas if any
) # Clean up the string

# Convert to numeric, setting errors to NaN for invalid entries
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce') # Convert to float

print(df) # Display the cleaned DataFrame
```

**💡 Explanation**:
- The code first removes the '$' symbol and any commas from the 'Amount' column.
- `pd.to_numeric(..., errors='coerce')` converts valid numbers to float and sets invalid entries (like 'NA') to NaN.
- This cleaned DataFrame is now ready for further analysis (e.g., sum, mean, etc.).
- Time complexity: O(n), where n is the number of rows.
- Space complexity: O(n), as a new column is created for cleaned values.

---

**Q: What are different techniques to handle missing values, and when should each be used?**

- There are several standard techniques to handle missing values in data:
 - **Remove Rows with Missing Values**:
 - Use when the dataset is large and missing values are few.
 - Prevents introducing bias from imputation.
 - **Impute with Mean/Median/Mode**:
 - Use mean for normally distributed numerical data.
 - Use median for skewed numerical data or when outliers are present.
 - Use mode for categorical data.
 - **Impute with a Constant Value**:
 - Use when a specific value (like 0 or "Unknown") makes sense for the business context.
 - **Forward/Backward Fill**:
 - Use for time series data where previous or next values can logically fill gaps.
 - **Model-Based Imputation**:
 - Use advanced methods (like KNN, regression, or ML models) when missingness is not random or when more accuracy is needed.
 - **Leave as Missing (NaN)**:
 - Some algorithms (like XGBoost, CatBoost) can handle NaN values natively.
- The choice depends on:
 - The amount and pattern of missing data.
 - The importance of the feature.
 - The impact on downstream analysis or model performance.
 - The business context and interpretability requirements.
- In practice, always analyze the reason for missingness and test the impact of different strategies on your results.

---

**Q: What technique would you use to handle missing values in the 'Amount' column for this user transaction data?**

- In this case, the 'Amount' column represents transaction values for each user, which are numerical and directly impact the calculation of the maximum transaction.
- Since the dataset is very small (only six users), removing rows with missing values could result in losing important user data.
- Imputing with mean or median is not ideal here because each transaction is unique and represents a real-world event, not a repeated measurement.
- For this scenario, the best approach is to:
 - Set missing or invalid 'Amount' values to 0 or NaN.
 - If the goal is to find the user with the maximum transaction, you should ignore users with missing or invalid amounts in the calculation.
 - This ensures that only valid transactions are considered, and the result reflects actual user activity.
- This approach is simple, transparent, and avoids introducing artificial values that could distort the analysis.
- In production pipelines (as in my experience with enterprise AI systems), we typically fill missing numerical values with 0 or drop them for aggregation tasks, unless business logic requires a different strategy. For example, in the UIDS project, nulls are filled or handled during data preparation to ensure data quality before downstream processing.

---

**Q: Should missing 'Amount' values be set to zero, and what does that imply in this context?**

- Setting missing 'Amount' values to zero will imply that the user made a transaction of zero value, not that the transaction data is missing.
- In this context, where 'Amount' represents the value of a transaction, replacing missing values with zero can mislead downstream analysis—especially when identifying the user with the maximum transaction.
- The best practice is to leave missing values as NaN (Not a Number) or remove those rows from calculations involving transaction amounts.
 - This ensures that only users with valid transaction amounts are considered for metrics like maximum transaction.
- This approach aligns with standard data preparation practices in enterprise AI pipelines, as seen in my experience with UIDS and RAG projects, where nulls are filled only when it makes logical sense, and otherwise, missing values are handled explicitly to maintain data integrity.
- In summary, do not replace missing 'Amount' values with zero unless business logic specifically requires it; otherwise, treat them as missing to avoid incorrect interpretations.

---

**Q: Statistically, what is the minimum number of samples needed for a machine learning model to work properly?**

- There is no strict universal rule for the minimum number of samples required for a machine learning model, as it depends on:
 - The complexity of the model (e.g., deep learning models need more data than simple linear models).
 - The number of features (dimensionality) in the dataset.
 - The variability and distribution of the data.
 - The specific use case and acceptable error margin.
- As a general guideline:
 - For simple models (like linear regression), you typically need at least 10 times as many samples as features (the "10:1 rule").
 - For deep learning or high-dimensional data, thousands to tens of thousands of samples are often needed for stable training and to avoid overfitting.
 - For classification tasks, having at least 50-100 samples per class is a common minimum, but more is always better.
- In industry practice, we often start with as much data as possible, then use techniques like cross-validation to assess if the model is generalizing well.
- In my experience with enterprise AI systems (like UIDS), we ensure data sufficiency by analyzing class balance, using data augmentation, and monitoring model performance metrics to decide if more data is needed before production deployment.

---

**Q: What is the minimum sample size required for statistical significance in experiments or datasets?**

- Statistically, the minimum sample size for significance depends on the test being used, the expected effect size, and the desired confidence level (typically 95%).
- As a general rule in statistics:
 - For most parametric tests (like t-tests), a minimum of 30 samples is often cited as the threshold for the Central Limit Theorem to apply, making the sample mean approximately normally distributed.
 - For very basic statistical analysis, at least 10-20 samples are sometimes used, but results are less reliable and have wider confidence intervals.
 - For machine learning, as mentioned earlier, at least 10 samples per feature is a common rule of thumb, but more is always better for model stability.
- In industry and research, power analysis is used to determine the minimum sample size needed to detect a given effect size with a specified confidence and power (commonly 80%).
- In summary:
 - **Below 30 samples**, statistical significance is weak and results are not robust.
 - **30 or more samples** is a practical minimum for most statistical tests to be considered reliable.
 - For production ML pipelines (like in UIDS), we always aim for larger datasets to ensure both statistical and practical significance before deploying models.

---

**Q: What is the minimum number of rows required for a dataset to be statistically significant for machine learning?**

- For any machine learning model to be statistically significant and reliable, a dataset with only 6 rows is far too small.
- As a practical industry guideline:
 - **Minimum 30 samples** is often cited for basic statistical significance (Central Limit Theorem).
 - For machine learning, **at least 50–100 rows** is a bare minimum for very simple models, but this is still quite low and may not generalize well.
 - For more robust results, especially with multiple features or classes, **hundreds or thousands of rows** are preferred.
- In my experience with enterprise AI projects like UIDS, we set minimum thresholds for training data size during preprocessing (for example, filtering out intents with fewer than a set number of samples).
- If I were advising a client, I would recommend **at least 30–50 rows per class or intent** as an absolute minimum, but ideally **several hundred rows** for meaningful model training and evaluation.
- With only 6 rows, any model built would not be statistically valid or reliable for production use.

---

**Q: What is the single minimum number of rows required for a machine learning model?**

- The absolute minimum number of rows I would recommend for any machine learning model is **30**.
 - This is based on statistical principles (Central Limit Theorem) and is commonly used as a threshold for basic significance.
- However, for practical and reliable results, especially in industry projects, I always aim for at least **50 rows** as a starting point, even for the simplest models.
- In my experience with enterprise AI pipelines (like UIDS), we set such minimum thresholds in preprocessing to ensure the model has enough data to learn meaningful patterns.
- So, the minimum single value I would give is **30 rows**, but ideally, more data is always better for model stability and accuracy.

---

**Q: Why is 50 chosen as the minimum number of rows, and what happens if you have less than that?**

- Choosing 50 as the minimum is based on practical experience and statistical guidelines:
 - With fewer than 50 samples, models are highly prone to overfitting and cannot generalize well.
 - Small datasets do not capture enough variability, so the model may learn noise instead of real patterns.
 - Many statistical tests and machine learning algorithms require a reasonable sample size to estimate parameters reliably and produce stable results.
- In my enterprise AI projects (like UIDS), we set such thresholds to ensure the model has enough data for meaningful training and evaluation.
- If you use less than 50 rows:
 - The model’s predictions will be unstable and unreliable.
 - Performance metrics (like accuracy or F1-score) will not be trustworthy.
 - The model may fail to detect real trends or patterns, leading to poor results in production.
- In summary, 50 is a practical minimum to balance statistical significance and model reliability, but more data is always better for robust AI solutions.

---

**Q: What is the impact of having 48 rows versus 52 rows in your dataset?**

- The difference between 48 and 52 rows is very small and will not have a significant impact on model performance or statistical significance.
- Both numbers are close to the practical minimum threshold (around 50) that I recommend for simple models.
- In practice:
 - With such small datasets, the model will still be sensitive to noise and may not generalize well.
 - The slight increase from 48 to 52 rows may help a little with stability, but it will not solve the core issue of limited data.
 - For robust and reliable results, it is always better to collect more data—ideally several hundred rows or more.
- In my experience with enterprise AI pipelines (like UIDS), we set minimum thresholds (for example, 50 rows per intent) to filter out classes with too little data, but always aim for higher counts for production models.
- So, while 52 is marginally better than 48, both are still at the lower end, and the impact is minimal unless you cross a threshold set in your pipeline configuration (such as a hard minimum of 50 rows).

---
