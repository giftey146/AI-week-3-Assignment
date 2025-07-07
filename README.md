Q1: TensorFlow vs. PyTorch

TensorFlow: Better for production (mobile/web), static graphs (mostly), strong industry use.

PyTorch: More Pythonic, dynamic graphs, preferred for research.

Choose TensorFlow for deployment; PyTorch for prototyping/research.

Q2: Jupyter Notebooks in AI

Data exploration & quick prototyping (run code in chunks, visualize results).

Documentation & sharing (mix code, visuals, and explanations for teams).

Q3: spaCy vs. Basic String Ops

Pre-built NLP features (POS tagging, NER) vs. manual string parsing.

Faster & more accurate (optimized Cython, handles context like "bank" = river or finance).

Linguistic awareness (lemmatization, dependencies) vs. simple substring matching.




Scikit-learn
Designed for classical machine learning (e.g., decision trees, SVM, k-NN, logistic regression).

Best suited for structured/tabular data and small to medium-sized datasets.

Very beginner-friendly with a simple, consistent API.

Requires minimal code to build and evaluate models.

Great for quick experimentation and prototyping.

Integrates smoothly with NumPy, pandas, and Jupyter.

Has strong community support and lots of tutorials, especially for students.

Does not support GPU acceleration natively.

Not ideal for deep learning tasks like image or speech recognition.

TensorFlow
Designed for deep learning and complex neural networks (e.g., CNNs, RNNs, transformers).

Suited for tasks like image classification, NLP, audio processing, and large-scale ML pipelines.

Initially has a steeper learning curve, especially for custom model building.

Offers Keras, a higher-level API, to simplify usage for beginners.

Supports GPU/TPU acceleration for high-performance training.

Maintained by Google with a massive ecosystem (e.g., TensorBoard, TFX, TensorFlow Lite).

Excellent for building models in production environments or deploying on the web/mobile.

Has a large and active global community and extensive documentation.

Scales well to big data and real-time applications.



A. Bias in MNIST Model (Digit Recognition)
Potential Biases:

Digit distribution imbalance: Some digits may appear more frequently than others, which can skew model performance.

Cultural handwriting bias: The dataset consists mainly of handwritten digits from U.S. students, potentially limiting generalizability to diverse populations.

Overfitting to clean samples: The model may not perform well on messy, real-world handwriting.

Mitigation Strategies:

Use data augmentation (e.g., rotation, distortion) to simulate a wider variety of handwriting styles.

Apply TensorFlow Fairness Indicators to evaluate model performance across different slices (if demographic metadata is available).

Continuously test the model with real-world or culturally diverse samples to ensure fairness.

B. Bias in Amazon Reviews Model (NER and Sentiment)
Potential Biases:

Brand favoritism: Rule-based sentiment tools may unintentionally favor well-known brands.

Lexicon limitations: Sentiment analysis using fixed rule-based tools like TextBlob may misinterpret slang, sarcasm, or non-standard dialects.

NER limitations: spaCy’s pre-trained models may not accurately recognize local or lesser-known product and brand names.

Mitigation Strategies:

Use custom rule-based patterns in spaCy (e.g., Matcher or EntityRuler) to identify products and brands specific to the domain.

Train a custom sentiment analysis model on a balanced set of Amazon reviews to reduce dependence on general-purpose tools.

Ensure the training data includes a diverse range of brands, product categories, and user demographics to minimize bias.

Summary
To develop fair and ethical machine learning models, it is important to:

Identify and monitor potential sources of bias in datasets and algorithms.

Use appropriate fairness and evaluation tools (e.g., TensorFlow Fairness Indicators).

Incorporate domain knowledge and diverse data when building NLP tools like NER and sentiment models.




