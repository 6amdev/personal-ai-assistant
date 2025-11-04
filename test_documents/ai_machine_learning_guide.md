# 🤖 คู่มือ AI และ Machine Learning ฉบับสมบูรณ์

> เอกสารสำหรับเริ่มต้นเรียนรู้ AI, Machine Learning, และ Deep Learning

---

## 📚 สารบัญ

1. [AI คืออะไร?](#ai-คืออะไร)
2. [Machine Learning คืออะไร?](#machine-learning-คืออะไร)
3. [ประเภทของ Machine Learning](#ประเภทของ-machine-learning)
4. [Deep Learning](#deep-learning)
5. [Neural Networks](#neural-networks)
6. [การประยุกต์ใช้งาน](#การประยุกต์ใช้งาน)
7. [เครื่องมือและ Libraries](#เครื่องมือและ-libraries)
8. [แหล่งเรียนรู้](#แหล่งเรียนรู้)

---

## AI คืออะไร?

**Artificial Intelligence (AI)** หรือ ปัญญาประดิษฐ์ คือการสร้างระบบคอมพิวเตอร์ที่สามารถทำงานที่ต้องใช้สติปัญญาของมนุษย์ได้

### ประวัติความเป็นมา

- **1950** - Alan Turing เสนอ Turing Test
- **1956** - John McCarthy ใช้คำว่า "Artificial Intelligence" ครั้งแรก
- **1997** - Deep Blue ชนะ Garry Kasparov ในหมากรุก
- **2011** - IBM Watson ชนะ Jeopardy
- **2016** - AlphaGo ชนะ Lee Sedol ในหมากล้อม
- **2022** - ChatGPT เปิดตัว สร้างปรากฏการณ์โลก

### ประเภทของ AI

#### 1. Narrow AI (Weak AI)
- AI ที่ทำงานเฉพาะด้าน
- ตัวอย่าง: Siri, Alexa, การจดจำใบหน้า
- **ใช้งานจริงในปัจจุบัน**

#### 2. General AI (Strong AI)
- AI ที่สามารถคิดและเรียนรู้เหมือนมนุษย์
- ยังไม่มีจริง อยู่ในขั้นทฤษฎี
- เป้าหมายในอนาคต

#### 3. Super AI
- AI ที่ฉลาดกว่ามนุษย์ทุกด้าน
- ยังเป็นจินตนาการ
- มีการถ่ายทำเป็นหนังมากมาย

---

## Machine Learning คืออะไร?

**Machine Learning (ML)** คือสาขาย่อยของ AI ที่เน้นการให้คอมพิวเตอร์เรียนรู้จากข้อมูล โดยไม่ต้องเขียนโปรแกรมแบบตายตัว

### หลักการทำงาน

```
ข้อมูล (Data) → Algorithm → โมเดล (Model) → ทำนาย (Prediction)
```

### สูตรพื้นฐาน

**Linear Regression:**
```
y = mx + b
```
- y = ผลลัพธ์ที่ต้องการทำนาย
- x = ข้อมูลนำเข้า (input)
- m = ความชัน (slope)
- b = ค่าคงที่ (intercept)

**Cost Function (Mean Squared Error):**
```
MSE = (1/n) Σ(y_actual - y_predicted)²
```

---

## ประเภทของ Machine Learning

### 1. Supervised Learning (การเรียนรู้แบบมีผู้สอน)

**คืออะไร:** ให้ข้อมูลที่มีคำตอบ (labeled data) แก่โมเดล

**ตัวอย่าง:**
- **Classification** - จำแนกประเภท
  - สแปมหรือไม่ใช่สแปม
  - แมวหรือหมา
  - โรคมะเร็งหรือไม่

- **Regression** - ทำนายค่าต่อเนื่อง
  - ทำนายราคาบ้าน
  - ทำนายยอดขาย
  - ทำนายอุณหภูมิ

**Algorithms ที่ใช้:**
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Neural Networks

**ตัวอย่างโค้ด Python:**
```python
from sklearn.linear_model import LinearRegression

# สร้างโมเดล
model = LinearRegression()

# ฝึกโมเดล
model.fit(X_train, y_train)

# ทำนาย
predictions = model.predict(X_test)
```

---

### 2. Unsupervised Learning (การเรียนรู้แบบไม่มีผู้สอน)

**คืออะไร:** ให้ข้อมูลที่ไม่มีคำตอบ (unlabeled data) แล้วให้โมเดลหาแพทเทิร์นเอง

**ตัวอย่าง:**
- **Clustering** - จัดกลุ่ม
  - แบ่งกลุ่มลูกค้า (Customer Segmentation)
  - จัดกลุ่มข่าว
  - จัดกลุ่มยีน

- **Dimensionality Reduction** - ลดมิติข้อมูล
  - Principal Component Analysis (PCA)
  - t-SNE
  - UMAP

- **Association Rules** - หาความสัมพันธ์
  - Market Basket Analysis
  - "คนที่ซื้อขนมปังมักซื้อนมด้วย"

**Algorithms ที่ใช้:**
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models
- PCA
- Autoencoders

**ตัวอย่างโค้ด:**
```python
from sklearn.cluster import KMeans

# สร้างโมเดล K-Means (3 กลุ่ม)
kmeans = KMeans(n_clusters=3)

# ฝึกและทำนาย
clusters = kmeans.fit_predict(X)
```

---

### 3. Reinforcement Learning (การเรียนรู้แบบเสริมแรง)

**คืออะไร:** เรียนรู้จากการลองผิดลองถูก โดยได้รับ reward หรือ penalty

**ตัวอย่าง:**
- เกม (AlphaGo, Dota 2, Chess)
- หุ่นยนต์เดิน
- รถยนต์ขับเคลื่อนอัตโนมัติ
- ระบบแนะนำ (Recommendation System)

**องค์ประกอบหลัก:**
- **Agent** - ตัวที่เรียนรู้
- **Environment** - สภาพแวดล้อม
- **State** - สถานะปัจจุบัน
- **Action** - การกระทำ
- **Reward** - รางวัล/โทษ

**Algorithms ที่ใช้:**
- Q-Learning
- Deep Q-Network (DQN)
- Policy Gradient
- Actor-Critic
- Proximal Policy Optimization (PPO)

**ตัวอย่างโค้ด (Q-Learning):**
```python
import numpy as np

# Q-Table
Q = np.zeros([state_space, action_space])

# Q-Learning Algorithm
for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        # เลือก action
        action = np.argmax(Q[state, :])
        
        # ทำ action
        new_state, reward, done = env.step(action)
        
        # Update Q-Table
        Q[state, action] = Q[state, action] + \
            learning_rate * (reward + discount_factor * 
            np.max(Q[new_state, :]) - Q[state, action])
        
        state = new_state
        
        if done:
            break
```

---

## Deep Learning

**Deep Learning** คือสาขาย่อยของ Machine Learning ที่ใช้ Neural Networks ที่มีหลายชั้น (deep)

### ทำไมต้อง Deep Learning?

**ข้อดี:**
- จัดการข้อมูลซับซ้อนได้ดี (รูปภาพ, เสียง, ข้อความ)
- เรียนรู้ feature เอง (automatic feature extraction)
- ประสิทธิภาพสูงเมื่อมีข้อมูลเยอะ
- State-of-the-art ในหลายงาน

**ข้อเสีย:**
- ต้องการข้อมูลเยอะมาก
- ต้องการ GPU (คำนวณนานและหนัก)
- ยากต่อการ interpret (black box)
- ต้องการความรู้เฉพาะทาง

### สถาปัตยกรรมที่นิยม

#### 1. Convolutional Neural Networks (CNN)
**ใช้สำหรับ:** รูปภาพ, วิดีโอ

**สถาปัตยกรรมที่มีชื่อ:**
- LeNet (1998)
- AlexNet (2012) - ชนะ ImageNet
- VGGNet (2014)
- ResNet (2015)
- EfficientNet (2019)

**ตัวอย่างโค้ด (Keras):**
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    # Convolution Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Fully Connected Layer
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

#### 2. Recurrent Neural Networks (RNN)
**ใช้สำหรับ:** ข้อมูลแบบ sequence (ข้อความ, เสียง, time series)

**ประเภท:**
- Vanilla RNN
- Long Short-Term Memory (LSTM) - ดีกว่า RNN
- Gated Recurrent Unit (GRU) - เร็วกว่า LSTM

**ตัวอย่างโค้ด (LSTM):**
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Embedding(vocab_size, 128),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

---

#### 3. Transformer
**ใช้สำหรับ:** NLP (Natural Language Processing)

**โมเดลที่มีชื่อ:**
- BERT (Google, 2018)
- GPT-2, GPT-3, GPT-4 (OpenAI)
- T5 (Google)
- LLaMA (Meta)
- Claude (Anthropic)

**สถาปัตยกรรมหลัก:**
- Self-Attention Mechanism
- Multi-Head Attention
- Positional Encoding
- Feed-Forward Networks

**ตัวอย่างการใช้ (Hugging Face):**
```python
from transformers import pipeline

# Text Generation
generator = pipeline('text-generation', model='gpt2')
result = generator("Once upon a time", max_length=50)

# Sentiment Analysis
classifier = pipeline('sentiment-analysis')
result = classifier("I love this product!")

# Question Answering
qa = pipeline('question-answering')
result = qa(question="What is AI?", 
            context="AI is artificial intelligence...")
```

---

#### 4. Generative Adversarial Networks (GANs)
**ใช้สำหรับ:** สร้างข้อมูลใหม่ (รูปภาพ, เสียง, วิดีโอ)

**องค์ประกอบ:**
- **Generator** - สร้างข้อมูลปลอม
- **Discriminator** - แยกของจริง/ปลอม
- **แข่งกัน** - Generator พยายามหลอก, Discriminator พยายามจับ

**ตัวอย่างการใช้:**
- สร้างรูปหน้าคนที่ไม่มีจริง (ThisPersonDoesNotExist.com)
- สร้างงานศิลปะ
- ปรับปรุงคุณภาพรูปภาพ (Super Resolution)
- แปลงรูปร่างหน้าคน (Face Swap)

---

#### 5. Diffusion Models
**ใช้สำหรับ:** สร้างรูปภาพคุณภาพสูง

**โมเดลที่มีชื่อ:**
- DALL-E 2 (OpenAI)
- Stable Diffusion (Stability AI)
- Midjourney
- Imagen (Google)

**วิธีทำงาน:**
1. เพิ่ม noise ให้รูปภาพทีละนิด
2. ฝึกโมเดลให้เอา noise ออก
3. เริ่มจาก noise → ค่อยๆ สร้างรูปภาพ

---

## Neural Networks

### โครงสร้างพื้นฐาน

```
Input Layer → Hidden Layers → Output Layer
```

**ส่วนประกอบ:**
- **Neurons (Nodes)** - หน่วยคำนวณ
- **Weights** - น้ำหนัก (ความสำคัญ)
- **Bias** - ค่าคงที่
- **Activation Function** - ฟังก์ชันกระตุ้น

### Activation Functions

#### 1. Sigmoid
```python
σ(x) = 1 / (1 + e^(-x))
```
- Output: 0 ถึง 1
- ใช้: Binary Classification (ชั้นสุดท้าย)

#### 2. Tanh
```python
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- Output: -1 ถึง 1
- ใช้: Hidden Layers (ดีกว่า Sigmoid)

#### 3. ReLU (Rectified Linear Unit)
```python
f(x) = max(0, x)
```
- Output: 0 ถึง ∞
- ใช้: Hidden Layers (นิยมมากที่สุด) ⭐
- เร็วและดีต่อการเรียนรู้

#### 4. Softmax
```python
σ(x_i) = e^(x_i) / Σ(e^(x_j))
```
- Output: ผลรวม = 1 (ความน่าจะเป็น)
- ใช้: Multi-class Classification (ชั้นสุดท้าย)

### Forward Propagation

```python
# Layer 1
z1 = W1 @ x + b1
a1 = relu(z1)

# Layer 2
z2 = W2 @ a1 + b2
a2 = relu(z2)

# Output Layer
z3 = W3 @ a2 + b3
output = softmax(z3)
```

### Backpropagation

**ขั้นตอน:**
1. คำนวณ Loss
2. หา Gradient (Chain Rule)
3. Update Weights

```python
# Gradient Descent
W = W - learning_rate * dL/dW
b = b - learning_rate * dL/db
```

### Optimizers

#### 1. Stochastic Gradient Descent (SGD)
```python
W = W - learning_rate * gradient
```

#### 2. Adam (นิยมมากที่สุด) ⭐
```python
# รวม momentum + adaptive learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
```

#### 3. RMSprop
#### 4. AdaGrad
#### 5. AdaDelta

---

## การประยุกต์ใช้งาน

### 1. Computer Vision (รู้จักภาพ)

**งาน:**
- Image Classification - จำแนกประเภทภาพ
- Object Detection - ตรวจจับวัตถุ (YOLO, Faster R-CNN)
- Semantic Segmentation - แยกส่วนภาพ
- Face Recognition - จดจำใบหน้า
- Optical Character Recognition (OCR) - อ่านตัวอักษร

**ตัวอย่างการใช้:**
- รถยนต์ขับเคลื่อนอัตโนมัติ
- ระบบรักษาความปลอดภัย
- แอพแปลภาษาจากรูปภาพ (Google Lens)
- ตรวจหาโรคจากภาพ X-Ray

---

### 2. Natural Language Processing (ประมวลผลภาษา)

**งาน:**
- Text Classification - จำแนกข้อความ
- Sentiment Analysis - วิเคราะห์ความรู้สึก
- Named Entity Recognition (NER) - หาชื่อเฉพาะ
- Machine Translation - แปลภาษา
- Question Answering - ตอบคำถาม
- Text Generation - สร้างข้อความ
- Summarization - สรุปข้อความ

**ตัวอย่างการใช้:**
- ChatGPT, Claude, Gemini
- Google Translate
- Grammarly (แก้ไขภาษา)
- Email Spam Filter
- Virtual Assistants (Siri, Alexa)

---

### 3. Speech Recognition (รู้จักเสียง)

**งาน:**
- Speech to Text - แปลงเสียงเป็นข้อความ
- Text to Speech - แปลงข้อความเป็นเสียง
- Speaker Recognition - จำแนกผู้พูด
- Emotion Detection - ตรวจจับอารมณ์จากเสียง

**ตัวอย่างการใช้:**
- Google Assistant
- Siri, Alexa
- Zoom Transcription
- Podcast Transcription

---

### 4. Recommendation Systems (ระบบแนะนำ)

**วิธีทำงาน:**
- Collaborative Filtering - ดูพฤติกรรมคนอื่น
- Content-Based Filtering - ดูคุณสมบัติสินค้า
- Hybrid - รวมทั้ง 2 แบบ

**ตัวอย่างการใช้:**
- Netflix - แนะนำหนัง
- YouTube - แนะนำวิดีโอ
- Amazon - แนะนำสินค้า
- Spotify - แนะนำเพลง

---

### 5. Time Series Forecasting (ทำนายอนาคต)

**งาน:**
- Stock Price Prediction - ทำนายราคาหุ้น
- Weather Forecasting - พยากรณ์อากาศ
- Sales Forecasting - ทำนายยอดขาย
- Energy Demand - ทำนายการใช้ไฟฟ้า

**Algorithms:**
- ARIMA
- LSTM
- Prophet (Facebook)
- Temporal Convolutional Networks (TCN)

---

### 6. Healthcare (สุขภาพ)

**งาน:**
- Disease Diagnosis - วินิจฉัยโรค
- Drug Discovery - ค้นหายาใหม่
- Medical Image Analysis - วิเคราะห์ภาพทางการแพทย์
- Patient Monitoring - ติดตามอาการผู้ป่วย

**ตัวอย่าง:**
- ตรวจมะเร็งจากภาพ X-Ray
- ทำนายโรคหัวใจ
- แนะนำแผนการรักษา

---

## เครื่องมือและ Libraries

### Python Libraries (นิยมสูงสุด)

#### 1. NumPy
**ใช้สำหรับ:** คำนวณเมทริกซ์, Array

```python
import numpy as np

# สร้าง array
arr = np.array([1, 2, 3, 4, 5])

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
```

---

#### 2. Pandas
**ใช้สำหรับ:** จัดการข้อมูล (DataFrame)

```python
import pandas as pd

# อ่านข้อมูล
df = pd.read_csv('data.csv')

# ดูข้อมูล
print(df.head())
print(df.describe())

# Filter
filtered = df[df['age'] > 25]
```

---

#### 3. Matplotlib & Seaborn
**ใช้สำหรับ:** Data Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Line plot
plt.plot(x, y)
plt.title('My Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

# Heatmap
sns.heatmap(correlation_matrix, annot=True)
```

---

#### 4. Scikit-learn
**ใช้สำหรับ:** Machine Learning แบบดั้งเดิม

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ฝึกโมเดล
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ทำนาย
predictions = model.predict(X_test)

# ประเมิน
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')
```

---

#### 5. TensorFlow & Keras
**ใช้สำหรับ:** Deep Learning (จาก Google)

```python
import tensorflow as tf
from tensorflow import keras

# สร้างโมเดล
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ฝึก
model.fit(X_train, y_train, epochs=10, batch_size=32)

# ประเมิน
model.evaluate(X_test, y_test)
```

---

#### 6. PyTorch
**ใช้สำหรับ:** Deep Learning (จาก Meta)

```python
import torch
import torch.nn as nn

# สร้างโมเดล
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# สร้าง instance
model = NeuralNetwork()
```

---

#### 7. Hugging Face Transformers
**ใช้สำหรับ:** NLP (Pre-trained Models)

```python
from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
result = classifier("I love machine learning!")

# Text Generation
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=30)

# Translation
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
```

---

### เครื่องมืออื่นๆ

- **Jupyter Notebook** - เขียนโค้ดแบบ interactive
- **Google Colab** - Jupyter บน Cloud (ฟรี GPU!)
- **Weights & Biases** - Track experiments
- **TensorBoard** - Visualize training
- **MLflow** - Manage ML lifecycle
- **Docker** - Deploy models

---

## แหล่งเรียนรู้

### หลักสูตรออนไลน์ (ฟรี)

#### 1. Coursera
- **Machine Learning** by Andrew Ng ⭐⭐⭐⭐⭐
- **Deep Learning Specialization** by Andrew Ng
- **AI For Everyone** (ไม่ต้องโค้ด)

#### 2. Fast.ai
- **Practical Deep Learning for Coders** (เน้นปฏิบัติ)
- ฟรี 100%
- ใช้ PyTorch

#### 3. YouTube Channels
- **3Blue1Brown** - Neural Networks มีภาพสวย
- **StatQuest** - อธิบายง่ายมาก
- **Sentdex** - Python & ML Tutorials
- **Two Minute Papers** - Paper รีวิว
- **Yannic Kilcher** - Paper Deep Dive

#### 4. Kaggle
- **Kaggle Learn** - Courses สั้นๆ
- **Competitions** - ลองแข่ง
- **Notebooks** - เรียนรู้จากคนอื่น

---

### หนังสือแนะนำ

#### เริ่มต้น
1. **"Hands-On Machine Learning"** by Aurélien Géron ⭐⭐⭐⭐⭐
2. **"Python Machine Learning"** by Sebastian Raschka
3. **"Deep Learning with Python"** by François Chollet

#### ขั้นสูง
1. **"Deep Learning"** by Ian Goodfellow (แบบเรียน)
2. **"Pattern Recognition and Machine Learning"** by Bishop
3. **"Reinforcement Learning"** by Sutton & Barto

---

### Websites & Blogs

- **Papers with Code** - Paper + Code implementations
- **Towards Data Science** - Medium publication
- **Machine Learning Mastery** - Tutorials
- **Distill.pub** - Interactive papers
- **OpenAI Blog** - Research updates
- **Google AI Blog** - Google's research

---

### Podcasts

- **Lex Fridman Podcast** - สัมภาษณ์นักวิจัย AI
- **The TWIML AI Podcast** - ML & AI discussions
- **Data Skeptic** - Data Science & ML

---

## 🎯 แผนการเรียนรู้ (Roadmap)

### ระดับเริ่มต้น (3-6 เดือน)

**เดือนที่ 1-2: พื้นฐาน**
- Python Programming
- NumPy, Pandas
- Data Visualization (Matplotlib)
- Statistics & Probability

**เดือนที่ 3-4: Machine Learning**
- Supervised Learning (Regression, Classification)
- Unsupervised Learning (Clustering)
- Scikit-learn
- ทำโปรเจคง่ายๆ

**เดือนที่ 5-6: ขั้นกลาง**
- Feature Engineering
- Model Evaluation
- Cross-Validation
- Kaggle Competitions (ง่ายๆ)

---

### ระดับกลาง (6-12 เดือน)

**เดือนที่ 7-9: Deep Learning**
- Neural Networks พื้นฐาน
- TensorFlow/Keras หรือ PyTorch
- CNN (Computer Vision)
- Transfer Learning

**เดือนที่ 10-12: เฉพาะทาง**
เลือก 1 ด