# Customer Segmentation using K-Means Clustering

This project implements **Customer Segmentation** using the **K-Means Clustering Algorithm** in Python. The model groups customers based on their **Age, Annual Income, and Spending Score** to help businesses understand different types of customers and apply targeted marketing strategies.

---

## 📌 Project Objective

The main objective of this project is to:
- Segment customers into different groups using **unsupervised machine learning**
- Analyze customer behavior
- Predict the cluster of a new customer based on input data

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📂 Project Structure












---

## 📊 Dataset Information

The dataset used in this project contains the following features:

- CustomerID
- Age
- Annual Income (k$)
- Spending Score (1–100)

This dataset is commonly used for customer segmentation problems.

---

## ⚙️ Steps Performed in the Project

1. Importing required libraries
2. Loading the dataset
3. Data preprocessing
4. Feature scaling using **StandardScaler**
5. Finding optimal number of clusters using **Elbow Method**
6. Training the **K-Means model**
7. Visualizing clusters
8. Saving trained scaler and model
9. Taking real-time user input
10. Predicting the cluster of a new customer

---

## 📈 Elbow Method

The Elbow Method is used to determine the optimal number of clusters (K). It plots the **Within-Cluster Sum of Squares (WCSS)** versus the number of clusters and selects the point where the curve bends.

---

## 🔍 K-Means Clustering

K-Means groups similar data points into clusters based on Euclidean distance from cluster centroids. Each customer is assigned to the nearest cluster.

---

## ▶️ How to Run This Project

### ✅ Step 1: Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn


✅ Step 2: Train the Model
open and run:
customer_segmentation.ipynb

This will:

Train the K-Means model

Save scaler.pkl and kmeans_model.pkl

Generate cluster visualizations

✅ Step 3: Predict New Customer Cluster

Run the prediction file:
python predict.py


Enter values like:

Enter Age: 20
Enter Annual Income (k$): 2000
Enter Spending Score (1–100): 56

Output:

This Customer Belongs To Cluster: 7



🧠 Interpretation of Output

The predicted cluster number indicates the group of customers with similar characteristics.
Each cluster can be labeled based on:

Low Income – Low Spending

High Income – High Spending

Medium Income – Medium Spending, etc.


🎯 Real-World Applications

Targeted Marketing

Customer Profiling

Product Recommendation

Business Strategy Planning

Sales Optimization




✅ Advantages of This Project

Simple and beginner-friendly

Strong machine learning concept (Unsupervised Learning)

Highly useful for internships and interviews

Real-world business application




🚀 Future Enhancements

Add GUI using Tkinter or Streamlit

Deploy on web using Flask

Add automatic cluster interpretation

Use advanced clustering algorithms




👨‍💻 Author

Name: [Your Name]
Branch: B.Tech (Computer Science / IT)
Role: Machine Learning Student
GitHub: https://github.com/kanha165

