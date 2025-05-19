import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# تحميل البيانات
df = pd.read_csv('final_Data.csv')
df.dropna(inplace=True)

# تحويل الأعمدة النصية إلى رقمية
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# مصفوفة الارتباط (اختياري)
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# فصل البيانات
X = df.drop(['QOL', 'index'], axis=1)
y = df['QOL']

# التقسيم والتوحيد
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# قائمة النماذج وملفات الحفظ
models = {
    "ann": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(probability=True),
    "naive_bayes": GaussianNB()
}

# تدريب النماذج وحفظ التوقعات
for name, model in models.items():
    print(f"\nTraining {name.upper()} model...")
    
    if name == "linear_regression":
        model.fit(X_train_scaled, y_train)
        y_pred_continuous = model.predict(X_test_scaled)
        y_pred = [round(p) for p in y_pred_continuous]
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    # طباعة النتائج
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # حفظ النتائج في ملف خاص لكل نموذج
    results_df = pd.DataFrame({
        'Actual': y_test.reset_index(drop=True),
        'Predicted': y_pred
    })
    results_df.to_csv(f'predictions_{name}.csv', index=False)

# حفظ بيانات التدريب والاختبار
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
