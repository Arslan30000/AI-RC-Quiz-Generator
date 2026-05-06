import os
import random
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title='AI Quiz & Reading Comprehension', layout='wide')

# ==========================================
# CACHED DATA & MODEL LOADING (Streamlit Guide Ch 5)
# ==========================================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv')).fillna("")
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv')).fillna("")
    return train_df, test_df

@st.cache_resource
def train_models(train_df, test_df):
    """Trains Model A and prepares TF-IDF vectorizers."""
    TARGET = 'answer'
    
    def combine_text(df):
        return df['article'] + " " + df['question'] + " " + df['A'] + " " + df['B'] + " " + df['C'] + " " + df['D']

    X_train_text = combine_text(train_df)
    y_train = train_df[TARGET]
    X_test_text = combine_text(test_df)
    y_test = test_df[TARGET]

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # Unsupervised: K-Means
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_train_vec)

    # Supervised Ensemble: Naive Bayes + Logistic Regression
    nb_model = MultinomialNB()
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    ensemble_model = VotingClassifier(estimators=[('nb', nb_model), ('lr', lr_model)], voting='soft')
    ensemble_model.fit(X_train_vec, y_train)

    # Calculate Test Metrics for Dashboard
    y_test_pred = ensemble_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, zero_division=0, output_dict=True)

    return vectorizer, ensemble_model, acc, report

# Load resources
with st.spinner("Loading data and training models... This will only happen once!"):
    train_df, test_df = load_data()
    vectorizer, ensemble_model, test_acc, test_report = train_models(train_df, test_df)

# ==========================================
# MODEL B LOGIC (Hints & Distractors)
# ==========================================
def generate_hint(article, question):
    sentences = [s.strip() for s in article.split('.') if len(s.strip()) > 10]
    if not sentences: return "No hint available."
    hint_vec = TfidfVectorizer(stop_words='english')
    tfidf_matrix = hint_vec.fit_transform(sentences + [question])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    return sentences[np.argmax(similarities)]

def generate_distractors(article, correct_answer):
    """Generates 3 plausible but wrong options using frequent words from the article."""
    words = [w.strip(',."\'') for w in article.split() if len(w) > 4]
    unique_words = list(set(words))
    distractors = [w for w in unique_words if w.lower() not in correct_answer.lower()]
    random.shuffle(distractors)
    return distractors[:3] if len(distractors) >= 3 else ["Option X", "Option Y", "Option Z"]

# ==========================================
# UI LAYOUT (Streamlit Guide Ch 4)
# ==========================================
st.title("📚 Intelligent Reading Comprehension System")
st.markdown("Powered by TF-IDF, K-Means, and Voting Classifiers (Naive Bayes & Logistic Regression)")

# Create the 4 required screens using Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 1. Article Input", "❓ 2. Quiz View", "💡 3. Hint Panel", "📊 4. Analytics Dashboard"])

# --- SCREEN 1: ARTICLE INPUT ---
with tab1:
    st.header("Input Reading Passage")
    
    if st.button("Load Random RACE Dataset Sample"):
        sample = test_df.sample(1).iloc[0]
        st.session_state['article'] = sample['article']
        st.session_state['question'] = sample['question']
        st.session_state['correct_ans_text'] = sample[sample['answer']] # Gets the actual text of A, B, C, or D
        st.success("Random sample loaded!")

    article_input = st.text_area("Article:", value=st.session_state.get('article', ''), height=200)
    question_input = st.text_input("Question:", value=st.session_state.get('question', ''))
    
    if st.button("Generate Quiz & Distractors", type="primary"):
        if article_input and question_input:
            st.session_state['article'] = article_input
            st.session_state['question'] = question_input
            
            # Generate Distractors (Model B)
            correct_ans = st.session_state.get('correct_ans_text', 'True Answer')
            distractors = generate_distractors(article_input, correct_ans)
            
            options = [correct_ans] + distractors
            random.shuffle(options)
            st.session_state['quiz_options'] = options
            st.session_state['correct_option'] = correct_ans
            
            # Generate Hint (Model B)
            st.session_state['hint'] = generate_hint(article_input, question_input)
            
            st.success("Quiz generated! Go to the 'Quiz View' tab to take the test.")
        else:
            st.error("Please provide both an article and a question.")

# --- SCREEN 2: QUIZ VIEW ---
with tab2:
    st.header("Take the Quiz")
    if 'quiz_options' in st.session_state:
        st.markdown(f"**Question:** {st.session_state['question']}")
        
        user_choice = st.radio("Select your answer:", st.session_state['quiz_options'])
        
        if st.button("Check Answer"):
            if user_choice == st.session_state['correct_option']:
                st.success("✅ Correct! Model A verified this answer.")
            else:
                st.error(f"❌ Incorrect. The correct answer is: {st.session_state['correct_option']}")
    else:
        st.info("Please generate a quiz in the Article Input tab first.")

# --- SCREEN 3: HINT PANEL ---
with tab3:
    st.header("Need help?")
    if 'hint' in st.session_state:
        with st.expander("Show Hint (Extracted via TF-IDF & Cosine Similarity)"):
            st.write(f"*{st.session_state['hint']}*")
    else:
        st.info("Generate a quiz first to see hints.")

# --- SCREEN 4: ANALYTICS DASHBOARD ---
with tab4:
    st.header("Developer Dashboard - Model A Performance")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{test_acc * 100:.2f}%")
    col2.metric("Macro F1-Score", f"{test_report['macro avg']['f1-score']:.2f}")
    col3.metric("Clustering Engine", "K-Means (4 Clusters)")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(test_report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='Blues'))