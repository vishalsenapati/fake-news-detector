"""
Streamlit App for Fake News Detection
Uses the exported TF-IDF + Logistic Regression model (fake_news_model.joblib)
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .fake-news {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    .real-news {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #22c55e;
        color: #166534;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
    }
    .stTextArea > div > div > textarea {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model pipeline"""
    model_path = 'fake_news_model.joblib'
    if not os.path.exists(model_path):
        return None
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict_news(text, pipeline):
    """Predict if news is fake or real"""
    if pipeline is None:
        return None, None
    
    # Get prediction
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    
    # probabilities[0] = P(Real), probabilities[1] = P(Fake)
    confidence = probabilities[prediction]
    
    return prediction, confidence


def plot_confidence(probabilities):
    """Create a confidence visualization"""
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = ['Real', 'Fake']
    colors = ['#22c55e', '#ef4444']
    bars = ax.barh(labels, probabilities, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1%}', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_title('Model Confidence', fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def show_confusion_matrix():
    """Display confusion matrix image if exists"""
    if os.path.exists('confusion_matrix.png'):
        st.image('confusion_matrix.png', caption='Confusion Matrix (Test Set)', use_container_width=True)


# Sidebar
with st.sidebar:
    st.markdown("### 🔍 Fake News Detector")
    st.markdown("---")
    
    # Model info
    st.markdown("#### Model Information")
    st.info("""
    **Algorithm**: TF-IDF + Logistic Regression
    
    **Features**: 
    - TF-IDF Vectorization (1-2 grams)
    - Max 10,000 features
    - Balanced class weights
    
    **Training Data**: 10,000 samples (5k real, 5k fake)
    """)
    
    st.markdown("---")
    
    # Show metrics
    st.markdown("#### Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "100%")
        st.metric("Precision", "100%")
    with col2:
        st.metric("Recall", "100%")
        st.metric("F1-Score", "100%")
    
    st.caption("Note: 100% on synthetic demo data. Real-world performance will vary.")
    
    st.markdown("---")
    
    # Confusion matrix
    if st.checkbox("Show Confusion Matrix"):
        show_confusion_matrix()
    
    st.markdown("---")
    st.markdown("#### About")
    st.markdown("""
    **Multi-Layered Misinformation Detection Platform**
    
    Phase 1: Baseline ML Classifier
    - TF-IDF + Logistic Regression
    - Model exported via joblib
    - Integrated with Streamlit
    
    *B.Tech 1st Year Multidisciplinary Project*
    """)


# Main content
st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Misinformation Detection | Phase 1: Baseline ML Model</p>', unsafe_allow_html=True)

# Load model
pipeline = load_model()

if pipeline is None:
    st.error("""
    ⚠️ **Model not found!** 
    
    Please run the training script first:
    ```bash
    python train_model.py
    ```
    This will create `fake_news_model.joblib` in the current directory.
    """)
    st.stop()

# Input area
st.markdown("### 📝 Enter News Text for Analysis")
st.markdown("Paste a news article, headline, or social media post below:")

# Example texts
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📰 Real News Example", use_container_width=True):
        st.session_state.example_text = "Government announces new policy to improve healthcare access affecting millions of citizens. According to Reuters sources, the initiative will expand coverage to rural areas."
with col2:
    if st.button("🚨 Fake News Example", use_container_width=True):
        st.session_state.example_text = "SHOCKING: Government secretly plans to control your children - mainstream media HIDES the truth! Secret documents reveal massive conspiracy."
with col3:
    if st.button("🧪 Custom Test", use_container_width=True):
        st.session_state.example_text = "Scientists discover breakthrough in renewable energy research that could reduce carbon emissions significantly. According to BBC sources."

# Text input
default_text = st.session_state.get('example_text', '')
user_input = st.text_area(
    "News Text",
    value=default_text,
    height=200,
    placeholder="Paste news article, headline, or social media post here...",
    label_visibility="collapsed"
)

# Clear example after use
if 'example_text' in st.session_state:
    del st.session_state.example_text

# Analyze button
analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
with analyze_col2:
    analyze_button = st.button("🔬 Analyze News", type="primary", use_container_width=True)

# Results
if analyze_button and user_input.strip():
    with st.spinner("Analyzing..."):
        prediction, confidence = predict_news(user_input, pipeline)
        
        # Get probabilities for visualization
        probabilities = pipeline.predict_proba([user_input])[0]
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    # Prediction box
    if prediction == 1:  # Fake
        st.markdown(f"""
        <div class="prediction-box fake-news">
            🚨 FAKE NEWS DETECTED
            <br><br>
            Confidence: {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("""
        **⚠️ This content appears to be misinformation.**
        
        Recommendations:
        - Verify with multiple trusted sources
        - Check the publication date and author
        - Look for supporting evidence from reputable outlets
        - Be cautious before sharing
        """)
    else:  # Real
        st.markdown(f"""
        <div class="prediction-box real-news">
            ✅ APPEARS CREDIBLE
            <br><br>
            Confidence: {confidence:.1%}
        </div>
        """, unsafe_allow_html=True)
        
        st.success("""
        **✅ This content appears to be legitimate news.**
        
        Still recommended:
        - Cross-reference with other sources
        - Check the publication date
        - Verify the author/organization credibility
        """)
    
    # Confidence visualization
    st.markdown("#### Confidence Breakdown")
    fig = plot_confidence(probabilities)
    st.pyplot(fig)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown(f"""
        **Model Prediction**: {'Fake (1)' if prediction == 1 else 'Real (0)'}
        
        **Class Probabilities**:
        - P(Real) = {probabilities[0]:.4f} ({probabilities[0]:.1%})
        - P(Fake) = {probabilities[1]:.4f} ({probabilities[1]:.1%})
        
        **Decision Threshold**: 0.5 (default)
        
        **Pipeline Components**:
        1. TF-IDF Vectorizer (10,000 features, 1-2 grams)
        2. Logistic Regression (balanced class weights)
        """)

elif analyze_button:
    st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>
    <p>Multi-Layered Misinformation Detection & Analysis Platform</p>
    <p>Phase 1: Baseline ML Classifier | B.Tech Multidisciplinary Project</p>
</div>
""", unsafe_allow_html=True)