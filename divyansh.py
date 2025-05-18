import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
import tempfile
import os
import pandas as pd
import time
import io
import base64
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import zipfile
import plotly.graph_objects as go
import math
from PIL import Image

# Set page config
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="wide")

# Add this CSS right after st.set_page_config()
st.markdown("""
<style>
    /* Main container with subtle gradient animation */
    .main {
        background: linear-gradient(-45deg, #f5f7fa, #e4e8eb, #eef2f7, #e8edf3);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    /* Animated title with shadow effect */
    .title-container {
        text-align: center;
        padding: 2rem 0;
        animation: fadeIn 1s ease-out;
    }
    
    .title-text {
        color: #2c3e50;
        font-size: 3.2rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: glowShadow 3s ease-in-out infinite;
    }
    
    .subtitle-text {
        color: #34495e;
        font-size: 1.2rem;
        max-width: 600px;
        margin: 1rem auto;
        line-height: 1.6;
        opacity: 0.9;
        text-align: center;
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: linear-gradient(to right, #ffffff, #f8f9fa);
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
        opacity: 0.8;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        opacity: 1;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #2980b9, #3498db) !important;
        opacity: 1 !important;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3) !important;
    }
    
    /* Animated button styling */
    .stButton > button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border-radius: 25px;
        padding: 0.75rem 2.5rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2);
        animation: buttonGlow 3s ease-in-out infinite;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(52, 152, 219, 0.3);
    }
    
    /* Card styling with hover effect */
    .css-1r6slb0 {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(52, 152, 219, 0.1);
    }
    
    .css-1r6slb0:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(52, 152, 219, 0.15);
        border-color: rgba(52, 152, 219, 0.2);
    }
    
    /* Progress bar with gradient */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3498db, #2980b9);
        transition: width 0.5s ease-out;
    }
    
    /* Message boxes with animations */
    .stInfo, .stSuccess, .stWarning, .stError {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .stInfo:hover, .stSuccess:hover, .stWarning:hover, .stError:hover {
        transform: translateX(5px);
    }
    
    /* Animations */
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    @keyframes glowShadow {
        0% { text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
        50% { text-shadow: 2px 2px 8px rgba(52, 152, 219, 0.2); }
        100% { text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    }
    
    @keyframes buttonGlow {
        0% { box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2); }
        50% { box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4); }
        100% { box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .dataframe:hover {
        box-shadow: 0 6px 12px rgba(52, 152, 219, 0.15);
    }
    
    .dataframe th {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        padding: 12px;
    }
    
    .dataframe td {
        padding: 10px;
        transition: all 0.2s ease;
    }
    
    .dataframe tr:hover td {
        background-color: rgba(52, 152, 219, 0.05);
    }
    
    /* Animated background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        transition: all 0.5s ease;
    }
    
    /* Semi-transparent white container for content */
    .stApp > header {
        background-color: transparent !important;
    }
    
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(10px);
    }
    
    /* Animated gradient background */
    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    /* Floating animation for cards */
    .css-1r6slb0 {
        animation: float 6s ease-in-out infinite;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    @keyframes float {
        0% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
        100% {
            transform: translateY(0px);
        }
    }
    
    /* Glowing effect for buttons */
    .stButton > button {
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0% {
            box-shadow: 0 0 5px rgba(52, 152, 219, 0.2);
        }
        50% {
            box-shadow: 0 0 20px rgba(52, 152, 219, 0.4);
        }
        100% {
            box-shadow: 0 0 5px rgba(52, 152, 219, 0.2);
        }
    }
    
    /* Particle effect background */
    .background-animation {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -1;
        pointer-events: none;
    }
    
    .particle {
        position: absolute;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        pointer-events: none;
        animation: particleFloat 20s infinite linear;
    }
    
    @keyframes particleFloat {
        0% {
            transform: translateY(0) translateX(0);
            opacity: 0;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            transform: translateY(-100vh) translateX(100vw);
            opacity: 0;
        }
    }
</style>

<div class="background-animation">
    <script>
        function createParticle() {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.width = Math.random() * 5 + 'px';
            particle.style.height = particle.style.width;
            particle.style.left = Math.random() * 100 + 'vw';
            particle.style.top = Math.random() * 100 + 'vh';
            document.querySelector('.background-animation').appendChild(particle);
            
            setTimeout(() => {
                particle.remove();
            }, 20000);
        }
        
        setInterval(createParticle, 500);
    </script>
</div>
""", unsafe_allow_html=True)

# Update the title section with centered subtitle
st.markdown("""
    <div class="title-container">
        <h1 class="title-text">🎙️ Speech Emotion Recognition</h1>
        <div style="display: flex; justify-content: center;">
            <p class="subtitle-text" style="text-align: center;">
                Analyze emotions in speech with advanced AI technology.<br>
                Discover the emotional content of any voice recording with precision and insight.
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("model.h5")
    except:
        st.error("Model file 'model.h5' not found. Please ensure the model file exists.")
        return None

# Function to extract MFCC features
def extract_features(y, sr=22050, fixed_length=200):
    try:
        if y is None or len(y) == 0:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < fixed_length:
            padded_mfcc = np.pad(mfcc, ((0, 0), (0, fixed_length - mfcc.shape[1])), mode="constant")
        else:
            padded_mfcc = mfcc[:, :fixed_length]
        return padded_mfcc.T
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Function to extract features from file
def extract_features_from_file(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        features = extract_features(y, sr)
        return features, y, sr
    except Exception as e:
        st.error(f"Error processing file {audio_path}: {str(e)}")
        return None, None, None

# Display waveform of audio
def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    return fig

# Generate PDF report
def generate_pdf_report(audio_name, emotion_results, emotion_labels, waveform_fig):
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Add waveform page
        pdf.savefig(waveform_fig)
        plt.close()
        
        # Create emotion probability chart
        fig, ax = plt.subplots(figsize=(10, 6))
        confidence_values = [emotion_results[2][i] * 100 for i in range(len(emotion_labels))]
        bars = ax.bar(emotion_labels, confidence_values)
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        ax.set_title(f'Emotion Analysis for "{audio_name}"')
        ax.set_ylabel('Probability (%)')
        ax.set_ylim(0, 100)
        pdf.savefig(fig)
        plt.close()
    
    buffer.seek(0)
    return buffer

# Export to CSV
def export_to_csv(results, emotion_labels, filename):
    buffer = io.BytesIO()
    
    data = {'Emotion': emotion_labels, 'Confidence (%)': [results[2][i] * 100 for i in range(len(emotion_labels))]}
    df = pd.DataFrame(data)
    
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer

# Add this new function after your existing functions
def create_emotion_spectrum_graph(predictions, emotions, main_emotion):
    # Convert predictions to distances from the main emotion
    main_emotion_idx = emotions.index(main_emotion)
    main_emotion_value = predictions[main_emotion_idx]
    
    # Calculate emotional distances (inverse of similarity)
    distances = [1 - abs(p - main_emotion_value) for p in predictions]
    
    # Create circular data by repeating the first value
    emotions_circular = emotions + [emotions[0]]
    distances_circular = distances + [distances[0]]
    
    # Create the radar chart
    fig = go.Figure()
    
    # Add the emotion spectrum trace
    fig.add_trace(go.Scatterpolar(
        r=distances_circular,
        theta=emotions_circular,
        fill='toself',
        name='Emotional Spectrum',
        line_color='rgba(255, 165, 0, 0.8)',
        fillcolor='rgba(255, 165, 0, 0.3)'
    ))
    
    # Update layout with better styling
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12, color="#2c3e50"),
                gridcolor="#ecf0f1"
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color="#2c3e50"),
                gridcolor="#ecf0f1"
            ),
            bgcolor="#f8f9fa"
        ),
        showlegend=False,
        title={
            'text': 'Emotional Spectrum Analysis',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color="#2c3e50")
        },
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=100, b=40, l=40, r=40)
    )
    
    return fig

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")

# Create tabs for different types of analysis
tabs = st.tabs([
    "🎤 Record Audio",
    "📁 Batch Processing",
    "📊 History"
])

# Try to load the model
model = load_model()
if model is None:
    st.warning("⚠️ Model could not be loaded. Using placeholder predictions.")
    # Create a placeholder model for testing
    class PlaceholderModel:
        def predict(self, features, verbose=0):
            batch_size = features.shape[0]
            num_emotions = 7
            random_preds = np.random.rand(batch_size, num_emotions)
            return random_preds / np.sum(random_preds, axis=1, keepdims=True)
    
    model = PlaceholderModel()

# Define emotion labels
emotions = ['Neutral','Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Tab 1: Record Audio
with tabs[0]:
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h3 style='color: #2c3e50; margin-bottom: 0.5rem;'>Voice Emotion Analyzer</h3>
            <p style='color: #7f8c8d;'>Record your voice to analyze the emotional content in your speech.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Record Audio")
        st.write("Click the microphone button below to start recording. Click again to stop.")
        
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_size="2x",
        )
    
    # Process the recorded audio
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        # Save audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
            tmpfile.write(audio_bytes)
            audio_path = tmpfile.name
        
        st.success("✅ Audio recorded successfully!")
        
        # Process button
        if st.button("🔍 Analyze Emotion"):
            with st.spinner("Processing audio..."):
                try:
                    # Load audio
                    y, sr = librosa.load(audio_path, sr=22050)
                    
                    if y is not None and len(y) > 0:
                        # Display waveform
                        st.subheader("Audio Waveform")
                        waveform_fig = plot_waveform(y, sr)
                        st.pyplot(waveform_fig)
                        
                        # Simple analysis for entire audio
                        features = extract_features(y, sr)
                        if features is not None:
                            features = np.expand_dims(features, axis=0)
                            prediction = model.predict(features)
                            predicted_class = np.argmax(prediction[0])
                            confidence = prediction[0][predicted_class] * 100
                            
                            if predicted_class < len(emotions):
                                emotion = emotions[predicted_class]
                            else:
                                emotion = f"Class {predicted_class}"
                            
                            # Display result
                            st.subheader("Result")
                            st.markdown(f"### Detected Emotion: **{emotion.upper()}**")
                            st.progress(float(min(confidence/100, 1.0)))
                            st.write(f"Confidence: {confidence:.1f}%")
                            
                            # Create a bar chart for all emotions
                            st.subheader("Emotion Probabilities")
                            chart_data = {emotions[i]: float(prediction[0][i]) * 100 
                                         for i in range(min(len(emotions), len(prediction[0])))}
                            st.bar_chart(chart_data)
                            
                            # Add the new Emotional Spectrum Graph
                            st.subheader("Emotional Spectrum Analysis")
                            spectrum_fig = create_emotion_spectrum_graph(
                                prediction[0],
                                emotions,
                                emotion
                            )
                            st.plotly_chart(spectrum_fig, use_container_width=True, key="record_spectrum")
                            
                            # Store in history
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.history.append({
                                'timestamp': timestamp,
                                'audio_length': len(y) / sr,
                                'dominant_emotion': emotion,
                                'prediction': prediction[0],
                                'audio_data': y,
                                'sample_rate': sr,
                                'name': f"Recording_{timestamp}"
                            })
                            
                            # Export options
                            st.subheader("Export Results")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                pdf_buffer = generate_pdf_report(
                                    "Recording", 
                                    (0, predicted_class, prediction[0]),
                                    emotions, 
                                    waveform_fig
                                )
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_buffer,
                                    file_name="emotion_analysis.pdf",
                                    mime="application/pdf"
                                )
                            
                            with col2:
                                csv_buffer = export_to_csv(
                                    (0, predicted_class, prediction[0]),
                                    emotions, 
                                    "emotion_analysis.csv"
                                )
                                st.download_button(
                                    label="Download CSV Data",
                                    data=csv_buffer,
                                    file_name="emotion_analysis.csv",
                                    mime="text/csv"
                                )
                    else:
                        st.error("Failed to load audio file. Please try recording again.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            
            # Clean up temporary file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                st.warning(f"Could not remove temporary file: {str(e)}")
    else:
        with col2:
            st.info("👈 Click the microphone button to start recording")

# Tab 2: Batch Processing
with tabs[1]:
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h3 style='color: #2c3e50; margin-bottom: 0.5rem;'>Batch Audio Analysis</h3>
            <p style='color: #7f8c8d;'>Process multiple audio files at once for emotion analysis.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Batch Process Audio Files")
    st.write("Upload multiple audio files for analysis")
    
    uploaded_files = st.file_uploader("Upload audio files", 
                                     type=["wav", "mp3", "ogg", "flac"], 
                                     accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        # Process button for batch analysis
        if st.button("🔍 Process All Files"):
            batch_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a container for all results
            results_container = st.container()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
                
                try:
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file.name.split(".")[-1]}') as tmpfile:
                        tmpfile.write(file.getvalue())
                        temp_path = tmpfile.name
                    
                    # Extract features
                    features, y, sr = extract_features_from_file(temp_path)
                    
                    if features is not None and y is not None:
                        # Create waveform
                        waveform_fig = plot_waveform(y, sr)
                        
                        # Add batch dimension for model prediction
                        features = np.expand_dims(features, axis=0)
                        
                        # Make prediction
                        prediction = model.predict(features, verbose=0)
                        predicted_class = np.argmax(prediction[0])
                        confidence = prediction[0][predicted_class] * 100
                        
                        # Get emotion label
                        if predicted_class < len(emotions):
                            emotion = emotions[predicted_class]
                        else:
                            emotion = f"Class {predicted_class}"
                        
                        # Store result
                        result = {
                            'filename': file.name,
                            'emotion': emotion,
                            'confidence': confidence,
                            'prediction': prediction[0],
                            'waveform': waveform_fig,
                            'audio_data': y,
                            'sample_rate': sr
                        }
                        batch_results.append(result)
                    else:
                        st.warning(f"Could not process file {file.name}. Skipping.")
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Clean up temp file
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
            
            status_text.text("Processing complete!")
            
            # Display batch results
            with results_container:
                if batch_results:
                    st.subheader("Batch Analysis Results")
                    
                    # Summary table of results
                    results_df = pd.DataFrame({
                        'Filename': [r['filename'] for r in batch_results],
                        'Detected Emotion': [r['emotion'] for r in batch_results],
                        'Confidence (%)': [r['confidence'] for r in batch_results]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Create download options
                    st.subheader("Export Results")
                    
                    # Export as CSV
                    csv_buffer = io.BytesIO()
                    results_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_buffer,
                        file_name="batch_analysis_results.csv",
                        mime="text/csv"
                    )
                    
                    # Create a ZIP file with individual PDF reports
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                        for result in batch_results:
                            # Generate PDF for this result
                            try:
                                pdf_buffer = generate_pdf_report(
                                    result['filename'],
                                    (0, emotions.index(result['emotion']), result['prediction']),
                                    emotions,
                                    result['waveform']
                                )
                                # Add PDF to ZIP file
                                pdf_filename = f"{result['filename'].split('.')[0]}_analysis.pdf"
                                zipf.writestr(pdf_filename, pdf_buffer.getvalue())
                            except Exception as e:
                                st.warning(f"Could not generate PDF for {result['filename']}: {str(e)}")
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download All PDF Reports (ZIP)",
                        data=zip_buffer,
                        file_name="batch_analysis_reports.zip",
                        mime="application/zip"
                    )
                    
                    # Show detailed results for each file in expanders
                    st.subheader("Detailed Results")
                    for result in batch_results:
                        with st.expander(f"{result['filename']} - {result['emotion']} ({result['confidence']:.1f}%)"):
                            st.pyplot(result['waveform'])
                            
                            # Create a bar chart for all emotions
                            chart_data = {emotions[i]: float(result['prediction'][i]) * 100 
                                         for i in range(min(len(emotions), len(result['prediction'])))}
                            st.bar_chart(chart_data)
                    
                    # Add results to history
                    for result in batch_results:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        history_entry = {
                            'timestamp': timestamp,
                            'name': result['filename'],
                            'audio_length': len(result['audio_data']) / result['sample_rate'],
                            'dominant_emotion': result['emotion'],
                            'prediction': result['prediction'],
                            'audio_data': result['audio_data'],
                            'sample_rate': result['sample_rate']
                        }
                        st.session_state.history.append(history_entry)
                else:
                    st.warning("No files could be processed successfully.")

# Tab 3: History
with tabs[2]:
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
            <h3 style='color: #2c3e50; margin-bottom: 0.5rem;'>Analysis History</h3>
            <p style='color: #7f8c8d;'>View and compare your previous emotion analysis results.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Analysis History")
    
    if not st.session_state.history:
        st.info("No analysis history yet. Record or upload audio to see results here.")
    else:
        # Display history in a table
        history_df = pd.DataFrame({
            'Time': [entry['timestamp'] for entry in st.session_state.history],
            'Name': [entry['name'] for entry in st.session_state.history],
            'Duration (s)': [f"{entry['audio_length']:.2f}" for entry in st.session_state.history],
            'Emotion': [entry['dominant_emotion'] for entry in st.session_state.history]
        })
        
        st.dataframe(history_df)
        
        # Select an entry to view details
        selected_idx = st.selectbox("Select entry to view details:", 
                                   range(len(st.session_state.history)),
                                   format_func=lambda i: f"{st.session_state.history[i]['timestamp']} - {st.session_state.history[i]['name']}")
        
        selected_entry = st.session_state.history[selected_idx]
        
        # Display details of selected entry
        st.subheader(f"Details for {selected_entry['name']}")
        
        # Plot waveform
        waveform_fig = plot_waveform(selected_entry['audio_data'], selected_entry['sample_rate'])
        st.pyplot(waveform_fig)
        
        # Display emotion
        st.markdown(f"### Detected Emotion: **{selected_entry['dominant_emotion'].upper()}**")
        
        # Create a bar chart for all emotions
        chart_data = {emotions[i]: float(selected_entry['prediction'][i]) * 100 
                     for i in range(min(len(emotions), len(selected_entry['prediction'])))}
        st.bar_chart(chart_data)
        
        # Add Emotional Spectrum Graph
        st.subheader("Emotional Spectrum Analysis")
        spectrum_fig = create_emotion_spectrum_graph(
            selected_entry['prediction'],
            emotions,
            selected_entry['dominant_emotion']
        )
        st.plotly_chart(spectrum_fig, use_container_width=True, key="history_spectrum")
        
        # Export options
        st.subheader("Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate PDF for selected entry
            pdf_buffer = generate_pdf_report(
                selected_entry['name'],
                (0, emotions.index(selected_entry['dominant_emotion']), selected_entry['prediction']),
                emotions,
                waveform_fig
            )
            
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name=f"{selected_entry['name']}_analysis.pdf",
                mime="application/pdf"
            )
        
        with col2:
            # Generate CSV for selected entry
            csv_buffer = export_to_csv(
                (0, emotions.index(selected_entry['dominant_emotion']), selected_entry['prediction']),
                emotions,
                f"{selected_entry['name']}_analysis.csv"
            )
            
            st.download_button(
                label="Download CSV Data",
                data=csv_buffer,
                file_name=f"{selected_entry['name']}_analysis.csv",
                mime="text/csv"
            )