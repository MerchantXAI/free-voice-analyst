import streamlit as st
import google.generativeai as genai
from gtts import gTTS
import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. ACCESSIBLE CLOUD CREDENTIALS ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("Setup incomplete! Please add your GEMINI_API_KEY to the Streamlit Advanced Secrets Panel.")
    st.stop()

# Configure the Gemini Engine
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. STREAMLIT INTERFACE AND STYLING ---
st.set_page_config(page_title="Cloud Voice Data Analyst AI", layout="wide")
st.title("📊 Cloud Data Analyst Voicebot")
st.caption("100% Free Cloud Engine — Controlled completely via Mobile and PC Browsers")

# Initialize Chat Memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "model", "content": "Hello! Use the top-left menu to upload a data file (.csv or .xlsx), then ask me questions via text or your mic!"}]
if "current_df" not in st.session_state:
    st.session_state["current_df"] = None

# Sidebar Upload Drawer (Hidden on phones behind a small arrow in top-left)
st.sidebar.header("📁 Data Upload Hub")
uploaded_file = st.sidebar.file_uploader("Upload your Dataset:", type=["csv", "xlsx"])

data_context = ""
if uploaded_file is not None:
    try:
        # Read the uploaded file into a Pandas DataFrame with standard web encoding and fallback
        if uploaded_file.name.endswith(".csv"):
            try:
                st.session_state["current_df"] = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                st.session_state["current_df"] = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            st.session_state["current_df"] = pd.read_excel(uploaded_file)
            
        df = st.session_state["current_df"]
        st.sidebar.success("Dataset synced to Cloud!")
        st.sidebar.dataframe(df.head(5)) 
        
        # Capture columns and data structure to feed Gemini
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        
        data_context = f"""
        You are a Cloud Data Analyst AI helping a user examine their spreadsheet data.
        
        DATASET SUMMARY:
        {df_info}
        
        FIRST 5 ROWS OF DATA:
        {df.head(5).to_string()}
        
        INSTRUCTIONS: 
        Keep your spoken answers brief and easy to understand aloud. 
        If the user asks for a chart, graph, plot, or trends, provide your text summary first. Then, finish your response by explaining what kind of chart should be rendered.
        """
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Render Old Chat History on screen refresh
for msg in st.session_state.messages:
    st_role = "assistant" if msg["role"] == "model" else msg["role"]
    with st.chat_message(st_role):
        st.write(msg["content"])
        if "plot_type" in msg:
            st.pyplot(msg["plot_fig"])

user_query = ""
audio_input_used = False

# --- 3. INPUT HANDLING ---
# Widget 1: Microphone (Handles phone/laptop audio hardware automatically)
audio_value = st.audio_input("🎤 Record Voice Command")
if audio_value:
    audio_input_used = True
    temp_voice_path = "user_voice.wav"
    with open(temp_voice_path, "wb") as f:
        f.write(audio_value.read())

# Widget 2: Text Chat Box
text_input = st.chat_input("⌨️ Or type your data or chart command here...")
if text_input:
    user_query = text_input

# --- 4. DATA PROCESSING AND ENGINE PIPELINE ---
if audio_input_used or user_query:
    with st.spinner("Processing request..."):
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        
        if audio_input_used:
            st.session_state.messages.append({"role": "user", "content": "🗣️ Sent a voice command"})
            with st.chat_message("user"):
                st.write("🗣️ Sent a voice command")
                
            try:
                with open(temp_voice_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                    
                audio_payload = {
                    "mime_type": "audio/wav",
                    "data": audio_data
                }
                
                response = model.generate_content([
                    data_context, 
                    "Transcribe the query hidden within this audio file, analyze the spreadsheet info matching it, and provide a short output answer summary.", 
                    audio_payload
                ])
                bot_response = response.text
            except Exception as e:
                bot_response = f"Sorry, I ran into an audio parsing error: {str(e)}. Please try typing your request instead!"
            
            if os.path.exists(temp_voice_path):
                os.remove(temp_voice_path)
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
                
            full_prompt = f"{data_context}\n\nUser Question: {user_query}"
            response = model.generate_content(full_prompt)
            bot_response = response.text

    msg_data = {"role": "model", "content": bot_response}
    df = st.session_state["current_df"]
    
    # --- 5. AUTOMATED VISUAL CHART PLOTTING ---
    check_text = user_query.lower() if not audio_input_used else bot_response.lower()
    is_chart_request = any(word in check_text for word in ["chart", "graph", "plot", "bar", "histogram", "scatter", "visualize", "trend"])
    
    if is_chart_request and df is not None:
        try:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.set_theme(style="darkgrid")
            
            # Smart data grouping matcher
            if len(cat_cols) >= 1 and len(num_cols) >= 1:
                top_cats = df[cat_cols].value_counts().nlargest(10).index
                filtered_df = df[df[cat_cols].isin(top_cats)]
                sns.barplot(data=filtered_df, x=cat_cols, y=num_cols, errorbar=None, ax=ax, palette="Blues_d")
                ax.set_title(f"{num_cols} Breakdown by Top {cat_cols}")
                plt.xticks(rotation=45)
            elif len(num_cols) >= 1:
                sns.lineplot(data=df, y=num_cols, x=df.index, ax=ax, color="#009688")
                ax.set_title(f"{num_cols} Trend Track")
            
            plt.tight_layout()
            msg_data["plot_type"] = "pyplot"
            msg_data["plot_fig"] = fig
        except Exception as chart_err:
            bot_response += f"\n*(Chart plotting error: {chart_err})*"
            msg_data["content"] = bot_response

    # Show Response elements simultaneously
    st.session_state.messages.append(msg_data)
    with st.chat_message("assistant"):
        st.write(bot_response)
        if "plot_type" in msg_data:
            st.pyplot(msg_data["plot_fig"])

    # --- 6. VOICE SYNTHESIS OUTPUT WITH ACRONYM FIXES ---
    with st.spinner("Speaking reply..."):
        # Strip structural markdown markers
        clean_text = bot_response.replace("**", "").replace("#", "").replace("`", "")
        
        if "|" in clean_text:
            clean_text = "I have generated and displayed the table breakdown on the screen for you to review."
        else:
            # SCRIPT-LEVEL DICTIONARY LOOKUP TO PREVENT ACCENT STUTTERS OR SPELLING GLITCHES
            pronunciation_fixes = {
                "ORDERNUMBER": "order number",
                "SALES": "sales",
                "QUANTITYORDERED": "quantity ordered",
                "PRICEEACH": "price each",
                "MSRP": "m s r p",
                "PRODUCTLINE": "product line",
                "CUSTOMERNAME": "customer name",
                "ORDERDATE": "order date",
                "DEALSIZE": "deal size",
                "COUNTRY": "country",
                "CITY": "city",
                "STATE": "state",
                "ADDRESSLINE2": "address line 2",
                "POSTALCODE": "postal code",
                "TERRITORY": "territory",
                "MSRP": "manufacturer retail price"
            }
            
            # Map upper structural tags to lowercase spaced strings before voice synthesis
            for upper_word, spoken_word in pronunciation_fixes.items():
                clean_text = clean_text.replace(upper_word, spoken_word)
                
            # Global lowercase converter fallback to ensure any remaining uppercase data prints correctly
            clean_text = clean_text.lower()
            
        tts = gTTS(text=clean_text, lang='en', slow=False)
        temp_audio = "cloud_response.mp3"
        tts.save(temp_audio)
        st.audio(temp_audio, format="audio/mp3", autoplay=True)
