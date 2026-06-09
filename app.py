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
st.set_page_config(page_title="Executive Data Analyst AI", layout="wide")
st.title("📊 Strategic Cloud Data Analyst Voicebot")
st.caption("100% Free Cloud Engine — Generating Executive Insights, Graphics & Voice Over")

# Initialize Chat & Chart Memory State Trackers
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "model", "content": "Hello Leader! Upload your .csv or .xlsx corporate log files into the sidebar menu, then drop a voice command or text query to trigger a comprehensive strategic analysis."}]
if "current_df" not in st.session_state:
    st.session_state["current_df"] = None

# Sidebar Upload Drawer (Hidden on phones behind a small arrow in top-left)
st.sidebar.header("📁 Corporate Data Upload Hub")
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
        st.sidebar.success("Dataset Synced to Cloud Core!")
        st.sidebar.dataframe(df.head(5)) 
        
        # Capture columns and data structure to feed Gemini
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        
        data_context = f"""
        You are an Elite Executive Data Analyst and Business Strategist. You are evaluating a high-stakes dataset for corporate decision makers.
        
        DATASET SUMMARY:
        {df_info}
        
        FIRST 5 ROWS OF DATA FOR CONTEXT:
        {df.head(5).to_string()}
        
        YOUR OBJECTIVE:
        When answering questions or providing an analysis, you must present an elaborate, structured response with these exact sections:
        1. **Executive Summary & High-Level Insights**: A concise overview of global operational health.
        2. **Business Stakeholder Recommendations**: Actionable strategies to drive revenue, capture market share, and optimize pricing/discounts (e.g., maximizing MSRP vs PRICEEACH captures).
        3. **Technical Stakeholder Recommendations**: Data engineering and data quality blueprints (e.g., addressing missing records in STATE, POSTALCODE, or TERRITORY fields, improving CRM logs).
        4. **Recommended Growth Strategies**: Specific, future-focused tactical steps to accelerate positive scaling.
        
        INSTRUCTIONS FOR VOICE COMPONENT:
        Keep the text clear and professional. Avoid reading long lists of individual data points aloud. If a chart is requested or generated, mention it briefly in your narrative.
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
audio_value = st.audio_input("🎤 Record Strategic Voice Command")
if audio_value:
    audio_input_used = True
    temp_voice_path = "user_voice.wav"
    with open(temp_voice_path, "wb") as f:
        f.write(audio_value.read())

text_input = st.chat_input("⌨️ Or type your strategic data command here...")
if text_input:
    user_query = text_input

# --- 4. DATA PROCESSING AND ENGINE PIPELINE ---
if audio_input_used or user_query:
    with st.spinner("Analyzing corporate performance matrix..."):
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        
        if audio_input_used:
            st.session_state.messages.append({"role": "user", "content": "🗣️ Sent an executive voice command"})
            with st.chat_message("user"):
                st.write("🗣️ Sent an executive voice command")
                
            try:
                with open(temp_voice_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                    
                audio_payload = {
                    "mime_type": "audio/wav",
                    "data": audio_data
                }
                
                response = model.generate_content([
                    data_context, 
                    "Transcribe and process this strategic voice command. Build a thorough data analysis with clear recommendations and insights for stakeholder growth.", 
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
    
    # --- 5. REFINED BULLETPROOF CHARTING PIPELINE ---
    check_text = user_query.lower() if not audio_input_used else bot_response.lower()
    is_chart_request = any(word in check_text for word in ["chart", "graph", "plot", "bar", "histogram", "scatter", "visualize", "trend", "analysis"])
    
    if is_chart_request and df is not None:
        try:
            # Standardize column queries to match uppercase profiles
            df.columns = [c.upper() for c in df.columns]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.set_theme(style="darkgrid")
            
            # Scenario A: Timeline Trend Tracking (SALES over ORDERDATE or tracking indices)
            if "SALES" in df.columns and any(x in check_text for x in ["trend", "time", "timeline", "date", "growth"]):
                if "ORDERDATE" in df.columns:
                    # Convert to standard format strings to smooth chart limits (YYYY-MM)
                    df['SHORT_DATE'] = df['ORDERDATE'].astype(str).str[:7]
                    timeline_df = df.groupby('SHORT_DATE')['SALES'].sum().reset_index().sort_values('SHORT_DATE')
                    sns.lineplot(data=timeline_df, x='SHORT_DATE', y='SALES', marker='o', ax=ax, color="#009688", linewidth=2.5)
                    ax.set_title("Global Sales Performance Trend Over Time", fontsize=12, fontweight='bold')
                    plt.xticks(rotation=45)
                else:
                    sns.lineplot(data=df.head(100), y='SALES', x=df.head(100).index, ax=ax, color="#009688")
                    ax.set_title("Sales Performance Volatility Tracking Index", fontsize=12, fontweight='bold')
            
            # Scenario B: Categorical Volume Breakdown (Slicing SALES by PRODUCTLINE or COUNTRY)
            elif "SALES" in df.columns and any(x in df.columns for x in ["PRODUCTLINE", "COUNTRY"]):
                target_cat = "PRODUCTLINE" if "PRODUCTLINE" in df.columns else "COUNTRY"
                
                # Safely pre-aggregate data vectors to prevent dimension vector mismatch errors
                grouped_df = df.groupby(target_cat)['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(10)
                
                sns.barplot(data=grouped_df, x=target_cat, y='SALES', ax=ax, palette="Blues_r", errorbar=None)
                ax.set_title(f"Top Revenue Contributors by Category Profiles ({target_cat})", fontsize=12, fontweight='bold')
                plt.xticks(rotation=35, ha='right')
                
            # Scenario C: Operational Size Mix (DEALSIZE counts)
            elif "DEALSIZE" in df.columns:
                grouped_deal = df['DEALSIZE'].value_counts().reset_index()
                grouped_deal.columns = ['DEALSIZE', 'COUNT']
                sns.barplot(data=grouped_deal, x='DEALSIZE', y='COUNT', ax=ax, palette="YlGnBu_r")
                ax.set_title("Operational Distribution Index by Transaction Deal Sizes", fontsize=12, fontweight='bold')
            
            else:
                # Catch-all baseline placeholder mapping numerical features safely
                num_cols = df.select_dtypes(include=['number']).columns.tolist()
                if num_cols:
                    sns.histplot(data=df, x=num_cols[0], kde=True, ax=ax, color="#4A90E2")
                    ax.set_title(f"Operational Metric Density Spread Profile: {num_cols[0]}", fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            msg_data["plot_type"] = "pyplot"
            msg_data["plot_fig"] = fig
        except Exception as chart_err:
bot_response += f"\n\n*(Visual Engine Alert: Could not auto-render plot layout: {chart_err})*"
msg_data["content"] = bot_response
# Show Complete Elements Simultaneously
st.session_state.messages.append(msg_data)
with st.chat_message("assistant"):
st.write(bot_response)
if "plot_type" in msg_data:
st.pyplot(msg_data["plot_fig"])
# --- 6. VOICE SYNTHESIS OUTPUT WITH PROMPT BRIDGES ---
with st.spinner("Speaking executive brief..."):
clean_text = bot_response.replace("**", "").replace("#", "").replace("`", "")
# If the text has a table format or is exceptionally long, give a summarized vocal briefing instead
if "|" in clean_text or len(clean_text) > 800:
clean_text = "I have successfully processed your dataset and generated an in-depth strategic analysis. I have broken down the high-level insights, provided tailored recommendations for your business and technical stakeholders, and plotted your performance metrics directly on the screen for you to review."
else:
pronunciation_fixes = {
"ORDERNUMBER": "order number", "SALES": "sales", "QUANTITYORDERED": "quantity ordered",
"PRICEEACH": "price each", "PRODUCTLINE": "product line", "CUSTOMERNAME": "customer name",
"ORDERDATE": "order date", "DEALSIZE": "deal size", "COUNTRY": "country", "CITY": "city",
"STATE": "state", "ADDRESSLINE2": "address line 2", "POSTALCODE": "postal code",
"TERRITORY": "territory", "MSRP": "manufacturer suggested retail price", "EMEA": "e m e a"
}
for upper_word, spoken_word in pronunciation_fixes.items():
clean_text = clean_text.replace(upper_word, spoken_word)
clean_text = clean_text.lower()
tts = gTTS(text=clean_text, lang='en', slow=False)
temp_audio = "cloud_response.mp3"
tts.save(temp_audio)
st.audio(temp_audio, format="audio/mp3", autoplay=True)
