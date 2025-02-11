import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
from fpdf import FPDF  # for PDF generation
from tempfile import NamedTemporaryFile
from wordcloud import WordCloud  # for generating word clouds

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize summarizer pipeline (this may take a moment on the first run)
@st.cache_resource(show_spinner=False)
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")
summarizer = get_summarizer()

def clean_text(text):
    """Replace problematic Unicode characters with ASCII equivalents."""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    return text

def summarize_text(text, max_tokens=500):
    """Summarize text using the Hugging Face summarization pipeline."""
    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return text
    chunk = ""
    summary = ""
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens > max_tokens:
            try:
                summarized = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                if summarized and isinstance(summarized, list) and 'summary_text' in summarized[0]:
                    summary += summarized[0]['summary_text'] + " "
                else:
                    summary += chunk + " "
            except Exception as e:
                summary += f"[Summarization error: {e}] "
            chunk = sentence
            current_tokens = sentence_tokens
        else:
            chunk += " " + sentence
            current_tokens += sentence_tokens
    if chunk.strip():
        try:
            summarized = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            if summarized and isinstance(summarized, list) and 'summary_text' in summarized[0]:
                summary += summarized[0]['summary_text'] + " "
            else:
                summary += chunk + " "
        except Exception as e:
            summary += f"[Summarization error: {e}] "
    return clean_text(summary.strip())

def extract_insights(text_series, top_n=5):
    """Extract the top `top_n` most common words from a pandas Series of text."""
    combined_text = " ".join(text_series)
    tokens = word_tokenize(combined_text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    if not tokens:
        return []
    freq_dist = nltk.FreqDist(tokens)
    return freq_dist.most_common(top_n)

def classify_sentiment(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def generate_excel_report(report_data):
    """Generate an Excel report (as a BytesIO object) that includes NPS and sentiment analysis."""
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    workbook = writer.book
    
    # --- NPS Summary Sheet ---
    if 'nps_summary' in report_data:
        try:
            worksheet = writer.sheets['NPS Summary']
        except KeyError:
            worksheet = workbook.add_worksheet('NPS Summary')
        start_row = 0
        if 'group_info' in report_data:
            worksheet.write(start_row, 0, "Group Column:")
            worksheet.write(start_row, 1, report_data['group_info']["Group Column"])
            worksheet.write(start_row+1, 0, "Selected Group:")
            worksheet.write(start_row+1, 1, report_data['group_info']["Selected Group"])
            start_row += 3
        worksheet.write(start_row, 0, "NPS Summary")
        df_nps = pd.DataFrame(list(report_data['nps_summary'].items()), columns=['Metric', 'Value'])
        df_nps.to_excel(writer, sheet_name='NPS Summary', index=False, startrow=start_row+2)
        if 'overall_nps_chart' in report_data:
            worksheet.insert_image(f'E{start_row+3}', 'overall_nps_chart.png', {'image_data': report_data['overall_nps_chart']})
        if 'nps_chart' in report_data:
            worksheet.insert_image(f'E{start_row+20}', 'nps_chart.png', {'image_data': report_data['nps_chart']})
        if 'nps_group_chart' in report_data:
            worksheet.insert_image(f'E{start_row+37}', 'nps_group_chart.png', {'image_data': report_data['nps_group_chart']})
    
    # --- Sentiment Analysis Sheets ---
    if 'sentiment' in report_data:
        for col, data in report_data['sentiment'].items():
            sheet_name = f"Sent_{col[:20]}"
            overall_counts = data.get('overall_counts', {})
            df_overall = pd.DataFrame(list(overall_counts.items()), columns=['Sentiment', 'Count'])
            df_overall.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
            worksheet = writer.sheets[sheet_name]
            worksheet.write(0, 0, f"Sentiment Analysis for '{col}'")
            if 'overall_chart' in data:
                worksheet.insert_image('E3', f"{col}_sentiment_chart.png", {'image_data': data['overall_chart']})
            row = df_overall.shape[0] + 4
            for sentiment in ['Negative', 'Neutral', 'Positive']:
                if sentiment in data:
                    details = data[sentiment]
                    worksheet.write(row, 0, f"{sentiment} Responses")
                    worksheet.write(row+1, 0, "Top Words (word: count)")
                    top_words = details.get('Top Words', [])
                    top_words_str = ", ".join([f"{word}: {freq}" for word, freq in top_words])
                    worksheet.write(row+1, 1, top_words_str)
                    worksheet.write(row+2, 0, "Summary")
                    worksheet.write(row+2, 1, details.get('Summary', ""))
                    row += 3
                    if f"{sentiment}_wordcloud" in data:
                        worksheet.insert_image(f'E{row}', f"{sentiment}_wordcloud.png", {'image_data': data[f"{sentiment}_wordcloud"]})
                        row += 15
    writer.close()
    output.seek(0)
    return output

def generate_pdf_report(report_data):
    """Generate a PDF report (as a BytesIO object) that includes NPS and sentiment analysis details."""
    pdf = FPDF()
    
    # Use clean_text for all text written into the PDF
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, clean_text("Survey Analysis Report"), ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    if 'group_info' in report_data:
        pdf.cell(0, 10, clean_text(f"Group Column: {report_data['group_info']['Group Column']}"), ln=True)
        pdf.cell(0, 10, clean_text(f"Selected Group: {report_data['group_info']['Selected Group']}"), ln=True)
        pdf.ln(5)
    
    pdf.cell(0, 10, clean_text("NPS Summary"), ln=True)
    pdf.set_font("Arial", '', 10)
    if 'nps_summary' in report_data:
        for key, value in report_data['nps_summary'].items():
            pdf.cell(0, 10, clean_text(f"{key}: {value}"), ln=True)
    
    if 'overall_nps_chart' in report_data:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_text("Overall NPS Score Chart"), ln=True)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(report_data['overall_nps_chart'].getvalue())
            tmp_img.flush()
            pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
    
    if 'nps_chart' in report_data:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_text("NPS Category Breakdown"), ln=True)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(report_data['nps_chart'].getvalue())
            tmp_img.flush()
            pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
    
    if 'nps_group_chart' in report_data:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_text("NPS Score by Group"), ln=True)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(report_data['nps_group_chart'].getvalue())
            tmp_img.flush()
            pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
    
    if 'sentiment' in report_data:
        for col, data in report_data['sentiment'].items():
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, clean_text(f"Sentiment Analysis for '{col}'"), ln=True)
            pdf.set_font("Arial", '', 10)
            overall_counts = data.get('overall_counts', {})
            for sentiment, count in overall_counts.items():
                pdf.cell(0, 10, clean_text(f"{sentiment}: {count}"), ln=True)
            pdf.ln(5)
            if 'overall_chart' in data:
                with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    tmp_img.write(data['overall_chart'].getvalue())
                    tmp_img.flush()
                    pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
            pdf.ln(5)
            for sentiment in ['Negative', 'Neutral', 'Positive']:
                if sentiment in data:
                    details = data[sentiment]
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(0, 10, clean_text(f"{sentiment} Responses"), ln=True)
                    pdf.set_font("Arial", '', 10)
                    top_words = details.get('Top Words', [])
                    top_words_str = ", ".join([f"{word}: {freq}" for word, freq in top_words])
                    pdf.cell(0, 10, clean_text(f"Top Words: {top_words_str}"), ln=True)
                    summary = details.get('Summary', "")
                    pdf.multi_cell(0, 10, clean_text(f"Summary: {summary}"))
                    if f"{sentiment}_wordcloud" in data:
                        with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                            tmp_img.write(data[f"{sentiment}_wordcloud"].getvalue())
                            tmp_img.flush()
                            pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
                        pdf.ln(5)
    pdf_bytes = pdf.output(dest='S').encode('latin1', errors='replace')
    pdf_output = io.BytesIO(pdf_bytes)
    pdf_output.seek(0)
    return pdf_output

# -------------------------------
# Main App Code
# -------------------------------
st.set_page_config(page_title="Survey Analysis Dashboard", layout="wide")
st.title("Survey Analysis Dashboard")

# Sidebar: CSV File Upload
st.sidebar.header("1. Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your survey CSV file", type="csv")

# Use session state to store computed report data and generated reports
if 'report_data' not in st.session_state:
    st.session_state.report_data = {}
if 'excel_report' not in st.session_state:
    st.session_state.excel_report = None
if 'pdf_report' not in st.session_state:
    st.session_state.pdf_report = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Auto-detect column types:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()

    st.sidebar.header("2. Configure Analysis Settings")
    if numeric_cols:
        nps_col = st.sidebar.selectbox("Select NPS rating column (0-10 scale)", options=numeric_cols)
    else:
        st.sidebar.error("No numeric columns detected for NPS calculation.")
        nps_col = None

    sample_col = st.sidebar.selectbox("Select sample grouping column (optional)", options=["None"] + text_cols)
    open_ended_cols = st.sidebar.multiselect("Select open-ended response columns", options=text_cols)

    selected_sample = "All"
    if sample_col != "None":
        unique_samples = df[sample_col].dropna().unique().tolist()
        selected_sample = st.sidebar.selectbox("Filter by sample population", options=["All"] + unique_samples)
        if selected_sample != "All":
            df = df[df[sample_col] == selected_sample]
            st.write(f"**Filtered Data:** {sample_col} = {selected_sample}")
        st.session_state.report_data['group_info'] = {"Group Column": sample_col, "Selected Group": selected_sample}

    st.write("---")
    st.header("NPS Calculation and Closed-Ended Analysis")
    if nps_col is not None:
        rating_series = pd.to_numeric(df[nps_col], errors='coerce')
        promoters = rating_series[rating_series >= 9].count()
        passives = rating_series[(rating_series >= 7) & (rating_series <= 8)].count()
        detractors = rating_series[rating_series <= 6].count()
        total_responses = rating_series.count()
        nps_score = ((promoters - detractors) / total_responses) * 100 if total_responses > 0 else np.nan

        st.write(f"**Total Responses:** {total_responses}")
        st.write(f"**Promoters (9-10):** {promoters}")
        st.write(f"**Passives (7-8):** {passives}")
        st.write(f"**Detractors (0-6):** {detractors}")
        st.write(f"**Calculated NPS:** {nps_score:.2f}")

        # NPS Category Breakdown chart
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Promoters', 'Passives', 'Detractors']
        counts = [promoters, passives, detractors]
        colors_nps = ['green', 'blue', 'red']
        bars = ax.bar(categories, counts, color=colors_nps)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', 
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), 
                        textcoords="offset points", 
                        ha='center', 
                        va='bottom')
        # Give some headroom so top labels don't get cut off
        ax.set_ylim([0, max(counts) * 1.1])
        ax.set_ylabel('Number of Respondents', fontsize=12)
        ax.set_title('NPS Category Breakdown', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        nps_img = io.BytesIO()
        fig.savefig(nps_img, format='png')
        nps_img.seek(0)
        st.session_state.report_data['nps_chart'] = nps_img

        # Overall NPS score chart
        fig_overall, ax_overall = plt.subplots(figsize=(3, 2.5))
        bar_container = ax_overall.bar("Overall NPS", nps_score, color="purple")
        ax_overall.set_ylim(-100, 100)
        ax_overall.set_ylabel("NPS Score")
        ax_overall.set_title("Overall NPS Score")
        for bar in bar_container:
            height = bar.get_height()
            ax_overall.annotate(f'{height:.2f}', 
                                xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 3), 
                                textcoords="offset points", 
                                ha='center', 
                                va='bottom')
        plt.tight_layout()
        st.pyplot(fig_overall)
        overall_nps_img = io.BytesIO()
        fig_overall.savefig(overall_nps_img, format="png")
        overall_nps_img.seek(0)
        st.session_state.report_data['overall_nps_chart'] = overall_nps_img

        st.session_state.report_data['nps_summary'] = {
            "Total Responses": total_responses,
            "Promoters (9-10)": promoters,
            "Passives (7-8)": passives,
            "Detractors (0-6)": detractors,
            "NPS Score": nps_score
        }
    else:
        st.error("NPS column not selected or not available.")

    # NPS by Group
    if sample_col != "None" and selected_sample == "All":
        st.write("---")
        st.subheader("NPS by Group")
        group_nps = df.groupby(sample_col)[nps_col].apply(
            lambda x: ((x >= 9).sum() - (x <= 6).sum()) / x.count() * 100
        )

        fig_group, ax_group = plt.subplots(figsize=(8, 4))
        bars_group = ax_group.bar(group_nps.index.astype(str), group_nps.values, color='cyan')
        for bar in bars_group:
            height = bar.get_height()
            ax_group.annotate(
                f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 6),
                textcoords="offset points",
                ha='center',
                va='bottom'
            )
        ax_group.set_ylabel("NPS Score", fontsize=12)
        ax_group.set_title("NPS Score by Group", fontsize=14)
        ax_group.set_ylim(-100, 100)
        ax_group.tick_params(axis='x', labelrotation=45, labelsize=10)
        ax_group.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
        st.pyplot(fig_group, use_container_width=True)

        group_nps_img = io.BytesIO()
        fig_group.savefig(group_nps_img, format="png")
        group_nps_img.seek(0)
        st.session_state.report_data['nps_group_chart'] = group_nps_img

    st.write("---")
    st.header("Sentiment Analysis of Open-Ended Responses")
    sia = SentimentIntensityAnalyzer()

    if open_ended_cols:
        st.session_state.report_data['sentiment'] = {}
        for col in open_ended_cols:
            st.subheader(f"Analysis for '{col}'")
            responses = df[col].dropna().astype(str)
            if responses.empty:
                st.write("No responses to analyze.")
                continue

            with st.spinner("Analyzing sentiment..."):
                sentiments = responses.apply(lambda text: sia.polarity_scores(text))
                sentiments_df = pd.DataFrame(list(sentiments))
                sentiments_df['Sentiment'] = sentiments_df['compound'].apply(classify_sentiment)
                sentiments_df['Response'] = responses.reset_index(drop=True)

            st.write("**Sentiment Counts:**")
            sentiment_counts = sentiments_df['Sentiment'].value_counts()
            colors_map = {"Positive": "green", "Neutral": "grey", "Negative": "red"}
            bar_colors = [colors_map.get(s, "blue") for s in sentiment_counts.index]

            # UPDATED to prevent cutoff
            fig_sent, ax_sent = plt.subplots(figsize=(6, 4))
            bars = ax_sent.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors)
            for bar in bars:
                height = bar.get_height()
                ax_sent.annotate(
                    f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom'
                )
            ax_sent.set_ylim([0, max(sentiment_counts.values) * 1.1])  # Add 10% headroom
            ax_sent.set_ylabel("Count", fontsize=12)
            ax_sent.set_title(f"Sentiment Distribution for '{col}'", fontsize=14)
            plt.tight_layout()

            st.pyplot(fig_sent, use_container_width=True)
            st.session_state.report_data['sentiment'][col] = {}
            # Save image to BytesIO
            sent_img = io.BytesIO()
            fig_sent.savefig(sent_img, format='png')
            sent_img.seek(0)
            st.session_state.report_data['sentiment'][col]['overall_chart'] = sent_img

            st.markdown("### Insights by Sentiment Category")
            sentiment_categories = sorted(sentiments_df['Sentiment'].unique())
            progress_bar = st.progress(0)
            for i, sentiment in enumerate(sentiment_categories):
                st.markdown(f"**{sentiment} Responses:**")
                sentiment_texts = sentiments_df[sentiments_df['Sentiment'] == sentiment]['Response']
                insights = extract_insights(sentiment_texts)
                if insights:
                    st.write("Top words and their frequencies:")
                    for word, freq in insights:
                        st.write(f"- **{word}**: {freq}")
                else:
                    st.write("Not enough data to extract word insights.")
                full_text = " ".join(sentiment_texts.tolist())
                if full_text.strip():
                    with st.spinner(f"Summarizing {sentiment} responses..."):
                        summary_text = summarize_text(full_text)
                    st.markdown(f"**Summary of {sentiment} responses:** {summary_text}")
                    wc = WordCloud(width=300, height=200, background_color='white').generate(full_text)
                    fig_wc, ax_wc = plt.subplots(figsize=(3, 2))
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)
                    wc_img = io.BytesIO()
                    fig_wc.savefig(wc_img, format='png')
                    wc_img.seek(0)
                    st.session_state.report_data['sentiment'][col][f"{sentiment}_wordcloud"] = wc_img
                else:
                    summary_text = ""
                    st.write("No text available for summarization.")
                st.session_state.report_data['sentiment'][col][sentiment] = {
                    "Top Words": insights,
                    "Summary": summary_text
                }
                progress_bar.progress((i + 1) / len(sentiment_categories))
    else:
        st.info("No open-ended columns selected for sentiment analysis.")

    # Regenerate report files with updated data
    st.session_state.excel_report = generate_excel_report(st.session_state.report_data)
    st.session_state.pdf_report = generate_pdf_report(st.session_state.report_data)
    
    st.write("---")
    st.header("Download Customized Report")
    st.download_button(
        label="Download Report as XLSX",
        data=st.session_state.excel_report,
        file_name="Survey_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.download_button(
        label="Download Report as PDF",
        data=st.session_state.pdf_report,
        file_name="Survey_Report.pdf",
        mime="application/pdf"
    )
else:
    st.info("Awaiting CSV file upload. Please use the sidebar to upload your survey data.")

st.write("### Developed with Educational Leadership in Mind")
st.caption("Designed to reflect the analytical rigor and intuitive data management approach familiar in academic environments.")
