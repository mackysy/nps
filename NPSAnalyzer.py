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

# For topic modeling
import gensim
from gensim import corpora
from bertopic import BERTopic

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------
# Summarizer
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = get_summarizer()

# -------------------------------
# Text Cleaning
# -------------------------------
def clean_text(text):
    """
    Replace problematic Unicode characters with ASCII equivalents,
    then fallback to 'replace' for any others that can't be encoded to latin-1.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Replace common fancy quotes, dashes, etc.
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2014", "--")  # Replace em dash with two hyphens

    # Finally, encode and decode in latin-1 with 'replace' to ensure no errors
    text = text.encode("latin-1", "replace").decode("latin-1")
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

# -------------------------------
# Topic Modeling Utilities
# -------------------------------
def run_lda_topic_modeling(docs, num_topics=3, passes=5):
    """
    Perform LDA topic modeling using Gensim on a list of documents.
    Returns a list of topics: (topic_id, [(word, weight), ...]).
    """
    # Tokenize
    tokens_list = [nltk.word_tokenize(doc.lower()) for doc in docs]
    # Remove stopwords, keep only alphabetic
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens_list = [
        [t for t in tokens if t.isalpha() and t not in stop_words]
        for tokens in tokens_list
    ]

    dictionary = corpora.Dictionary(tokens_list)
    if len(dictionary) == 0:
        return []

    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]
    if not corpus:
        return []

    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes
    )
    topics = lda_model.show_topics(num_topics=num_topics, num_words=5, formatted=False)
    return topics

def run_bertopic_modeling(docs):
    """
    Perform topic modeling using BERTopic on a list of documents.
    Returns the topic info DataFrame from BERTopic (topic #, size, name, etc.).
    """
    if not docs:
        return pd.DataFrame()

    topic_model = BERTopic(verbose=False)
    topics, _ = topic_model.fit_transform(docs)
    topic_info = topic_model.get_topic_info()
    return topic_info

# -------------------------------
# Report Generation
# -------------------------------
def generate_excel_report(report_data):
    """
    Generate an Excel report (as a BytesIO object) that includes:
      - NPS Summary
      - Sentiment Analysis (but no BERTopic)
      - Satisfaction/Likert charts
    """
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
    
    # --- Sentiment Analysis Sheets (NO BERTopic) ---
    if 'sentiment' in report_data:
        for col, data in report_data['sentiment'].items():
            sheet_name = f"Sent_{col[:20]}"
            overall_counts = data.get('overall_counts', {})
            df_overall = pd.DataFrame(list(overall_counts.items()), columns=['Sentiment', 'Count'])
            df_overall.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
            worksheet = writer.sheets[sheet_name]
            worksheet.write(0, 0, f"Sentiment Analysis for '{col}'")
            if 'overall_chart' in data:
                # Insert image
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
                    
                    # Summaries
                    worksheet.write(row+2, 0, "Summary")
                    worksheet.write(row+2, 1, details.get('Summary', ""))

                    # We skip BERTopic info here entirely
                    # We skip LDA topics? We included them before, let's keep them if you want:
                    lda_topics = details.get('lda_topics', [])
                    if lda_topics:
                        worksheet.write(row+4, 0, "LDA Topics")
                        row_lda = row+5
                        for (topic_id, words_list) in lda_topics:
                            words_str = ", ".join([w for w, _ in words_list])
                            worksheet.write(row_lda, 0, f"Topic {topic_id}")
                            worksheet.write(row_lda, 1, words_str)
                            row_lda += 1
                        row = row_lda + 1
                    else:
                        row += 5

                    row += 3

    # --- Satisfaction / Likert Sheet ---
    if 'satisfaction' in report_data:
        try:
            worksheet2 = writer.sheets['Satisfaction']
        except KeyError:
            worksheet2 = workbook.add_worksheet('Satisfaction')
        
        row_s = 0
        worksheet2.write(row_s, 0, "Satisfaction / Likert Analysis")
        row_s += 2

        # Each column gets a small block
        for col, col_data in report_data['satisfaction'].items():
            worksheet2.write(row_s, 0, f"Column: {col}")
            row_s += 1

            # If there's a stored bar chart or histogram, insert it
            if 'chart' in col_data:
                worksheet2.insert_image(row_s, 2, f"{col}_satisfaction_chart.png", {'image_data': col_data['chart']})
                row_s += 15  # move down for next item

    writer.close()
    output.seek(0)
    return output

def generate_pdf_report(report_data):
    """
    Generate a PDF report (as a BytesIO object) that includes:
      - NPS Summary
      - Sentiment Analysis (NO BERTopic)
      - Satisfaction charts
    """
    pdf = FPDF()
    
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
    
    # Overall NPS chart
    if 'overall_nps_chart' in report_data:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_text("Overall NPS Score Chart"), ln=True)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(report_data['overall_nps_chart'].getvalue())
            tmp_img.flush()
            pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
    
    # NPS Category Breakdown
    if 'nps_chart' in report_data:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_text("NPS Category Breakdown"), ln=True)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(report_data['nps_chart'].getvalue())
            tmp_img.flush()
            pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
    
    # NPS by Group
    if 'nps_group_chart' in report_data:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_text("NPS Score by Group"), ln=True)
        with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            tmp_img.write(report_data['nps_group_chart'].getvalue())
            tmp_img.flush()
            pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
    
    # Sentiment Analysis & Topics (NO BERTopic in PDF)
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
                    
                    # Top Words
                    top_words = details.get('Top Words', [])
                    top_words_str = ", ".join([f"{word}: {freq}" for word, freq in top_words])
                    pdf.cell(0, 10, clean_text(f"Top Words: {top_words_str}"), ln=True)
                    
                    # Summary
                    summary = details.get('Summary', "")
                    pdf.multi_cell(0, 10, clean_text(f"Summary: {summary}"))
                    
                    # LDA Topics
                    lda_topics = details.get('lda_topics', [])
                    if lda_topics:
                        pdf.set_font("Arial", 'B', 10)
                        pdf.cell(0, 10, clean_text("LDA Topics:"), ln=True)
                        pdf.set_font("Arial", '', 10)
                        for (topic_id, words_list) in lda_topics:
                            words_only = [w for w, _ in words_list]
                            pdf.cell(0, 10, clean_text(f"Topic {topic_id}: {', '.join(words_only)}"), ln=True)
                        pdf.ln(5)

                    # We skip any BERTopic info
    
    # Satisfaction / Likert in PDF
    if 'satisfaction' in report_data:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, clean_text("Satisfaction / Likert Analysis"), ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", '', 10)
        for col, col_data in report_data['satisfaction'].items():
            pdf.cell(0, 10, clean_text(f"Column: {col}"), ln=True)
            if 'chart' in col_data:
                with NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                    tmp_img.write(col_data['chart'].getvalue())
                    tmp_img.flush()
                    pdf.image(tmp_img.name, x=10, y=None, w=pdf.w - 40)
            pdf.ln(10)

    # Output PDF as BytesIO
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

# We'll store satisfaction data in st.session_state['satisfaction'] if not present
if 'satisfaction' not in st.session_state.report_data:
    st.session_state.report_data['satisfaction'] = {}

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

    group_col_1 = st.sidebar.selectbox("Select sample grouping column (optional)", ["None"] + text_cols)
    group_col_2 = st.sidebar.selectbox("Select additional grouping column 1 (optional)", ["None"] + text_cols)
    group_col_3 = st.sidebar.selectbox("Select additional grouping column 2 (optional)", ["None"] + text_cols)

    # Filter by group_col_1 if user chooses
    selected_sample = "All"
    if group_col_1 != "None":
        unique_samples = df[group_col_1].dropna().unique().tolist()
        selected_sample = st.sidebar.selectbox("Filter by sample population", ["All"] + unique_samples)
        if selected_sample != "All":
            df = df[df[group_col_1] == selected_sample]
            st.write(f"**Filtered Data:** {group_col_1} = {selected_sample}")
        st.session_state.report_data['group_info'] = {"Group Column": group_col_1, "Selected Group": selected_sample}

    open_ended_cols = st.sidebar.multiselect(
        "Select open-ended response columns (for sentiment & topic modeling)",
        options=text_cols
    )

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
        if sum(counts) == 0:
            ax.text(0.5, 0.5, "No responses available", ha='center', va='center')
        else:
            bars = ax.bar(categories, counts, color=colors_nps)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center',
                            va='bottom')
            ax.set_ylim([0, max(counts) * 1.1] if max(counts) > 0 else [0, 1])
        ax.set_ylabel('Number of Respondents', fontsize=12)
        ax.set_title('NPS Category Breakdown', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

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
        st.pyplot(fig_overall, use_container_width=False)
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

        # NPS by multiple grouping columns
        group_cols = []
        if group_col_1 != "None":
            group_cols.append(group_col_1)
        if group_col_2 != "None":
            group_cols.append(group_col_2)
        if group_col_3 != "None":
            group_cols.append(group_col_3)

        if group_cols:
            st.write("---")
            st.subheader("NPS by Grouping Columns")
            group_nps_df = (
                df.groupby(group_cols)[nps_col]
                  .apply(lambda x: ((x >= 9).sum() - (x <= 6).sum()) / x.count() * 100 if x.count() else np.nan)
                  .reset_index(name='NPS_Score')
            )
            st.dataframe(group_nps_df)

            if len(group_cols) == 1:
                fig_group, ax_group = plt.subplots(figsize=(8, 4))
                x_vals = group_nps_df[group_cols[0]].astype(str)
                y_vals = group_nps_df['NPS_Score']
                bars_group = ax_group.bar(x_vals, y_vals, color='cyan')
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
                ax_group.set_title(f"NPS by {group_cols[0]}", fontsize=14)
                ax_group.set_ylim(-100, 100)
                ax_group.set_xticklabels(ax_group.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_group, use_container_width=False)

                g_img = io.BytesIO()
                fig_group.savefig(g_img, format='png')
                g_img.seek(0)
                st.session_state.report_data['nps_group_chart'] = g_img

            elif len(group_cols) == 2:
                pivoted = group_nps_df.pivot(index=group_cols[0], columns=group_cols[1], values='NPS_Score')
                st.write("NPS Pivot Table:")
                st.dataframe(pivoted)

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                pivoted.plot(kind='bar', ax=ax2)
                ax2.set_ylabel("NPS Score")
                ax2.set_title(f"NPS by {group_cols[0]} & {group_cols[1]}")
                ax2.set_ylim(-100, 100)
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=False)

                g2_img = io.BytesIO()
                fig2.savefig(g2_img, format='png')
                g2_img.seek(0)
                st.session_state.report_data['nps_group_chart'] = g2_img

            else:
                st.write("#### 3-Level Grouping Charts")
                for i, row_data in group_nps_df.iterrows():
                    combo_values = [str(row_data[col]) for col in group_cols]
                    combo_label = " / ".join(combo_values)
                    nps_val = row_data['NPS_Score']

                    fig3, ax3 = plt.subplots(figsize=(4, 2.5))
                    ax3.bar("NPS", nps_val, color="cyan")
                    ax3.set_ylim(-100, 100)
                    ax3.set_ylabel("NPS Score")
                    ax3.set_title(combo_label)
                    ax3.text(0, nps_val, f"{nps_val:.2f}", ha="center", va="bottom")
                    plt.tight_layout()
                    st.pyplot(fig3, use_container_width=False)
        else:
            st.write("No additional grouping selected.")

    else:
        st.error("NPS column not selected or not available.")

    # -------------------------------
    # Sentiment Analysis & Topic Modeling
    # -------------------------------
    st.write("---")
    st.header("Sentiment Analysis & Topic Modeling of Open-Ended Responses")
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
            if len(sentiment_counts) > 0:
                ax_sent.set_ylim([0, max(sentiment_counts.values) * 1.1])
            else:
                ax_sent.set_ylim([0, 1])
            ax_sent.set_ylabel("Count", fontsize=12)
            ax_sent.set_title(f"Sentiment Distribution for '{col}'", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig_sent, use_container_width=False)

            st.session_state.report_data['sentiment'][col] = {}
            # Save chart to BytesIO
            sent_img = io.BytesIO()
            fig_sent.savefig(sent_img, format='png')
            sent_img.seek(0)
            st.session_state.report_data['sentiment'][col]['overall_chart'] = sent_img

            # Store overall counts for Excel/PDF
            st.session_state.report_data['sentiment'][col]['overall_counts'] = sentiment_counts.to_dict()

            st.markdown("### Insights & Topics by Sentiment Category")
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
                
                summary_text = ""
                if full_text.strip():
                    with st.spinner(f"Summarizing {sentiment} responses..."):
                        summary_text = summarize_text(full_text)
                    st.markdown(f"**Summary of {sentiment} responses:** {summary_text}")
                else:
                    st.write("No text available for summarization.")

                # LDA
                with st.spinner(f"Running LDA topic modeling on {sentiment} responses..."):
                    lda_topics = run_lda_topic_modeling(sentiment_texts.tolist(), num_topics=3)
                if lda_topics:
                    st.write("**LDA Topics** (top 5 words per topic):")
                    for (topic_id, words_list) in lda_topics:
                        words_only = [w for w, _ in words_list]
                        st.write(f"- **Topic {topic_id}**: {', '.join(words_only)}")
                else:
                    st.write("No LDA topics found (possibly not enough data).")

                # BERTopic (excluded from final downloads, but still displayed)
                with st.spinner(f"Running BERTopic on {sentiment} responses..."):
                    bertopic_info = run_bertopic_modeling(sentiment_texts.tolist())
                if not bertopic_info.empty:
                    st.write("**BERTopic Topic Info** (excluded from PDF/Excel)")
                    st.dataframe(bertopic_info)

                # Save results to session state (except we won't insert BERTopic in final output)
                st.session_state.report_data['sentiment'][col][sentiment] = {
                    "Top Words": insights,
                    "Summary": summary_text,
                    "lda_topics": lda_topics,
                    # 'bertopic_info': bertopic_info  # We'll keep it if you want, but won't show in final
                }

                progress_bar.progress((i + 1) / len(sentiment_categories))
    else:
        st.info("No open-ended columns selected for sentiment analysis.")

    # ---------------------------------------
    # Satisfaction / Likert Analysis
    # ---------------------------------------
    st.write("---")
    st.header("Satisfaction / Likert Analysis")

    satisfaction_cols = st.sidebar.multiselect(
        "Select additional satisfaction/likert columns (optional)",
        options=text_cols + numeric_cols
    )

    # We'll store the results in st.session_state.report_data['satisfaction']
    if satisfaction_cols:
        for col in satisfaction_cols:
            st.subheader(f"Satisfaction distribution for: {col}")
            data_col = df[col].dropna()

            # Prepare a figure
            fig_sat, ax_sat = plt.subplots(figsize=(6, 3))

            if col in numeric_cols:
                # numeric approach
                # assume 1-5 scale (adjust bins if needed)
                ax_sat.hist(data_col, bins=range(1, 7), color='lightblue', edgecolor='black')
                ax_sat.set_title(f"Distribution of {col}")
                ax_sat.set_xlabel("Rating")
                ax_sat.set_ylabel("Frequency")
            else:
                # text/categorical approach
                value_counts = data_col.value_counts(dropna=False)
                top_10 = value_counts.head(10)
                # Truncate labels
                truncated_labels = [
                    (label[:30] + "...") if len(label) > 30 else label
                    for label in top_10.index
                ]
                top_10.index = truncated_labels

                ax_sat.bar(top_10.index, top_10.values, color="lightblue", edgecolor="black")
                ax_sat.set_title(f"Top 10 categories for {col} (Truncated)")
                plt.xticks(rotation=45, ha='right', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig_sat, use_container_width=False)

            # Save chart to BytesIO for the final downloads
            chart_io = io.BytesIO()
            fig_sat.savefig(chart_io, format='png')
            chart_io.seek(0)

            # Store in session state
            st.session_state.report_data['satisfaction'][col] = {
                "chart": chart_io
            }

            # If user wants group stats by group_col_1
            if group_col_1 != "None":
                st.write(f"**Group-level summary by {group_col_1}**")
                if col in numeric_cols:
                    group_vals = df.groupby(group_col_1)[col].mean(numeric_only=False)
                    st.write(group_vals)
                else:
                    grouped = df.groupby(group_col_1)[col].value_counts(dropna=False)
                    st.write(grouped)

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
