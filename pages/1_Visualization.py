import streamlit as st
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import time
import folium
import branca.colormap as cm
from streamlit_folium import folium_static
import plotly.express as px

st.title("Data Exploration")

st.subheader('Sentiment Percentage')

df_gopay = pd.read_csv('dataset/gopay_app_after_full_cleaning.csv')

@st.cache_data
def create_pie_chart():
    fig = px.pie(df_gopay, names='label')
    return fig

# Call the function and display the chart
fig = create_pie_chart()
st.plotly_chart(fig)

# Set the order of months
month_order = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November']
df_gopay['month'] = pd.Categorical(df_gopay['month'], categories=month_order, ordered=True)

# Group by month and label to get counts
review_counts_by_month_label = df_gopay.groupby(['month', 'label'])['content'].count().unstack().reset_index()

# Create a Streamlit app
st.subheader('Number of Sentiment Reviews')

# Create tabs
tab_positive, tab_negative = st.tabs(["Positive", "Negative"])

# Define the content for Tab Positive
with tab_positive:
    filtered_data = review_counts_by_month_label[['month', 'positive']]
    fig_positive = px.line(filtered_data, x='month', y='positive', title='Positive Reviews', markers=True)
    st.plotly_chart(fig_positive)

# Define the content for Tab Negative
with tab_negative:
    filtered_data = review_counts_by_month_label[['month', 'negative']]
    fig_negative = px.line(filtered_data, x='month', y='negative', title='Negative Reviews', markers=True)
    fig_negative.update_traces(line_shape='linear', line=dict(color='red'))
    st.plotly_chart(fig_negative)

from nltk.probability import FreqDist
positive_words = ' '.join(df_gopay[df_gopay['label'] == 'positive']['stemmed_text'])
positive_freq_dist = FreqDist(positive_words.split())

@st.cache_data
def compute_top_words_frequency(num_top_words, freq_dist):
    top_words = freq_dist.most_common(num_top_words)
    df_top_words = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    df_top_words = df_top_words.sort_values(by='Frequency', ascending=False)
    return df_top_words

st.subheader('Positive Words Frequency')

# Generate a unique key for the positive words slider
slider_key_pos = 'slider_pos_' + str(hash(df_gopay.to_string()))
num_top_words_pos = st.slider('Select the number of top positive words to display', min_value=1, max_value=50, value=5, key=slider_key_pos)
df_top_words_pos = compute_top_words_frequency(num_top_words_pos, positive_freq_dist)

fig_pos = px.bar(df_top_words_pos, x='Frequency', y='Word', orientation='h', color_discrete_sequence=['green'],
                 labels={'Frequency': 'Frequency', 'Word': 'Word'},
                 title=f'Top {num_top_words_pos} Positive Words Frequency Distribution')
fig_pos.update_layout(yaxis=dict(autorange="reversed"))

st.plotly_chart(fig_pos)


# Negative Words

negative_words = ' '.join(df_gopay[df_gopay['label'] == 'negative']['stemmed_text'])
negative_freq_dist = FreqDist(negative_words.split())

st.subheader('Negative Words Frequency')

# Generate a unique key for the negative words slider
slider_key_neg = 'slider_neg_' + str(hash(df_gopay.to_string()))
num_top_words_neg = st.slider('Select the number of top negative words to display', min_value=1, max_value=50, value=5, key=slider_key_neg)
df_top_words_neg = compute_top_words_frequency(num_top_words_neg, negative_freq_dist)

fig_neg = px.bar(df_top_words_neg, x='Frequency', y='Word', orientation='h', color_discrete_sequence=['darkred'],
                 labels={'Frequency': 'Frequency', 'Word': 'Word'},
                 title=f'Top {num_top_words_neg} Negative Words Frequency Distribution')
fig_neg.update_layout(yaxis=dict(autorange="reversed"))

st.plotly_chart(fig_neg)