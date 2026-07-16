import re

import nltk

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

nltk.download('punkt')


def clean_html(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script/style first
    for script in soup(["script", "style"]):
        script.extract()

    # ── KEY FIX: strip navigation boilerplate before extracting text ──────────
    # Without this, get_text() mixes page body with nav menus, headers, and
    # footers. The keyword window then lands on nav text (e.g. "Skip To Main
    # Content Departments Info For Giving Contact Search Aerospace...") instead
    # of the actual answer text (e.g. "Dean Robert H. Bishop Vice Chancellor...").
    for nav_tag in soup(["nav", "header", "footer", "aside"]):
        nav_tag.extract()

    text = soup.get_text()

    # Replace multiple whitespace characters with single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n', ' ', text)  # Replace newline characters with space
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters

    return text


def extract_relevant_context(html_content, query, window_size=1500):
    """
    Extracts a window of words from the text that contains the most keywords from the query.
    This avoids sending 30,000 words to the LLM (which causes 10s-20s rate limit and generation latency).
    """
    # Use standard string split instead of nltk.word_tokenize to avoid GIL contention
    tokens = html_content.split()

    if len(tokens) <= window_size:
        return html_content

    # Clean query into keywords
    stop_words = {"what", "is", "the", "who", "where", "how", "for", "in", "and", "a", "to", "of", "are", "do", "does", "on"}
    query_words = set(query.lower().replace("?", "").replace("!", "").replace(".", "").split())
    keywords = query_words - stop_words

    if not keywords:
        return ' '.join(tokens[:window_size])

    best_score = -1
    best_start = 0

    # Slide a window across the text, advancing by half the window size
    for i in range(0, len(tokens), window_size // 2):
        window = tokens[i:i + window_size]
        window_lower = [w.lower() for w in window]
        
        # Calculate score (prioritize unique keywords first, then total frequency)
        unique_hits = 0
        total_hits = 0
        for k in keywords:
            hits = sum(1 for w in window_lower if k in w)
            if hits > 0:
                unique_hits += 1
                total_hits += hits
        
        score = (unique_hits * 1000) + total_hits
                
        if score > best_score:
            best_score = score
            best_start = i

    # Return the start of the page + the best keyword window (if the best window isn't already at the start)
    start_window = tokens[:window_size // 2]
    
    if best_start <= window_size // 2:
        best_window = tokens[:window_size]
        return ' '.join(best_window)
    else:
        best_window = tokens[best_start:best_start + window_size]
        return ' '.join(start_window) + " ... " + ' '.join(best_window)
