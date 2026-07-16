import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.language_models.openai_language_model import OpenAILanguageModel
from src.search.data_processor import clean_html, extract_relevant_context

logger = logging.getLogger(__name__)


def _summarize_single_result(result_item, query, template_path, model_name):
    """Helper function to summarize a single result in a thread."""
    try:
        html_content = result_item.get("metadata", {}).get("content", "")
        cleaned_html = clean_html(html_content)
        truncated_html = extract_relevant_context(cleaned_html, query, window_size=1500)
        
        summarization_model = OpenAILanguageModel(template_path, model_name=model_name)
        summary_prompt = summarization_model.generate_prompt(
            context=truncated_html, question=query)
            
        response = summarization_model.invoke(summary_prompt)
        
        result_item['truncated_html'] = truncated_html
        result_item['llm_summary'] = response.content
        return result_item
    except Exception as e:
        logger.error(f"Error summarizing result: {e}")
        result_item['truncated_html'] = ""
        result_item['llm_summary'] = ""
        return result_item


def summarize_search_results_with_llm(config, query, search_results):
    search_results = search_results[:7]
    summarized_context_string = ""
    filtered_results = []

    print("\n")
    logger.info("Picking relevant content from search results (in parallel)..")
    
    # Use ThreadPoolExecutor to summarize all chunks concurrently
    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = [
            executor.submit(_summarize_single_result, item, query, config.SUMMARY_TEMPLATE_PATH, config.FAST_MODEL)
            for item in search_results
        ]
        
        # We need to preserve the original ordering based on the search_results list
        # so we don't just use as_completed blindly.
        # Instead, we map futures to their index to sort them back later, or just wait.
        # Let's just wait for them in order.
        
        for future in futures:
            result_item = future.result()
            filtered_results.append(result_item)
            summarized_context_string += result_item.get('llm_summary', '')

    logger.debug(f"Length of processed search context: "
                 f"{len(summarized_context_string)}")

    return filtered_results, summarized_context_string
