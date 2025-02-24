import tiktoken

def trim_text_to_token_limit(text: str, token_limit: int = 6000, model: str = "llama-3.1-8b-instant") -> str:
    """
    Trims the input text to fit within the token limit for the specified model.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) <= token_limit:
        return text
    # Truncate tokens and decode back to text; you may also want to summarize instead of blunt truncation.
    trimmed_tokens = tokens[:token_limit]
    return encoding.decode(trimmed_tokens)