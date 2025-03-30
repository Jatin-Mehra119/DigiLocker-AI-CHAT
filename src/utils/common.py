import tiktoken

def trim_text_to_token_limit(text, token_limit=6000):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) > token_limit:
            tokens = tokens[:token_limit]
            return encoding.decode(tokens)
        return text
    except Exception as e:
        print(f"Warning: Token counting failed, using character-based fallback: {e}")
        char_limit = token_limit * 4
        return text[:char_limit]