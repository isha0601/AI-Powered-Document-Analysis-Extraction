from src.services.analyzers import analyze_sentiment, extract_entities, summarize_text


def analyze_text(text: str) -> dict:
    summary = summarize_text(text)
    entities = extract_entities(text)
    sentiment = analyze_sentiment(text)
    return {
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment,
    }
