import spacy
from typing import List, Tuple

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> List[str]:
    doc = nlp(text)
    preprocessed_tokens = [token.lemma_.lower() for token in doc if not token.is_stop]
    return preprocessed_tokens

def pos_tagging(text: str) -> List[Tuple[str, str]]:
    doc = nlp(text)
    pos_tags = [(token.text, token.pos_) for token in doc]
    return pos_tags

def named_entity_recognition(text: str) -> List[Tuple[str, str]]:
    doc = nlp(text)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]
    return named_entities

def dependency_parsing(text: str) -> List[Tuple[str, str, str]]:
    doc = nlp(text)
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
    return dependencies

def noun_phrase_extraction(text: str) -> List[str]:
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return noun_phrases

if __name__ == "__main__":
    text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Steve Jobs was one of the co-founders of Apple."

    preprocessed_tokens = preprocess_text(text)
    print("Preprocessed tokens:")
    print(preprocessed_tokens)

    pos_tags = pos_tagging(text)
    print("POS tags:")
    for token, pos_tag in pos_tags:
        print(f"{token} ({pos_tag})")

    named_entities = named_entity_recognition(text)
    print("Named entities found:")
    for entity, label in named_entities:
        print(f"{entity} ({label})")

    dependencies = dependency_parsing(text)
    print("Dependency parsing:")
    for token, dep, head in dependencies:
        print(f"{token} ({dep} - {head})")

    noun_phrases = noun_phrase_extraction(text)
    print("Noun phrases:")
    for np in noun_phrases:
        print(np)
