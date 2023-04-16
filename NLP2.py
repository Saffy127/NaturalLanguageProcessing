import nltk

def named_entity_recognition(text):
    # Tokenize the text into sentences and words
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Perform part-of-speech tagging
    pos_tagged_sentences = [nltk.pos_tag(tokens) for tokens in tokenized_sentences]

    # Perform named entity recognition using NLTK's ne_chunk method
    ne_chunks = [nltk.ne_chunk(pos_tags) for pos_tags in pos_tagged_sentences]

    # Extract named entities
    named_entities = []
    for tree in ne_chunks:
        for subtree in tree.subtrees():
            if subtree.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
                entity = " ".join([leaf[0] for leaf in subtree.leaves()])
                named_entities.append((entity, subtree.label()))

    return named_entities

if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

    text = "Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Steve Jobs was one of the co-founders of Apple."

    named_entities = named_entity_recognition(text)
    print("Named entities found:")
    for entity, label in named_entities:
        print(f"{entity} ({label})")
