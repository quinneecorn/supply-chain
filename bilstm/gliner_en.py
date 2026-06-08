import os
import json
from supabase import create_client, Client

# bilstm/gliner.py

_model = None

def get_model():
    """Lazy loads the GLiNER model only when needed."""
    global _model
    if _model is None:
        from gliner import GLiNER
        print(f"[i] Loading GLiNER model (urchade/gliner_medium-v2.1)...")
        _model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    return _model

def process_with_gliner(text):
    """Identifies entities locally using GLiNER and returns structured data."""
    model = get_model()
    
    labels = [
        "Company Name", "Public Company", "Private Company", 
        "Government Body", "Brand Name", "Product Model", "Project Name"
    ]
    
    FROM_LABELS = ["Company Name", "Public Company", "Private Company", "Government Body"]
    TO_LABELS = ["Brand Name", "Product Model", "Project Name"]

    ROLE_MAPPING = {
        "Acquiring Corporation": "[__NE_TO__]",
        "Recipient": "[__NE_TO__]",
        "Department": "[__NE_FROM__]",
        "The Committee": "[__NE_FROM__]"
    }

    try:
        entities = model.predict_entities(text, labels, threshold=0.3)
    except Exception:
        return None
    
    proper_entities = []
    entities = sorted(entities, key=lambda x: x['score'], reverse=True)
    
    seen_indices = set()
    for ent in entities:
        name = ent['text'].strip()
        start, end = ent['start'], ent['end']
        
        if any(i in seen_indices for i in range(start, end)):
            continue
            
        if name and name[0].isupper():
            proper_entities.append(ent)
            for i in range(start, end):
                seen_indices.add(i)

    sorted_entities = sorted(proper_entities, key=lambda x: x['start'], reverse=True)
    
    masked_sentence = text
    from_entities = []
    to_entities = []
    primary_from = None
    primary_to = None

    for ent in sorted_entities:
        name = ent['text']
        label = ent['label']
        start, end = ent['start'], ent['end']
        
        token = ""
        if name in ROLE_MAPPING:
            token = ROLE_MAPPING[name]
        elif label in FROM_LABELS:
            token = "[__NE_FROM__]"
        elif label in TO_LABELS:
            token = "[__NE_TO__]"
        
        if token:
            if token == "[__NE_FROM__]":
                from_entities.append(name)
                primary_from = name
            else:
                to_entities.append(name)
                primary_to = name
            masked_sentence = masked_sentence[:start] + token + masked_sentence[end:]

    return {
        "masked_sentence": masked_sentence,
        "entity_from": primary_from,
        "entity_to": primary_to
    }