import spacy
from supabase import Client

print("Loading shared spaCy NLP model...")
nlp = spacy.load("en_core_web_sm")

# Unified keyword list (merged News and SEC keywords)
SUPPLY_KEYWORDS = [
    "suppli", "manufactur", "vendor", "contract", "procure",
    "outsourc", "distribut", "partner", "acqui", "subsidiar",
    "joint venture", "licens", "assembly", "component", "source",
    "tier-1", "tier-2", "oem", "odm", "wafer", "foundry",
    "accounted for", "accounting for", "percent of our net", 
    "percent of consolidated", "major customer", "principal customer",
    "concentration of credit", "customer concentration",
    "sole source", "single-source", "relies heavily on", "dependence on",
    "supply agreement", "purchase agreement", "manufacturing agreement",
    "master agreement", "offtake", "sub-contract"
]

class BaseCrawler:
    def __init__(self, supabase: Client):
        self.supabase = supabase

    def crawl(self, company_name: str, **kwargs):
        """Must be implemented by child classes (SECCrawler, NewsCrawler, CHSCrawler)"""
        raise NotImplementedError

    def extract_sentences(self, text: str) -> list[str]:
        """Universal text processing with memory protection for SEC/CHS documents."""
        if not text: return []
        
        # Split text into chunks to avoid spaCy 1,000,000 character limit
        chunk_size = 500000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        matched_sentences = []
        for chunk in chunks:
            if not chunk.strip(): continue
            doc = nlp(chunk)
            
            for sent in doc.sents:
                sentence_clean = " ".join(sent.text.strip().split())
                if 40 <= len(sentence_clean) <= 1000:
                    sentence_lower = sentence_clean.lower()
                    if any(kw in sentence_lower for kw in SUPPLY_KEYWORDS):
                        matched_sentences.append(sentence_clean)
                        
        return list(dict.fromkeys(matched_sentences))

    def save_sentences(self, company_name: str, source_url: str, doc_type: str, date: str, sentences: list[str], accession: str = None, root_seed: str = None):
        """Shared database writing logic. Accepts optional accession number for SEC."""
        if not sentences: return 0
        
        rows = [{
            "source_company": company_name,
            "accession_number": "N/A",
            "source_url": source_url,
            "document_type": doc_type,
            "filing_date": date,
            "raw_sentence": s,
            "accession_number": accession if accession else "N/A",
            "root_seed": root_seed,
            "llm_processed": "not_started"
        } for s in sentences]
        
        total_saved = 0
        for i in range(0, len(rows), 500):
            batch = rows[i:i + 500]
            res = self.supabase.table("scraped_sentences").upsert(
                batch, on_conflict="source_company,accession_number,raw_sentence"
            ).execute()
            
            if hasattr(res, "data") and res.data:
                total_saved += len(res.data)
            else:
                total_saved += len(batch)
                
        return total_saved