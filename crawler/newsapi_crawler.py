import os
import requests
from dotenv import load_dotenv
from supabase import create_client, Client
from newspaper import Article
import spacy
from .base_crawler import BaseCrawler

# 1. Environment Setup
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

print("Loading spaCy NLP model for News Crawler...")
nlp = spacy.load("en_core_web_sm")

class NewsCrawler(BaseCrawler):
    def __init__(self, supabase: Client):
        super().__init__(supabase)
        # We define our own keywords here just for the override function, 
        # but it keeps the class self-contained.
        self.supply_keywords = [
            "suppli", "manufactur", "vendor", "contract", "procure",
            "outsourc", "distribut", "partner", "acqui", "subsidiar",
            "joint venture", "licens", "assembly", "component", "source",
            "tier-1", "tier-2", "oem", "odm", "wafer", "foundry"
        ]

    # OVERRIDE: News needs paragraphs, not isolated sentences
    def extract_sentences(self, text: str) -> list[str]:
        """
        Overrides the BaseCrawler extraction.
        Groups sentences into a 3-sentence sliding window to preserve journalistic context,
        so the LLM can resolve pronouns like 'they' or 'the company'.
        """
        if not text: return []
        
        doc = nlp(text)
        sentences = list(doc.sents)
        matched_blocks = []
        
        # Create a sliding window of up to 3 sentences
        for i in range(len(sentences)):
            # Grab current sentence + up to 2 previous sentences
            start_idx = max(0, i - 2)
            block = " ".join([s.text.strip() for s in sentences[start_idx:i+1]])
            
            block_clean = " ".join(block.split())
            
            # Allow longer blocks (up to 1500 chars) since we are joining sentences
            if 40 <= len(block_clean) <= 1500:
                # We only check if the *current* (final) sentence in the window contains the keyword.
                # If it does, we save the whole block for context.
                target_sent_lower = sentences[i].text.strip().lower()
                
                if any(kw in target_sent_lower for kw in self.supply_keywords):
                    matched_blocks.append(block_clean)
                    
        return list(dict.fromkeys(matched_blocks))

    def crawl(self, company_name: str, root_seed: str = None, **kwargs):
        print(f"\n[NewsCrawler] Searching news for: {company_name}")
        
        if not NEWS_API_KEY:
            print("  [!] Missing NEWS_API_KEY. Skipping.")
            return

        # Query structure optimized for business/supply chain news
        query = f'"{company_name}" AND (supplier OR partnership OR contract OR supply chain OR manufacturer)'
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&pageSize=10"
        
        headers = {"X-Api-Key": NEWS_API_KEY}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            articles_data = response.json().get("articles", [])
            
            if not articles_data:
                print("  [i] No relevant supply chain news found.")
                return
                
            print(f"  [✓] Found {len(articles_data)} relevant articles. Scraping full text...")
            total_company_sentences = 0
            
            for index, art in enumerate(articles_data, start=1):
                article_url = art.get("url")
                publish_date = art.get("publishedAt", "")[:10]
                source_name = art.get("source", {}).get("name", "Unknown Source")
                
                print(f"    -> [{index}/{len(articles_data)}] Scraping: {source_name}")
                
                try:
                    # newspaper3k strips out HTML, ads, and navigation menus
                    news_article = Article(article_url)
                    news_article.download()
                    news_article.parse()
                    full_text = news_article.text
                    
                    # Uses our overridden Context Block extractor
                    context_blocks = self.extract_sentences(full_text)
                    
                    if context_blocks:
                        # Inherits the database saving logic from BaseCrawler!
                        saved = self.save_sentences(
                            company_name=company_name,
                            source_url=article_url,
                            doc_type=f"News ({source_name})",
                            date=publish_date if publish_date else "1970-01-01",
                            sentences=context_blocks, 
                            root_seed=root_seed
                        )
                        total_company_sentences += saved
                        print(f"       [+] Extracted and saved {saved} context blocks.")
                except Exception as scrape_err:
                    print(f"       [✗] Failed to scrape {source_name}: {str(scrape_err)[:80]}...")
                    
            print(f"  [✓] Finished {company_name} News | Total context blocks saved: {total_company_sentences}")
            
        except Exception as e:
            print(f"  [✗] NewsAPI request failed: {str(e)}")

if __name__ == "__main__":
    if SUPABASE_URL and SUPABASE_KEY:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        crawler = NewsCrawler(client)
        # Test it directly!
        crawler.crawl("Sony Group Corporation")