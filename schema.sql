-- SQL Schema setup for SEC EDGAR Crawler
-- Copy and run this script in your Supabase SQL Editor.

-- 1. Create company_queue table
CREATE TABLE IF NOT EXISTS company_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name TEXT UNIQUE NOT NULL,
    naics_code TEXT DEFAULT '334',
    tier_level INT DEFAULT 0,
    status TEXT DEFAULT 'not_started' CHECK (status IN ('not_started', 'in_progress', 'done', 'failed')),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Create raw_sentences table
CREATE TABLE IF NOT EXISTS raw_sentences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name TEXT NOT NULL,
    cik TEXT NOT NULL,
    accession_number TEXT NOT NULL,
    form_type TEXT NOT NULL CHECK (form_type IN ('10-K', '10-Q', '8-K', '20-F', '6-K')),
    filing_date DATE NOT NULL,
    sentence TEXT NOT NULL,
    llm_processed BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now(),
    
    -- Unique constraint to prevent duplicate sentences per filing
    CONSTRAINT unique_sentence_per_filing UNIQUE (company_name, accession_number, sentence)
);

-- Create index on status for faster querying of company_queue
CREATE INDEX IF NOT EXISTS idx_company_queue_status ON company_queue(status);

-- Create index on llm_processed for downstream workers
CREATE INDEX IF NOT EXISTS idx_raw_sentences_llm_processed ON raw_sentences(llm_processed);
