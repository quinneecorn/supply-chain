# NLP NER & Masking Project

This project provides two independent, high-performance pipelines for Named Entity Recognition (NER) and automated masking of corporate and contractual text.

## Modules

### 1. [NER_Gemini_Pipeline](./NER_Gemini_Pipeline/)
- **Method**: Cloud-based (Gemini 2.0 Flash via OpenRouter).
- **Quality**: Highest (uses full LLM reasoning).
- **Requirement**: Requires an OpenRouter API key.
- **Table**: Configured for `NLP_copy_test` by default.

### 2. [NER_GLiNER_Pipeline](./NER_GLiNER_Pipeline/)
- **Method**: Local Zero-Shot Model (urchade/gliner_medium-v2.1).
- **Benefit**: Completely free and unlimited (no API costs).
- **Requirement**: Runs locally on your CPU/GPU.
- **Table**: Configured for `NLP Gliner` by default.

---

## Workspace Folders
- **`legacy/`**: Contains all previous versions, diagnostic tools, data pull/convert utilities, and generated CSV backups.
- **`NER_Gemini_Pipeline/`**: Standalone production folder for Gemini.
- **`NER_GLiNER_Pipeline/`**: Standalone production folder for GLiNER.

Refer to the `INSTRUCTIONS.md` within each module for detailed setup and execution steps.
