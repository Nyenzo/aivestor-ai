# Import libraries for bill analysis
import spacy  # For NLP and keyword extraction
import re  # For text cleaning

# Step 1: Load the processed bill data from processed_data.txt
def load_bills(file_path):
    """
    Load bills from the processed data file.
    Args:
        file_path (str): Path to the processed_data.txt file
    Returns:
        list: List of processed bill texts
    """
    bills = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("BILL:"):
                bills.append(line.replace("BILL:", "").strip())
    return bills

# Step 2: Clean garbled text (e.g., spaces between letters)
def clean_garbled_text(text):
    """
    Clean text with spaces between letters (e.g., 'f r k' -> 'frk').
    Args:
        text (str): Raw text to clean
    Returns:
        str: Cleaned text
    """
    # Remove extra spaces between letters (e.g., 'f r k' -> 'frk')
    cleaned_text = re.sub(r'\b(\w\s+){2,}\w\b', lambda m: m.group(0).replace(" ", ""), text)
    # Remove excessive spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Step 3: Extract key terms using SpaCy
def extract_key_terms(text, nlp):
    """
    Extract key terms (nouns, entities, phrases) from text using SpaCy.
    Args:
        text (str): Text to analyze
        nlp: SpaCy language model
    Returns:
        list: List of key terms
    """
    # Clean the text first
    text = clean_garbled_text(text)
    
    # Process the text with SpaCy
    doc = nlp(text)
    
    # Extract nouns, entities, and significant phrases
    key_terms = []
    
    # Extract named entities (e.g., organizations, laws)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "LAW", "GPE", "MONEY", "DATE"]:  # Relevant entity types
            key_terms.append(ent.text)
    
    # Extract noun chunks (significant phrases)
    for chunk in doc.noun_chunks:
        # Exclude overly generic terms
        if len(chunk.text.split()) > 1 and chunk.text.lower() not in ["kenya", "act", "bill"]:
            key_terms.append(chunk.text)
    
    # Extract individual nouns not already in entities or chunks
    for token in doc:
        if token.pos_ == "NOUN" and token.text.lower() not in [kt.lower() for kt in key_terms]:
            if token.text.lower() not in ["kenya", "act", "bill", "laws"]:  # Filter generic terms
                key_terms.append(token.text)
    
    return list(set(key_terms))  # Remove duplicates

# Step 4: Map key terms to sectors and impacts
# Step 4: Map key terms to sectors and impacts
def map_terms_to_sectors(key_terms):
    """
    Map key terms to affected sectors and potential stock impacts.
    Args:
        key_terms (list): List of key terms
    Returns:
        dict: Mapping of sectors to potential impacts
    """
    sector_impacts = {
        "Banking": [],
        "Capital Markets": [],
        "General Business": [],
        "Government Securities": [],
        "Taxation": [],
        "Public Finance": [],
        "Telecom": [],
        "Energy": [],
        "Fintech": [],
        "Tech": [],
        "Health": []
    }
    
    # Define keyword mappings to sectors
    sector_keywords = {
        "Banking": ["bank", "finance", "investment", "loan", "credit"],
        "Capital Markets": ["capital markets", "securities", "stock", "exchange"],
        "General Business": ["business", "company", "enterprise", "market"],
        "Government Securities": ["government securities", "bond", "treasury"],
        "Taxation": ["tax", "taxation", "revenue", "duty"],
        "Public Finance": ["public finance", "budget", "expenditure", "revenue"],
        "Telecom": ["telecom", "communication", "mobile", "network"],
        "Energy": ["energy", "power", "electricity", "fuel", "renewable"],
        "Fintech": ["fintech", "financial technology", "digital payment", "blockchain"],
        "Tech": ["tech", "technology", "software", "hardware", "ai"],
        "Health": ["health", "healthcare", "medical", "pharma", "hospital"]
    }
    
    # Map terms to sectors
    for term in key_terms:
        term_lower = term.lower()
        for sector, keywords in sector_keywords.items():
            if any(keyword in term_lower for keyword in keywords):
                # Determine potential impact based on term
                if "amendment" in term_lower or "regulation" in term_lower:
                    impact = "Potential Regulatory Change"
                elif "tax" in term_lower or "duty" in term_lower:
                    impact = "Potential Cost Increase"
                else:
                    impact = "Potential Sector Influence"
                sector_impacts[sector].append((term, impact))
    
    # Remove sectors with no impacts
    return {sector: impacts for sector, impacts in sector_impacts.items() if impacts}

# Step 5: Main Function to Run Bill Analysis
def main():
    # Load SpaCy's English model
    print("Loading SpaCy model...")
    nlp = spacy.load('en_core_web_sm')
    print("Model loaded successfully.")

    # Load the processed bills
    bills = load_bills("processed_data.txt")
    print(f"Loaded {len(bills)} bills.")

    # Analyze each bill
    print("\nAnalyzing bills for key terms and sector impacts...")
    for i, bill_text in enumerate(bills):
        print(f"\nBill {i + 1}: {bill_text[:50]}...")
        
        # Extract key terms
        key_terms = extract_key_terms(bill_text, nlp)
        # Convert key_terms to string with UTF-8 encoding
        key_terms_str = ', '.join(str(term).encode('utf-8', errors='replace').decode('utf-8') for term in key_terms)
        print(f"Key Terms: {key_terms_str}")
        
        # Map terms to sectors and impacts
        sector_impacts = map_terms_to_sectors(key_terms)
        print("Sector Impacts:")
        for sector, impacts in sector_impacts.items():
            print(f"  {sector}:")
            for term, impact in impacts:
                print(f"    - {term}: {impact}")

    # Save results to a file
    with open("bill_analysis_results.txt", "w", encoding="utf-8") as file:
        for i, bill_text in enumerate(bills):
            file.write(f"\nBill {i + 1}: {bill_text[:50]}...\n")
            key_terms = extract_key_terms(bill_text, nlp)
            file.write(f"Key Terms: {', '.join(key_terms)}\n")
            sector_impacts = map_terms_to_sectors(key_terms)
            file.write("Sector Impacts:\n")
            for sector, impacts in sector_impacts.items():
                file.write(f"  {sector}:\n")
                for term, impact in impacts:
                    file.write(f"    - {term}: {impact}\n")
    print("\nBill analysis results saved to bill_analysis_results.txt")

if __name__ == "__main__":
    main()