import pdfplumber
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        # Fallback to older punkt if punkt_tab not available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except:
        # Fallback to older tagger if new one not available
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    return text


def extract_keywords(text):
    """Extract keywords from text using NLP techniques."""
    # Convert to lowercase and clean text
    text = text.lower()
    
    # Remove special characters but keep spaces and alphanumeric
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Filter out stopwords and short words
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # POS tagging to extract nouns and proper nouns (likely to be skills/technologies)
    tagged = pos_tag(filtered_tokens)
    
    # Extract nouns (NN, NNS, NNP, NNPS)
    keywords = [word for word, pos in tagged if pos.startswith('NN')]
    
    # Also extract common tech keywords (often appear as capitalized in resumes)
    # Extract capitalized words/phrases (common for technologies)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Combine and count
    all_keywords = keywords + [w.lower() for w in capitalized if len(w) > 2]
    
    # Count frequency
    keyword_counts = Counter(all_keywords)
    
    # Return top keywords
    return keyword_counts.most_common(30)


def extract_tech_keywords(text):
    """Extract technology-related keywords using common tech terms."""
    # Common technology keywords
    tech_keywords = [
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'node', 'django', 'flask', 'spring', 'express', 'sql', 'mysql', 'postgresql',
        'mongodb', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github',
        'jenkins', 'ci/cd', 'agile', 'scrum', 'rest', 'api', 'graphql', 'microservices',
        'machine learning', 'ai', 'data science', 'tensorflow', 'pytorch', 'pandas',
        'numpy', 'scikit-learn', 'html', 'css', 'sass', 'bootstrap', 'tailwind',
        'redux', 'webpack', 'npm', 'yarn', 'linux', 'unix', 'bash', 'shell',
        'terraform', 'ansible', 'elasticsearch', 'kafka', 'redis', 'rabbitmq'
    ]
    
    text_lower = text.lower()
    found_keywords = []
    
    for keyword in tech_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    return found_keywords


def extract_keywords_from_resume(pdf_path):
    """Main function to extract keywords from resume PDF."""
    # Extract text
    text = extract_text_from_pdf(pdf_path)
    
    if not text.strip():
        return {"error": "No text found in PDF"}
    
    # Extract general keywords
    general_keywords = extract_keywords(text)
    
    # Extract tech-specific keywords
    tech_keywords = extract_tech_keywords(text)
    
    return {
        "general_keywords": dict(general_keywords),
        "tech_keywords": tech_keywords,
        "total_keywords": len(general_keywords) + len(tech_keywords)
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python keyword_extractor.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        result = extract_keywords_from_resume(pdf_path)
        
        print("\n=== KEYWORD EXTRACTION RESULTS ===\n")
        print("Tech Keywords Found:")
        for keyword in result.get("tech_keywords", []):
            print(f"  - {keyword}")
        
        print("\nTop General Keywords:")
        for keyword, count in list(result.get("general_keywords", {}).items())[:20]:
            print(f"  - {keyword}: {count}")
        
        print(f"\nTotal unique keywords: {result.get('total_keywords', 0)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

