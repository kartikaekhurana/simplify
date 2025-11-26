import pdfplumber
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Try to import transformers for pre-trained models
TRANSFORMERS_AVAILABLE = False
domain_classifier = None
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
    # Initialize domain classifier (lazy loading to avoid startup delay)
    print("Transformers available. Pre-trained models can be used.")
except ImportError:
    print("Note: transformers library not installed. Using keyword-based approach.")
    print("For better accuracy, install: pip install transformers torch")

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


def detect_domain_with_model(text):
    """Detect domain using pre-trained model (if available)."""
    global domain_classifier
    
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # Lazy load the classifier
        if domain_classifier is None:
            # Using a general text classification model
            # You can replace with a resume-specific model if available
            domain_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # Use CPU
            )
        
        # Truncate text to model's max length (512 tokens)
        text_truncated = text[:2000]  # Approximate truncation
        
        # Note: This is a general model, not resume-specific
        # For production, use a fine-tuned resume classification model
        result = domain_classifier(text_truncated)
        
        # Since this is a general model, we'll still use keyword-based for domain
        # This is a placeholder for when you have a resume-specific model
        return None
        
    except Exception as e:
        print(f"Model-based detection failed: {e}")
        return None


def detect_domain(text):
    """Detect the professional domain based on resume content."""
    # Try model-based detection first (if available)
    model_result = detect_domain_with_model(text)
    if model_result:
        return model_result
    
    # Fall back to keyword-based approach
    text_lower = text.lower()
    
    # Domain keywords with weights
    domains = {
        'Data Science / Machine Learning': {
            'keywords': ['machine learning', 'data science', 'data scientist', 'ml engineer', 
                        'deep learning', 'neural network', 'tensorflow', 'pytorch', 'keras',
                        'pandas', 'numpy', 'scikit-learn', 'jupyter', 'data analysis',
                        'statistical analysis', 'predictive modeling', 'nlp', 'natural language processing',
                        'computer vision', 'recommendation system', 'data mining', 'big data'],
            'score': 0
        },
        'Software Engineering / Full Stack': {
            'keywords': ['software engineer', 'full stack', 'fullstack', 'web development',
                        'frontend', 'backend', 'full-stack developer', 'software developer',
                        'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
                        'javascript', 'typescript', 'html', 'css', 'rest api', 'graphql'],
            'score': 0
        },
        'DevOps / Cloud Engineering': {
            'keywords': ['devops', 'cloud engineer', 'aws', 'azure', 'gcp', 'docker',
                        'kubernetes', 'ci/cd', 'jenkins', 'terraform', 'ansible', 'infrastructure',
                        'cloud architecture', 'microservices', 'containerization', 'orchestration',
                        'monitoring', 'logging', 'deployment', 'automation'],
            'score': 0
        },
        'Backend Engineering': {
            'keywords': ['backend', 'api development', 'server-side', 'database design',
                        'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
                        'microservices', 'distributed systems', 'system design', 'architecture',
                        'java', 'python', 'go', 'scala', 'ruby on rails'],
            'score': 0
        },
        'Frontend Engineering': {
            'keywords': ['frontend', 'ui/ux', 'user interface', 'react', 'angular', 'vue',
                        'javascript', 'typescript', 'html', 'css', 'sass', 'bootstrap',
                        'responsive design', 'web design', 'user experience', 'ui developer'],
            'score': 0
        },
        'Mobile Development': {
            'keywords': ['mobile developer', 'ios', 'android', 'swift', 'kotlin', 'react native',
                        'flutter', 'mobile app', 'app development', 'xcode', 'android studio'],
            'score': 0
        },
        'Product Management': {
            'keywords': ['product manager', 'product management', 'product strategy', 'roadmap',
                        'stakeholder', 'agile', 'scrum', 'product owner', 'user stories',
                        'product development', 'go-to-market', 'product launch'],
            'score': 0
        },
        'Cybersecurity': {
            'keywords': ['cybersecurity', 'security engineer', 'penetration testing', 'vulnerability',
                        'encryption', 'firewall', 'security audit', 'threat analysis', 'security analyst',
                        'information security', 'network security'],
            'score': 0
        },
        'QA / Testing': {
            'keywords': ['qa engineer', 'quality assurance', 'test automation', 'selenium',
                        'testing', 'test engineer', 'qa analyst', 'manual testing', 'automated testing'],
            'score': 0
        }
    }
    
    # Calculate scores for each domain
    for domain, data in domains.items():
        for keyword in data['keywords']:
            if keyword in text_lower:
                data['score'] += 1
    
    # Find domain with highest score
    if not any(d['score'] > 0 for d in domains.values()):
        return "General / Other"
    
    best_domain = max(domains.items(), key=lambda x: x[1]['score'])
    return best_domain[0] if best_domain[1]['score'] > 0 else "General / Other"


def detect_seniority(text):
    """Detect seniority level based on resume content."""
    text_lower = text.lower()
    
    # Seniority indicators
    seniority_keywords = {
        'Intern / Entry Level': {
            'titles': ['intern', 'internship', 'entry level', 'junior', 'trainee', 'graduate'],
            'indicators': ['0-1 years', '0-2 years', 'fresh graduate', 'recent graduate'],
            'score': 0
        },
        'Junior / Associate': {
            'titles': ['junior', 'associate', 'entry', 'level 1', 'i'],
            'indicators': ['1-2 years', '1-3 years', '2 years', '2+ years'],
            'score': 0
        },
        'Mid-Level': {
            'titles': ['mid-level', 'mid level', 'engineer', 'developer', 'analyst', 'specialist'],
            'indicators': ['3-5 years', '3+ years', '4 years', '5 years', '4-6 years'],
            'score': 0
        },
        'Senior': {
            'titles': ['senior', 'sr.', 'sr ', 'lead', 'principal', 'staff'],
            'indicators': ['5+ years', '6+ years', '7+ years', '8+ years', '5-8 years'],
            'score': 0
        },
        'Lead / Principal': {
            'titles': ['lead', 'principal', 'staff engineer', 'architect', 'tech lead', 'engineering lead'],
            'indicators': ['8+ years', '10+ years', '12+ years', '15+ years', 'team lead', 'leading team'],
            'score': 0
        },
        'Director / VP / Executive': {
            'titles': ['director', 'vp', 'vice president', 'head of', 'chief', 'cto', 'cpo', 'executive'],
            'indicators': ['managing', 'strategy', 'executive', 'c-level', 'vice president'],
            'score': 0
        }
    }
    
    # Extract years of experience
    years_pattern = r'(\d+)\+?\s*(?:years?|yrs?|yr)'
    years_matches = re.findall(years_pattern, text_lower)
    max_years = 0
    if years_matches:
        max_years = max([int(y) for y in years_matches if y.isdigit()])
    
    # Check for leadership indicators
    leadership_indicators = ['led', 'leading', 'managed', 'mentored', 'architected', 'designed system',
                            'team lead', 'tech lead', 'engineering lead', 'oversaw', 'directed']
    has_leadership = any(indicator in text_lower for indicator in leadership_indicators)
    
    # Calculate scores
    for level, data in seniority_keywords.items():
        # Check titles
        for title in data['titles']:
            if title in text_lower:
                data['score'] += 2
        
        # Check indicators
        for indicator in data['indicators']:
            if indicator in text_lower:
                data['score'] += 1
    
    # Adjust based on years of experience
    if max_years >= 10:
        seniority_keywords['Lead / Principal']['score'] += 3
        seniority_keywords['Director / VP / Executive']['score'] += 2
    elif max_years >= 7:
        seniority_keywords['Senior']['score'] += 3
        seniority_keywords['Lead / Principal']['score'] += 2
    elif max_years >= 5:
        seniority_keywords['Senior']['score'] += 2
        seniority_keywords['Mid-Level']['score'] += 1
    elif max_years >= 3:
        seniority_keywords['Mid-Level']['score'] += 2
    elif max_years >= 1:
        seniority_keywords['Junior / Associate']['score'] += 2
    
    # Boost for leadership indicators
    if has_leadership:
        seniority_keywords['Senior']['score'] += 2
        seniority_keywords['Lead / Principal']['score'] += 3
    
    # Find best match
    if not any(d['score'] > 0 for d in seniority_keywords.values()):
        return "Not Specified"
    
    best_level = max(seniority_keywords.items(), key=lambda x: x[1]['score'])
    return best_level[0] if best_level[1]['score'] > 0 else "Not Specified"


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
    
    # Detect domain
    domain = detect_domain(text)
    
    # Detect seniority level
    seniority = detect_seniority(text)
    
    return {
        "general_keywords": dict(general_keywords),
        "tech_keywords": tech_keywords,
        "total_keywords": len(general_keywords) + len(tech_keywords),
        "domain": domain,
        "seniority": seniority
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

