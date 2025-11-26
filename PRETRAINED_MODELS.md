# Pre-trained Models for Resume Analysis

Yes, there are pre-trained models available for resume analysis! Here are the best options:

## Available Models

### 1. **Resume Analyser - BERT** (Recommended)
- **Model**: `SwaKyxd/resume-analyser-bert`
- **Hugging Face**: https://huggingface.co/SwaKyxd/resume-analyser-bert
- **Features**: Classifies resumes into 25 job categories
- **Accuracy**: 100% validation accuracy on test dataset
- **Use Case**: Domain/Job Category Classification

### 2. **ESCOXLM-R**
- **Features**: Multilingual (27 languages), skill extraction, job title classification
- **Use Case**: International resumes, skill matching

### 3. **CareerBERT**
- **Features**: Pre-trained on job-related data, resume-job matching
- **Use Case**: Resume-job description matching

## Installation

To use pre-trained models, install:

```bash
pip install transformers torch sentencepiece
```

## Usage

The current implementation uses a **keyword-based approach** which is:
- ✅ Fast and lightweight
- ✅ No GPU required
- ✅ Works offline
- ✅ Good accuracy for most cases

For **better accuracy** with pre-trained models, you can:

1. Install transformers: `pip install transformers torch`
2. The code will automatically detect if transformers is available
3. You can integrate a resume-specific model like `SwaKyxd/resume-analyser-bert`

## Current Implementation

The code uses a hybrid approach:
- **Primary**: Keyword-based detection (fast, reliable)
- **Optional**: Pre-trained models (if transformers is installed)

This ensures the app works immediately while allowing for ML model integration when needed.

## Note

Pre-trained models require:
- More memory (1-2GB+)
- Slower initial load time
- May require GPU for best performance

For a web application, the keyword-based approach is often more practical unless you need very high accuracy or are processing many resumes.

