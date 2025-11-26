import random
import requests

def generate_interview_questions(domain, seniority, tech_keywords=None):
    # Base questions for all levels
    base_questions = [
        "Tell me about yourself and your background.",
        "What interests you most about this role?",
        "Why are you looking for a new opportunity?",
        "What are your greatest strengths?",
        "What is your biggest weakness and how do you work on it?",
    ]
    
    # Domain-specific questions
    domain_questions = {
        'Data Science / Machine Learning': {
            'junior': [
                "Explain the difference between supervised and unsupervised learning.",
                "What is overfitting and how do you prevent it?",
                "Walk me through a machine learning project you've worked on.",
                "What evaluation metrics would you use for a classification problem?",
                "Explain cross-validation and why it's important.",
            ],
            'mid': [
                "How do you handle imbalanced datasets?",
                "Explain the bias-variance tradeoff.",
                "Describe your experience with feature engineering.",
                "How do you select the right algorithm for a problem?",
                "Walk me through your approach to model deployment.",
            ],
            'senior': [
                "How do you design an ML system architecture for production?",
                "Explain how you would handle model drift in production.",
                "Describe your experience with MLOps and model monitoring.",
                "How do you ensure fairness and ethics in ML models?",
                "Walk me through a complex ML problem you solved and the tradeoffs involved.",
            ],
        },
        'Software Engineering / Full Stack': {
            'junior': [
                "Explain the difference between REST and GraphQL.",
                "What is the difference between let, const, and var in JavaScript?",
                "Explain how React components work.",
                "What is version control and why is it important?",
                "Describe the difference between SQL and NoSQL databases.",
            ],
            'mid': [
                "How do you handle state management in a large application?",
                "Explain microservices architecture and its tradeoffs.",
                "Describe your approach to API design.",
                "How do you ensure code quality and maintainability?",
                "Walk me through your debugging process for a complex bug.",
            ],
            'senior': [
                "How do you design a scalable system architecture?",
                "Explain your approach to technical leadership and mentoring.",
                "Describe a time you had to make a difficult technical decision.",
                "How do you balance technical debt with feature development?",
                "Walk me through how you would design a distributed system.",
            ],
        },
        'DevOps / Cloud Engineering': {
            'junior': [
                "What is CI/CD and why is it important?",
                "Explain the difference between Docker and Kubernetes.",
                "What is infrastructure as code?",
                "Describe your experience with cloud platforms.",
                "How do you monitor application performance?",
            ],
            'mid': [
                "How do you design a disaster recovery plan?",
                "Explain your approach to container orchestration.",
                "Describe your experience with infrastructure automation.",
                "How do you ensure security in cloud deployments?",
                "Walk me through your CI/CD pipeline design.",
            ],
            'senior': [
                "How do you design a multi-cloud architecture?",
                "Explain your approach to infrastructure cost optimization.",
                "Describe your experience with Kubernetes at scale.",
                "How do you handle incident response and post-mortems?",
                "Walk me through your approach to infrastructure security and compliance.",
            ],
        },
        'Backend Engineering': {
            'junior': [
                "Explain the difference between SQL and NoSQL databases.",
                "What is an API and how does it work?",
                "Describe database indexing and why it's important.",
                "What is caching and when would you use it?",
                "Explain the difference between synchronous and asynchronous operations.",
            ],
            'mid': [
                "How do you design a database schema for scalability?",
                "Explain your approach to API rate limiting and throttling.",
                "Describe your experience with message queues.",
                "How do you handle database migrations?",
                "Walk me through your approach to API security.",
            ],
            'senior': [
                "How do you design a distributed system architecture?",
                "Explain your approach to database sharding and partitioning.",
                "Describe your experience with system design at scale.",
                "How do you ensure data consistency in distributed systems?",
                "Walk me through how you would design a high-throughput API.",
            ],
        },
        'Frontend Engineering': {
            'junior': [
                "Explain the difference between HTML, CSS, and JavaScript.",
                "What is the DOM and how does it work?",
                "Describe responsive design principles.",
                "What is the difference between CSS Grid and Flexbox?",
                "Explain how JavaScript closures work.",
            ],
            'mid': [
                "How do you optimize frontend performance?",
                "Explain your approach to state management.",
                "Describe your experience with frontend testing.",
                "How do you ensure accessibility in web applications?",
                "Walk me through your approach to component architecture.",
            ],
            'senior': [
                "How do you design a frontend architecture for large applications?",
                "Explain your approach to frontend performance optimization at scale.",
                "Describe your experience with frontend build tools and bundlers.",
                "How do you ensure maintainability in large codebases?",
                "Walk me through your approach to frontend security.",
            ],
        },
        'Mobile Development': {
            'junior': [
                "Explain the difference between native and cross-platform development.",
                "What is the app lifecycle in iOS/Android?",
                "Describe your experience with mobile UI frameworks.",
                "How do you handle different screen sizes?",
                "Explain mobile app state management.",
            ],
            'mid': [
                "How do you optimize mobile app performance?",
                "Describe your approach to mobile app testing.",
                "How do you handle offline functionality?",
                "Explain your experience with push notifications.",
                "Walk me through your approach to mobile app architecture.",
            ],
            'senior': [
                "How do you design a mobile app architecture for scale?",
                "Explain your approach to mobile app security.",
                "Describe your experience with app store optimization.",
                "How do you handle cross-platform code sharing?",
                "Walk me through your approach to mobile CI/CD.",
            ],
        },
    }
    
    # Determine seniority category
    seniority_lower = seniority.lower()
    if 'intern' in seniority_lower or 'entry' in seniority_lower or 'junior' in seniority_lower:
        level = 'junior'
    elif 'senior' in seniority_lower or 'lead' in seniority_lower or 'principal' in seniority_lower or 'director' in seniority_lower:
        level = 'senior'
    else:
        level = 'mid'
    
    # Get domain-specific questions
    questions = base_questions.copy()
    
    if domain in domain_questions:
        if level in domain_questions[domain]:
            questions.extend(domain_questions[domain][level])
    
    # Add technology-specific questions if tech keywords are provided
    if tech_keywords:
        tech_questions = generate_tech_specific_questions(tech_keywords, level)
        questions.extend(tech_questions)
    
    # Shuffle and return 10-12 questions
    random.shuffle(questions)
    return questions[:12]


def generate_tech_specific_questions(tech_keywords, level):
    """Generate questions based on specific technologies mentioned."""
    questions = []
    
    tech_question_map = {
        'python': [
            "Explain Python's GIL (Global Interpreter Lock).",
            "What are Python decorators and when would you use them?",
        ] if level != 'junior' else [
            "What are Python list comprehensions?",
            "Explain the difference between lists and tuples in Python.",
        ],
        'react': [
            "Explain React's virtual DOM and how it works.",
            "What are React hooks and when would you use them?",
        ] if level != 'junior' else [
            "What are React components?",
            "Explain props and state in React.",
        ],
        'aws': [
            "How do you design a highly available architecture on AWS?",
            "Explain your experience with AWS services for scaling.",
        ] if level != 'junior' else [
            "What AWS services are you familiar with?",
            "Explain the difference between EC2 and Lambda.",
        ],
        'docker': [
            "How do you optimize Docker images for production?",
            "Explain Docker networking and volumes.",
        ] if level != 'junior' else [
            "What is Docker and why is it used?",
            "Explain the difference between Docker and virtual machines.",
        ],
        'kubernetes': [
            "How do you design a Kubernetes cluster for high availability?",
            "Explain your approach to Kubernetes resource management.",
        ] if level != 'junior' else [
            "What is Kubernetes and what problems does it solve?",
            "Explain the basic Kubernetes components.",
        ],
    }
    
    # Add questions for technologies found
    for keyword in tech_keywords[:5]:  # Limit to top 5 technologies
        keyword_lower = keyword.lower()
        for tech, tech_questions in tech_question_map.items():
            if tech in keyword_lower:
                questions.extend(tech_questions[:1])  # Add 1 question per tech
                break
    
    return questions


def get_interview_flow(domain, seniority):
    """Get interview flow structure based on domain and seniority."""
    return {
        "duration_minutes": 45 if 'senior' in seniority.lower() or 'lead' in seniority.lower() else 30,
        "sections": [
            {"name": "Introduction", "duration": 5, "questions": 2},
            {"name": "Technical Skills", "duration": 15 if 'senior' in seniority.lower() else 10, "questions": 4},
            {"name": "Experience & Projects", "duration": 15, "questions": 4},
            {"name": "Problem Solving", "duration": 10, "questions": 2},
        ]
    }

