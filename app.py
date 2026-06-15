import re
import random
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load Model Assets
try:
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    models = joblib.load("mbti_models.joblib")
    print("Model assets loaded successfully.")
except Exception as e:
    print(f"Error loading model assets: {e}")
    vectorizer = None
    models = None

# MBTI Types Metadata
MBTI_INFO = {
    "INTJ": {
        "title": "The Architect",
        "description": "Imaginative and strategic thinkers, with a plan for everything.",
        "strengths": ["Rational", "Informed", "Independent", "Determined", "Versatile"],
        "weaknesses": ["Arrogant", "Dismissive of Emotions", "Overly Critical", "Combative"],
        "summary": "Architects are one of the rarest and most strategically capable personality types. They pride themselves on their minds and their ability to see through fluff to find the underlying truth."
    },
    "INTP": {
        "title": "The Logician",
        "description": "Innovative inventors with an unquenchable thirst for knowledge.",
        "strengths": ["Analytical", "Original", "Open-Minded", "Curious", "Objective"],
        "weaknesses": ["Disconnected", "Insensitive", "Absent-Minded", "Condescending"],
        "summary": "Logicians pride themselves on their unique perspectives and vigorous intellect. They love analyzing patterns, solving complex puzzles, and finding logical connections in everything."
    },
    "ENTJ": {
        "title": "The Commander",
        "description": "Bold, imaginative and strong-willed leaders, always finding a way.",
        "strengths": ["Efficient", "Energetic", "Self-Confident", "Strong-Willed", "Strategic"],
        "weaknesses": ["Stubborn", "Dominant", "Intolerant", "Impatient", "Cold"],
        "summary": "Commanders are natural-born leaders. People with this type embody the gifts of charisma and confidence, and project authority in a way that draws crowds together behind a common goal."
    },
    "ENTP": {
        "title": "The Debater",
        "description": "Smart and curious thinkers who cangitnot resist an intellectual challenge.",
        "strengths": ["Knowledgeable", "Quick Thinker", "Original", "Charismatic", "Energetic"],
        "weaknesses": ["Very Argumentative", "Insensitive", "Intolerant", "Can Find It Hard to Focus"],
        "summary": "Debaters love the process of mental sparring. They are quick-witted and ready to tear down established conventions to rebuild them into something better."
    },
    "INFJ": {
        "title": "The Advocate",
        "description": "Quiet and mystical, yet very inspiring and tireless idealists.",
        "strengths": ["Creative", "Insightful", "Principled", "Passionate", "Altruistic"],
        "weaknesses": ["Sensitive to Criticism", "Extremely Private", "Perfectionist", "Prone to Burnout"],
        "summary": "Advocates are the rarest personality type of all. They have a deep sense of idealism and integrity, but they are not idle dreamers – they take concrete steps to realize their goals and leave a lasting positive impact."
    },
    "INFP": {
        "title": "The Mediator",
        "description": "Poetic, kind and altruistic people, always eager to help a good cause.",
        "strengths": ["Empathetic", "Generous", "Open-Minded", "Creative", "Passionate"],
        "weaknesses": ["Unrealistic", "Self-Isolating", "Vulnerable to Stress", "Overly Self-Critical"],
        "summary": "Mediators are true idealists, always looking for the hint of good in even the worst of people and events, searching for ways to make things better."
    },
    "ENFJ": {
        "title": "The Protagonist",
        "description": "Charismatic and inspiring leaders, able to mesmerize their listeners.",
        "strengths": ["Receptive", "Reliable", "Passionate", "Altruistic", "Charismatic"],
        "weaknesses": ["Unrealistic", "Overly Idealistic", "Condescending", "Prone to Self-Sacrifice"],
        "summary": "Protagonists are natural-born leaders, full of passion and charisma. They love helping others grow, and are driven by a deep sense of moral purpose."
    },
    "ENFP": {
        "title": "The Campaigner",
        "description": "Enthusiastic, creative and sociable free spirits, who can always find a reason to smile.",
        "strengths": ["Curious", "Observant", "Energetic", "Excellent Communicators", "Festive"],
        "weaknesses": ["People-Pleasing", "Disorganized", "Overly Accommodating", "Restless"],
        "summary": "Campaigners are true free spirits. They are often the life of the party, but unlike Explorers, they are less interested in the sheer excitement of the moment than in making social and emotional connections."
    },
    "ISTJ": {
        "title": "The Logistician",
        "description": "Practical and fact-minded individuals, whose reliability cannot be doubted.",
        "strengths": ["Honest and Direct", "Strong-Willed", "Very Responsible", "Calm and Practical", "Create Order"],
        "weaknesses": ["Stubborn", "Judgmental", "Often Reason by the Book", "Can Blame Themselves Excessively"],
        "summary": "Logisticians are orderly, reliable, and dedicated. They value rules and guidelines, and take pride in their work and moral integrity."
    },
    "ISFJ": {
        "title": "The Defender",
        "description": "Very dedicated and warm protectors, always ready to defend their loved ones.",
        "strengths": ["Supportive", "Reliable", "Observant", "Enthusiastic", "Hardworking"],
        "weaknesses": ["Humble and Shy", "Take Things Personally", "Overload Themselves", "Reluctant to Change"],
        "summary": "Defenders are efficient and responsible, giving careful attention to practical details in their daily lives. They are warm-hearted, altruistic, and devoted to protecting those they care about."
    },
    "ESTJ": {
        "title": "The Executive",
        "description": "Excellent administrators, unsurpassed at managing things or people.",
        "strengths": ["Dedicated", "Honest", "Direct", "Reliable", "Excellent Organizers"],
        "weaknesses": ["Inflexible", "Uncomfortable with Unconventional Situations", "Too Focused on Social Status"],
        "summary": "Executives are representatives of order and tradition, organizing people and projects around clear guidelines and expectations."
    },
    "ESFJ": {
        "title": "The Consul",
        "description": "Extraordinarily caring, social and popular people, always eager to help.",
        "strengths": ["Strong Sense of Duty", "Very Loyal", "Sensitive and Warm", "Good at Connecting with Others"],
        "weaknesses": ["Worried about Their Social Status", "Inflexible", "Reluctant to Innovate", "Needy"],
        "summary": "Consuls are social creatures, keeping track of what their friends are up to and doing their best to bring harmony to their communities."
    },
    "ISTP": {
        "title": "The Virtuoso",
        "description": "Bold and practical experimenters, masters of all kinds of tools.",
        "strengths": ["Optimistic and Energetic", "Creative and Practical", "Spontaneous and Rational", "Know How to Prioritize"],
        "weaknesses": ["Stubborn", "Private and Reserved", "Easily Bored", "Dislike Commitment"],
        "summary": "Virtuosos love to explore with their hands and their eyes, touching and examining the world around them with cool rationalism and spirited curiosity."
    },
    "ISFP": {
        "title": "The Adventurer",
        "description": "Flexible and charming artists, always ready to explore and experience something new.",
        "strengths": ["Charming", "Sensitive to Others", "Imaginative", "Passionate", "Artistic"],
        "weaknesses": ["Fiercely Independent", "Unpredictable", "Easily Stressed", "Overly Competitive"],
        "summary": "Adventurers are true artists, using aesthetics, design, and even their choices to push the limits of social convention."
    },
    "ESTP": {
        "title": "The Persuader",
        "description": "Smart, energetic and very perceptive people, who truly enjoy living on the edge.",
        "strengths": ["Bold", "Rational and Practical", "Great People Skills", "Perceptive", "Direct"],
        "weaknesses": ["Insensitive", "Impatient", "Risk-Prone", "May Miss the Bigger Picture"],
        "summary": "Persuaders have a massive impact on their immediate surroundings. They love being the center of attention and appreciate action, living in the moment and seizing opportunities."
    },
    "ESFP": {
        "title": "The Entertainer",
        "description": "Spontaneous, energetic and enthusiastic people – life is never boring around them.",
        "strengths": ["Bold", "Original", "Practical", "Excellent People Skills", "Observant"],
        "weaknesses": ["Sensitive", "Easily Bored", "Poor Planners", "Unfocused"],
        "summary": "Entertainers love to perform, putting on a show for others and basking in the excitement. They are warm, social, and enjoy making others happy."
    }
}

# Guided Topics Pool (Variant 2)
TOPICS = [
    "Describe your ideal day from morning to night. What activities make you feel truly energized and content?",
    "How do you handle conflict or arguments with close friends or family? Walk us through your thought process.",
    "Describe a book, movie, or personal experience that deeply changed your perspective on life or humanity.",
    "What is a major goal you have for your future? How do you plan to achieve it, and what motivates you?",
    "When you are working on a project in a group, what role do you naturally fall into and why?",
    "Talk about your relationship with solitude. Do you find comfort in being alone, or do you actively seek out social interaction?"
]

# MBTI types list for regex masking
MBTI_TYPES = [
    'infj', 'enfp', 'intj', 'entp', 'intp', 'infp', 'entj', 'enfj',
    'isfj', 'istj', 'isfp', 'istp', 'estp', 'esfp', 'estj', 'esfj'
]
MASK_REGEX = re.compile(r'\b(' + '|'.join(MBTI_TYPES) + r')s?\b', re.IGNORECASE)

def preprocess_text(text):
    """Clean and mask input text in the same way as training preprocessing"""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Mask explicit type names
    text = MASK_REGEX.sub('[MASK]', text)
    # Clean spacing and tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/get-topic', methods=['GET'])
def get_topic():
    # Return a random topic from the pool
    topic = random.choice(TOPICS)
    return jsonify({"topic": topic})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not vectorizer or not models:
        return jsonify({"error": "Models not loaded. Please train the models first."}), 500
        
    data = request.json
    raw_text = data.get("text", "")
    
    if not raw_text.strip():
        return jsonify({"error": "Empty text provided."}), 400
        
    # Preprocess text
    cleaned_text = preprocess_text(raw_text)
    
    # Check word count
    words = cleaned_text.split()
    word_count = len(words)
    if word_count < 10:
        return jsonify({"error": "Please write at least 10 words to analyze."}), 400
        
    # Transform text
    tfidf_features = vectorizer.transform([cleaned_text])
    
    # Store predictions and probabilities
    trait_percentages = {}
    trait_confidences = []
    
    # Traits mapping: Introvert/Extrovert, Intuitive/Sensing, Feeling/Thinking, Judging/Perceiving
    traits_info = {
        'I/E': ('Introvert', 'Extrovert'),
        'N/S': ('Intuitive', 'Sensing'),
        'F/T': ('Feeling', 'Thinking'),
        'J/P': ('Judging', 'Perceiving')
    }
    
    # Cartesian product components for multiple personalities
    options = {
        'I/E': [],
        'N/S': [],
        'F/T': [],
        'J/P': []
    }
    
    # Margin for multiple probable personalities (e.g. 5% near 50%, which is [45%, 55%])
    margin = 5.0
    
    for trait in ['I/E', 'N/S', 'F/T', 'J/P']:
        clf = models[trait]
        prob = clf.predict_proba(tfidf_features)[0] # [prob_class_0, prob_class_1]
        
        # class 1 = I, N, F, J; class 0 = E, S, T, P
        pos_label, neg_label = traits_info[trait]
        pos_prob = prob[1] * 100
        neg_prob = prob[0] * 100
        
        trait_percentages[trait] = {
            "pos_label": pos_label,
            "neg_label": neg_label,
            "pos_percent": round(pos_prob, 1),
            "neg_percent": round(neg_prob, 1)
        }
        
        # Calculate axis confidence: how far is the probability from 50%?
        axis_confidence = abs(prob[1] - 0.5) * 2 # ranges from 0 to 1
        trait_confidences.append(axis_confidence)
        
        # Determine options for this dimension based on the margin
        if abs(pos_prob - 50.0) <= margin:
            options[trait] = [pos_label[0], neg_label[0]]
        else:
            options[trait] = [pos_label[0]] if pos_prob > 50.0 else [neg_label[0]]

    # 1. Overall Confidence Score (mean of trait confidences)
    mean_confidence = np.mean(trait_confidences) * 100 # percentage (0% to 100%)
    
    # 2. Cartesian product of options to build list of probable MBTI types
    probable_types = []
    for c1 in options['I/E']:
        for c2 in options['N/S']:
            for c3 in options['F/T']:
                for c4 in options['J/P']:
                    probable_types.append(f"{c1}{c2}{c3}{c4}")
                    
    # Primary MBTI type (based on strictly max value)
    primary_type = "".join([
        'I' if trait_percentages['I/E']['pos_percent'] >= 50 else 'E',
        'N' if trait_percentages['N/S']['pos_percent'] >= 50 else 'S',
        'F' if trait_percentages['F/T']['pos_percent'] >= 50 else 'T',
        'J' if trait_percentages['J/P']['pos_percent'] >= 50 else 'P'
    ])
    
    # Ensure primary type is first in the list
    if primary_type in probable_types:
        probable_types.remove(primary_type)
    probable_types.insert(0, primary_type)
    
    # Retrieve details for the predicted types
    results_info = []
    for mbti in probable_types:
        info = MBTI_INFO.get(mbti, {})
        results_info.append({
            "mbti": mbti,
            "title": info.get("title", "Unknown"),
            "description": info.get("description", ""),
            "summary": info.get("summary", ""),
            "strengths": info.get("strengths", []),
            "weaknesses": info.get("weaknesses", [])
        })

    # 3. Linguistic Marker Explainability
    # We find words in the input that had the most influence on each trait
    influence_details = {}
    feature_names = vectorizer.get_feature_names_out()
    
    # Process text into words present in vocabulary
    # Using simple split and filtering
    input_tokens = list(set(cleaned_text.split()))
    
    for trait in ['I/E', 'N/S', 'F/T', 'J/P']:
        clf = models[trait]
        coefs = clf.coef_[0]
        pos_label, neg_label = traits_info[trait]
        
        word_influences = []
        for word in input_tokens:
            # Check if word is in vocabulary
            if word in vectorizer.vocabulary_:
                idx = vectorizer.vocabulary_[word]
                # Weight of the word in this specific document
                word_tfidf = tfidf_features[0, idx]
                if word_tfidf > 0:
                    weight = coefs[idx]
                    influence = word_tfidf * weight
                    word_influences.append((word, influence))
                    
        # Sort word influences
        # Positive influences push towards Class 1 (I, N, F, J)
        # Negative influences push towards Class 0 (E, S, T, P)
        word_influences.sort(key=lambda x: x[1])
        
        # Extract top 5 words associated with positive and negative labels in this input text
        neg_markers = [{"word": w, "weight": round(inf, 4)} for w, inf in word_influences[:5] if inf < -0.005]
        pos_markers = [{"word": w, "weight": round(inf, 4)} for w, inf in word_influences[-5:][::-1] if inf > 0.005]
        
        influence_details[trait] = {
            "pos_label": pos_label,
            "neg_label": neg_label,
            "pos_markers": pos_markers, # associated with I, N, F, J
            "neg_markers": neg_markers  # associated with E, S, T, P
        }

    return jsonify({
        "word_count": word_count,
        "primary_type": primary_type,
        "probable_types": probable_types,
        "results_info": results_info,
        "trait_percentages": trait_percentages,
        "confidence": round(mean_confidence, 1),
        "influence_details": influence_details
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
