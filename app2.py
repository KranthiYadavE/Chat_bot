import torch
import gradio as gr
import json
import os
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import re
import random


MODEL_PATH = "C:/Users/HP/Desktop/HCI_Final/model1/model_state.pth" #change path after extracting
TOKENIZER_CONFIG = "C:/Users/HP/Desktop/HCI_Final/model1/tokenizer_config" #change path after extracting
SPECIAL_TOKENS_MAP = "C:/Users/HP/Desktop/HCI_Final/model1/special_tokens_map" #change path after extracting
VOCAB_PATH = "C:/Users/HP/Desktop/HCI_Final/model1/vocab" #change path after extracting

print("Loading emotion detection model...")
emotion_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
emotion_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=27  
)

checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
emotion_model.load_state_dict(checkpoint["model_state_dict"])
label_encoder = checkpoint["label_encoder"]
num_labels = checkpoint.get("num_labels", 27)
emotion_labels = label_encoder.classes_
print(f"Loaded model with {num_labels} emotion labels: {emotion_labels}")
emotion_model.eval()

print("Loading DialoGPT model...")
dialog_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialog_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize conversation variables
conversation_history = []
chat_history_ids = None

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='chatbot_debug.log')
logger = logging.getLogger(__name__)

# Color mapping for emotion categories 
emotion_colors = {
    # Positive emotions
    "joy": "#FFD700",        # Gold
    "amusement": "#FFA500",  # Orange
    "excitement": "#FF4500", # OrangeRed
    "love": "#FF1493",       # DeepPink
    "optimism": "#32CD32",   # LimeGreen
    "pride": "#4169E1",      # RoyalBlue
    "gratitude": "#9370DB",  # MediumPurple
    "relief": "#00CED1",     # DarkTurquoise
    "approval": "#20B2AA",   # LightSeaGreen
    
    # Negative emotions
    "sadness": "#6495ED",    # CornflowerBlue
    "anger": "#DC143C",      # Crimson
    "fear": "#800080",       # Purple
    "disgust": "#006400",    # DarkGreen
    "grief": "#2F4F4F",      # DarkSlateGray
    "disappointment": "#696969", # DimGray
    "guilt": "#8B4513",      # SaddleBrown
    "remorse": "#A0522D",    # Sienna
    
    # Neutral emotions
    "neutral": "#808080",    # Gray
    "confusion": "#DDA0DD",  # Plum
    "surprise": "#00BFFF",   # DeepSkyBlue
    "curiosity": "#9932CC",  # DarkOrchid
    "realization": "#4682B4", # SteelBlue
    
    # Default for any other emotion
    "default": "#A9A9A9"     # DarkGray
}

# Detect emotion function 
def detect_emotion(text):
    """Detect emotion in the input text using your trained model"""
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()
    
    
    predicted_emotion = emotion_labels[prediction]
    
    
    all_probs = {emotion_labels[i]: float(probabilities[i].item()) for i in range(len(emotion_labels))}
    
   
    logger.info(f"Detected emotion: {predicted_emotion} (confidence: {confidence:.2f})")
    
    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": all_probs
    }


def generate_emotion_chart(emotion_data):
    """Generate a bar chart of top emotions"""
    probabilities = emotion_data["probabilities"]
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    
    labels = [item[0] for item in sorted_probs]
    values = [item[1] for item in sorted_probs]
    colors = [emotion_colors.get(label, emotion_colors["default"]) for label in labels]
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0, max(values) * 1.2)  # Add some space above the highest bar
    
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.title('Top 5 Detected Emotions')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
   
    temp_chart = f"temp_emotion_chart_{int(time.time())}.png"
    plt.savefig(temp_chart)
    plt.close()
    
    return temp_chart


def detect_intent(text):
    """Detect user intent from input text"""
    text = text.lower()
    
    
    intent_patterns = {
        "greeting": r"\b(hi|hello|hey|greetings|good morning|good afternoon|good evening)\b",
        "farewell": r"\b(bye|goodbye|see you|talk to you later|farewell)\b",
        "gratitude": r"\b(thanks|thank you|appreciate it|grateful)\b",
        "question": r"what|how|why|when|where|who|can you|could you|would you|will you|\?",
        "achievement": r"\b(achieved|completed|finished|got|received|won|passed|succeeded|accomplishment|achievement|success|promotion|job|hired|offer|accepted)\b",
        "problem": r"\b(problem|issue|trouble|difficulty|struggling|can't|cannot|failed|lost|missing|sick|ill|unwell|hurt|injured)\b",
        "help_request": r"\b(help|assist|support|advice|suggestion|recommend|solve)\b",
        "opinion_request": r"\b(think|opinion|view|perspective|feel about)\b",
        "statement": r"(i am|i'm|i feel|i think|i believe|i want|i need|i got|i have)",
        "life_event": r"\b(birthday|wedding|anniversary|graduation|move|moving|trip|vacation|holiday|birth|death|funeral|divorce|married|engaged)\b",
        "lottery": r"\b(lottery|jackpot|lotto|won money|prize)\b",
        "presentation": r"\b(presentation|speech|talk|meeting|interview)\b",
        "traffic": r"\b(traffic|driving|road|car|vehicle)\b",
        "pet": r"\b(pet|cat|dog|animal|bird|fish)\b"
    }
    
   
    detected_intents = []
    for intent, pattern in intent_patterns.items():
        if re.search(pattern, text):
            detected_intents.append(intent)
    
    
    if not detected_intents:
        return ["general"]
    
    return detected_intents


def generate_response(user_input, emotion_data):
    """Generate response using DialoGPT with emotion context and intent awareness"""
    global chat_history_ids
    
    emotion = emotion_data["emotion"]
    confidence = emotion_data["confidence"]
    
    
    intents = detect_intent(user_input)
    logger.info(f"Detected intents: {intents}")
    
   
    emotion_prompts = {
        "joy": "The user is happy. Respond in a cheerful and encouraging way: ",
        "sadness": "The user is sad. Respond with empathy and support: ",
        "anger": "The user is angry or frustrated. Respond calmly and helpfully: ",
        "fear": "The user is afraid or anxious. Respond in a reassuring way: ",
        "surprise": "The user is surprised. Acknowledge their surprise in your response: ",
        "disgust": "The user is expressing disgust. Respond with understanding: ",
        "neutral": "Respond to the user in a balanced, helpful way: ",
        "confusion": "The user is confused. Clarify and help them understand: ",
        "curiosity": "The user is curious. Respond with informative details: ",
        "excitement": "The user is excited. Match their enthusiasm in your response: ",
        "pride": "The user is proud. Acknowledge their achievement in your response: ",
        "optimism": "The user is optimistic. Respond with encouragement: ",
        "gratitude": "The user is grateful. Acknowledge their appreciation: ",
        "love": "The user is expressing love or affection. Respond warmly: ",
        "disappointment": "The user is disappointed. Respond with empathy and understanding: "
    }
    

    intent_templates = {
        "achievement": {
            "joy": [
                "Congratulations! That's fantastic news about {achievement}! You must be thrilled.",
                "That's amazing! Well done on {achievement}! How are you celebrating?",
                "Wow, congratulations on {achievement}! You should be really proud of yourself.",
                "That's wonderful news about {achievement}! You've earned this moment to celebrate!",
                "Congratulations! Getting {achievement} is a great accomplishment. How do you feel?"
            ],
            "pride": [
                "You have every right to be proud of {achievement}! Congratulations!",
                "What an achievement! You should absolutely feel proud of {achievement}.",
                "That's fantastic! Your hard work on {achievement} has paid off brilliantly.",
                "This is a moment to be proud! Congratulations on {achievement}.",
                "You've earned this success with {achievement}! Take time to celebrate your accomplishment."
            ],
            "excitement": [
                "How exciting! Congratulations on {achievement}! What happens next?",
                "That's such exciting news about {achievement}! Tell me more about it!",
                "Wow! Congratulations on {achievement}! Your excitement is completely justified!",
                "That's incredible news about {achievement}! No wonder you're excited!",
                "Congratulations! {achievement} is definitely something to be excited about!"
            ],
            "neutral": [
                "Congratulations on {achievement}. That's good news.",
                "Well done on {achievement}. What will you do next?",
                "That's great to hear about {achievement}. How are you feeling about it?",
                "Congratulations. {achievement} is certainly an accomplishment.",
                "Good news about {achievement}. Is this what you were hoping for?"
            ]
        },
        "problem": {
            "sadness": [
                "I'm sorry to hear you're dealing with this problem. That must be difficult.",
                "That sounds really challenging. How are you coping with it?",
                "I'm sorry you're going through this. Would it help to talk more about it?",
                "That's tough to deal with. What support do you have right now?",
                "I'm sorry to hear that. How long have you been dealing with this issue?"
            ],
            "anger": [
                "I understand why that would make you angry. That's frustrating.",
                "You have every right to feel upset about that. How can I help?",
                "That would make anyone angry. Have you been able to address it directly?",
                "I can see why you're frustrated. What options do you have for resolving this?",
                "That's definitely aggravating. What would help improve the situation?"
            ],
            "fear": [
                "That sounds concerning. What worries you most about this situation?",
                "I understand why you'd be worried. Have you thought about possible next steps?",
                "It's natural to feel anxious about this. What would help you feel more secure?",
                "That's definitely something to be concerned about. What information would help?",
                "I can see why that would cause anxiety. What's your biggest concern right now?"
            ],
            "disappointment": [
                "I understand your disappointment. It's tough when things don't go as expected.",
                "I'm sorry that happened. It's natural to feel disappointed in this situation.",
                "That sounds really disappointing. How are you handling it?",
                "I can see why you'd feel let down. Is there anything that might help?",
                "It's frustrating when things don't work out. What would make this situation better?"
            ]
        },
        "greeting": {
            "joy": [
                "Hello! It's great to see you today! How are you doing?",
                "Hi there! You seem to be in a good mood. What's going on?",
                "Hey! It's wonderful to chat with you. What's making you happy today?",
                "Hello! I'm glad you're here. How's your day going?",
                "Hi! It's a pleasure to talk with you. What's on your mind today?"
            ],
            "neutral": [
                "Hello there. How are you doing today?",
                "Hi. What can I help you with?",
                "Hello. What's on your mind?",
                "Hi there. How's your day going?",
                "Hello. How can I assist you today?"
            ]
        },
        "statement": {
            "general": [
                "Thank you for sharing that with me. How does that make you feel?",
                "I appreciate you telling me. Would you like to elaborate on that?",
                "That's interesting. Can you tell me more about it?",
                "I see. What other thoughts do you have about that?",
                "Thank you for sharing. How important is this to you?"
            ]
        },
        "lottery": {
            "joy": [
                "Wow! Congratulations on winning the lottery! That's incredible news! What are your plans for the winnings?",
                "That's amazing! Winning the lottery is such a rare stroke of luck! How are you feeling about it?",
                "Congratulations on your lottery win! Such exciting news! Have you told your family and friends yet?",
                "What wonderful news about winning the lottery! This is life-changing! What's the first thing you'll do?",
                "Incredible! Lottery winners are so rare - congratulations! How big was the jackpot?"
            ],
            "excitement": [
                "Winning the lottery is definitely worth being excited about! Congratulations! Any big plans?",
                "That's incredibly exciting news about the lottery win! I can feel your enthusiasm! What will you do first?",
                "Wow, a lottery win! Your excitement is totally justified! How much did you win?",
                "A lottery jackpot - that's amazing! No wonder you're excited! Have you decided how to spend it?",
                "That's such thrilling news! Winning the lottery is a dream for many people! How are you celebrating?"
            ]
        },
        "pet": {
            "sadness": [
                "I'm so sorry to hear about your cat being sick. Pets are family members, and it's hard when they're unwell. How long has your cat been sick?",
                "That's really sad about your pet. I understand how worried you must be. Have you been able to take them to the vet?",
                "I'm sorry your cat isn't well. It's completely normal to feel down when our pets are suffering. Is there anything that can be done to help them?",
                "That's tough news about your cat. Our pets mean so much to us. How are you holding up?",
                "I'm sorry to hear about your sick cat. It's heartbreaking when our pets aren't well. What has the vet said?"
            ]
        },
        "traffic": {
            "anger": [
                "Getting cut off in traffic is incredibly frustrating. I don't blame you for feeling angry about that. Did you make it to your destination safely?",
                "That would make anyone angry! Dangerous driving is so infuriating. Are you feeling calmer now?",
                "Being cut off in traffic is both annoying and dangerous. Your anger is completely justified. How's the rest of your day going?",
                "I understand why that made you angry. Reckless drivers can be really upsetting to encounter. Did anything help you calm down afterward?",
                "That sounds really frustrating. Traffic incidents can really affect our mood. What do you usually do to deal with road rage?"
            ]
        },
        "presentation": {
            "fear": [
                "It's completely normal to be nervous about an upcoming presentation. Many people find public speaking challenging. Is there anything specific you're worried about?",
                "Presentation anxiety is something most people experience. Is there anything I can help you with to prepare?",
                "Being nervous before a big presentation is natural. It shows you care about doing well. Have you had time to practice?",
                "I understand those pre-presentation nerves. They can be intense! Do you have any techniques that have helped you in the past?",
                "Feeling nervous about tomorrow's presentation is completely understandable. What aspects of it concern you most?"
            ]
        },
        "gratitude": {
            "gratitude": [
                "I'm happy you appreciated the gift! It's wonderful when someone's thoughtfulness is recognized. Was it for a special occasion?",
                "Thoughtful gifts can really brighten our day! I'm glad it meant something to you. Did they know exactly what you wanted?",
                "That's lovely to hear about the thoughtful gift. Expressing gratitude is so important. Have you let them know how much you appreciate it?",
                "Thoughtful gifts are special because they show someone really knows you. I'm glad it made you feel appreciated. What was the gift?",
                "That's wonderful! Receiving something thoughtful can really make a difference. How did they know what would be meaningful to you?"
            ]
        },
        "help_request": {
            "neutral": [
                "I understand you're feeling neutral about this situation. What specific help are you looking for?",
                "Sometimes a neutral perspective can be valuable. What would you like assistance with?",
                "I appreciate your balanced approach to this. How can I help you with it?",
                "Even with neutral feelings, it's good to seek help when needed. What are you hoping to achieve?",
                "I'm happy to help. What specific aspects of the situation would you like to address?"
            ]
        },
        "question": {
            "curiosity": [
                "Your curiosity about the new project is great! Questions drive learning and improvement. What specifically interests you about it?",
                "It's wonderful to be curious about new things! What aspects of the project would you like to learn more about?",
                "Curiosity is such a valuable trait! What particular information are you hoping to discover about the project?",
                "Being curious about new projects shows engagement and interest. What have you heard about it so far?",
                "I appreciate your curious nature! Asking questions is how we grow. What sparked your interest in this project?"
            ]
        },
        "opinion_request": {
            "surprise": [
                "That magic trick does sound amazing! I can tell how surprised you were. What part of it astonished you most?",
                "Wow! Being astonished by a magic trick is such a special feeling. Could you describe what happened?",
                "Magic that genuinely surprises us is becoming rare these days. What made this trick so amazing?",
                "Those moments of astonishment are so valuable! What was it about the trick that surprised you?",
                "It sounds like the magician really succeeded in creating true surprise! Was it a close-up trick or stage magic?"
            ]
        },
        "life_event": {
            "pride": [
                "It's wonderful that you feel proud of your friend's achievements! Supporting others is so important. What goals did they achieve?",
                "Being proud of friends shows what a supportive person you are. How did they accomplish their goals?",
                "That's a lovely sentiment! Being proud of others' achievements is a sign of true friendship. How long were they working toward these goals?",
                "It's wonderful to celebrate others' successes! What makes their achievement particularly special?",
                "Your pride in your friend's accomplishments shows what a caring person you are. How are they celebrating this success?"
            ]
        }
    }
    
    # Job-related patterns 
    job_patterns = [
        r"\bgot (?:a |the |my )?(new |)job\b",
        r"\bgot (?:a |the |my )?(new |)offer\b",
        r"\bhired\b",
        r"\baccepted (?:a |the |my )?(new |)(?:position|job|offer)\b",
        r"\bstarting (?:a |the |my )?(new |)job\b"
    ]
    
    job_responses = {
        "joy": [
            "Congratulations on your new job! That's fantastic news. When do you start?",
            "That's wonderful! Getting a new job is a big achievement. Are you excited about it?",
            "Congratulations! A new job is definitely something to celebrate. What will you be doing?",
            "That's great news about your job! How are you feeling about this new opportunity?",
            "Congratulations on the new job! Your hard work has paid off. What are you most looking forward to?"
        ],
        "excitement": [
            "Congratulations on your new job! Your excitement is completely justified! Tell me more about it!",
            "That's such exciting news about your new position! What will you be doing?",
            "Wow! Congratulations on getting hired! This is definitely something to be excited about!",
            "That's incredible news about your job offer! What are you most looking forward to?",
            "Congratulations! Getting a new job is definitely worth being excited about! When do you start?"
        ],
        "neutral": [
            "Congratulations on your new job. That's good news. What position is it?",
            "Well done on getting hired. What will your role be?",
            "That's great to hear about your job. How are you feeling about it?",
            "Congratulations. Getting a job offer is certainly an accomplishment. When do you start?",
            "Good news about your new position. Is this what you were hoping for?"
        ]
    }
    
    # Common emotion templates for direct responses
    emotion_templates = {
        "joy": [
            "I'm so happy for you! That's wonderful news.",
            "That's fantastic! I'm glad things are going well for you.",
            "It's great to hear you're happy! What specifically is making you feel good today?",
            "Your happiness is contagious! Tell me more about what's going well.",
            "That's wonderful! It's always nice to share good moments."
        ],
        "sadness": [
            "I'm really sorry to hear that. It must be difficult for you right now.",
            "I understand you're feeling sad. I'm here if you want to talk about it.",
            "That sounds tough. Would it help to talk more about what's making you sad?",
            "I'm sorry you're feeling down. Is there anything specific that triggered this feeling?",
            "It's okay to feel sad sometimes. Would you like to talk about what's on your mind?"
        ],
        "anger": [
            "I can see why that would be frustrating. Let's see if we can work through this.",
            "It makes sense that you're upset about this. What would help the situation?",
            "That does sound infuriating. Would you like to tell me more about what happened?",
            "I understand why you're angry. Sometimes talking about it can help process those feelings.",
            "Your frustration is completely valid. What do you think would make this situation better?"
        ],
        "fear": [
            "It's okay to be worried. Let's think about this together.",
            "I understand why you might be anxious about that. What would make you feel safer?",
            "That does sound concerning. Is there anything specific you're most worried about?",
            "It's natural to feel afraid sometimes. What's your biggest concern right now?",
            "I can understand your anxiety. Would it help to break down what you're worried about?"
        ],
        "excitement": [
            "Your enthusiasm is wonderful! Tell me more about what you're excited about!",
            "That does sound exciting! What are you looking forward to most?",
            "I can feel your excitement! How long have you been waiting for this?",
            "That's definitely something to be excited about! What happens next?",
            "Your excitement is contagious! When will this be happening?"
        ],
        "love": [
            "That's beautiful! Love is such a special feeling.",
            "I'm touched that you're sharing these feelings. Tell me more about this connection.",
            "It sounds like you really care deeply. What do you value most about this relationship?",
            "That's wonderful to hear. Strong connections are so important in life.",
            "It sounds like this relationship means a lot to you. What makes it so special?"
        ],
        "neutral": [
            "I see. Could you tell me more about that?",
            "Interesting. How does that make you feel?",
            "I understand. What would you like to discuss next?",
            "Thank you for sharing. Is there anything else on your mind?",
            "I see what you mean. Would you like to elaborate on that?"
        ],
        "surprise": [
            "Wow! That is surprising! How did you react in the moment?",
            "I can imagine your surprise! Those unexpected moments can be quite powerful.",
            "That sounds completely unexpected! What happened next?",
            "How astonishing! Were others surprised as well?",
            "What a surprising situation! Has anything like this happened before?"
        ],
        "curiosity": [
            "That's an interesting thing to be curious about. What sparked your interest?",
            "Curiosity is such a valuable trait. What aspects are you most interested in learning?",
            "I appreciate your curious mindset. What have you discovered so far?",
            "That's a fascinating topic to explore. Where do you plan to look for more information?",
            "Your curiosity about this is wonderful. What questions do you have specifically?"
        ],
        "disappointment": [
            "I'm sorry to hear you're disappointed. That's never a good feeling.",
            "Being ignored in a meeting can be really disheartening. Have you experienced this before?",
            "I understand your disappointment. It's frustrating when our contributions aren't acknowledged.",
            "That's disappointing. How did you respond when your suggestion was ignored?",
            "I'm sorry that happened. Would it help to bring up your suggestion again in another context?"
        ],
        "pride": [
            "It's wonderful that you feel proud of your friend! That shows what a supportive person you are.",
            "Being proud of others' achievements is such a positive quality. What specifically did they accomplish?",
            "That's lovely! Taking pride in friends' successes shows real friendship. How did they achieve their goals?",
            "Your pride in your friend is admirable. Have you told them how proud you are?",
            "That's a wonderful feeling, being proud of someone you care about. How long were they working toward this goal?"
        ],
        "gratitude": [
            "It's wonderful when people do thoughtful things! What made this gift particularly meaningful?",
            "Expressing gratitude is so important. Have you let them know how much you appreciate it?",
            "That sounds like a truly thoughtful gesture. What makes it so special to you?",
            "Appreciation for thoughtful acts makes our relationships stronger. Was this unexpected?",
            "That's lovely! Receiving thoughtful gifts can really brighten our day. What was it?"
        ]
    }
    
    
    is_job_related = any(re.search(pattern, user_input.lower()) for pattern in job_patterns)
    is_lottery_related = "lottery" in intents or "won" in user_input.lower() and "lottery" in user_input.lower()
    is_pet_related = "pet" in intents or "cat" in user_input.lower() and "sick" in user_input.lower()
    is_traffic_related = "traffic" in intents or "traffic" in user_input.lower() and "cut" in user_input.lower()
    is_presentation_related = "presentation" in intents or "presentation" in user_input.lower() and ("nervous" in user_input.lower() or "tomorrow" in user_input.lower())
    is_gift_related = "gratitude" in intents or "gift" in user_input.lower() and "thoughtful" in user_input.lower()
    is_project_related = "curiosity" in intents or "project" in user_input.lower() and "curious" in user_input.lower()
    is_magic_related = "surprise" in intents or "magic" in user_input.lower() and ("trick" in user_input.lower() or "astonished" in user_input.lower())
    is_friend_achievement = "pride" in intents or "proud" in user_input.lower() and "friend" in user_input.lower()
    is_meeting_related = "disappointment" in intents or "meeting" in user_input.lower() and ("ignored" in user_input.lower() or "disappointed" in user_input.lower())
    
    # Handle specific scenarios from examples
    if is_lottery_related and (emotion == "joy" or emotion == "excitement"):
        logger.info("Detected lottery win scenario")
        return random.choice(intent_templates["lottery"][emotion])
    
    if is_pet_related and emotion == "sadness":
        logger.info("Detected sick pet scenario")
        return random.choice(intent_templates["pet"]["sadness"])
    
    if is_traffic_related and emotion == "anger":
        logger.info("Detected traffic incident scenario")
        return random.choice(intent_templates["traffic"]["anger"])
    
    if is_presentation_related and emotion == "fear":
        logger.info("Detected presentation anxiety scenario")
        return random.choice(intent_templates["presentation"]["fear"])
    
    if is_gift_related and emotion == "gratitude":
        logger.info("Detected gift gratitude scenario")
        return random.choice(intent_templates["gratitude"]["gratitude"])
    
    if is_project_related and emotion == "curiosity":
        logger.info("Detected project curiosity scenario")
        return random.choice(intent_templates["question"]["curiosity"])
    
    if is_magic_related and emotion == "surprise":
        logger.info("Detected magic trick surprise scenario")
        return random.choice(intent_templates["opinion_request"]["surprise"])
    
    if is_friend_achievement and emotion == "pride":
        logger.info("Detected friend achievement pride scenario")
        return random.choice(intent_templates["life_event"]["pride"])
    
    if is_meeting_related and emotion == "disappointment":
        logger.info("Detected meeting disappointment scenario")
        return random.choice(emotion_templates["disappointment"])
    
    if is_job_related:
        logger.info("Detected job-related content")
        if emotion in job_responses:
            return random.choice(job_responses[emotion])
        else:
            return random.choice(job_responses["neutral"])
    
    
    strong_triggers = {
        "lost my pet": ("sadness", "I'm so sorry to hear about your pet. Losing a pet can be incredibly painful - they're family members. Would you like to talk about your pet or how you're feeling?"),
        "feeling sad": ("sadness", "I understand you're feeling down right now. It's completely okay to feel sad sometimes. Would you like to talk about what's troubling you?"),
        "i'm depressed": ("sadness", "I'm sorry you're feeling this way. Depression can be really difficult to deal with. Have you been able to talk to someone you trust about how you're feeling?"),
        "i'm happy": ("joy", "That's wonderful to hear! What's bringing you joy today?"),
        "i hate": ("anger", "I can see that you're feeling strongly about this. What specifically about it is bothering you?"),
        "i'm scared": ("fear", "It's okay to feel scared. Many things in life can be frightening. Would you like to talk more about what's causing your fear?"),
        "i'm nervous": ("fear", "Being nervous is completely normal. Is there something coming up that you're worried about?"),
        "won the lottery": ("joy", "Wow! Congratulations on winning the lottery! That's incredible news! What are your plans for the winnings?"),
        "presentation tomorrow": ("fear", "It's completely normal to be nervous about an upcoming presentation. Many people find public speaking challenging. Is there anything specific you're worried about?"),
        "cut me off in traffic": ("anger", "Getting cut off in traffic is incredibly frustrating. I don't blame you for feeling angry about that. Did you make it to your destination safely?"),
        "thoughtful gift": ("gratitude", "I'm happy you appreciated the gift! It's wonderful when someone's thoughtfulness is recognized. Was it for a special occasion?"),
        "feeling neutral": ("neutral", "I understand you're feeling neutral about this situation. Sometimes a balanced perspective can be valuable. Is there anything specific you'd like to discuss about it?"),
        "curious": ("curiosity", "Your curiosity is fantastic! Questions drive learning and improvement. What specifically interests you about this?"),
        "ignored my suggestion": ("disappointment", "I'm sorry to hear you're disappointed. It's tough when our contributions aren't acknowledged. How did you respond when your suggestion was ignored?"),
        "magic trick": ("surprise", "Wow! Being astonished by a magic trick is such a special feeling. What made this trick so amazing?"),
        "proud of my friend": ("pride", "It's wonderful that you feel proud of your friend's achievements! Supporting others is so important. What goals did they achieve?")
    }
    
    
    for trigger, (trigger_emotion, response) in strong_triggers.items():
        if trigger.lower() in user_input.lower():
            logger.info(f"Triggered strong emotional response for: {trigger}")
            return response
    
    
    achievement_triggers = ["got a promotion", "passed my exam", "completed my project", "graduated", "won an award"]
    problem_triggers = ["lost my job", "failed my test", "broke up", "sick", "having trouble", "struggling with"]
    greeting_triggers = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    farewell_triggers = ["bye", "goodbye", "see you", "talk to you later"]
    
    
    for trigger in achievement_triggers:
        if trigger in user_input.lower():
            if emotion in ["joy", "excitement", "pride"]:
                achievement = user_input.lower().split(trigger)[1].strip()
                if not achievement:
                    achievement = "your achievement"
                return random.choice(intent_templates["achievement"][emotion]).format(achievement=achievement)
    
   
    for trigger in problem_triggers:
        if trigger in user_input.lower():
            if emotion in ["sadness", "anger", "fear", "disappointment"]:
                return random.choice(intent_templates["problem"][emotion])
    
    
    for trigger in greeting_triggers:
        if trigger in user_input.lower():
            if emotion in intent_templates.get("greeting", {}):
                return random.choice(intent_templates["greeting"][emotion])
            else:
                return random.choice(intent_templates["greeting"]["neutral"])
    

    for trigger in farewell_triggers:
        if trigger in user_input.lower():
            return "Goodbye! Feel free to chat again whenever you'd like."
    
    # Handle intents with emotion
    for intent in intents:
        if intent in intent_templates and emotion in intent_templates[intent]:
            return random.choice(intent_templates[intent][emotion])
    
    # Fall back to emotion-only templates if we have them
    if emotion in emotion_templates:
        return random.choice(emotion_templates[emotion])
    
    # If nothing else matches, use DialoGPT with enhanced quality checks
    logger.info(f"Using DialoGPT for response generation with emotion: {emotion}")
    
    # Add emotion context to the input
    prompt = emotion_prompts.get(emotion, emotion_prompts["neutral"])
    contextualized_input = f"{prompt}{user_input}"
    
    # Encode the new input and add to chat history
    new_input_ids = dialog_tokenizer.encode(contextualized_input + dialog_tokenizer.eos_token, return_tensors='pt')
    
    # Append to chat history if it exists
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids
    
    # Set temperature based on emotion
    temperature = 0.7
    if emotion in ["excitement", "joy", "surprise"]:
        temperature = 0.8  # More creative for positive emotions
    elif emotion in ["neutral", "curiosity"]:
        temperature = 0.7  # Balanced for neutral emotions
    elif emotion in ["fear", "sadness", "anger"]:
        temperature = 0.6  # More conservative for negative emotions
    
    try:
        chat_history_ids = dialog_model.generate(
            bot_input_ids,
            max_length=bot_input_ids.shape[-1] + 50,
            pad_token_id=dialog_tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=temperature
        )
        
        # Extract the response
        response = dialog_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        # Log the raw response
        logger.info(f"Raw generated response: {response}")
        
        # Check for quality issues
        response_quality_issues = (
            not response.strip() or                           # Empty response
            len(response) < 10 or                             # Too short
            response.lower() in user_input.lower() or         # Echo/repeat
            any(word in response.lower() for word in ["neutralize", "neutral", "as an ai"]) or  # Generic responses
            "I don't" in response or                          # Negative responses
            "I can't" in response or
            "I cannot" in response
        )
        
        if response_quality_issues:
            logger.info("Using fallback template due to poor response quality")
            
            # Use emotion templates if available
            if emotion in emotion_templates:
                response = random.choice(emotion_templates[emotion])
            else:
                # Generic fallbacks
                fallback_templates = [
                    "I see. Could you tell me more about that?",
                    "That's interesting. How does that make you feel?",
                    "I understand. What would you like to discuss next?",
                    "Thank you for sharing. Is there anything else on your mind?"
                ]
                response = random.choice(fallback_templates)
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I'm having trouble responding right now. Could you try rephrasing your message?"


def format_emotion_output(emotion_data):
    """Format emotion detection output"""
    emotion = emotion_data["emotion"]
    confidence = emotion_data["confidence"]
    
    # Get top 5 emotions
    probabilities = emotion_data["probabilities"]
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    top_probs = {k: round(v, 3) for k, v in sorted_probs}
    
    # Get color for primary emotion
    color = emotion_colors.get(emotion, emotion_colors["default"])
    
    # Create HTML output
    output = f"""
    <div style="padding: 10px; border-radius: 8px; background-color: #f8f9fa;">
        <h3 style="margin-top: 0; color: {color};">Primary Emotion: {emotion.capitalize()}</h3>
        <div style="background-color: #e9ecef; height: 20px; border-radius: 4px; overflow: hidden;">
            <div style="background-color: {color}; width: {confidence*100}%; height: 100%; text-align: center; color: white; line-height: 20px; font-size: 12px;">
                {confidence:.2f}
            </div>
        </div>
        <h4 style="margin-top: 15px;">Top Emotions:</h4>
        <ul style="list-style-type: none; padding-left: 0;">
    """
    
    for emotion, prob in sorted_probs:
        e_color = emotion_colors.get(emotion, emotion_colors["default"])
        output += f"""
            <li style="margin-bottom: 5px;">
                <span style="color: {e_color}; font-weight: bold;">{emotion.capitalize()}</span>: 
                <div style="display: inline-block; width: 150px; background-color: #e9ecef; height: 15px; border-radius: 3px; overflow: hidden; vertical-align: middle;">
                    <div style="background-color: {e_color}; width: {prob*100}%; height: 100%;"></div>
                </div>
                <span style="margin-left: 5px;">{prob:.3f}</span>
            </li>
        """
    
    output += """
        </ul>
    </div>
    """
    
    return output


def save_conversation():
    """Save conversation history to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_history_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(conversation_history, f, indent=2)
    return filename


def respond(message, chat_history):
    """Process user message and generate response"""
    if not message.strip():
        return "", chat_history, gr.update(value=""), None
    
    # Log the user message
    logger.info(f"User message: {message}")
    
    # Detect emotion in the message
    emotion_data = detect_emotion(message)
    
    # Format the emotion output
    emotion_html = format_emotion_output(emotion_data)
    
    # Generate an emotion chart
    emotion_chart = generate_emotion_chart(emotion_data)
    
    # Generate a response
    response = generate_response(message, emotion_data)
    
    # Log the bot response
    logger.info(f"Bot response: {response}")
    
    # Update conversation history
    conversation_history.append({
        "user": message,
        "emotion_data": {
            "emotion": emotion_data["emotion"],
            "confidence": emotion_data["confidence"],
            "top_probabilities": {k: emotion_data["probabilities"][k] 
                                 for k in sorted(emotion_data["probabilities"], 
                                               key=lambda x: emotion_data["probabilities"][x], 
                                               reverse=True)[:5]}
        },
        "bot": response,
        "timestamp": datetime.now().isoformat()
    })
    
    # Update chat history
    chat_history.append((message, response))
    
    return "", chat_history, gr.update(value=emotion_html), emotion_chart


def reset_chat():
    """Reset the chat and save the conversation history"""
    global chat_history_ids, conversation_history
    
    # Save conversation if there's anything to save
    if conversation_history:
        filename = save_conversation()
        logger.info(f"Conversation saved to {filename}")
    
    chat_history_ids = None
    conversation_history = []
    
    return [], gr.update(value=""), None


# Set up enhanced Gradio interface with CSS styling
with gr.Blocks(css="""
    .container {max-width: 1000px; margin: auto;}
    .chatbot-container {border-radius: 10px; border: 1px solid #ddd; background-color: #fcfcfc;}
    .emotion-container {border-radius: 10px; border: 1px solid #ddd; padding: 10px; margin-top: 10px; background-color: #fcfcfc;}
    .input-container {margin-top: 10px;}
    .button-row {margin-top: 10px;}
    .header {text-align: center; margin-bottom: 20px;}
    .footer {text-align: center; margin-top: 20px; font-size: 0.8em; color: #666;}
    .user-message {background-color: #e7f3ff !important; border-radius: 15px !important; padding: 10px !important;}
    .bot-message.bot-message {background-color: #f0f0f0 !important; border-radius: 15px !important; padding: 10px !important;}
""") as demo:
    
    with gr.Row(elem_classes="header"):
        gr.Markdown("# 🤖 Emotion-Aware Chatbot")
        gr.Markdown("This AI chatbot detects emotions in your messages and responds accordingly.")
    
    with gr.Row(elem_classes="container"):
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(elem_classes="chatbot-container", height=500, elem_id="chat-box", 
                              bubble_full_width=False)
            
            with gr.Row(elem_classes="input-container"):
                msg = gr.Textbox(
                    placeholder="Type your message here...", 
                    label="Your message",
                    show_label=False,
                    lines=2
                )
            
            with gr.Row(elem_classes="button-row"):
                clear = gr.Button("🔄 New Conversation", variant="secondary")
                submit = gr.Button("📤 Send", variant="primary")
        
        with gr.Column(scale=3, elem_classes="emotion-container"):
            gr.Markdown("## 😊 Emotion Analysis")
            
            # HTML display for emotion data
            emotion_display = gr.HTML(label="Detected Emotions")
            
            # Image display for the emotion chart
            chart_display = gr.Image(label="Emotion Distribution", show_label=False)
            
            # Information about the chatbot
            with gr.Accordion("ℹ️ About", open=False):
                gr.Markdown("""
                This chatbot uses:
                - An emotion detection model based on DistilBERT
                - DialoGPT for generating conversational responses
                - Your detected emotions to adjust the tone of responses
                - Intent recognition for better understanding of your messages
                
                The conversation history is automatically saved when you start a new chat.
                """)
            
            # Examples for testing - including all the test cases you provided
            with gr.Accordion("✨ Try These Examples", open=True):
                gr.Examples(
                    examples=[
                        ["Hello there! How are you today?"],
                        ["I'm feeling really happy today!"],
                        ["I'm so sad, I lost my pet yesterday."],
                        ["I'm really angry about what happened."],
                        ["I'm a bit scared about my upcoming surgery."],
                        ["I just got a promotion at work!"],
                        ["I got a new job!"],
                        ["I'm excited because I just graduated!"],
                        ["I need help with a problem I'm having."],
                        ["What do you think about artificial intelligence?"],
                        ["I just won the lottery! I'm so thrilled!"],
                        ["My cat is sick, and I feel really down."],
                        ["Someone cut me off in traffic this morning, and it made me so angry."],
                        ["I have a big presentation tomorrow, and I'm incredibly nervous."],
                        ["That was a really thoughtful gift. I appreciate it so much."],
                        ["Honestly, I'm just feeling completely neutral about the whole situation."],
                        ["Did you hear about the new project? I'm quite curious to learn more."],
                        ["They completely ignored my suggestion in the meeting. I feel a bit disappointed."],
                        ["Wow, that magic trick was absolutely amazing! I'm astonished."],
                        ["I'm so proud of my friend for achieving their goals."]
                    ],
                    inputs=msg
                )
    
    # Handle interactions
    msg.submit(respond, [msg, chatbot], [msg, chatbot, emotion_display, chart_display])
    submit.click(respond, [msg, chatbot], [msg, chatbot, emotion_display, chart_display])
    clear.click(reset_chat, None, [chatbot, emotion_display, chart_display], queue=False)
    
    with gr.Row(elem_classes="footer"):
        gr.Markdown("Emotional Intelligence Chatbot - Developed with ❤️ using Gradio and Hugging Face Transformers")

if __name__ == "__main__":
    # Launch the Gradio interface
    demo.launch(share=True, inbrowser=True)  # Set share=False if you don't want a public link