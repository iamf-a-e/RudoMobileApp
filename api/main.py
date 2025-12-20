# api/main.py

import os
import json
import re
import random
import string
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import google.generativeai as genai
from upstash_redis import Redis
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin

# =====================
# App & Logging
# =====================
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
application = app

# =====================
# Firebase Initialization
# =====================
try:
    # Initialize Firebase (for Firestore)
    if not firebase_admin._apps:
        # Use environment variable or service account JSON
        firebase_creds = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if firebase_creds:
            cred = credentials.Certificate(json.loads(firebase_creds))
        else:
            # For local development, you might have a file
            cred = credentials.Certificate("serviceAccountKey.json")
        
        initialize_app(cred)
    
    firestore_db = firestore.client()
    logging.info("Firebase Firestore initialized")
except Exception as e:
    logging.error(f"Firebase initialization failed: {e}")
    firestore_db = None

# =====================
# Environment & Config
# =====================
GEN_API = os.environ.get("GEN_API")
if not GEN_API:
    raise RuntimeError("GEN_API environment variable not set")

genai.configure(api_key=GEN_API)
MODEL_NAME = "gemini-2.5-flash"

# =====================
# Redis (Upstash)
# =====================
redis_client = None
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

if redis_url and redis_token:
    try:
        redis_client = Redis(url=redis_url, token=redis_token)
        redis_client.ping()
        logging.info("Connected to Upstash Redis")
    except Exception as e:
        logging.error(f"Redis connection failed: {e}")
        redis_client = None

# =====================
# Import Instructions & Data
# =====================
try:
    from .instructions import instructions
    from .products_data import products_data
    from .pregnancy_data import pregnancy_data
    from .pregnancy_data_shona import pregnancy_data_shona
    from .pregnancy_data_ndebele import pregnancy_data_ndebele
    from .pregnancy_data_tonga import pregnancy_data_tonga
    from .pregnancy_data_chinyanja import pregnancy_data_chinyanja
    from .pregnancy_data_bemba import pregnancy_data_bemba
    from .pregnancy_data_lozi import pregnancy_data_lozi
    from .cervical_cancer_data import cervical_cancer_data
    
    # Create language maps from the imported data
    maternal_map = {
        "english": pregnancy_data,
        "shona": pregnancy_data_shona,
        "ndebele": pregnancy_data_ndebele,
        "tonga": pregnancy_data_tonga,
        "chinyanja": pregnancy_data_chinyanja,
        "bemba": pregnancy_data_bemba,
        "lozi": pregnancy_data_lozi
    }
    
    cancer_map = {
        "english": cervical_cancer_data
    }
    
    logging.info("Successfully imported all data files")
    
except Exception as e:
    logging.error(f"Failed to import data files: {e}")
    # Create empty structures if imports fail
    maternal_map = {}
    cancer_map = {}
    products_data = ""
    instructions = ""

# =====================
# Global State (cached)
# =====================
user_states = {}
chat_sessions = {}

# =====================
# Helpers: State Persistence
# =====================

def load_user_states():
    global user_states
    if not redis_client:
        return
    try:
        data = redis_client.get("user_states")
        user_states = json.loads(data) if data else {}
    except Exception as e:
        logging.error(f"Failed to load states: {e}")
        user_states = {}

def save_user_states():
    if not redis_client:
        return
    try:
        redis_client.set("user_states", json.dumps(user_states))
    except Exception as e:
        logging.error(f"Failed to save states: {e}")

# =====================
# Language Detection
# =====================

def detect_language(text: str) -> str:
    language_keywords = {
        "english": ["hie", "hi", "hey", "hello", "good morning", "good afternoon"],
        "shona": ["mhoro", "mhoroi", "makadini", "hesi"],
        "ndebele": ["sawubona", "unjani", "salibonani"],
        "tonga": ["mwabuka buti", "mwalibizya buti", "kwasiya", "mulibuti"],
        "chinyanja": ["bwanji", "muli bwanji", "mukuli bwanji"],
        "bemba": ["muli shani", "mulishani", "mwashibukeni"],
        "lozi": ["muzuhile", "mutozi", "muzuhile cwani"]
    }
    
    text_lower = text.lower()
    for lang, keywords in language_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return lang
    return "english"

# =====================
# Gemini AI Helper
# =====================

def get_gemini_response(user_message: str, user_id: str, language: str = "english") -> str:
    """Get response from Gemini AI using the instructions"""
    try:
        # Build context from user state
        state = user_states.get(user_id, {})
        context = f"""
        User Language: {language}
        User ID: {user_id}
        Current Step: {state.get('step', 'new')}
        User Registered: {state.get('registered', False)}
        User Code: {state.get('user_code', 'Not registered')}
        
        Previous Messages: {state.get('chat_history', [])[-3:] if state.get('chat_history') else 'None'}
        
        Instructions to follow:
        {instructions}
        
        Available Pregnancy Data by Language:
        {json.dumps({k: "Available" for k in maternal_map.keys()}, indent=2)}
        
        Available Cervical Cancer Data:
        {"Available" if cancer_map.get('english') else "Not available"}
        
        Products Data:
        {products_data[:500]}...  # Truncated for context
        
        User Message: {user_message}
        
        Respond according to the instructions above. Remember your identity is Rudo, Dawa Health's Virtual Assistant.
        """
        
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(context)
        
        # Save to chat history
        if user_id not in chat_sessions:
            chat_sessions[user_id] = []
        
        chat_sessions[user_id].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        chat_sessions[user_id].append({
            "role": "assistant",
            "content": response.text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit chat history to last 20 messages
        if len(chat_sessions[user_id]) > 20:
            chat_sessions[user_id] = chat_sessions[user_id][-20:]
        
        return response.text.strip()
        
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        return "I apologize, but I'm having trouble processing your request. Please try again."

# =====================
# Firestore Helpers
# =====================

def save_message_to_firestore(user_id: str, message: str, is_user: bool = True):
    """Save message to Firestore"""
    if not firestore_db:
        logging.warning("Firestore not initialized, skipping save")
        return
    
    try:
        chat_ref = firestore_db.collection("whatsapp_chats").document(user_id)
        messages_ref = chat_ref.collection("messages")
        
        # Ensure chat document exists
        chat_ref.set({
            "user_id": user_id,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "last_message": message[:100]  # Truncate long messages
        }, merge=True)
        
        # Add message
        messages_ref.add({
            "content": message,
            "is_user": is_user,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "created_at": datetime.now().isoformat()
        })
        
        logging.info(f"Message saved to Firestore for user {user_id}")
    except Exception as e:
        logging.error(f"Failed to save to Firestore: {e}")

def get_chat_history(user_id: str, limit: int = 50):
    """Get chat history from Firestore"""
    if not firestore_db:
        return []
    
    try:
        messages_ref = firestore_db.collection("whatsapp_chats").document(user_id).collection("messages")
        query = messages_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit)
        docs = query.stream()
        
        messages = []
        for doc in docs:
            data = doc.to_dict()
            messages.append({
                "id": doc.id,
                "content": data.get("content", ""),
                "is_user": data.get("is_user", False),
                "timestamp": data.get("timestamp", ""),
                "created_at": data.get("created_at", "")
            })
        
        # Return in chronological order
        messages.reverse()
        return messages
    except Exception as e:
        logging.error(f"Failed to get chat history: {e}")
        return []

# =====================
# Core Conversation Logic
# =====================

def handle_chat_message(user_id: str, message: str, phone_number: str = None) -> str:
    """Main chat handler for mobile app"""
    load_user_states()
    
    # Initialize user state if not exists
    if user_id not in user_states:
        language = detect_language(message)
        user_states[user_id] = {
            "step": "language_detection",
            "language": language,
            "registered": False,
            "user_code": None,
            "phone_number": phone_number,
            "chat_history": [],
            "last_active": datetime.now().isoformat()
        }
    
    state = user_states[user_id]
    state["last_active"] = datetime.now().isoformat()
    
    # Get AI response
    ai_response = get_gemini_response(message, user_id, state["language"])
    
    # Update chat history
    state["chat_history"] = state.get("chat_history", [])[-9:] + [
        {"role": "user", "message": message},
        {"role": "assistant", "message": ai_response}
    ]
    
    # Save message to Firestore
    save_message_to_firestore(user_id, message, is_user=True)
    save_message_to_firestore(user_id, ai_response, is_user=False)
    
    save_user_states()
    
    return ai_response

# =====================
# API Routes
# =====================

@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint for mobile app"""
    try:
        data = request.get_json(force=True)
        
        user_id = data.get("user_id")
        message = data.get("message")
        session_id = data.get("session_id")
        phone_number = data.get("phone_number")
        
        if not user_id and session_id:
            user_id = session_id
        
        if not user_id or not message:
            return jsonify({"error": "user_id and message are required"}), 400
        
        logging.info(f"Chat request from user {user_id}: {message[:50]}...")
        
        # Get AI response
        response = handle_chat_message(user_id, message, phone_number)
        
        return jsonify({
            "reply": response,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/send-message", methods=["POST"])
def send_message():
    """Endpoint for sending messages (used by mobile app)"""
    try:
        data = request.get_json(force=True)
        
        user_id = data.get("user_id")
        phone_number = data.get("phone_number")
        message = data.get("message")
        
        if not user_id or not message:
            return jsonify({"error": "user_id and message are required"}), 400
        
        logging.info(f"Sending message from user {user_id} ({phone_number}): {message[:50]}...")
        
        # Get AI response
        response = handle_chat_message(user_id, message, phone_number)
        
        return jsonify({
            "success": True,
            "message": "Message sent successfully",
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Send message error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat-history/<user_id>", methods=["GET"])
def chat_history(user_id):
    """Get chat history for a user"""
    try:
        limit = request.args.get("limit", default=50, type=int)
        
        messages = get_chat_history(user_id, limit)
        
        return jsonify({
            "user_id": user_id,
            "messages": messages,
            "count": len(messages),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Chat history error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini
        model = genai.GenerativeModel(MODEL_NAME)
        test_response = model.generate_content("Health check")
        
        # Test Firestore if available
        firestore_status = "connected" if firestore_db else "not_configured"
        
        # Test Redis if available
        redis_status = "connected" if redis_client else "not_configured"
        
        return jsonify({
            "status": "healthy",
            "service": "Dawa Health AI Chat API",
            "model": MODEL_NAME,
            "firestore": firestore_status,
            "redis": redis_status,
            "timestamp": datetime.now().isoformat(),
            "test_response": test_response.text[:50] if test_response.text else "No response"
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/", methods=["GET"])
def home():
    """Home endpoint"""
    return jsonify({
        "service": "Dawa Health AI Chat API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/chat": "Main chat endpoint",
            "POST /api/send-message": "Send message endpoint",
            "GET /api/chat-history/<user_id>": "Get chat history",
            "GET /api/health": "Health check"
        }
    })

# =====================
# Application Startup
# =====================

if __name__ == "__main__":
    load_user_states()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)



