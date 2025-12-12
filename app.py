import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
import requests
import os
import fitz
import sched
import time 
import logging
from mimetypes import guess_type
from datetime import datetime, timedelta
from urlextract import URLExtract
from training import instructions, product_images
from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from google.api_core.exceptions import ResourceExhausted
from training import products, instructions, pregnancy_data, pregnancy_data_shona, pregnancy_data_ndebele, pregnancy_data_tonga, pregnancy_data_chinyanja, pregnancy_data_bemba, pregnancy_data_lozi, cervical_cancer_data
from products_data import products_by_category
from upstash_redis import Redis  # ✅ Upstash Redis client
import json
import re
import random
import string

logging.basicConfig(level=logging.INFO)

# Initialize Upstash Redis connection
redis_url = os.environ.get("UPSTASH_REDIS_URL")
redis_token = os.environ.get("UPSTASH_REDIS_TOKEN")

if redis_url and redis_token:
    try:
        redis_client = Redis(url=redis_url, token=redis_token)
        # Test connection
        redis_client.ping()
        logging.info("Successfully connected to Upstash Redis")
    except Exception as e:
        logging.error(f"Failed to connect to Upstash Redis: {e}")
        redis_client = None
else:
    redis_client = None
    logging.warning("UPSTASH_REDIS_URL or UPSTASH_REDIS_TOKEN not set, Redis functionality disabled")

# Global user states dictionary
user_states = {}

wa_token = os.environ.get("WA_TOKEN")
phone_id = os.environ.get("PHONE_ID")
gen_api = os.environ.get("GEN_API")
owner_phone = os.environ.get("OWNER_PHONE")
model_name = "gemini-2.0-flash"
genai.configure(api_key=gen_api)
name = "Fae"
bot_name = "Rudo"
AGENT = "+263719835124"

app = Flask(__name__)
genai.configure(api_key=gen_api)

class CustomURLExtract(URLExtract):
    def _get_cache_file_path(self):
        cache_dir = "/tmp"
        return os.path.join(cache_dir, "tlds-alpha-by-domain.txt")

extractor = CustomURLExtract(limit=1)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(model_name=model_name,
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat(history=[])

def save_user_states():
    """Save all user states to Upstash Redis"""
    if redis_client:
        try:
            redis_client.set("user_states", json.dumps(user_states))
            logging.info("User states saved to Redis")
        except Exception as e:
            logging.error(f"Error saving user states: {e}")

def load_user_states():
    """Load all user states from Upstash Redis"""
    global user_states
    if redis_client:
        try:
            states_data = redis_client.get("user_states")
            if states_data:
                user_states = json.loads(states_data)
                logging.info("User states loaded from Redis")
            else:
                user_states = {}
                logging.info("No user states found in Redis, initializing empty")
        except Exception as e:
            logging.error(f"Error loading user states: {e}")
            user_states = {}
    else:
        user_states = {}

def get_user_conversation(sender):
    """Get user conversation history from Upstash Redis"""
    if redis_client:
        try:
            history = redis_client.get(f"conversation:{sender}")
            return json.loads(history) if history else []
        except Exception as e:
            logging.error(f"Error getting conversation: {e}")
            return []
    return []

def save_user_conversation(sender, role, message):
    """Save user conversation to Upstash Redis"""
    if redis_client:
        try:
            conversation = get_user_conversation(sender)
            conversation.append({
                "role": role,
                "message": message,
                "timestamp": datetime.now().isoformat()
            })
            if len(conversation) > 100:
                conversation = conversation[-100:]
            redis_client.set(f"conversation:{sender}", json.dumps(conversation), ex=60*60*24*30)
            logging.debug(f"Saved conversation for {sender}")
        except Exception as e:
            logging.error(f"Error saving conversation: {e}")

def detect_language(message):
    language_keywords = {
        "english": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
        "shona": ["mhoro", "mhoroi", "makadini", "hesi", "ndinonzi"],
        "ndebele": ["sawubona", "unjani", "salibonani", "yebo"],
        "tonga": ["mwabuka buti", "mwalibizya buti", "kwasiya", "mulibuti"],
        "chinyanja": ["bwanji", "muli bwanji", "mukuli bwanji", "moni"],
        "bemba": ["muli shani", "mulishani", "mwashibukeni", "shani"],
        "lozi": ["muzuhile", "mutozi", "muzuhile cwani", "lwani"]
    }
    message_lower = message.lower()
    for lang, keywords in language_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            return lang
    return "english"

def send(answer, sender, phone_id):
    url = f"https://graph.facebook.com/v19.0/{phone_id}/messages"
    headers = {
        'Authorization': f'Bearer {wa_token}',
        'Content-Type': 'application/json'
    }
    type = "text"
    body = "body"
    content = answer
    image_urls = product_images.image_urls

    if "product_image" in answer:
        product_match = re.search(r'product_image_(\w+)', answer)
        if product_match:
            product_name = product_match.group(1)
            if product_name in image_urls:
                image_url = image_urls[product_name]
                mime_type, _ = guess_type(image_url.split("/")[-1])
                if mime_type and mime_type.startswith("image"):
                    type = "image"
                    body = "link"
                    content = image_url
                    answer = re.sub(r'product_image_\w+', '', answer)

    data = {
        "messaging_product": "whatsapp",
        "to": sender,
        "type": type,
        type: {
            body: content,
            **({"caption": answer.strip()} if type != "text" else {})
        },
    }

    response = requests.post(url, headers=headers, json=data)
    save_user_conversation(sender, "bot", answer)
    return response

def remove(*file_paths):
    for file in file_paths:
        if os.path.exists(file):
            os.remove(file)

def reset_conversation(sender):
    """Reset user conversation state"""
    if sender in user_states:
        # Keep registration info but reset conversation flow
        user_states[sender].update({
            "step": "main_menu",
            "topic": None,
            "conversation_history": []
        })
        save_user_states()
    
    # Clear conversation history from Redis
    if redis_client:
        try:
            redis_client.delete(f"conversation:{sender}")
            logging.info(f"Conversation reset for {sender}")
        except Exception as e:
            logging.error(f"Error clearing conversation history: {e}")

# Database setup (optional)
db = False
if os.environ.get("DB_URL"):
    try:
        db = True
        db_url = os.environ.get("DB_URL")
        if db_url and "redis" not in db_url:
            engine = create_engine(db_url)
            Session = sessionmaker(bind=engine)
            Base = declarative_base()
            scheduler = sched.scheduler(time.time, time.sleep)
            report_time = datetime.now().replace(hour=22, minute=00, second=0, microsecond=0)

            class Chat(Base):
                __tablename__ = 'chats'
                Chat_no = Column(Integer, primary_key=True)
                Sender = Column(String(255), nullable=False)
                Message = Column(String, nullable=False)
                Chat_time = Column(DateTime, default=datetime.utcnow)

            logging.info("Creating tables if they do not exist...")
            Base.metadata.create_all(engine)

            def insert_chat(sender, message):
                logging.info("Inserting chat into database")
                try:
                    session = Session()
                    chat = Chat(Sender=sender, Message=message)
                    session.add(chat)
                    session.commit()
                    logging.info("Chat inserted successfully")
                except Exception as e:
                    logging.error(f"Error inserting chat: {e}")
                    session.rollback()
                finally:
                    session.close()

            def get_chats(sender):
                try:
                    session = Session()
                    chats = session.query(Chat.Message).filter(Chat.Sender == sender).all()
                    return [chat[0] for chat in chats]
                except Exception as e:
                    logging.error(f"Error getting chats: {e}")
                    return []
                finally:
                    session.close()

            def delete_old_chats():
                try:
                    session = Session()
                    cutoff_date = datetime.now() - timedelta(days=14)
                    session.query(Chat).filter(Chat.Chat_time < cutoff_date).delete()
                    session.commit()
                    logging.info("Old chats deleted successfully")
                except Exception as e:
                    logging.error(f"Error deleting old chats: {e}")
                    session.rollback()
                finally:
                    session.close()

            def create_report(phone_id):
                logging.info("Creating report")
                try:
                    today = datetime.today().strftime('%d-%m-%Y')
                    session = Session()
                    query = session.query(Chat.Message).filter(func.date_trunc('day', Chat.Chat_time) == today).all()
                    if query:
                        chats = '\n\n'.join([chat[0] for chat in query])
                        send(chats, owner_phone, phone_id)
                except Exception as e:
                    logging.error(f"Error creating report: {e}")
                finally:
                    session.close()
        else:
            logging.warning("DB_URL appears to be a Redis URL, SQLAlchemy database disabled")
            db = False
    except Exception as e:
        logging.error(f"Error setting up database: {e}")
        db = False
else:
    db = False
    logging.info("DB_URL not set, database functionality disabled")


def handle_language_detection(sender, prompt, phone_id):
    detected_lang = detect_language(prompt)
    user_states[sender]["language"] = detected_lang
    user_states[sender]["step"] = "registration"
    user_states[sender]["needs_language_confirmation"] = False

    if detected_lang == "shona":
        send("Mhoro! Ndinonzi Rudo, mubatsiri wepamhepo weDawa Health. Reggai titange nekunyoresa. Ndapota ndipe manhamba mana ekupedzisira enhare yenyu.", sender, phone_id)
    elif detected_lang == "ndebele":
        send("Sawubona! Ngingu Rudo, isiphathamandla se-Dawa Health. Masige saqala ngokubhalisa. Ngicela unginike amadijithi amane okugcina efoni yakho.", sender, phone_id)
    elif detected_lang == "tonga":
        send("Mwabuka buti! Nine Rudo, munisanga wa Dawa Health. Tuyambile mukubhaliska. Ndapota mba pe manamba yane yakupela ya foni yobe.", sender, phone_id)
    elif detected_lang == "chinyanja":
        send("Moni! Ndine Rudo, katandizi wa Dawa Health. Tiyambireni ndikulembetsani. Chonde ndipatseni manambala anayi okupela a foni yanu.", sender, phone_id)
    elif detected_lang == "bemba":
        send("Mwashibukeni! Nine Rudo, umushishi wa Dawa Health. Tulembefye. Napela mpepe manamba shine yakulekelesha ya foni yobe.", sender, phone_id)
    elif detected_lang == "lozi":
        send("Muzuhile! Nine Rudo, musiyami wa Dawa Health. Re kae ku sa felisize. Ni kope mina ya manamba a mane a feleti ya fele ni yahao.", sender, phone_id)
    else:
        send("Hello! I'm Rudo, Dawa Health's virtual assistant. Let's start with registration. What is the last 4 digits of your number?", sender, phone_id)
    
    save_user_states()


def handle_registration(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    
    if state.get("phone_digits") is None:
        # First step: Get the last 4 digits of phone number
        state["phone_digits"] = prompt
        
        # Generate random 4 letters
        random_letters = ''.join(random.choices(string.ascii_uppercase, k=4))
        
        # Create user ID in format DH-XXXX-ABCD
        user_id = f"DH-{prompt}-{random_letters}"
        state["user_id"] = user_id
        
        # Tell user to keep the ID safe and proceed to main menu
        if lang == "shona":
            send(f"Ndatenda! ID yenyu yakagadzirwa ndeye: {user_id}. Chengetedza ID iyi nekuti ichakumbirwa kumaDawa clinics. Ndingakubatsirei nhasi? Sarudza imwe yesarudzo inotevera:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        elif lang == "ndebele":
            send(f"Ngiyabonga! I-ID yakho eyakhiwe ithi: {user_id}. Gcina le ID ngoba izocelwa kumaDawa clinics. Ngingakusiza ngani namuhla? Khetha okukodwa:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        elif lang == "tonga":
            send(f"Twatotela! ID yobe eyasungika ndiye: {user_id}. Sungila ID iyi pakuti izokumbidwa kumaDawa clinics. Ndingakusebelesya shani lelo? Santha imwe:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        elif lang == "chinyanja":
            send(f"Zikomo! ID yanu yopangidwa ndi: {user_id}. Sungani ID iyi chifukwa idzafunsidwa kumakilinki a Dawa. Ndingakuthandizani lero? Sankhani imodzi:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        elif lang == "bemba":
            send(f"Natotela! ID yobe eapangwa ni: {user_id}. Eshaleni ID iyi pantu icalefwaya kumaDawa clinics. Nshingafye uli shani lelo? Palamina imo:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        elif lang == "lozi":
            send(f"Ni itumezi! ID yahao e etile ki: {user_id}. Kabula ID ye hobane i ta kopwa kwa Dawa clinics. Ni ka ku thusa jaha ki? Kopa sina:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        else:
            send(f"Thank you! Your generated ID is: {user_id}. Keep this ID safe because it'll be asked for at the Dawa clinics. How can I help you today? Please choose one:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        
        state["registered"] = True
        state["step"] = "main_menu"
    
    save_user_states()
    

def handle_follow_up(sender, prompt, phone_id):
    """
    Handle follow-up conversation after answering a question.
    Asks if user needs more help, then offers products if they say no.
    """
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    # Check for negative responses
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde"]
    
    if any(response in prompt_lower for response in no_responses):
        # User doesn't need more help, offer products
        topic = state.get("topic")
        
        if topic == "maternal":
            if lang == "shona":
                send("Ndatenda! Ungada here kutenga zvigadzirwa zvehutano hwepamuviri?", sender, phone_id)
            elif lang == "ndebele":
                send("Ngiyabonga! Ungathanda ukuthengwa izinto zokunakekela isisu?", sender, phone_id)
            elif lang == "tonga":
                send("Twatotela! Ungalanda kulemba zinto zyakusebelesya mupamumba?", sender, phone_id)
            elif lang == "chinyanja":
                send("Zikomo! Kodi mukufuna kugula zinthu zopezera zamimana?", sender, phone_id)
            elif lang == "bemba":
                send("Natotela! Mulefwaya ukutenga ifyakosa pa pamimba?", sender, phone_id)
            elif lang == "lozi":
                send("Ni itumezi! Ni ta ka rata ku rekela zwa pabuzwa bwa like?", sender, phone_id)
            else:
                send("Thank you! Would you like to purchase maternal health products?", sender, phone_id)
                
        elif topic == "cervical":
            if lang == "shona":
                send("Ndatenda! Ungada here kutenga zvigadzirwa zve cervical cancer?", sender, phone_id)
            elif lang == "ndebele":
                send("Ngiyabonga! Ungathanda ukuthengwa izinto zokuvikela isilonda somlomo wesibeletho?", sender, phone_id)
            elif lang == "tonga":
                send("Twatotela! Ungalanda kulemba zinto zya cancer ya cervix?", sender, phone_id)
            elif lang == "chinyanja":
                send("Zikomo! Kodi mukufuna kugula zinthu zopezera za cervical cancer?", sender, phone_id)
            elif lang == "bemba":
                send("Natotela! Mulefwaya ukutenga ifyakosa pa cervical cancer?", sender, phone_id)
            elif lang == "lozi":
                send("Ni itumezi! Ni ta ka rata ku rekela zwa pabuzwa bwa cancer ya cervix?", sender, phone_id)
            else:
                send("Thank you! Would you like to purchase cervical cancer products?", sender, phone_id)
        else:
            # No specific topic, offer general products
            if lang == "shona":
                send("Ndatenda! Ungada here kutenga zvigadzirwa zvehutano?", sender, phone_id)
            else:
                send("Thank you! Would you like to purchase health products?", sender, phone_id)
        
        # Set state to product inquiry
        state["step"] = "product_inquiry"
        save_user_states()
        
    elif any(response in prompt_lower for response in yes_responses):
        # User needs more help, return to main menu
        state["step"] = "main_menu"
        if lang == "shona":
            send("Ndingakubatsirei zvakare? Sarudza:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        else:
            send("How else can I help you? Please choose:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        save_user_states()
        
    else:
        # Unclear response, ask again
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Pane chimwe chandingakubatsira nacho here?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Is there anything else I can help you with?", sender, phone_id)


def handle_general_followup(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "inde"]
    no_responses = ["no", "nah", "aiwa", "kwete", "hapana", "nope"]

    if any(r in prompt_lower for r in yes_responses):
        # Stay in general-question mode
        if lang == "shona":
            send("Bvunza mubvunzo wenyu.", sender, phone_id)
        else:
            send("Please ask your question.", sender, phone_id)
        state["step"] = "general_question"
        save_user_states()
        return

    if any(r in prompt_lower for r in no_responses):
        if lang == "shona":
            send("Ndatenda! Iva nezuva rakanaka.", sender, phone_id)
        else:
            send("Thank you! Have a good day.", sender, phone_id)
        reset_conversation(sender)
        return

    # unclear
    if lang == "shona":
        send("Ndapota pindura: Hongu kana Aiwa.", sender, phone_id)
    else:
        send("Please reply Yes or No.", sender, phone_id)



def ask_follow_up_question(sender, phone_id):
    """
    Ask the follow-up question after answering a user's query
    """
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Pane chimwe chandingakubatsira nacho here?", sender, phone_id)
    elif lang == "ndebele":
        send("Ingabe kukhona okunye engingakusiza ngakho?", sender, phone_id)
    elif lang == "tonga":
        send("Kuli chinco nchingakusebelesya nacho", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi pali zina zomwe ndingakuthandizireni?", sender, phone_id)
    elif lang == "bemba":
        send("Kuli fintu fyalumo nshingafye", sender, phone_id)
    elif lang == "lozi":
        send("Ki sina sika ni ka thusa ka sona", sender, phone_id)
    else:
        send("Is there anything else I can help you with?", sender, phone_id)
    
    # Set state to wait for follow-up response
    state["step"] = "follow_up"
    save_user_states()

def ask_cervical_more_info(sender, phone_id):
    """Ask if user wants more cervical cancer information"""
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Ungada here kuwana rumwe ruzivo rwe cervical cancer?", sender, phone_id)
    elif lang == "ndebele":
        send("Ungathanda ukuthola eminye imininingwane nge-cervical cancer?", sender, phone_id)
    elif lang == "tonga":
        send("Ungalanda kuwana umwe umanyisi wa cancer ya cervix", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi mukufuna kupereka zina zambiri za cervical cancer?", sender, phone_id)
    elif lang == "bemba":
        send("Mulefwaya ukupokelela ifyakumanina fyalumo pa cervical cancer", sender, phone_id)
    elif lang == "lozi":
        send("Ni ta ka rata ku fumana li ta ni linye za cancer ya cervix", sender, phone_id)
    else:
        send("Would you like to get more information about cervical cancer?", sender, phone_id)
    
    state["step"] = "cervical_more_info"
    save_user_states()

def ask_cervical_question_number(sender, phone_id):
    """Ask user to enter a cervical cancer question number"""
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Pinda nhamba yemubvunzo kubva pa 1 kusvika pa 100:", sender, phone_id)
    elif lang == "ndebele":
        send("Faka inombolo yombuzo kusuka ku-1 kuya ku-100:", sender, phone_id)
    elif lang == "tonga":
        send("Leta inombola yambuzo kufuma pa 1 mpaka 100:", sender, phone_id)
    elif lang == "chinyanja":
        send("Lowetsani nambala yafunso kuchokera pa 1 mpaka 100:", sender, phone_id)
    elif lang == "bemba":
        send("Lete inombola yacipuna kufuma pa 1 mpaka 100:", sender, phone_id)
    elif lang == "lozi":
        send("Kenisa nombola ya lipuzo kusuka pa 1 ku fita ku 100:", sender, phone_id)
    else:
        send("Enter a question number from 1 to 100:", sender, phone_id)
    
    state["step"] = "cervical_question_number"
    save_user_states()

def ask_keep_learning(sender, phone_id):
    """Ask if user wants to keep learning more cervical cancer topics"""
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Ungada here kuramba uchidzidza zvimwe zvinhu zve cervical cancer?", sender, phone_id)
    elif lang == "ndebele":
        send("Ungathanda ukuqhubeka nokufunda ezinye izindaba ze-cervical cancer?", sender, phone_id)
    elif lang == "tonga":
        send("Ungalanda kubulelela ukusambilila izinye izintu za cancer ya cervix", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi mukufuna kupitiriza kuphunzira zina zambiri za cervical cancer?", sender, phone_id)
    elif lang == "bemba":
        send("Mulefwaya ukupitilila ukusambilila ifyakumanina fyalumo pa cervical cancer", sender, phone_id)
    elif lang == "lozi":
        send("Ni ta ka rata ku sa lielela ku ijaluka li ta ni linye za cancer ya cervix", sender, phone_id)
    else:
        send("Would you like to keep learning more about cervical cancer?", sender, phone_id)
    
    state["step"] = "keep_learning"
    save_user_states()

def handle_cervical_more_info(sender, prompt, phone_id):
    """Handle response to cervical cancer more information question"""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really"]
    
    if any(response in prompt_lower for response in yes_responses):
        # User wants more info, ask for question number
        ask_cervical_question_number(sender, phone_id)
    elif any(response in prompt_lower for response in no_responses):
        # User doesn't want more info, move to product inquiry
        state["step"] = "product_inquiry"
        handle_follow_up(sender, "no", phone_id)  # This will trigger product offering
    else:
        # Unclear response, ask again
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kuwana rumwe ruzivo?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to get more information?", sender, phone_id)

def handle_cervical_question_number(sender, prompt, phone_id):
    """Handle cervical cancer question number input"""
    state = user_states[sender]
    lang = state["language"]
    
    try:
        question_num = int(re.sub(r"\D", "", prompt))
        if 1 <= question_num <= 100:
            # Access the cervical cancer data tuple
            data_tuple = cervical_cancer_data.cervical_cancer_data
            
            # Find the question in the tuple
            question_found = False
            for i, item in enumerate(data_tuple):
                # Look for the question pattern in each tuple item
                if f"*Question {question_num}:" in str(item):
                    # Found the question, now get the full content
                    question_content = str(item)
                    
                    # If this is not the last item, check if we need to include the next item for the answer
                    if i + 1 < len(data_tuple) and "Answer" in str(data_tuple[i + 1]):
                        question_content += "\n" + str(data_tuple[i + 1])
                    
                    send(question_content, sender, phone_id)
                    question_found = True
                    
                    # Ask if they want to keep learning
                    ask_keep_learning(sender, phone_id)
                    break
            
            if not question_found:
                if lang == "shona":
                    send(f"Ndine urombo, handina kuwana mubvunzo wenhamba {question_num}. Edza imwe nhamba kubva pa 1 kusvika pa 100.", sender, phone_id)
                else:
                    send(f"Sorry, I couldn't find question number {question_num}. Please try another number from 1 to 100.", sender, phone_id)
                ask_cervical_question_number(sender, phone_id)
        else:
            if lang == "shona":
                send("Ndapota pinda nhamba kubva pa 1 kusvika pa 100 chete.", sender, phone_id)
            else:
                send("Please enter a number between 1 and 100 only.", sender, phone_id)
            ask_cervical_question_number(sender, phone_id)
            
    except ValueError:
        if lang == "shona":
            send("Ndapota pinda nhamba chaiyo kubva pa 1 kusvika pa 100.", sender, phone_id)
        else:
            send("Please enter a valid number between 1 and 100.", sender, phone_id)
        ask_cervical_question_number(sender, phone_id)

def handle_keep_learning(sender, prompt, phone_id):
    """Handle response to keep learning question"""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really"]
    
    if any(response in prompt_lower for response in yes_responses):
        # User wants to keep learning, ask for another question number
        ask_cervical_question_number(sender, phone_id)
    elif any(response in prompt_lower for response in no_responses):
        # User doesn't want to keep learning, move to product inquiry
        state["step"] = "product_inquiry"
        handle_follow_up(sender, "no", phone_id)  # This will trigger product offering
    else:
        # Unclear response, ask again
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kuramba uchidzidza?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to keep learning?", sender, phone_id)

def ask_another_week(sender, phone_id):
    """Ask if user wants to learn about other pregnancy weeks"""
    state = user_states[sender]
    lang = state["language"]
    
    if lang == "shona":
        send("Ungada here kudzidza nezve mamwe mavhiki epamuviri?", sender, phone_id)
    elif lang == "ndebele":
        send("Ungathanda ukufunda ngamanye amasonto esisu?", sender, phone_id)
    elif lang == "tonga":
        send("Ungalanda kusambilila za linji zvina vwiki zyapamumba", sender, phone_id)
    elif lang == "chinyanja":
        send("Kodi mukufuna kuphunzira za masabata ena a pamimba?", sender, phone_id)
    elif lang == "bemba":
        send("Mulefwaya ukusambilila ifyakumanina fya sabala shalemo pa pamimba", sender, phone_id)
    elif lang == "lozi":
        send("Ni ta ka rata ku ijaluka za masabata a mangwe a pabuzwa", sender, phone_id)
    else:
        send("Would you like to learn about other pregnancy weeks?", sender, phone_id)
    
    state["step"] = "ask_another_week"
    save_user_states()

def handle_another_week(sender, prompt, phone_id):
    """Handle response to 'another week' question"""
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde"]
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really"]
    
    if any(response in prompt_lower for response in yes_responses):
        # User wants another week, return to ask_week step
        state["step"] = "ask_week"
        if lang == "shona":
            send("Ndapota isa vhiki re pamuviri (1-40):", sender, phone_id)
        else:
            send("Please enter your pregnancy week number (1-40):", sender, phone_id)
        save_user_states()
        
    elif any(response in prompt_lower for response in no_responses):
        # User doesn't want more weeks, move to product inquiry
        state["step"] = "product_inquiry"
        state["topic"] = "maternal"  # Ensure topic is set for product offering
        
        # Offer maternal health products
        if lang == "shona":
            send("Ndatenda! Ungada here kutenga zvigadzirwa zvehutano hwepamuviri? Tinopa:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits", sender, phone_id)
        elif lang == "ndebele":
            send("Ngiyabonga! Ungathanda ukuthengwa izinto zokunakekela isisu? Sinakho:\n- Ama-Prenatal Vitamins\n- Izinto zokuhlola isisu\n- Amakhithi okunakekela isisu", sender, phone_id)
        elif lang == "tonga":
            send("Twatotela! Ungalanda kulemba zinto zyakusebelesya mupamumba? Tuli nazyo:\n- Prenatal Vitamins\n- Zintu zyakulinga upumumbe\n- Makiti yokusebelesya mupamumba", sender, phone_id)
        elif lang == "chinyanja":
            send("Zikomo! Kodi mukufuna kugula zinthu zopezera zamimana? Tili ndi:\n- Prenatal Vitamins\n- Zoyesera zamimana\n- Makiti apezera zamimana", sender, phone_id)
        elif lang == "bemba":
            send("Natotela! Mulefwaya ukutenga ifyakosa pa pamimba? Tuli na:\n- Prenatal Vitamins\n- Ifyakosa pa pamimba\n- Makiti yakosa pa pamimba", sender, phone_id)
        elif lang == "lozi":
            send("Ni itumezi! Ni ta ka rata ku rekela zwa pabuzwa bwa like? Re na:\n- Prenatal Vitamins\n- Litila ku li pumile\n- Makiti ya pabuzwa", sender, phone_id)
        else:
            send("Thank you! Would you like to purchase maternal health products? We offer:\n- Prenatal Vitamins\n- Pregnancy Tests\n- Maternal Care Kits", sender, phone_id)
        save_user_states()
        
    else:
        # Unclear response, ask again
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kudzidza nezve mamwe mavhiki?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to learn about other weeks?", sender, phone_id)

def handle_main_menu(sender, prompt, phone_id):
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()

    logging.info(f"User {sender} said: '{prompt}' (lowercase: '{prompt_lower}')")
    logging.info(f"Current state: step={state.get('step')}, topic={state.get('topic')}")

    # Handle reset/restart commands - make matching more precise
    reset_keywords = ["start over", "restart", "new conversation", "main menu", "menu", "reset", "help"]
    reset_phrases = ["hie", "hey", "hi"]  # Only match these as exact phrases
    
    # Check for exact matches for short words
    if (any(keyword in prompt_lower for keyword in reset_keywords) or
        any(prompt_lower.strip() == phrase for phrase in reset_phrases)):
        reset_conversation(sender)
        state = user_states[sender]  # Refresh state reference
        lang = state["language"]
        if lang == "shona":
            send("Ndingakubatsirei nhasi? Sarudza imwe yesarudzo inotevera:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        else:
            send("How can I help you today? Please choose one:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        save_user_states()
        return

    # --- Route based on current step FIRST ---
    current_step = state.get("step")
    
    # Handle specific states that have dedicated handlers
    if current_step == "ask_another_week":
        handle_another_week(sender, prompt, phone_id)
        return
        
    if current_step == "cervical_more_info":
        handle_cervical_more_info(sender, prompt, phone_id)
        return
        
    if current_step == "cervical_question_number":
        handle_cervical_question_number(sender, prompt, phone_id)
        return
        
    if current_step == "keep_learning":
        handle_keep_learning(sender, prompt, phone_id)
        return
        
    if current_step == "follow_up":
        handle_follow_up(sender, prompt, phone_id)
        return
        
    if current_step == "product_inquiry":
        # Handle product selection logic here
        if lang == "shona":
            send("Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo.", sender, phone_id)
        else:
            send("Thank you! We'll contact you shortly for more details.", sender, phone_id)
        state["step"] = "main_menu"
        save_user_states()
        return

    # --- THEN handle the step-based flows ---
    
    # --- Step A: Ask for general info or specific question ---
    if state.get("step") == "choose_info_type":
        if prompt_lower in ["1", "general", "information", "info"]:
            if state.get("topic") == "maternal":
                state["step"] = "ask_week"
                if lang == "shona":
                    send("Ndapota isa vhiki re pamuviri (1-40):", sender, phone_id)
                else:
                    send("Please enter your pregnancy week number (1–40):", sender, phone_id)
            elif state.get("topic") == "cervical":
                # Default to Question 1 info
                if lang == "shona":
                    send(
                        "*Mubvunzo 1:*\n"
                        "- Chii chinonzi cervical cancer?\n\n"
                        "Mhinduro:\n"
                        "- Cervical cancer chirwere che cervix, chikamu chezasi chechibereko chinobatana nechibereko. "
                        "Ndicho chirwere chegomarara chechipiri chinowanikwa zvakanyanya pasi rose uye ndicho chinonyanya kuitika kuvakadzi muZambia. "
                        "Chirwere chinodzivirika uye chinorapika, kunyanya kana chikaonekwa nekukurumidza.\n\n",
                        sender,
                        phone_id
                    )
                else:
                    send(
                        "*Question 1:*\n"
                        "- What is cervical cancer?\n\n"
                        "Answer:\n"
                        "- Cervical cancer is a disease of the cervix, the lower part of the uterus that connects to the vagina. "
                        "It is the second most common female malignancy worldwide and the most common in females in Zambia. "
                        "It is a preventable and treatable disease, especially when detected early.\n\n",
                        sender,
                        phone_id
                    )
                # Ask if they want more information
                ask_cervical_more_info(sender, phone_id)
            save_user_states()
            return

        elif prompt_lower in ["2", "specific", "question", "questions"]:
            if state.get("topic") == "maternal":
                # Show a static list of common pregnancy questions
                state["step"] = "maternal_question_choice"
                if lang == "shona":
                    send(
                        "Sarudza mubvunzo:\n"
                        "1. Ndezvikaita zviratidzo zvepamuviri?\n"
                        "2. Ndeapi marairiro ezvokudya?\n"
                        "3. Ndingafanire kuona chiremba riini?",
                        sender,
                        phone_id
                    )
                else:
                    send(
                        "You can choose a question or ask any of your own.\n"
                        "1. What are common pregnancy symptoms?\n"
                        "2. What nutrition tips should I follow?\n"
                        "3. When should I see a doctor?",
                        sender,
                        phone_id
                    )
            elif state.get("topic") == "cervical":
                # Show a static list of common cervical cancer questions
                state["step"] = "cervical_question_choice"
                if lang == "shona":
                    send(
                        "Sarudza mubvunzo:\n"
                        "1. Chii chinonzi cervical cancer?\n"
                        "2. Ndezvipi zviratidzo zvekutanga zvecervical cancer?\n"
                        "3. Chii chinokonzera cervical cancer?",
                        sender,
                        phone_id
                    )
                else:
                    send(
                        "You can choose a question or ask any of your own.\n"
                        "1. What is cervical cancer?\n"
                        "2. What are the early symptoms of cervical cancer?\n"
                        "3. What causes cervical cancer?",
                        sender,
                        phone_id
                    )
            save_user_states()
            return

        else:
            if lang == "shona":
                send("Pindura ne '1' kuti uwane ruzivo kana '2' kuti ubvunze mibvunzo.", sender, phone_id)
            else:
                send("Please reply with '1' for general information or '2' for specific questions.", sender, phone_id)
            return

   
    # --- Step B: Handle maternal week selection ---
    if state.get("step") == "ask_week":
        try:
            week = int(re.sub(r"\D", "", prompt_lower))
            if 1 <= week <= 40:
                info_text = pregnancy_data.pregnancy_data
                pattern = rf"\*Week {week}:.*?(?=\*Week {week+1}:|\Z)"
                match = re.search(pattern, info_text, re.S)
                if match:
                    send(f"Here's information for *Week {week}:*\n\n{match.group(0)}", sender, phone_id)
                    
                    # Ask if they want to learn about other weeks
                    ask_another_week(sender, phone_id)
                else:
                    send("No data available for that week.", sender, phone_id)
                    ask_another_week(sender, phone_id)
        except ValueError:
            send("Please enter a valid week number between 1 and 40.", sender, phone_id)
            ask_another_week(sender, phone_id)
        return  

    # --- Step C: Handle specific maternal question choice ---
    if state.get("step") == "maternal_question_choice":
        if prompt_lower in ["1", "symptoms", "zviratidzo"]:
            if lang == "shona":
                send("Zviratidzo zvepamuviri zvinosanganisira kusvotwa, kuneta, kuvava mazamu, uye kuchinja mweya.", sender, phone_id)
            else:
                send("Common pregnancy symptoms include nausea, fatigue, breast tenderness, and mood swings.", sender, phone_id)
    
        elif prompt_lower in ["2", "nutrition", "zvokudya"]:
            if lang == "shona":
                send("Marairiro ezvokudya: Idya chikafu chakaringana, wedzera folic acid uye iron, uye nwa mvura yakawanda.", sender, phone_id)
            else:
                send("Nutrition tips: Eat balanced meals, increase folic acid and iron intake, and stay hydrated.", sender, phone_id)
    
        elif prompt_lower in ["3", "doctor", "chiremba"]:
            if lang == "shona":
                send("Enda kuchiremba kana uine kurwadziwa kwakanyanya, kubuda ropa kwakawanda, kana fivha yepamusoro.", sender, phone_id)
            else:
                send("See a doctor immediately if you experience severe pain, heavy bleeding, or high fever.", sender, phone_id)
    
        else:
            logging.info("DEBUG: Processing free-text maternal question")
            if lang == "shona":
                send("Kufunga...", sender, phone_id)
            else:
                send("Thinking...", sender, phone_id)
    
            gemini_response = ask_gemini(prompt)  
            send(gemini_response, sender, phone_id)
        
        logging.info(f"DEBUG: Before ask_follow_up_question, step is: {state.get('step')}")
        ask_follow_up_question(sender, phone_id)
        logging.info(f"DEBUG: After ask_follow_up_question, step is: {state.get('step')}")
        save_user_states()
        return

    # --- Step D: Handle cervical specific question choice ---
    if state.get("step") == "cervical_question_choice":
        if prompt_lower in ["1", "what is it", "what is cervical cancer", "chii"]:
            if lang == "shona":
                send("Cervical cancer chirwere che cervix, chikamu chezasi chechibereko chinobatana nechibereko. Ndicho chirwere chegomarara chechipiri chinowanikwa zvakanyanya pasi rose uye ndicho chinonyanya kuitika kuvakadzi muZambia. Chirwere chinodzivirika uye chinorapika, kunyanya kana chikaonekwa nekukurumidza.", sender, phone_id)
            else:
                send("Cervical cancer is a disease of the cervix, the lower part of the uterus that connects to the vagina. It is the second most common female malignancy worldwide and the most common in females in Zambia. It is a preventable and treatable disease, especially when detected early.", sender, phone_id)
        elif prompt_lower in ["2", "symptoms", "early symptoms", "zviratidzo"]:
            if lang == "shona":
                send("Mumatanho ekutanga, cervical cancer kazhinji haina zviratidzo zvinooneka. Ndokusaka kuongororwa nguva nenguva kwakakosha. Sezvo cancer ichikura, zviratidzo zvinogona kusanganisira kubuda ropa kusingawanzo (pakati penguva, mushure mekuita bonde, kana mushure mekuenda kumwedzi), kubuda kwezvipembenene zvinonhuwa, kana kurwadziwa panguva yekuita bonde.", sender, phone_id)
            else:
                send("In its early stages, cervical cancer often has no noticeable symptoms. This is why regular screening is so important. As the cancer progresses, symptoms may include unusual vaginal bleeding (between periods, after sex, or after menopause), foul-smelling vaginal discharge, or pain during sexual intercourse.", sender, phone_id)
        elif prompt_lower in ["3", "causes", "what causes it", "chikonzero"]:
            if lang == "shona":
                send("Kazhinji, cervical cancer inokonzerwa nehutachiona husingaperi hweHuman Papilloma Virus (HPV). HPV ihutachiona hwakajairika, hunotapuriranwa nekusangana pabonde. Kunyange immune system yemuviri ichibvisa hutachiona muvanhu vazhinji, hutachiona husingaperi hunogona kukonzera shanduko yemasero inogona kuzopedzisira yaita cancer.", sender, phone_id)
            else:
                send("In almost all cases, cervical cancer is caused by persistent infection with the Human Papilloma Virus (HPV). HPV is a very common, sexually transmitted virus. While the body's immune system clears the virus in most people, a persistent infection can lead to abnormal cell changes that may eventually develop into cancer.", sender, phone_id)
        else:
            # ✅ Forward free-text question to Gemini
            if lang == "shona":
                send("Kufunga...", sender, phone_id)
            else:
                send("Thinking...", sender, phone_id)
    
            # Call Gemini here — e.g., a function like:
            gemini_response = ask_gemini_cancer(prompt)  
            send(gemini_response, sender, phone_id)
            
        ask_follow_up_question(sender, phone_id)
        save_user_states()
        return

    # --- Step E: Topic selection ---
    if "maternal" in prompt_lower or "maternal health" in prompt_lower:
        state["topic"] = "maternal"
        state["step"] = "choose_info_type"
        if lang == "shona":
            send("Ungada ruzivo here kana kuti une mubvunzo chaiwo?\n1. Ruzivo\n2. Mubvunzo Chaiwo", sender, phone_id)
        else:
            send("Would you like general information or do you have a specific question?\n1. General Information\n2. Specific Question", sender, phone_id)
        save_user_states()
        return

    if "cervical" in prompt_lower or "cervical cancer" in prompt_lower:
        state["topic"] = "cervical"
        state["step"] = "choose_info_type"
        if lang == "shona":
            send("Ungada ruzivo here kana kuti une mubvunzo chaiwo?\n1. Ruzivo\n2. Mubvunzo Chaiwo", sender, phone_id)
        else:
            send("Would you like general information or do you have a specific question?\n1. General Information\n2. Specific Question", sender, phone_id)
        save_user_states()
        return

    # --- Step F: Fallback ---
    lang = state["language"]

    gemini_reply = ask_gemini_general(prompt, lang)
    send(gemini_reply, sender, phone_id)
    
    # Ask follow-up question
    if lang == "shona":
        send("Pane chimwe chandingakubatsira nacho here? (Hongu / Aiwa)", sender, phone_id)
    else:
        send("Do you have any more questions?", sender, phone_id)
    
    state["step"] = "general_followup"
    save_user_states()
    return


def detect_language(text: str) -> str:
    """
    Simple language detection: returns 'shona' if Shona words are detected, otherwise 'english'.
    """
    shona_keywords = [
        "zviratidzo", "chiremba", "pamuviri", "mvura",
        "mazamu", "kubuda", "ropa", "mwana", "kusvotwa", "kurwadziwa"
    ]
    if any(word in text.lower() for word in shona_keywords):
        return "shona"
    return "english"

def ask_gemini(question: str) -> str:
    """
    Sends a free-text maternal question to Gemini and returns a response
    in the same language as the user's question.
    """
    try:
        lang = detect_language(question)

        # 🧠 Language-specific instruction
        if lang == "shona":
            instruction = (
                "Iwe uri mubatsiri wezvehutano hwepamuviri. "
                "Pindura mubvunzo uyu muShona yakajeka uye yakapfava:\n\n"
            )
        else:
            instruction = (
                "You are a maternal health assistant. "
                "Answer the following question clearly and simply in English:\n\n"
            )

        # ✅ Use your model
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(instruction + question)

        if response and response.text:
            return response.text.strip()
        else:
            return (
                "Ndine hurombo, handina kuwana mhinduro." if lang == "shona"
                else "Sorry, I couldn't find an answer."
            )

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return (
            "Pane dambudziko pakupindura mubvunzo wako." if lang == "shona"
            else "Sorry, there was a problem getting an answer."
        )

def detect_language_cancer(text: str) -> str:
    """
    Simple language detection: returns 'shona' if Shona words are detected, otherwise 'english'.
    """
    shona_keywords = [
        "zviratidzo", "chiremba", "gomarara", "mvura", "ropa", "kurwadziwa"
    ]
    if any(word in text.lower() for word in shona_keywords):
        return "shona"
    return "english"

def ask_gemini_cancer(question: str) -> str:
    """
    Sends a free-text maternal question to Gemini and returns a response
    in the same language as the user's question.
    """
    try:
        lang = detect_language_cancer(question)

        # 🧠 Language-specific instruction
        if lang == "shona":
            instruction = (
                "Iwe uri mubatsiri wezvehutano hwegomarara rechibereko. "
                "Pindura mubvunzo uyu muShona yakajeka uye yakapfava:\n\n"
            )
        else:
            instruction = (
                "You are a cervical cancer health assistant. "
                "Answer the following question clearly and simply in English:\n\n"
            )

        # ✅ Use your model
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(instruction + question)

        if response and response.text:
            return response.text.strip()
        else:
            return (
                "Ndine hurombo, handina kuwana mhinduro." if lang == "shona"
                else "Sorry, I couldn't find an answer."
            )

    except Exception as e:
        print(f"[Gemini Error] {e}")
        return (
            "Pane dambudziko pakupindura mubvunzo wako." if lang == "shona"
            else "Sorry, there was a problem getting an answer."
        )


def ask_gemini_general(question: str, lang: str) -> str:
    """
    Handles ANY unexpected general question.
    Responds in same language (Shona or English).
    """
    try:
        # Language–specific instructions
        if lang == "shona":
            instruction = (
                "Pindura mubvunzo uyu muShona yakapfava uye iri nyore kunzwisisa:\n\n"
            )
        else:
            instruction = (                
                "You are a professional health assistant specializing in maternal health and cervical cancer. "
                "Answer the user's question using correct and evidence-based health information. "
                "IMPORTANT: The response must be detailed, factual, and professional. "
                "DO NOT start with phrases like 'Okay', 'Sure', 'Here’s', or 'Let me explain'. "
                "DO NOT include any conversational fillers. "
                "Start directly with the answer. "
                "Include a brief disclaimer at the end stating that this information does not replace a doctor's evaluation. "
                "Respond in clear, simple English:\n\n"
            )

         

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(instruction + question)

        if response and response.text:
            return response.text.strip()
        else:
            return "Ndine urombo, handina kuwana mhinduro." if lang == "shona" else "Sorry, I couldn't find an answer."

    except Exception as e:
        print(f"[Gemini General Error] {e}")
        return (
            "Pane dambudziko pakupindura mubvunzo wako." if lang == "shona"
            else "Sorry, there was an error answering your question."
        )


def handle_conversation_state(sender, prompt, phone_id):
    state = user_states[sender]
    prompt_lower = prompt.lower().strip()
    
    # Handle reset keywords
    reset_keywords = ["start over", "restart", "new conversation", "main menu", "reset", "help"]
    if any(keyword in prompt_lower for keyword in reset_keywords):
        reset_conversation(sender)
        state = user_states[sender]  # Refresh state reference
        lang = state["language"]
        if lang == "shona":
            send("Ndingakubatsirei nhasi? Sarudza imwe yesarudzo inotevera:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        else:
            send("How can I help you today? Please choose one:\n- Maternal Health\n- Cervical Cancer", sender, phone_id)
        return

    # Route based on current step
    current_step = state.get("step")
    
    if current_step == "language_detection" and state.get("first_message", True):
        handle_language_detection(sender, prompt, phone_id)
    elif current_step == "registration":
        handle_registration(sender, prompt, phone_id)
    elif current_step in ["ask_another_week", "cervical_more_info", "cervical_question_number", "keep_learning", "follow_up"]:
        # These have dedicated handlers
        if current_step == "ask_another_week":
            handle_another_week(sender, prompt, phone_id)
        elif current_step == "cervical_more_info":
            handle_cervical_more_info(sender, prompt, phone_id)
        elif current_step == "cervical_question_number":
            handle_cervical_question_number(sender, prompt, phone_id)
        elif current_step == "keep_learning":
            handle_keep_learning(sender, prompt, phone_id)
        elif current_step == "follow_up":
            handle_follow_up(sender, prompt, phone_id)
    elif current_step == "product_inquiry":
        # Handle product purchase responses
        handle_purchase_response(sender, prompt, phone_id)
    elif current_step == "confirm_purchase":
        # Handle final purchase confirmation
        handle_purchase_confirmation(sender, prompt, phone_id)

    elif current_step == "general_followup":
        handle_general_followup(sender, prompt, phone_id)
        return
    
    elif current_step == "general_question":
        # Any question goes straight to Gemini
        lang = state["language"]
        reply = ask_gemini_general(prompt, lang)
        send(reply, sender, phone_id)
    
        # Ask again if they have more questions
        if lang == "shona":
            send("Pane chimwe chamunoda kubvunza here? (Hongu / Aiwa)", sender, phone_id)
        else:
            send("Do you have any more questions?", sender, phone_id)
    
        state["step"] = "general_followup"
        save_user_states()
        return

    else:
        # All other steps go to main menu handler
        handle_main_menu(sender, prompt, phone_id)

def handle_purchase_confirmation(sender, prompt, phone_id):
    """
    Handle final purchase confirmation.
    """
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde"]
    
    if any(response in prompt_lower for response in no_responses):
        # User doesn't want to purchase
        if lang == "shona":
            send("Zvakanaka. Tinokutendai! Kana uine mimwe mibvunzo, tanga patsva nekuti 'hi'.", sender, phone_id)
        else:
            send("Alright. Thank you! If you have more questions, start over by saying 'hi'.", sender, phone_id)
        reset_conversation(sender)
        
    elif any(response in prompt_lower for response in yes_responses):
        # User wants to purchase
        if lang == "shona":
            send("Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo nezvekutenga.", sender, phone_id)
        else:
            send("Thank you! We'll contact you shortly for more details about your purchase.", sender, phone_id)
        reset_conversation(sender)
        
    else:
        # Unclear response
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kuenderera mberi nekutenga?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to proceed with purchasing?", sender, phone_id)


def handle_purchase_response(sender, prompt, phone_id):
    """
    Handle yes/no responses to the purchase question.
    If yes, show relevant products from products.py based on user's topic.
    If no, thank the user and end the conversation.
    """
    state = user_states[sender]
    lang = state["language"]
    prompt_lower = prompt.lower().strip()
    
    # Check for negative responses
    no_responses = ["no", "nah", "nope", "hapana", "kwete", "aiwa", "a'a", "not really"]
    yes_responses = ["yes", "yeah", "yep", "please", "ehe", "hongu", "ndizvo", "inde"]
    
    if any(response in prompt_lower for response in no_responses):
        # User doesn't want to purchase, thank them and end conversation
        if lang == "shona":
            send("Ndatenda! Iva nezuva rakanaka. Kana uine mimwe mibvunzo, tanga patsva nekuti 'hi'.", sender, phone_id)
        elif lang == "ndebele":
            send("Ngiyabonga! Ube nosuku oluhle. Uma uneminye imibuzo, qala ingxoxo entsha ngo-'hi'.", sender, phone_id)
        elif lang == "tonga":
            send("Twatotela! Ba ne zuva li limpe. Ngati uli na minye imibuzo, yambila patsva uleke 'hi'.", sender, phone_id)
        elif lang == "chinyanja":
            send("Zikomo! Khalani ndi tsiku labwino. Ngati muli ndi mafunso ena, yambitsaninso pogwiritsa ntchito 'hi'.", sender, phone_id)
        elif lang == "bemba":
            send("Natotela! Mubeleko ubushiku bwawama. Ngati muli na ifyakupuzanya fimbi, tembeni nakutandila na 'hi'.", sender, phone_id)
        elif lang == "lozi":
            send("Ni itumezi! Ba ni zuwa le li ne. Ha ni na lipuzo ni line, alusa nakuli 'hi'.", sender, phone_id)
        else:
            send("Thank you! Have a nice day. If you have more questions, start over by saying 'hi'.", sender, phone_id)
        
        # Reset conversation state
        reset_conversation(sender)
        
    elif any(response in prompt_lower for response in yes_responses):
        # User wants to purchase, show relevant products
        topic = state.get("topic")
        
        if topic == "maternal":
            # Show maternal health products
            maternal_products = extract_products_by_category("Maternal Health")
            if maternal_products:
                products_text = format_products_for_display(maternal_products, lang)
                send(products_text, sender, phone_id)
            else:
                if lang == "shona":
                    send("Ndine urombo, hapana zvigadzirwa zvehutano hwepamuviri zvazvino onekwa. Tinokurudzira kuenda kukiriniki yedu kuti uwane rumwe ruzivo.", sender, phone_id)
                else:
                    send("Sorry, no maternal health products are currently available. We recommend visiting our clinic for more information.", sender, phone_id)
                
        elif topic == "cervical":
            # Show cervical cancer products
            cervical_products = extract_products_by_category("Cervical Cancer")
            if cervical_products:
                products_text = format_products_for_display(cervical_products, lang)
                send(products_text, sender, phone_id)
            else:
                if lang == "shona":
                    send("Ndine urombo, hapana zvigadzirwa zvecervical cancer zvazvino onekwa. Tinokurudzira kuenda kukiriniki yedu kuti uwane rumwe ruzivo.", sender, phone_id)
                else:
                    send("Sorry, no cervical cancer products are currently available. We recommend visiting our clinic for more information.", sender, phone_id)
        else:
            # Show general products if no specific topic
            general_products = extract_products_by_category("General")
            if general_products:
                products_text = format_products_for_display(general_products, lang)
                send(products_text, sender, phone_id)
            else:
                if lang == "shona":
                    send("Tinokutendai! Tichakubatai mukati memaminitsi mashoma kuti muwedzere ruzivo.", sender, phone_id)
                else:
                    send("Thank you! We'll contact you shortly for more details.", sender, phone_id)
        
        # After showing products, ask if they want to proceed with purchase
        if lang == "shona":
            send("Ungada here kuenderera mberi nekutenga chimwe chezvigadzirwa izvi?", sender, phone_id)
        else:
            send("Would you like to proceed with purchasing any of these products?", sender, phone_id)
        
        state["step"] = "confirm_purchase"
        save_user_states()
        
    else:
        # Unclear response, ask again
        if lang == "shona":
            send("Handina kunzwisisa. Pindura ndapota: Ungada here kutenga zvigadzirwa?", sender, phone_id)
        else:
            send("I didn't understand. Please reply: Would you like to purchase products?", sender, phone_id)

def extract_products_by_category(category_name):
    """
    Extract products from the products_by_category dictionary.
    Returns a list of product dictionaries.
    """
    try:
        return products_by_category.get(category_name, [])
    except Exception as e:
        logging.error(f"Error extracting products for category {category_name}: {e}")
        return []
        

def format_products_for_display(products_list, lang):
    """
    Format products list into a readable string for display.
    """
    if not products_list:
        if lang == "shona":
            return "Hapana zvigadzirwa zvazvino onekwa."
        else:
            return "No products currently available."
    
    if lang == "shona":
        header = "🏥 Zvigadzirwa Zvehutano:\n\n"
    else:
        header = "🏥 Health Products:\n\n"
    
    products_text = header
    for i, product in enumerate(products_list, 1):
        name = product.get('name', 'Unknown Product')
        price = product.get('price', 'Price not available')
        availability = product.get('availability', 'Availability not specified')
        
        if lang == "shona":
            products_text += f"{i}. {name}\n"
            products_text += f"   💰 Mutengo: {price}\n"
            products_text += f"   📦 Kuwanikwa: {availability}\n\n"
        else:
            products_text += f"{i}. {name}\n"
            products_text += f"   💰 Price: {price}\n"
            products_text += f"   📦 Availability: {availability}\n\n"
    
    if lang == "shona":
        products_text += "Sarudza chirongwa nekuudza nhamba yacho."
    else:
        products_text += "Select a product by telling us the number."
    
    return products_text


def message_handler(data, phone_id):
    global user_states
    sender = data["from"]
    load_user_states()
    
    if sender not in user_states:
        user_states[sender] = {
            "step": "language_detection",
            "language": "english",
            "needs_language_confirmation": False,
            "registered": False,
            "full_name": None,
            "address": None,
            "conversation_history": [],
            "first_message": True  # ✅ New flag to track first message
        }
        save_user_states()
    else:
        # ✅ Mark as not first message for returning users
        user_states[sender]["first_message"] = False
    
    if data["type"] == "text":
        prompt = data["text"]["body"]
    else:
        prompt = ""
    
    save_user_conversation(sender, "user", prompt)
    handle_conversation_state(sender, prompt, phone_id)
    
    if db:
        scheduler.enterabs(report_time.timestamp(), 1, create_report, (phone_id,))
        scheduler.run(blocking=False)
        delete_old_chats()

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("connected.html")

@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")
        if mode == "subscribe" and token == "BOT":
            return challenge, 200
        else:
            return "Failed", 403

    elif request.method == "POST":
        try:
            entry = request.get_json()["entry"][0]["changes"][0]["value"]

            # ✅ only handle if a real message exists
            if "messages" in entry:
                data = entry["messages"][0]
                phone_id = entry["metadata"]["phone_number_id"]
                message_handler(data, phone_id)
            else:
                logging.info("Webhook received non-message event (statuses, delivery, etc). Ignored.")

        except Exception as e:
            logging.error(f"Error in webhook: {e}")
        return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    load_user_states()
    app.run(debug=True, port=8000)










