from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
from datetime import datetime
import hashlib
import secrets
import os
import re
import json

try:
    import firebase_admin
    from firebase_admin import credentials, auth as firebase_auth
except ImportError:
    firebase_admin = None
    credentials = None
    firebase_auth = None

# Try to import SHAP - handle import errors gracefully
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SHAP could not be imported: {e}")
    print("SHAP explanations will be disabled. Install dependencies: pip install shap scipy scikit-learn")
    SHAP_AVAILABLE = False
    shap = None

app = Flask(__name__)

# ----------------------------
# Secret Key Configuration
# ----------------------------
# Generate a secure secret key for production
# In production, consider using environment variable: os.environ.get('SECRET_KEY')
if os.environ.get('SECRET_KEY'):
    app.secret_key = os.environ.get('SECRET_KEY')
else:
    # Generate a random secret key (32 bytes = 256 bits)
    app.secret_key = secrets.token_hex(32)

# ----------------------------
# Session Configuration
# ----------------------------
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours in seconds

# ----------------------------
# Load model files
# ----------------------------
try:
    model = joblib.load("model/yield_model.pkl")
    encoders = joblib.load("model/encoders.pkl")
    y = joblib.load("model/target_values.pkl")
    
    # Initialize SHAP explainer only if available
    explainer = None
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            print("SHAP explainer initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize SHAP explainer: {e}")
            SHAP_AVAILABLE = False
except FileNotFoundError as e:
    print(f"Error: Model files not found: {e}")
    print("Please ensure model files exist in the 'model' directory:")
    print("  - model/yield_model.pkl")
    print("  - model/encoders.pkl")
    print("  - model/target_values.pkl")
    model = None
    encoders = None
    y = None
    explainer = None
except Exception as e:
    print(f"Error loading model files: {e}")
    model = None
    encoders = None
    y = None
    explainer = None

# ----------------------------
# Load dropdown values from dataset
# ----------------------------
try:
    _df = pd.read_csv(os.path.join(os.path.dirname(__file__), "crop_yield1.csv"))
    for _col in ["Crop", "Season", "State"]:
        _df[_col] = _df[_col].astype(str).str.strip().str.title()
    CROP_LIST = sorted(_df["Crop"].unique().tolist())
    SEASON_LIST = sorted(_df["Season"].unique().tolist())
    STATE_LIST = sorted(_df["State"].unique().tolist())
    del _df
except Exception as e:
    print(f"Warning: Could not load dropdown values from dataset: {e}")
    CROP_LIST = []
    SEASON_LIST = []
    STATE_LIST = []

# ----------------------------
# User Storage (Simple dictionary - replace with database in production)
# ----------------------------
users = {}  # Format: {username: {'password': hashed_password, 'email': email, 'phone': phone, 'join_date': date, 'last_login': date, 'predictions': count, 'notes': text}}
uid_to_username = {}  # Maps Firebase UID to local username

FIREBASE_ENABLED = False
FIREBASE_INIT_ERROR = None


def initialize_firebase():
    """Initialize Firebase Admin SDK from service account path."""
    global FIREBASE_ENABLED, FIREBASE_INIT_ERROR

    if firebase_admin is None:
        FIREBASE_ENABLED = False
        FIREBASE_INIT_ERROR = "firebase-admin dependency is missing. Install requirements.txt."
        return

    service_account_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    if not service_account_path:
        FIREBASE_ENABLED = False
        FIREBASE_INIT_ERROR = "Set FIREBASE_SERVICE_ACCOUNT_PATH to enable Firebase auth."
        return

    if not os.path.exists(service_account_path):
        FIREBASE_ENABLED = False
        FIREBASE_INIT_ERROR = "Firebase service account file path is invalid."
        return

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
        FIREBASE_ENABLED = True
        FIREBASE_INIT_ERROR = None
    except Exception as exc:
        FIREBASE_ENABLED = False
        FIREBASE_INIT_ERROR = f"Firebase initialization failed: {exc}"


def get_firebase_web_config():
    """Build Firebase Web SDK config from environment variables."""
    return {
        "apiKey": os.environ.get("FIREBASE_WEB_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_WEB_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_WEB_PROJECT_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_WEB_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_WEB_MESSAGING_SENDER_ID", ""),
        "appId": os.environ.get("FIREBASE_WEB_APP_ID", ""),
    }


def firebase_web_config_ready():
    """Return True when all required web config values are present."""
    cfg = get_firebase_web_config()
    return all(cfg.values())


def translate_feature_key(feature_name: str) -> str:
    """Map model feature name to translation key (or empty string)."""
    mapping = {
        "Crop": "feature_crop",
        "Season": "feature_season",
        "State": "feature_state",
        "Area": "feature_area",
        "Annual_Rainfall": "feature_annual_rainfall",
        "Fertilizer": "feature_fertilizer",
        "Pesticide": "feature_pesticide",
        "Model Prediction": "feature_model_prediction",
    }
    return mapping.get(feature_name, "")


def suggestion_key_for_feature(feature_name: str) -> str:
    mapping = {
        "Fertilizer": "suggest_increase_fertilizer",
        "Pesticide": "suggest_improve_pest_management",
        "Area": "suggest_optimize_land_usage",
        "Annual_Rainfall": "suggest_consider_irrigation",
        "Season": "suggest_choose_optimal_season",
        "State": "suggest_regional_climate_matters",
        "Crop": "suggest_consider_high_yield_varieties",
        "Model Prediction": "suggest_review_inputs",
    }
    return mapping.get(feature_name, "")


def verify_firebase_id_token(id_token):
    """Verify Firebase ID token and return (decoded_token, error_message)."""
    if not FIREBASE_ENABLED:
        if FIREBASE_INIT_ERROR:
            return None, FIREBASE_INIT_ERROR
        return None, "Firebase auth is not configured."

    if not id_token:
        return None, "Missing Firebase ID token."

    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded, None
    except Exception:
        return None, "Invalid or expired Firebase token. Please authenticate again."


def create_or_get_user_from_firebase(decoded_token, requested_username=""):
    """Return existing username for Firebase UID or create a new local profile."""
    uid = decoded_token.get("uid")
    if not uid:
        return None

    if uid in uid_to_username:
        return uid_to_username[uid]

    email = decoded_token.get("email")
    phone = decoded_token.get("phone_number")

    base_username = requested_username.strip()
    if not base_username:
        if email and "@" in email:
            base_username = email.split("@", 1)[0]
        elif phone:
            base_username = f"user{phone[-4:]}"
        else:
            base_username = f"user{uid[:8]}"

    safe_username = re.sub(r"[^a-zA-Z0-9_]", "_", base_username)[:30] or "user"
    username = safe_username
    suffix = 1
    while username in users:
        username = f"{safe_username}_{suffix}"
        suffix += 1

    users[username] = {
        "password": None,
        "email": email if email else None,
        "phone": phone[3:] if phone and phone.startswith("+91") else phone,
        "firebase_uid": uid,
        "join_date": datetime.now().strftime('%Y-%m-%d'),
        "last_login": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "predictions": 0,
        "notes": ''
    }
    uid_to_username[uid] = username
    return username

def hash_password(password):
    """Simple password hashing (use bcrypt in production)"""
    return hashlib.sha256(password.encode()).hexdigest()


def is_valid_email(email: str) -> bool:
    """
    Validate Gmail address:
    - must start with a small alphabet
    - may contain lowercase letters, digits, or # . _ - in the middle
    - must end with '@gmail.com'
    """
    if not email:
        return False
    pattern = r'^[a-z][a-z0-9#._-]*@gmail\.com$'
    return re.match(pattern, email) is not None


def is_valid_phone(phone: str) -> bool:
    """Validate phone number as exactly 10 digits."""
    if not phone:
        return False
    return re.fullmatch(r'\d{10}', phone) is not None

def check_login():
    """Check if user is logged in"""
    return 'username' in session


initialize_firebase()

# ----------------------------
# Simple Multi-language Support
# ----------------------------

# Available UI languages shown in the dropdown
LANGUAGES = {
    'en': 'English',
    'te': 'తెలుగు / Telugu',
    'hi': 'हिन्दी / Hindi',
    'ta': 'தமிழ் / Tamil',
    'bn': 'বাংলা / Bengali',
    'mr': 'मराठी / Marathi',
    'gu': 'ગુજરાતી / Gujarati',
    'pa': 'ਪੰਜਾਬੀ / Punjabi',
}

DEFAULT_LANGUAGE = 'en'

# Translation keys used in templates
TRANSLATIONS = {
    # Common / navigation
    'app_title': {
        'en': 'AgriXAI',
        'te': 'అగ్రిXAI',
        'hi': 'एग्रीXAI',
        'ta': 'அக்ரிXAI',
        'bn': 'এগ্রিXAI',
        'mr': 'अ‍ॅग्रीXAI',
        'gu': 'એગ્રીXAI',
        'pa': 'ਐਗਰੀXAI',
    },
    'nav_home': {
        'en': 'Home',
        'te': 'హోమ్',
        'hi': 'होम',
        'ta': 'முகப்பு',
        'bn': 'হোম',
        'mr': 'होम',
        'gu': 'હોમ',
        'pa': 'ਹੋਮ',
    },
    'nav_profile': {
        'en': 'Profile',
        'te': 'ప్రొఫైల్',
        'hi': 'प्रोफ़ाइल',
        'ta': 'சுயவிவரம்',
        'bn': 'প্রোফাইল',
        'mr': 'प्रोफाइल',
        'gu': 'પ્રોફાઇલ',
        'pa': 'ਪ੍ਰੋਫ਼ਾਈਲ',
    },
    'nav_logout': {
        'en': 'Logout',
        'te': 'లాగ్ అవుట్',
        'hi': 'लॉग आउट',
        'ta': 'வெளியேறு',
        'bn': 'লগ আউট',
        'mr': 'लॉग आउट',
        'gu': 'લૉગ આઉટ',
        'pa': 'ਲੌਗ ਆਉਟ',
    },
    'nav_login': {
        'en': 'Login',
        'te': 'లాగిన్',
        'hi': 'लॉगिन',
        'ta': 'உள்நுழைக',
        'bn': 'লগইন',
        'mr': 'लॉगिन',
        'gu': 'લૉગિન',
        'pa': 'ਲੌਗ ਇਨ',
    },
    'nav_signup': {
        'en': 'Sign Up',
        'te': 'సైన్ అప్',
        'hi': 'साइन अप',
        'ta': 'பதிவு செய்க',
        'bn': 'সাইন আপ',
        'mr': 'साइन अप',
        'gu': 'સાઇન અપ',
        'pa': 'ਸਾਈਨ ਅੱਪ',
    },

    # Predict form
    'predict_enter_farm_details': {
        'en': 'Enter Farm Details',
        'te': 'ఫారం వివరాలు నమోదు చేయండి',
        'hi': 'खेत का विवरण दर्ज करें',
        'ta': 'பண்ணை விவரங்களை உள்ளிடவும்',
        'bn': 'খামারের বিবরণ লিখুন',
        'mr': 'शेतीची माहिती भरा',
        'gu': 'ફાર્મની વિગતો દાખલ કરો',
        'pa': 'ਖੇਤ ਦੀ ਜਾਣਕਾਰੀ ਦਾਖਲ ਕਰੋ',
    },
    'predict_label_crop': {
        'en': 'Crop',
        'te': 'పంట',
        'hi': 'फसल',
        'ta': 'பயிர்',
        'bn': 'ফসল',
        'mr': 'पीक',
        'gu': 'પાક',
        'pa': 'ਫਸਲ',
    },
    'predict_label_season': {
        'en': 'Season',
        'te': 'రుతువు',
        'hi': 'मौसम',
        'ta': 'காலம்',
        'bn': 'মৌসুম',
        'mr': 'हंगाम',
        'gu': 'મોસમ',
        'pa': 'ਮੌਸਮ',
    },
    'predict_label_state': {
        'en': 'State',
        'te': 'రాష్ట్రం',
        'hi': 'राज्य',
        'ta': 'மாநிலம்',
        'bn': 'রাজ্য',
        'mr': 'राज्य',
        'gu': 'રાજ્ય',
        'pa': 'ਰਾਜ',
    },
    'predict_label_area': {
        'en': 'Area (hectares)',
        'te': 'విస్తీర్ణం (హెక్టార్లు)',
        'hi': 'क्षेत्रफल (हेक्टेयर)',
        'ta': 'பரப்பு (ஹெக்டேயர்)',
        'bn': 'এলাকা (হেক্টর)',
        'mr': 'क्षेत्रफळ (हेक्टर)',
        'gu': 'વિસ્તાર (હેક્ટર)',
        'pa': 'ਖੇਤਰਫਲ (ਹੈਕਟੀਅਰ)',
    },
    'predict_label_rainfall': {
        'en': 'Annual Rainfall (mm)',
        'te': 'వార్షిక వర్షపాతం (మి.మీ)',
        'hi': 'वार्षिक वर्षा (मि.मी.)',
        'ta': 'ஆண்டு மழை (மிமீ)',
        'bn': 'বার্ষিক বৃষ্টি (মি.মি.)',
        'mr': 'वार्षिक पाऊस (मि.मी.)',
        'gu': 'વાર્ષિક વરસાદ (મીમી)',
        'pa': 'ਸਾਲਾਨਾ ਵਰਖਾ (ਮਿਮੀ)',
    },
    'predict_label_fertilizer': {
        'en': 'Fertilizer (kg)',
        'te': 'ఎరువు (కి.గ్రా.)',
        'hi': 'उर्वरक (किग्रा.)',
        'ta': 'உரங்கள் (கிலோ)',
        'bn': 'সার (কেজি)',
        'mr': 'खत (कि.ग्राम)',
        'gu': 'ખાતર (કિ.ગ્રા.)',
        'pa': 'ਖਾਦ (ਕਿਲੋ)',
    },
    'predict_label_pesticide': {
        'en': 'Pesticide (kg)',
        'te': 'పురుగు మందు (కి.గ్రా.)',
        'hi': 'कीटनाशक (किग्रा.)',
        'ta': 'பூச்சிக்கொல்லி (கிலோ)',
        'bn': 'কীটনাশক (কেজি)',
        'mr': 'कीटकनाशक (कि.ग्राम)',
        'gu': 'કીટનાશક (કિ.ગ્રા.)',
        'pa': 'ਕੀਟਨਾਸ਼ਕ (ਕਿਲੋ)',
    },
    'predict_button_predict': {
        'en': 'Predict Yield',
        'te': 'ఉత్పత్తిని అంచనా వేయండి',
        'hi': 'उपज का अनुमान लगाएं',
        'ta': 'விளைச்சலை கணிக்கவும்',
        'bn': 'ফলন অনুমান করুন',
        'mr': 'उत्पन्नाचा अंदाज लावा',
        'gu': 'પાક ઉત્પાદનનો અંદાજ લગાવો',
        'pa': 'ਪੈਦਾਵਾਰ ਦਾ ਅੰਦਾਜ਼ਾ ਲਗਾਓ',
    },

    # App tagline shown in headers
    'app_tagline': {
        'en': 'Intelligent Crop Yield Advisory System',
        'te': 'స్మార్ట్ పంట దిగుబడి సలహా వ్యవస్థ',
        'hi': 'स्मार्ट फ़सल उपज सलाह प्रणाली',
        'ta': 'செயல்முறை பயிர் விளைச்சல் ஆலோசனை அமைப்பு',
        'bn': 'স্মার্ট ফসল ফলন পরামর্শ ব্যবস্থা',
        'mr': 'स्मार्ट पिक उत्पादन सल्ला प्रणाली',
        'gu': 'સ્માર્ટ પાક ઉપજ સલાહ પ્રણાલી',
        'pa': 'ਸਮਾਰਟ ਫ਼ਸਲ ਪੈਦਾਵਾਰ ਸਲਾਹ ਪ੍ਰਣਾਲੀ',
    },

    # Result page texts
    'result_title': {
        'en': 'Crop Yield Prediction Result',
        'te': 'పంట దిగుబడి అంచనా ఫలితం',
        'hi': 'फसल उपज पूर्वानुमान परिणाम',
        'ta': 'பயிர் விளைச்சல் கணிப்பு முடிவு',
        'bn': 'ফসল ফলন পূর্বাভাসের ফলাফল',
        'mr': 'पीक उत्पादन अंदाज निकाल',
        'gu': 'પાક ઉપજ અંદાજ પરિણામ',
        'pa': 'ਫਸਲ ਪੈਦਾਵਾਰ ਅਨੁਮਾਨ ਨਤੀਜਾ',
    },
    'result_predicted_yield': {
        'en': 'Predicted Yield',
        'te': 'అంచనా దిగుబడి',
        'hi': 'अनुमानित उपज',
        'ta': 'கணிக்கப்பட்ட விளைச்சல்',
        'bn': 'অনুমানিত ফলন',
        'mr': 'अनुमानित उत्पादन',
        'gu': 'અનુમાનિત ઉપજ',
        'pa': 'ਅਨੁਮਾਨਿਤ ਪੈਦਾਵਾਰ',
    },
    'result_status_high': {
        'en': 'Status: HIGH Yield',
        'te': 'స్థితి: అధిక దిగుబడి',
        'hi': 'स्थिति: उच्च उपज',
        'ta': 'நிலை: அதிக விளைச்சல்',
        'bn': 'অবস্থা: উচ্চ ফলন',
        'mr': 'स्थिती: जास्त उत्पादन',
        'gu': 'સ્થિતિ: ઊંચી ઉપજ',
        'pa': 'ਸਥਿਤੀ: ਵੱਧ ਪੈਦਾਵਾਰ',
    },
    'result_status_low': {
        'en': 'Status: LOW Yield',
        'te': 'స్థితి: తక్కువ దిగుబడి',
        'hi': 'स्थिति: कम उपज',
        'ta': 'நிலை: குறைந்த விளைச்சல்',
        'bn': 'অবস্থা: কম ফলন',
        'mr': 'स्थिती: कमी उत्पादन',
        'gu': 'સ્થિતિ: ઓછી ઉપજ',
        'pa': 'ਸਥਿਤੀ: ਘੱਟ ਪੈਦਾਵਾਰ',
    },
    'result_key_positive': {
        'en': 'Key Positive Factors',
        'te': 'ముఖ్యమైన అనుకూల కారకాలు',
        'hi': 'मुख्य सकारात्मक कारक',
        'ta': 'முக்கிய நேர்மறை காரணங்கள்',
        'bn': 'প্রধান ইতিবাচক উপাদান',
        'mr': 'मुख्य सकारात्मक घटक',
        'gu': 'મુખ્ય સકારાત્મક ઘટકો',
        'pa': 'ਮੁਖ਼ ਸਾਕਾਰਾਤਮਕ ਕਾਰਕ',
    },
    'result_key_negative': {
        'en': 'Factors Reducing Yield',
        'te': 'దిగుబడిని తగ్గించే కారకాలు',
        'hi': 'उपज घटाने वाले कारक',
        'ta': 'விளைச்சலைக் குறைக்கும் காரணங்கள்',
        'bn': 'ফলন কমানোর উপাদান',
        'mr': 'उत्पादन घटविणारे घटक',
        'gu': 'ઉપજ ઘટાડનારાં ઘટકો',
        'pa': 'ਪੈਦਾਵਾਰ ਘਟਾਉਣ ਵਾਲੇ ਕਾਰਨ',
    },
    'result_recommendations': {
        'en': 'Smart Recommendations',
        'te': 'స్మార్ట్ సూచనలు',
        'hi': 'स्मार्ट सुझाव',
        'ta': 'செயல்முறை பரிந்துரைகள்',
        'bn': 'স্মার্ট পরামর্শ',
        'mr': 'स्मार्ट शिफारसी',
        'gu': 'સ્માર્ટ ભલામણો',
        'pa': 'ਸਮਾਰਟ ਸਿਫ਼ਾਰਸ਼ਾਂ',
    },
    'result_predict_again': {
        'en': 'Predict Again',
        'te': 'మళ్లీ అంచనా వేయండి',
        'hi': 'फिर से अनुमान लगाएं',
        'ta': 'மீண்டும் கணிக்கவும்',
        'bn': 'পুনরায় পূর্বাভাস দিন',
        'mr': 'पुन्हा अंदाज लावा',
        'gu': 'ફરીથી અંદાજ લગાવો',
        'pa': 'ਮੁੜ ਅਨੁਮਾਨ ਲਗਾਓ',
    },
    'result_impact': {
        'en': 'Impact',
        'te': 'ప్రభావం',
        'hi': 'प्रभाव',
        'ta': 'தாக்கம்',
        'bn': 'প্রভাব',
        'mr': 'प्रभाव',
        'gu': 'પ્રભાવ',
        'pa': 'ਅਸਰ',
    },
    'feature_crop': {'en': 'Crop', 'te': 'పంట', 'hi': 'फसल', 'ta': 'பயிர்', 'bn': 'ফসল', 'mr': 'पीक', 'gu': 'પાક', 'pa': 'ਫਸਲ'},
    'feature_season': {'en': 'Season', 'te': 'రుతువు', 'hi': 'मौसम', 'ta': 'காலம்', 'bn': 'মৌসুম', 'mr': 'हंगाम', 'gu': 'મોસમ', 'pa': 'ਮੌਸਮ'},
    'feature_state': {'en': 'State', 'te': 'రాష్ట్రం', 'hi': 'राज्य', 'ta': 'மாநிலம்', 'bn': 'রাজ্য', 'mr': 'राज्य', 'gu': 'રાજ્ય', 'pa': 'ਰਾਜ'},
    'feature_area': {'en': 'Area', 'te': 'విస్తీర్ణం', 'hi': 'क्षेत्रफल', 'ta': 'பரப்பு', 'bn': 'এলাকা', 'mr': 'क्षेत्रफळ', 'gu': 'વિસ્તાર', 'pa': 'ਖੇਤਰਫਲ'},
    'feature_annual_rainfall': {'en': 'Annual Rainfall', 'te': 'వార్షిక వర్షపాతం', 'hi': 'वार्षिक वर्षा', 'ta': 'ஆண்டு மழை', 'bn': 'বার্ষিক বৃষ্টি', 'mr': 'वार्षिक पाऊस', 'gu': 'વાર્ષિક વરસાદ', 'pa': 'ਸਾਲਾਨਾ ਵਰਖਾ'},
    'feature_fertilizer': {'en': 'Fertilizer', 'te': 'ఎరువు', 'hi': 'उर्वरक', 'ta': 'உரம்', 'bn': 'সার', 'mr': 'खत', 'gu': 'ખાતર', 'pa': 'ਖਾਦ'},
    'feature_pesticide': {'en': 'Pesticide', 'te': 'పురుగు మందు', 'hi': 'कीटनाशक', 'ta': 'பூச்சிக்கொல்லி', 'bn': 'কীটনাশক', 'mr': 'कीटकनाशक', 'gu': 'કીટનાશક', 'pa': 'ਕੀਟਨਾਸ਼ਕ'},
    'feature_model_prediction': {'en': 'Model Prediction', 'te': 'మోడల్ అంచనా', 'hi': 'मॉडल अनुमान', 'ta': 'மாதிரி கணிப்பு', 'bn': 'মডেল পূর্বাভাস', 'mr': 'मॉडेल अंदाज', 'gu': 'મોડેલ અંદાજ', 'pa': 'ਮਾਡਲ ਅਨੁਮਾਨ'},
    'suggest_increase_fertilizer': {
        'en': 'Increase fertilizer carefully based on soil condition.',
        'te': 'నేల పరిస్థితిని బట్టి జాగ్రత్తగా ఎరువును పెంచండి.',
        'hi': 'मिट्टी की स्थिति के अनुसार उर्वरक सावधानी से बढ़ाएँ।',
        'ta': 'மண் நிலையைப் பொறுத்து உரத்தை கவனமாக அதிகரிக்கவும்.',
        'bn': 'মাটির অবস্থার ভিত্তিতে সার সতর্কভাবে বাড়ান।',
        'mr': 'मातीच्या परिस्थितीनुसार खत काळजीपूर्वक वाढवा.',
        'gu': 'માટી સ્થિતિ અનુસાર ખાતર સાવધાનીથી વધારો.',
        'pa': 'ਮਿੱਟੀ ਦੀ ਹਾਲਤ ਮੁਤਾਬਕ ਖਾਦ ਸਾਵਧਾਨੀ ਨਾਲ ਵਧਾਓ।',
    },
    'suggest_improve_pest_management': {
        'en': 'Improve pest management practices.',
        'te': 'పురుగు నియంత్రణ విధానాలను మెరుగుపరచండి.',
        'hi': 'कीट प्रबंधन के तरीकों में सुधार करें।',
        'ta': 'பூச்சி மேலாண்மை முறைகளை மேம்படுத்தவும்.',
        'bn': 'পোকামাকড় ব্যবস্থাপনা পদ্ধতি উন্নত করুন।',
        'mr': 'कीड व्यवस्थापन पद्धती सुधार करा.',
        'gu': 'કીટ વ્યવસ્થાપન પદ્ધતિઓ સુધારો.',
        'pa': 'ਕੀੜੇ ਪ੍ਰਬੰਧਨ ਤਰੀਕੇ ਸੁਧਾਰੋ।',
    },
    'suggest_optimize_land_usage': {
        'en': 'Optimize land usage for better productivity.',
        'te': 'మంచి ఉత్పాదకత కోసం భూమి వినియోగాన్ని మెరుగుపరచండి.',
        'hi': 'बेहतर उत्पादकता के लिए भूमि उपयोग को अनुकूलित करें।',
        'ta': 'மேம்பட்ட உற்பத்திக்காக நிலப் பயன்பாட்டை ஒழுங்குபடுத்தவும்.',
        'bn': 'ভালো উৎপাদনের জন্য জমির ব্যবহার অনুকূল করুন।',
        'mr': 'उत्तम उत्पादकतेसाठी जमीन वापर अनुकूल करा.',
        'gu': 'વધુ ઉત્પાદન માટે જમીન ઉપયોગને અનુકૂળ બનાવો.',
        'pa': 'ਵਧੀਆ ਉਤਪਾਦਕਤਾ ਲਈ ਜ਼ਮੀਨ ਦੀ ਵਰਤੋਂ ਸੁਧਾਰੋ।',
    },
    'suggest_consider_irrigation': {
        'en': 'Consider irrigation support during low rainfall.',
        'te': 'వర్షపాతం తక్కువగా ఉన్నప్పుడు నీటి పారుదల సహాయాన్ని పరిగణించండి.',
        'hi': 'कम वर्षा के दौरान सिंचाई की व्यवस्था पर विचार करें।',
        'ta': 'மழை குறைவாக இருக்கும் போது பாசன உதவியைப் பரிசீலிக்கவும்.',
        'bn': 'কম বৃষ্টিপাতের সময় সেচ ব্যবস্থার কথা বিবেচনা করুন।',
        'mr': 'पाऊस कमी असताना सिंचनाची सोय विचारात घ्या.',
        'gu': 'ઓછા વરસાદ દરમિયાન સિંચાઈ સહાય વિચાર કરો.',
        'pa': 'ਘੱਟ ਵਰਖਾ ਦੌਰਾਨ ਸਿੰਚਾਈ ਸਹਾਇਤਾ ਬਾਰੇ ਸੋਚੋ।',
    },
    'suggest_choose_optimal_season': {
        'en': 'Choose an optimal growing season.',
        'te': 'అనుకూలమైన సాగు రుతువును ఎంచుకోండి.',
        'hi': 'उपयुक्त खेती का मौसम चुनें।',
        'ta': 'சரியான பயிர் வளர்ப்பு காலத்தைத் தேர்வு செய்யவும்.',
        'bn': 'উপযুক্ত চাষের মৌসুম বেছে নিন।',
        'mr': 'योग्य लागवडीचा हंगाम निवडा.',
        'gu': 'ઉત્તમ વાવણી મોસમ પસંદ કરો.',
        'pa': 'ਉਚਿਤ ਬਿਜਾਈ ਮੌਸਮ ਚੁਣੋ।',
    },
    'suggest_regional_climate_matters': {
        'en': 'Regional climate conditions may affect yield.',
        'te': 'ప్రాంతీయ వాతావరణ పరిస్థితులు దిగుబడిని ప్రభావితం చేయవచ్చు.',
        'hi': 'क्षेत्रीय जलवायु परिस्थितियाँ उपज को प्रभावित कर सकती हैं।',
        'ta': 'பிராந்திய காலநிலை நிலைகள் விளைச்சலை பாதிக்கலாம்.',
        'bn': 'আঞ্চলিক জলবায়ু পরিস্থিতি ফলনকে প্রভাবিত করতে পারে।',
        'mr': 'प्रादेशिक हवामान परिस्थिती उत्पादनावर परिणाम करू शकते.',
        'gu': 'પ્રાદેશિક હવામાન પરિસ્થિતિઓ ઉપજને અસર કરી શકે છે.',
        'pa': 'ਖੇਤਰੀ ਮੌਸਮੀ ਹਾਲਾਤ ਪੈਦਾਵਾਰ ’ਤੇ ਅਸਰ ਕਰ ਸਕਦੇ ਹਨ।',
    },
    'suggest_consider_high_yield_varieties': {
        'en': 'Consider high-yield crop varieties.',
        'te': 'అధిక దిగుబడి పంట రకాలను పరిగణించండి.',
        'hi': 'उच्च उपज वाली फसल किस्मों पर विचार करें।',
        'ta': 'அதிக விளைச்சல் தரும் பயிர் வகைகளை பரிசீலிக்கவும்.',
        'bn': 'উচ্চ ফলনশীল ফসলের জাত বিবেচনা করুন।',
        'mr': 'जास्त उत्पादन देणाऱ्या पिकांच्या जाती विचारात घ्या.',
        'gu': 'ઉંચી ઉપજ આપતી પાક જાતો વિચાર કરો.',
        'pa': 'ਵੱਧ ਪੈਦਾਵਾਰ ਵਾਲੀਆਂ ਫਸਲੀ ਕਿਸਮਾਂ ਬਾਰੇ ਸੋਚੋ।',
    },
    'suggest_review_inputs': {
        'en': 'Review your input values and try again.',
        'te': 'మీ ఇన్‌పుట్ విలువలను పరిశీలించి మళ్లీ ప్రయత్నించండి.',
        'hi': 'अपने इनपुट मान जांचें और फिर से कोशिश करें।',
        'ta': 'உங்கள் உள்ளீட்டு மதிப்புகளை சரிபார்த்து மீண்டும் முயற்சிக்கவும்.',
        'bn': 'আপনার ইনপুট মানগুলো যাচাই করে আবার চেষ্টা করুন।',
        'mr': 'तुमची इनपुट मूल्ये तपासा आणि पुन्हा प्रयत्न करा.',
        'gu': 'તમારા ઇનપુટ મૂલ્યો તપાસો અને ફરી પ્રયાસ કરો.',
        'pa': 'ਆਪਣੀਆਂ ਇਨਪੁੱਟ ਵੈਲਿਊਜ਼ ਚੈੱਕ ਕਰਕੇ ਮੁੜ ਕੋਸ਼ਿਸ਼ ਕਰੋ।',
    },

    # Profile page
    'profile_page_title': {
        'en': 'Farmer Profile',
        'te': 'రైతు ప్రొఫైల్',
        'hi': 'किसान प्रोफ़ाइल',
        'ta': 'விவசாயி சுயவிவரம்',
        'bn': 'কৃষক প্রোফাইল',
        'mr': 'शेतकरी प्रोफाइल',
        'gu': 'ખેડૂત પ્રોફાઇલ',
        'pa': 'ਕਿਸਾਨ ਪ੍ਰੋਫ਼ਾਈਲ',
    },
    'profile_welcome': {
        'en': 'Welcome, {username}!',
        'te': 'స్వాగతం, {username}!',
        'hi': 'स्वागत है, {username}!',
        'ta': 'வரவேற்பு, {username}!',
        'bn': 'স্বাগতম, {username}!',
        'mr': 'स्वागत आहे, {username}!',
        'gu': 'સ્વાગત છે, {username}!',
        'pa': 'ਜੀ ਆਇਆਂ ਨੂੰ, {username}!',
    },
    'profile_dashboard': {
        'en': 'Your Agricultural Intelligence Dashboard',
        'te': 'మీ వ్యవసాయ మేధస్సు డాష్‌బోర్డ్',
        'hi': 'आपका कृषि बुद्धिमत्ता डैशबोर्ड',
        'ta': 'உங்கள் விவசாய அறிவு டாஷ்போர்டு',
        'bn': 'আপনার কৃষি বুদ্ধিমত্তা ড্যাশবোর্ড',
        'mr': 'तुमचे कृषी बुद्धिमत्ता डॅशबोर्ड',
        'gu': 'તમારું કૃષિ બુદ્ધિમત્તા ડેશબોર્ડ',
        'pa': 'ਤੁਹਾਡਾ ਖੇਤੀਬਾੜੀ ਇੰਟੈਲੀਜੈਂਸ ਡੈਸ਼ਬੋਰਡ',
    },
    'profile_account_info': {
        'en': 'Account Information',
        'te': 'ఖాతా సమాచారం',
        'hi': 'खाता जानकारी',
        'ta': 'கணக்கு தகவல்',
        'bn': 'অ্যাকাউন্ট তথ্য',
        'mr': 'खातेची माहिती',
        'gu': 'ખાતાની માહિતી',
        'pa': 'ਖਾਤਾ ਜਾਣਕਾਰੀ',
    },
    'profile_username': {
        'en': 'Username',
        'te': 'వినియోగదారు పేరు',
        'hi': 'उपयोगकर्ता नाम',
        'ta': 'பயனர் பெயர்',
        'bn': 'ব্যবহারকারীর নাম',
        'mr': 'वापरकर्ता नाव',
        'gu': 'વપરાશકર્તા નામ',
        'pa': 'ਯੂਜ਼ਰਨੇਮ',
    },
    'profile_email': {
        'en': 'Email',
        'te': 'ఇమెయిల్',
        'hi': 'ईमेल',
        'ta': 'மின்னஞ்சல்',
        'bn': 'ইমেইল',
        'mr': 'ईमेल',
        'gu': 'ઈમેલ',
        'pa': 'ਈਮੇਲ',
    },
    'profile_member_since': {
        'en': 'Member Since',
        'te': 'సభ్యత్వం నుండి',
        'hi': 'सदस्य since',
        'ta': 'உறுப்பினர் என்பதிலிருந்து',
        'bn': 'সদস্য হওয়ার তারিখ',
        'mr': 'सदस्यत्वापासून',
        'gu': 'સભ્ય તરીકેથી',
        'pa': 'ਮੈਂਬਰ ਹੋਣ ਤੋਂ',
    },
    'profile_your_statistics': {
        'en': 'Your Statistics',
        'te': 'మీ గణాంకాలు',
        'hi': 'आपके आंकड़े',
        'ta': 'உங்கள் புள்ளிவிவரங்கள்',
        'bn': 'আপনার পরিসংখ্যান',
        'mr': 'तुमची आकडेवारी',
        'gu': 'તમારા આંકડા',
        'pa': 'ਤੁਹਾਡੇ ਅੰਕੜੇ',
    },
    'profile_total_predictions': {
        'en': 'Total Predictions',
        'te': 'మొత్తం అంచనాలు',
        'hi': 'कुल पूर्वानुमान',
        'ta': 'மொத்த கணிப்புகள்',
        'bn': 'মোট পূর্বাভাস',
        'mr': 'एकूण अंदाज',
        'gu': 'કુલ અંદાજો',
        'pa': 'ਕੁੱਲ ਅਨੁਮਾਨ',
    },
    'profile_account_status': {
        'en': 'Account Status',
        'te': 'ఖాతా స్థితి',
        'hi': 'खाता स्थिति',
        'ta': 'கணக்கு நிலை',
        'bn': 'অ্যাকাউন্টের অবস্থা',
        'mr': 'खातेची स्थिती',
        'gu': 'ખાતાની સ્થિતિ',
        'pa': 'ਖਾਤਾ ਸਥਿਤੀ',
    },
    'profile_status_active': {
        'en': 'Active',
        'te': 'సక్రియ',
        'hi': 'सक्रिय',
        'ta': 'செயலில்',
        'bn': 'সক্রিয়',
        'mr': 'सक्रिय',
        'gu': 'સક્રિય',
        'pa': 'ਸਰਗਰਮ',
    },
    'profile_last_login': {
        'en': 'Last Login',
        'te': 'చివరి లాగిన్',
        'hi': 'अंतिम लॉगिन',
        'ta': 'கடைசி உள்நுழைவு',
        'bn': 'শেষ লগইন',
        'mr': 'शेवटचे लॉगिन',
        'gu': 'છેલ્લું લોગિન',
        'pa': 'ਆਖਰੀ ਲੌਗਇਨ',
    },
    'profile_popular_crops': {
        'en': 'Popular Crops',
        'te': 'జనాదరణ పంటలు',
        'hi': 'लोकप्रिय फसलें',
        'ta': 'பிரபல பயிர்கள்',
        'bn': 'জনপ্রিয় ফসল',
        'mr': 'लोकप्रिय पिके',
        'gu': 'લોકપ્રિય પાક',
        'pa': 'ਪ੍ਰਸਿੱਧ ਫਸਲਾਂ',
    },
    'profile_logout_btn': {
        'en': 'Logout',
        'te': 'లాగ్ అవుట్',
        'hi': 'लॉग आउट',
        'ta': 'வெளியேறு',
        'bn': 'লগ আউট',
        'mr': 'लॉग आउट',
        'gu': 'લૉગ આઉટ',
        'pa': 'ਲੌਗ ਆਉਟ',
    },
    'profile_alt_farmer': {
        'en': 'Farmer Profile',
        'te': 'రైతు ప్రొఫైల్',
        'hi': 'किसान प्रोफ़ाइल',
        'ta': 'விவசாயி சுயவிவரம்',
        'bn': 'কৃষক প্রোফাইল',
        'mr': 'शेतकरी प्रोफाइल',
        'gu': 'ખેડૂત પ્રોફાઇલ',
        'pa': 'ਕਿਸਾਨ ਪ੍ਰੋਫ਼ਾਈਲ',
    },
    'profile_crop_wheat': {'en': 'Wheat', 'te': 'గోధుమ', 'hi': 'गेहूं', 'ta': 'கோதுமை', 'bn': 'গম', 'mr': 'गहू', 'gu': 'ઘઉં', 'pa': 'ਕਣਕ'},
    'profile_crop_tomato': {'en': 'Tomato', 'te': 'టమోటా', 'hi': 'टमाटर', 'ta': 'தக்காளி', 'bn': 'টমেটো', 'mr': 'टोमॅटो', 'gu': 'ટમેટા', 'pa': 'ਟਮਾਟਰ'},
    'profile_crop_banana': {'en': 'Banana', 'te': 'అరటి', 'hi': 'केला', 'ta': 'வாழைப்பழம்', 'bn': 'কলা', 'mr': 'केळे', 'gu': 'કેળું', 'pa': 'ਕੇਲਾ'},
    'profile_crop_sugarcane': {'en': 'Sugarcane', 'te': 'చెరకు', 'hi': 'गन्ना', 'ta': 'கரும்பு', 'bn': 'আখ', 'mr': 'ऊस', 'gu': 'શેરડી', 'pa': 'ਗੰਨਾ'},
    'profile_update_contact_notebook': {
        'en': 'Update Contact & Notebook',
        'te': 'సంప్రదింపు & నోట్‌బుక్‌ను నవీకరించండి',
        'hi': 'संपर्क और नोटबुक अपडेट करें',
        'ta': 'தொடர்பு & குறிப்புப்புத்தகத்தை புதுப்பிக்கவும்',
        'bn': 'যোগাযোগ ও নোটবুক আপডেট করুন',
        'mr': 'संपर्क आणि नोटबुक अद्यतनित करा',
        'gu': 'સંપર્ક અને નોટબુક અપડેટ કરો',
        'pa': 'ਸੰਪਰਕ ਅਤੇ ਨੋਟਬੁੱਕ ਅੱਪਡੇਟ ਕਰੋ',
    },
    'profile_email_gmail_label': {
        'en': 'Email (Gmail)',
        'te': 'ఇమెయిల్ (Gmail)',
        'hi': 'ईमेल (Gmail)',
        'ta': 'மின்னஞ்சல் (Gmail)',
        'bn': 'ইমেইল (Gmail)',
        'mr': 'ईमेल (Gmail)',
        'gu': 'ઈમેલ (Gmail)',
        'pa': 'ਈਮੇਲ (Gmail)',
    },
    'profile_phone_label': {
        'en': 'Phone Number (10 digits)',
        'te': 'ఫోన్ నంబర్ (10 అంకెలు)',
        'hi': 'फ़ोन नंबर (10 अंक)',
        'ta': 'தொலைபேசி எண் (10 இலக்கங்கள்)',
        'bn': 'ফোন নাম্বার (১০ সংখ্যা)',
        'mr': 'फोन नंबर (१० अंक)',
        'gu': 'ફોન નંબર (10 અંક)',
        'pa': 'ਫੋਨ ਨੰਬਰ (10 ਅੰਕ)',
    },
    'profile_farmer_notebook_label': {
        'en': 'Farmer Notebook',
        'te': 'రైతు నోట్‌బుక్',
        'hi': 'किसान नोटबुक',
        'ta': 'விவசாயி குறிப்புப்புத்தகம்',
        'bn': 'কৃষক নোটবুক',
        'mr': 'शेतकरी वही',
        'gu': 'ખેડૂત નોટબુક',
        'pa': 'ਕਿਸਾਨ ਨੋਟਬੁੱਕ',
    },
    'profile_farmer_notebook_placeholder': {
        'en': 'Write your farming notes, observations, or plans here...',
        'te': 'మీ వ్యవసాయ గమనికలు, పరిశీలనలు లేదా ప్రణాళికలను ఇక్కడ వ్రాయండి...',
        'hi': 'अपनी खेती से जुड़ी नोट्स, निरीक्षण या योजनाएँ यहाँ लिखें...',
        'ta': 'உங்கள் விவசாய குறிப்புகள், கவனிப்புகள் அல்லது திட்டங்களை இங்கே எழுதுங்கள்...',
        'bn': 'আপনার কৃষি নোট, পর্যবেক্ষণ বা পরিকল্পনা এখানে লিখুন...',
        'mr': 'तुमचे शेतीसंबंधी टिपण, निरीक्षणे किंवा योजना येथे लिहा...',
        'gu': 'તમારા ખેતીના નોંધ, નિરીક્ષણો અથવા યોજનાઓ અહીં લખો...',
        'pa': 'ਆਪਣੇ ਖੇਤੀਬਾੜੀ ਨੋਟ, ਵੇਖਭਾਲ ਜਾਂ ਯੋਜਨਾਵਾਂ ਇੱਥੇ ਲਿਖੋ...',
    },
    'profile_save_profile_button': {
        'en': 'Save Profile',
        'te': 'ప్రొఫైల్‌ను సేవ్ చేయండి',
        'hi': 'प्रोफ़ाइल सहेजें',
        'ta': 'சுயவிவரத்தை சேமிக்கவும்',
        'bn': 'প্রোফাইল সংরক্ষণ করুন',
        'mr': 'प्रोफाइल जतन करा',
        'gu': 'પ્રોફાઇલ સાચવો',
        'pa': 'ਪ੍ਰੋਫ਼ਾਈਲ ਸੇਵ ਕਰੋ',
    },
}


def get_user_language():
    """Return current UI language code from session or default."""
    lang = session.get('language')
    if lang in LANGUAGES:
        return lang
    return DEFAULT_LANGUAGE


def translate(key):
    """Translate a simple UI string based on the active language."""
    lang = get_user_language()
    per_lang = TRANSLATIONS.get(key, {})
    return per_lang.get(lang) or per_lang.get(DEFAULT_LANGUAGE) or key


@app.context_processor
def inject_language():
    """Make language data and translator available in all templates."""
    return {
        'languages': LANGUAGES,
        'current_language': get_user_language(),
        't': translate,
    }


@app.route('/set_language', methods=['POST'])
def set_language():
    """Set the preferred UI language after login and redirect back."""
    if not check_login():
        return redirect(url_for('login'))
    
    lang = request.form.get('language')
    if lang in LANGUAGES:
        session['language'] = lang
    
    # Return to the previous page or prediction page as a fallback
    next_url = request.referrer or url_for('predict')
    return redirect(next_url)

# ----------------------------
# Authentication Routes
# ----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        id_token = request.form.get('id_token', '').strip()
        if id_token:
            decoded_token, token_error = verify_firebase_id_token(id_token)
            if token_error:
                return render_template(
                    'login.html',
                    error=token_error,
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )

            requested_username = request.form.get('username', '').strip()
            username = create_or_get_user_from_firebase(decoded_token, requested_username=requested_username)
            if not username:
                return render_template(
                    'login.html',
                    error='Unable to create local session from Firebase account.',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )

            session.permanent = True
            session['username'] = username
            users[username]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return redirect(url_for('predict'))

        email = request.form.get('email', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not email or not username or not password:
            return render_template(
                'login.html',
                error='Please fill in all fields',
                firebase_config=get_firebase_web_config(),
                firebase_enabled=FIREBASE_ENABLED,
                firebase_web_ready=firebase_web_config_ready()
            )

        if not is_valid_email(email):
            return render_template(
                'login.html',
                error='Please enter a valid Gmail address (starts with a small letter and ends with @gmail.com)',
                firebase_config=get_firebase_web_config(),
                firebase_enabled=FIREBASE_ENABLED,
                firebase_web_ready=firebase_web_config_ready()
            )
        
        hashed_password = hash_password(password)
        
        if username in users and users[username]['password'] == hashed_password:
            session.permanent = True
            session['username'] = username
            users[username]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return redirect(url_for('predict'))
        else:
            return render_template(
                'login.html',
                error='Invalid username or password',
                firebase_config=get_firebase_web_config(),
                firebase_enabled=FIREBASE_ENABLED,
                firebase_web_ready=firebase_web_config_ready()
            )
    
    return render_template(
        'login.html',
        firebase_config=get_firebase_web_config(),
        firebase_enabled=FIREBASE_ENABLED,
        firebase_web_ready=firebase_web_config_ready()
    )

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        id_token = request.form.get('id_token', '').strip()
        if id_token:
            decoded_token, token_error = verify_firebase_id_token(id_token)
            if token_error:
                return render_template(
                    'signup.html',
                    error=token_error,
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )

            requested_username = request.form.get('username', '').strip()
            if not requested_username:
                return render_template(
                    'signup.html',
                    error='Username is required to create your local profile.',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )

            username = create_or_get_user_from_firebase(decoded_token, requested_username=requested_username)
            if not username:
                return render_template(
                    'signup.html',
                    error='Unable to create local session from Firebase account.',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )

            session.permanent = True
            session['username'] = username
            users[username]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return redirect(url_for('predict'))

        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        signup_method = request.form.get('signup_method', 'email')
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        if not username or not password:
            return render_template(
                'signup.html',
                error='Username and password are required',
                firebase_config=get_firebase_web_config(),
                firebase_enabled=FIREBASE_ENABLED,
                firebase_web_ready=firebase_web_config_ready()
            )

        # User must choose exactly one method: email OR phone
        if signup_method not in ('email', 'phone'):
            return render_template(
                'signup.html',
                error='Please choose how you want to sign up: with email or phone number',
                firebase_config=get_firebase_web_config(),
                firebase_enabled=FIREBASE_ENABLED,
                firebase_web_ready=firebase_web_config_ready()
            )

        if signup_method == 'email':
            if not email:
                return render_template(
                    'signup.html',
                    error='Please enter your Gmail address to sign up with email',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )
            if phone:
                return render_template(
                    'signup.html',
                    error='You selected email signup. Please leave the phone field empty',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )
            if not is_valid_email(email):
                return render_template(
                    'signup.html',
                    error='Please enter a valid Gmail address (starts with a small letter and ends with @gmail.com)',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )

        if signup_method == 'phone':
            if not phone:
                return render_template(
                    'signup.html',
                    error='Please enter your 10-digit phone number to sign up with phone',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )
            if email:
                return render_template(
                    'signup.html',
                    error='You selected phone signup. Please leave the email field empty',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )
            if not is_valid_phone(phone):
                return render_template(
                    'signup.html',
                    error='Phone number must be exactly 10 digits',
                    firebase_config=get_firebase_web_config(),
                    firebase_enabled=FIREBASE_ENABLED,
                    firebase_web_ready=firebase_web_config_ready()
                )
        
        if password != confirm_password:
            return render_template(
                'signup.html',
                error='Passwords do not match',
                firebase_config=get_firebase_web_config(),
                firebase_enabled=FIREBASE_ENABLED,
                firebase_web_ready=firebase_web_config_ready()
            )
        
        if len(password) < 6:
            return render_template(
                'signup.html',
                error='Password must be at least 6 characters',
                firebase_config=get_firebase_web_config(),
                firebase_enabled=FIREBASE_ENABLED,
                firebase_web_ready=firebase_web_config_ready()
            )
        
        if username in users:
            return render_template(
                'signup.html',
                error='Username already exists',
                firebase_config=get_firebase_web_config(),
                firebase_enabled=FIREBASE_ENABLED,
                firebase_web_ready=firebase_web_config_ready()
            )

        # Ensure email is unique if provided
        if email:
            for existing_user in users.values():
                if existing_user.get('email') == email:
                    return render_template(
                        'signup.html',
                        error='Email is already associated with another account',
                        firebase_config=get_firebase_web_config(),
                        firebase_enabled=FIREBASE_ENABLED,
                        firebase_web_ready=firebase_web_config_ready()
                    )

        # Ensure phone is unique if provided
        if phone:
            for existing_user in users.values():
                if existing_user.get('phone') == phone:
                    return render_template(
                        'signup.html',
                        error='Phone number is already associated with another account',
                        firebase_config=get_firebase_web_config(),
                        firebase_enabled=FIREBASE_ENABLED,
                        firebase_web_ready=firebase_web_config_ready()
                    )
        
        users[username] = {
            'password': hash_password(password),
            'email': email if email else None,
            'phone': phone if phone else None,
            'join_date': datetime.now().strftime('%Y-%m-%d'),
            'last_login': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': 0,
            'notes': ''
        }
        
        session.permanent = True
        session['username'] = username
        return redirect(url_for('predict'))
    
    return render_template(
        'signup.html',
        firebase_config=get_firebase_web_config(),
        firebase_enabled=FIREBASE_ENABLED,
        firebase_web_ready=firebase_web_config_ready()
    )

@app.route('/guest')
def guest_login():
    """One-click guest/demo access — no registration required."""
    guest_username = 'guest_demo'
    if guest_username not in users:
        users[guest_username] = {
            'password': None,
            'email': 'demo@agrixai.example',
            'phone': None,
            'join_date': datetime.now().strftime('%Y-%m-%d'),
            'last_login': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predictions': 0,
            'notes': 'Welcome to AgriXAI! This is a demo account for project demonstration.'
        }
    users[guest_username]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    session.permanent = True
    session['username'] = guest_username
    session['is_guest'] = True
    return redirect(url_for('predict'))

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('username', None)
    session.pop('is_guest', None)
    return redirect(url_for('welcome'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if not check_login():
        return redirect(url_for('login'))
    
    username = session['username']
    user_data = users.get(username, {})

    profile_error = None
    profile_success = None

    if request.method == 'POST':
        new_email = request.form.get('email', '').strip()
        new_phone = request.form.get('phone', '').strip()
        new_notes = request.form.get('notes', '').strip()

        # Validate email format if provided
        if new_email and not is_valid_email(new_email):
            profile_error = 'Please enter a valid Gmail address (starts with a small letter and ends with @gmail.com)'
        # Validate phone format if provided
        elif new_phone and not is_valid_phone(new_phone):
            profile_error = 'Phone number must be exactly 10 digits'
        else:
            # Ensure email is unique among other users
            if new_email:
                for other_username, existing_user in users.items():
                    if other_username == username:
                        continue
                    if existing_user.get('email') == new_email:
                        profile_error = 'Email is already associated with another account'
                        break

            # Ensure phone is unique among other users
            if not profile_error and new_phone:
                for other_username, existing_user in users.items():
                    if other_username == username:
                        continue
                    if existing_user.get('phone') == new_phone:
                        profile_error = 'Phone number is already associated with another account'
                        break

        if not profile_error:
            # Apply updates
            if new_email:
                user_data['email'] = new_email
            if new_phone:
                user_data['phone'] = new_phone

            user_data['notes'] = new_notes
            users[username] = user_data
            profile_success = 'Profile updated successfully'

    return render_template(
        'profile.html',
        username=username,
        email=user_data.get('email'),
        phone=user_data.get('phone'),
        notes=user_data.get('notes', ''),
        join_date=user_data.get('join_date', 'N/A'),
        last_login=user_data.get('last_login', 'N/A'),
        total_predictions=user_data.get('predictions', 0),
        profile_error=profile_error,
        profile_success=profile_success,
    )

# ----------------------------
# Welcome/Landing Route
# ----------------------------
@app.route('/')
def welcome():
    # If already logged in, redirect to prediction page
    if check_login():
        return redirect(url_for('predict'))
    return render_template("welcome.html")

# ----------------------------
# Prediction Page Route (GET) and Submission Route (POST)
# ----------------------------
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if user is logged in
    if not check_login():
        # If POST request without login, redirect to login with a message
        if request.method == 'POST':
            flash('Please login to make predictions', 'warning')
            return redirect(url_for('login'))
        # If GET request without login, redirect to login
        return redirect(url_for('login'))
    
    # Handle GET request - show prediction form
    if request.method == 'GET':
        return render_template("predict.html", logged_in=check_login(), username=session.get('username'),
                               crops=CROP_LIST, seasons=SEASON_LIST, states=STATE_LIST)
    
    # Handle POST request - process prediction
    # Double check login status (should already be checked above, but just in case)
    if not check_login():
        return redirect(url_for('login'))
    
    # Check if model is loaded
    if model is None or encoders is None or y is None:
        return render_template(
            "result.html",
            yield_value=0,
            status="ERROR",
            positive=[],
            negative=[],
            suggestions=["Error: Model files not loaded. Please check the model directory."],
            logged_in=check_login(),
            username=session.get('username')
        )
    
    # Update prediction count
    username = session.get('username')
    if username in users:
        users[username]['predictions'] = users[username].get('predictions', 0) + 1

    # Collect user input
    farmer_input = {
        "Crop": request.form["Crop"],
        "Season": request.form["Season"],
        "State": request.form["State"],
        "Area": float(request.form["Area"]),
        "Annual_Rainfall": float(request.form["Annual_Rainfall"]),
        "Fertilizer": float(request.form["Fertilizer"]),
        "Pesticide": float(request.form["Pesticide"])
    }

    input_df = pd.DataFrame([farmer_input])

    # Clean categorical columns
    for col in ["Crop", "Season", "State"]:
        input_df[col] = input_df[col].astype(str).str.strip().str.title()

    # Encode categorical features
    for col in ["Crop", "Season", "State"]:
        input_df[col] = encoders[col].transform(input_df[col])

    # Match feature order
    input_df = input_df[model.feature_names_in_]

    # ----------------------------
    # Prediction
    # ----------------------------
    predicted_yield = model.predict(input_df)[0]

    # Use median instead of mean so roughly half of
    # predictions are classified as HIGH and half as LOW.
    # This avoids everything being marked LOW when the
    # dataset has a few very large yield values.
    threshold_yield = y.median()
    yield_status = "HIGH" if predicted_yield >= threshold_yield else "LOW"

    # ----------------------------
    # SHAP Explanation (if available)
    # ----------------------------
    top_positive = []
    top_negative = []
    
    if SHAP_AVAILABLE and explainer is not None:
        try:
            shap_values = explainer.shap_values(input_df)[0]

            # Pair features with impact values
            feature_impact = sorted(
                zip(input_df.columns, shap_values),
                key=lambda x: abs(x[1]),
                reverse=True
            )

            for feature, value in feature_impact:
                if value > 0:
                    top_positive.append((feature, round(value, 3)))
                elif value < 0:
                    top_negative.append((feature, round(value, 3)))

            # Take top 3 most important features
            top_positive = top_positive[:3]
            top_negative = top_negative[:3]
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            # Fallback: use feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = list(zip(input_df.columns, model.feature_importances_))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                top_positive = [(feat, round(imp, 3)) for feat, imp in feature_importance[:3]]
    else:
        # Fallback: use feature importance if SHAP is not available
        if hasattr(model, 'feature_importances_'):
            feature_importance = list(zip(input_df.columns, model.feature_importances_))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_positive = [(feat, round(imp, 3)) for feat, imp in feature_importance[:3]]
            top_negative = []
        else:
            # If no feature importance available, provide generic suggestions
            top_positive = [("Model Prediction", round(predicted_yield, 2))]
            top_negative = []

    # ----------------------------
    # Smart Suggestions
    # ----------------------------
    suggestions = []

    for feature, value in top_negative:

        if feature == "Fertilizer":
            suggestions.append(suggestion_key_for_feature(feature) or "suggest_increase_fertilizer")

        elif feature == "Pesticide":
            suggestions.append(suggestion_key_for_feature(feature) or "suggest_improve_pest_management")

        elif feature == "Area":
            suggestions.append(suggestion_key_for_feature(feature) or "suggest_optimize_land_usage")

        elif feature == "Annual_Rainfall":
            suggestions.append(suggestion_key_for_feature(feature) or "suggest_consider_irrigation")

        elif feature == "Season":
            suggestions.append(suggestion_key_for_feature(feature) or "suggest_choose_optimal_season")

        elif feature == "State":
            suggestions.append(suggestion_key_for_feature(feature) or "suggest_regional_climate_matters")

        elif feature == "Crop":
            suggestions.append(suggestion_key_for_feature(feature) or "suggest_consider_high_yield_varieties")

    # ----------------------------
    # Render Result Page
    # ----------------------------
    return render_template(
        "result.html",
        yield_value=round(predicted_yield, 2),
        status=yield_status,
        positive=[(translate_feature_key(f), v, f) for f, v in top_positive],
        negative=[(translate_feature_key(f), v, f) for f, v in top_negative],
        suggestions=suggestions,
        logged_in=check_login(),
        username=session.get('username')
    )

# ----------------------------
# Health Check
# ----------------------------
@app.route('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None, 'shap_available': SHAP_AVAILABLE}, 200

# ----------------------------
# Run Application
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)