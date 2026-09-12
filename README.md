# 🌾 AgriXAI - Intelligent Crop Yield Advisory System

A web-based application that uses Machine Learning and Explainable AI (XAI) to predict crop yields and provide actionable insights for farmers. Built with Flask, featuring SHAP explanations for transparent AI decision-making.

## 📋 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [API Routes](#api-routes)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)

## ✨ Features

### 🔐 Authentication System  
- **User Registration**: Create farmer accounts with username and password
- **Secure Login**: Session-based authentication with password hashing
- **Firebase Integration**: Email/password and phone OTP authentication support
- **User Profiles**: View account information and prediction statistics
- **Password Visibility Toggle**: Eye icon to show/hide passwords

### 🌾 Crop Yield Prediction
- **AI-Powered Predictions**: Machine learning model predicts crop yields based on:
  - Crop type
  - Season
  - State/Region
  - Area (hectares)
  - Annual Rainfall (mm)
  - Fertilizer usage (kg)
  - Pesticide usage (kg)

### 📊 Explainable AI (XAI)
- **SHAP Explanations**: Understand which factors positively or negatively impact yield
- **Feature Importance**: Visual breakdown of contributing factors
- **Smart Recommendations**: Actionable suggestions to improve crop yield

### 🎨 User Interface
- **Green Theme**: Farm-friendly green color scheme throughout
- **Responsive Design**: Works on desktop and mobile devices
- **Farmer & Crop Images**: Visual elements showcasing agricultural themes
- **Intuitive Navigation**: Easy-to-use interface for farmers

## 🛠 Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: 
  - scikit-learn (Random Forest/XGBoost model)
  - SHAP (SHapley Additive exPlanations)
  - pandas (Data manipulation)
  - joblib (Model serialization)
- **Frontend**: HTML5, CSS3, JavaScript
- **Authentication**: Flask sessions + Firebase Authentication (email and phone)
- **Styling**: Custom CSS with green agricultural theme

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone or Download the Project
```bash
cd "c:\Users\abhis\OneDrive\Documents\Class\MP\Project"
```

### Step 2: Install Dependencies

**Option 1: Using requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option 2: Using installation script**
```powershell
# For Windows PowerShell
.\install_dependencies.ps1

# For Windows Command Prompt
install_dependencies.bat
```

**Option 3: Manual Installation**
```bash
pip install Flask==3.0.0 pandas==2.1.4 scikit-learn==1.3.2 scipy==1.11.4 shap==0.44.0 joblib==1.3.2 numpy==1.26.2
```

### Step 3: Prepare Model Files
Ensure you have the following model files in the `model/` directory:
- `model/yield_model.pkl` - Trained ML model
- `model/encoders.pkl` - Label encoders for categorical features
- `model/target_values.pkl` - Target values for comparison

> **Note**: If you don't have these files, train the model using `train.ipynb` first.

## 📁 Project Structure

```
Project/
│
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── install_dependencies.ps1    # PowerShell installation script
├── install_dependencies.bat    # Batch installation script
│
├── templates/                  # HTML templates
│   ├── welcome.html           # Landing page with slogan
│   ├── login.html             # Login page
│   ├── signup.html            # Registration page
│   ├── predict.html           # Prediction form
│   ├── result.html            # Prediction results
│   └── profile.html           # User profile page
│
├── model/                      # ML model files (create this folder)
│   ├── yield_model.pkl
│   ├── encoders.pkl
│   └── target_values.pkl
│
└── train.ipynb                 # Jupyter notebook for model training
```

## 🚀 Usage Guide

### Starting the Application

1. **Navigate to project directory**
   ```bash
   cd "c:\Users\abhis\OneDrive\Documents\Class\MP\Project"
   ```

2. **Run the Flask application**
   ```bash
   python app.py
   ```

3. **Access the application**
   - Open your web browser
   - Navigate to: `http://localhost:5000` or `http://127.0.0.1:5000`

### User Flow

1. **Welcome Page** (`/`)
   - View the landing page with the slogan: "ALWAYS DO YOUR BEST - WHAT YOU PLAN NOW, YOU WILL HARVEST LATER"
   - Click "Login" or "Sign Up" buttons

2. **Registration** (`/signup`)
   - Create a new account with:
     - Username (required)
     - Email (optional)
     - Password (minimum 6 characters)
     - Confirm Password
   - Click "Create Account" button

3. **Login** (`/login`)
   - Enter username and password
   - Click "Login to AgriXAI"
   - Use eye icon (👁️) to toggle password visibility

4. **Prediction Page** (`/predict`)
   - After login, you'll be redirected to the prediction form
   - Fill in the farm details:
     - 🌱 Crop (e.g., Rice, Wheat, Corn)
     - 🌦 Season (e.g., Kharif, Rabi)
     - 📍 State (e.g., Andhra Pradesh, Punjab)
     - 🌾 Area in hectares
     - 🌧 Annual Rainfall in mm
     - 🧪 Fertilizer in kg
     - 🐛 Pesticide in kg
   - Click "🌿 Predict Yield"

5. **Results Page**
   - View predicted yield value
   - See HIGH or LOW yield status
   - Review positive factors affecting yield
   - Check negative factors reducing yield
   - Read smart recommendations
   - Click "Predict Again" to make another prediction

6. **Profile Page** (`/profile`)
   - View account information
   - Check prediction statistics
   - See popular crops

## 🔌 API Routes

| Route | Method | Description | Auth Required |
|-------|--------|-------------|---------------|
| `/` | GET | Welcome/landing page | No |
| `/login` | GET, POST | User login | No |
| `/signup` | GET, POST | User registration | No |
| `/logout` | GET, POST | User logout | Yes |
| `/predict` | GET | Show prediction form | Yes |
| `/predict` | POST | Process prediction | Yes |
| `/profile` | GET | User profile page | Yes |

## 📋 Requirements

### Python Packages
```
Flask==3.0.0
pandas==2.1.4
scikit-learn==1.3.2
scipy==1.11.4
shap==0.44.0
joblib==1.3.2
numpy==1.26.2
```

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- Web browser (Chrome, Firefox, Edge, Safari)

## ⚙️ Configuration

### Secret Key
The application generates a random secret key automatically. For production:

1. Set environment variable:
   ```bash
   # Windows PowerShell
   $env:SECRET_KEY="your-secret-key-here"
   
   # Linux/Mac
   export SECRET_KEY="your-secret-key-here"
   ```

2. Or modify `app.py` to use a fixed key (not recommended for production)

### Firebase Authentication Setup

1. Create a Firebase project and enable:
   - Email/Password provider
   - Phone provider
2. Download a Firebase service account JSON file.
3. Set these environment variables before running `python app.py`:

```powershell
$env:FIREBASE_SERVICE_ACCOUNT_PATH="C:\path\to\serviceAccountKey.json"
$env:FIREBASE_WEB_API_KEY="your-web-api-key"
$env:FIREBASE_WEB_AUTH_DOMAIN="your-project.firebaseapp.com"
$env:FIREBASE_WEB_PROJECT_ID="your-project-id"
$env:FIREBASE_WEB_STORAGE_BUCKET="your-project.appspot.com"
$env:FIREBASE_WEB_MESSAGING_SENDER_ID="1234567890"
$env:FIREBASE_WEB_APP_ID="1:1234567890:web:abcdef123456"
```

4. In Firebase Console -> Authentication -> Settings, add your local domain:
   - `localhost`
   - `127.0.0.1`
5. For phone OTP in development, configure reCAPTCHA and test phone numbers in Firebase Console.

### Session Configuration
- Sessions are permanent by default
- Session lifetime: 24 hours
- Stored in browser cookies

### Model Files
Place your trained model files in the `model/` directory:
- `yield_model.pkl` - Your trained scikit-learn model
- `encoders.pkl` - Dictionary of LabelEncoders for categorical features
- `target_values.pkl` - pandas Series with target values for comparison

## 🐛 Troubleshooting

### Issue: SHAP Import Error
**Error**: `ModuleNotFoundError: No module named 'shap'`

**Solution**:
```bash
pip install shap scipy scikit-learn
```

The application will still work without SHAP, but explanations will be limited.

### Issue: Model Files Not Found
**Error**: `FileNotFoundError: model/yield_model.pkl`

**Solution**:
1. Train your model using `train.ipynb`
2. Save model files in `model/` directory
3. Ensure filenames match exactly

### Issue: Redirected to Login When Clicking Predict
**Possible Causes**:
- Session expired
- Not logged in
- Cookies disabled

**Solution**:
1. Log in again
2. Enable cookies in browser
3. Clear browser cache and try again

### Issue: Port Already in Use
**Error**: `Address already in use`

**Solution**:
```bash
# Change port in app.py
app.run(debug=True, port=5001)
```

## 🔮 Future Improvements

- [ ] Database integration (SQLite/PostgreSQL) for user storage
- [ ] Password reset functionality
- [ ] Email notifications
- [ ] Historical prediction tracking
- [ ] Export predictions to PDF/Excel
- [ ] Multi-language support
- [ ] Advanced visualizations (charts, graphs)
- [ ] Weather API integration
- [ ] Mobile app version
- [ ] Admin dashboard
- [ ] Crop recommendation system
- [ ] Soil analysis integration

## 📝 Notes

### Security Considerations
- **Current**: Simple password hashing (SHA-256)
- **Production**: Use bcrypt or Argon2 for password hashing
- **Current**: In-memory user storage
- **Production**: Use a proper database (PostgreSQL, MySQL)

### Model Training
To train your own model:
1. Open `train.ipynb` in Jupyter Notebook
2. Follow the notebook cells to:
   - Load and preprocess data
   - Train models (Random Forest, XGBoost, etc.)
   - Evaluate model performance
   - Save model and encoders

### Data Format
The model expects input in this format:
- **Crop**: String (e.g., "Rice", "Wheat")
- **Season**: String (e.g., "Kharif", "Rabi")
- **State**: String (e.g., "Andhra Pradesh")
- **Area**: Float (hectares)
- **Annual_Rainfall**: Float (mm)
- **Fertilizer**: Float (kg)
- **Pesticide**: Float (kg)

## 👥 Contributing

This is an academic project. For improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes as part of a Master's Project.

## 🙏 Acknowledgments

- SHAP library for explainable AI
- Flask community for excellent documentation
- Unsplash for agricultural images
- scikit-learn for machine learning tools

## 📧 Support

For issues or questions:
- Check the Troubleshooting section above
- Review the code comments in `app.py`
- Check Flask and SHAP documentation

---

**Slogan**: *"ALWAYS DO YOUR BEST - WHAT YOU PLAN NOW, YOU WILL HARVEST LATER"* 🌾

**Built with ❤️ for Farmers**
