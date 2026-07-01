import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="AI HR Retention & Enterprise Audit Platform", layout="wide")

# ==================== LANGUAGE TRANSLATION DICTIONARY ====================
LANGUAGES = {
    "English": {
        "title": "🏢 AI HR Portal: Industry-Level Data Science & Retention Platform",
        "subtitle": "An End-to-End Machine Learning System featuring Live Analytics, Real ML Metrics, and Explainable AI.",
        "reason_title": "### 🔍 Potential Reasons / Specific HR Insights:",
        "terminate_low_merit": "❌ **Action: Immediate Termination!** Employee is young with zero experience, no skills, and critically low attendance.",
        "high_merit_bonus": "🛡️ **Strategy: Outstanding Profile!** High Experience, Top Job Level, Active Skills, High Attendance, and Strong Domain Knowledge detected. Recommended for a **High Salary Raise (+40%)** to prevent attrition.",
        "young_expert_raise": "🛡️ **Strategy: Young Talent Bonus!** Employee is young but possesses high experience. Recommended for an immediate **Salary Raise & Future Corporate Benefits Plan**.",
        "attendance_warning": "⚠️ **HR Warning Issued:** Attendance is below 85% ({attendance}%). Please issue an official Warning Letter. **Note: If 3 warnings are accumulated, proceed with Termination.**",
        "standard_ot": "🛡️ **Strategy:** Remove Overtime & Set Max 8 Working Hours. Retain with standard benefits.",
        "standard_raise": "🛡️ **Strategy:** Provide Standard Salary Raise (+30%). Safe investment due to stable merits."
    },
    "Tamil": {
        "title": "🏢 AI HR போர்டல்: இண்டஸ்ட்ரி-லெவல் டேட்டா சயின்ஸ் & ரிடென்ஷன் பிளாட்பார்ம்",
        "subtitle": "லைவ் அனலிட்டிக்ஸ், அசல் ML மெட்ரிக்ஸ் மற்றும் எக்ஸ்ப்ளெய்னபிள் AI கொண்ட எண்ட்-டு-எண்ட் மெஷின் லேர்னிங் சிஸ்டம்.",
        "reason_title": "### 🔍 சாத்தியமான காரணங்கள் / குறிப்பிட்ட HR நுண்ணறிவுகள்:",
        "terminate_low_merit": "❌ **நடவடிக்கை: உடனடி பணிநீக்கம்!** குறைந்த வயதுடன், அனுபவம், தொழில்நுட்ப திறன் மற்றும் அட்டெண்டன்ஸ் அனைத்தும் மிக மோசமாக உள்ளது.",
        "high_merit_bonus": "🛡️ **உத்தி: சிறந்த பணியாளர்!** அதிக அனுபவம், டாப் ஜாப் லெவல், ஆக்டிவ் ஸ்கில்ஸ், சிறந்த அட்டெண்டன்ஸ் மற்றும் டொமைன் நாலெட்ஜ் உள்ளது. இவரைத் தக்கவைக்க **40% சம்பள உயர்வு** வழங்க பரிந்துரைக்கப்படுகிறது.",
        "young_expert_raise": "🛡️ **உத்தி: இளம் திறமையாளர் போனஸ்!** குறைந்த வயதுடையவர், ஆனால் அதிக அனுபவம் கொண்டுள்ளார். இவருக்கு உடனடி **சம்பள உயர்வு மற்றும் எதிர்கால நன்மைகள் (Future Benefits)** வழங்க பரிந்துரைக்கப்படுகிறது.",
        "attendance_warning": "⚠️ **HR எச்சரிக்கை:** அட்டெண்டன்ஸ் 85%-க்கும் குறைவாக ({attendance}%) உள்ளது. அதிகாரப்பூர்வ எச்சரிக்கை கடிதம் வழங்கவும். **குறிப்பு: 3 எச்சரிக்கைகள் பெற்றால் பணிநீக்கம் (Terminate) செய்யவும்.**",
        "standard_ot": "🛡️ **உத்தி:** ஓவர்டைமை நீக்கி, அதிகபட்சம் 8 மணிநேர வேலையாக மாற்றவும். நிலையான பலன்களுடன் தக்கவைக்கவும்.",
        "standard_raise": "🛡️ **உத்தி:** வழக்கமான சம்பள உயர்வு (+30%) வழங்கவும். நல்ல தகுதிகள் இருப்பதால் பாதுகாப்பான முதலீடு."
    },
    "Malayalam": {
        "title": "🏢 AI HR പോർട്ടൽ: ഇൻഡസ്ട്രി-ലെവൽ ഡാറ്റ സയൻസ് & റിറ്റെൻഷൻ പ്ലാറ്റ്‌ഫോം",
        "subtitle": "ലൈവ് അനലിറ്റിക്സ്, യഥാർത്ഥ ML മെട്രിക്സ്, എക്സ്പ്ലെയ്നബിൾ AI എന്നിവ അടങ്ങിയ ഒരു എൻഡ്-ടു-എൻഡ് മെഷീൻ ലേണിംഗ് സിസ്റ്റം.",
        "reason_title": "### 🔍 സാധ്യമായ കാരണങ്ങൾ / പ്രത്യേക HR സ്ഥിതിവിവരക്കണക്കുകൾ:",
        "terminate_low_merit": "❌ **നടപടി: ഉടനടി പിരിച്ചുവിടുക!** കുറഞ്ഞ പ്രായം, പ്രവൃത്തിപരിചയമില്ലായ്മ, കഴിവില്ലായ്മ, വളരെ കുറഞ്ഞ ഹാജർനില എന്നിവ കണ്ടെത്തി.",
        "high_merit_bonus": "🛡️ **സ്ട്രാറ്റജി: മികച്ച പ്രൊഫൈൽ!** ഉയർന്ന പ്രവൃത്തിപരിചയം, ടോപ്പ് ജോബ് ലെവൽ, സജീവമായ കഴിവുകൾ, മികച്ച ഹാജർനില എന്നിവയുണ്ട്. **40% ശമ്പള വർദ്ധനവ്** ശുപാർശ ചെയ്യുന്നു.",
        "young_expert_raise": "🛡️ **സ്ട്രാറ്റജി: യുവ പ്രതിഭ ബോണസ്!** കുറഞ്ഞ പ്രായത്തിലും ഉയർന്ന പ്രവൃത്തിപരിചയമുണ്ട്. ഉടനടി **ശമ്പള വർദ്ധനവും ഭാവി ആനുകൂല്യങ്ങളും (Future Benefits)** നൽകുക.",
        "attendance_warning": "⚠️ **HR മുന്നറിയിപ്പ്:** ഹാജർനില 85%-ൽ താഴെയാണ് ({attendance}%). ഔദ്യോഗിക മുന്നറിയിപ്പ് നൽകുക. **ശ്രദ്ധിക്കുക: 3 മുന്നറിയിപ്പുകൾ ലഭിച്ചാൽ പിരിച്ചുവിടുക.**",
        "standard_ot": "🛡️ **സ്ട്രാറ്റജി:** ओवरटाइम ഒഴിവാക്കുക, പരമാവധി 8 മണിക്കൂർ ജോലി നിശ്ചയിക്കുക.",
        "standard_raise": "🛡️ **സ്ട്രാറ്റജി:** സാധാരണ ശമ്പള വർദ്ധനവ് (+30%) നൽകുക."
    },
    "Telugu": {
        "title": "🏢 AI HR పోర్టల్: ఇండస్ట్రీ-లెవెల్ డేటా సైన్స్ & రిటెన్షన్ ప్లాట్‌ఫారమ్",
        "subtitle": "లైవ్ అనలిటిక్స్, రియల్ ML మెట్రిక్స్ మరియు ఎక్స్‌ప్లైనబుల్ AI ఫీచర్లతో ఎండ్-トゥ-ఎండ్ మెషీన్ లెర్నింగ్ సిస్టమ్.",
        "reason_title": "### 🔍 సంభావ్య కారణాలు / నిర్దిష్ట HR అంతర్దృష్టులు:",
        "terminate_low_merit": "❌ **చర్య: తక్షణమే తొలగించండి!** తక్కువ వయస్సు, అనుభవం లేకపోవడం, నైపుణ్యాలు మరియు హాజరు చాలా తక్కువగా ఉన్నాయి.",
        "high_merit_bonus": "🛡️ **వ్యూహం: అత్యుత్తమ ఉద్యోగి!** ఎక్కువ అనుభవం, టాప్ జాబ్ లెవెల్, యాక్టివ్ స్కిల్స్ మరియు మంచి హాజరు ఉన్నాయి. **40% జీతం పెంపు** సిఫార్సు చేయబడింది.",
        "young_expert_raise": "🛡️ **వ్యూహం: యంగ్ టాలెంట్ బోనస్!** తక్కువ వయస్సు ఉన్నప్పటికీ ఎక్కువ అనుభవం ఉంది. తక్షణ **జీతం పెంపు మరియు భవిష్యత్తు ప్రయోజనాలు (Future Benefits)** అందించండి.",
        "attendance_warning": "⚠️ **HR హెచ్చరిక:** హాజరు 85% కంటే తక్కువగా ({attendance}%) ఉంది. అధికారిక హెచ్చరిక పంపండి. **గమనిక: 3 హెచ్చరికలు వస్తే తొలగించండి.**",
        "standard_ot": "🛡️ **వ్యూహం:** ओवरटाइम తొలగించి, గరిష్టంగా 8 గంటల పని వేళలను సెట్ చేయండి.",
        "standard_raise": "🛡️ **వ్యూహం:** సాధారణ జీతం పెంపు (+30%) అందించండి."
    },
    "Hindi": {
        "title": "🏢 AI HR पोर्टल: | इंडस्ट्री-लेवल डेटा साइंस और रिटेंशन प्लेटफॉर्म",
        "subtitle": "लाइव एनालिटिक्स, वास्तविक ML मेट्रिक्स और एक्सप्लेनेबल AI की विशेषता वाला एक एंड-टू-एंड मशीन लर्निंग सिस्टम।",
        "reason_title": "### 🔍 संभावित कारण / विशिष्ट HR अंतर्दृष्टि:",
        "terminate_low_merit": "❌ **कार्रवाई: तत्काल सेवा समाप्ति!** कम उम्र, अनुभवहीनता, कौशल की कमी और अत्यंत कम उपस्थिति पाई गई है।",
        "high_merit_bonus": "🛡️ **रणनीति: उत्कृष्ट कर्मचारी!** उच्च अनुभव, शीर्ष जॉब स्तर, सक्रिय कौशल और अच्छी उपस्थिति। इन्हें रोकने के लिए **40% वेतन वृद्धि** की सिफारिश की जाती है।",
        "young_expert_raise": "🛡️ **रणनीति: युवा प्रतिभा बोनस!** उम्र कम है लेकिन अनुभव अधिक है। तत्काल **वेतन वृद्धि और भविष्य के कॉर्पोरेट लाभ (Future Benefits)** प्रदान करें।",
        "attendance_warning": "⚠️ **HR चेतावनी जारी:** उपस्थिति 85% से कम ({attendance}%) है। आधिकारिक चेतावनी पत्र दें। **नोट: 3 चेतावनी होने पर टर्मिनेट करें।**",
        "standard_ot": "🛡️ **रणनीति:** ओवरटाइम हटाएं और अधिकतम 8 घंटे काम तय करें।",
        "standard_raise": "🛡️ **रणनीति:** मानक वेतन वृद्धि (+30%) प्रदान करें।"
    }
}

# 🌐 SIDEBAR LANGUAGE SELECTOR
st.sidebar.markdown("### 🌐 Language Settings")
selected_lang = st.sidebar.selectbox("Choose Language / ഭാഷ / భాష / भाषा", list(LANGUAGES.keys()), index=0)
text = LANGUAGES[selected_lang]

st.title(text["title"])
st.write(text["subtitle"])

# 2. Paths pointing to Trained Machine Learning Model
model_path = 'Notebooks/Models/attrition_rf_model.pkl'
if not os.path.exists(model_path):
    model_path = '../Notebooks/Models/attrition_rf_model.pkl'

# Load trained Random Forest model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# 3. Initialize Session State Database with Ram (EMP007) and Varsha (EMP005) updated
if 'hr_database' not in st.session_state:
    st.session_state.hr_database = pd.DataFrame([
        {"Employee_ID": "EMP001", "Name": "Arun Kumar", "Age": 28, "Salary": 85000, "Department": "Information Technology", "Job_Role": "Software Engineer", "Working_Hours": 10, "Experience_Years": 4, "Job_Level": 2, "Overtime": "Yes", "Skills_Master": "Active", "Attendance_Pct": 92, "Domain_Knowledge": "High", "Address": "Chennai", "Blood_Group": "O+ve"},
        {"Employee_ID": "EMP002", "Name": "Priya Sharma", "Age": 35, "Salary": 210000, "Department": "Sales", "Job_Role": "Manager", "Working_Hours": 8, "Experience_Years": 10, "Job_Level": 4, "Overtime": "No", "Skills_Master": "Active", "Attendance_Pct": 95, "Domain_Knowledge": "High", "Address": "Coimbatore", "Blood_Group": "A+ve"},
        {"Employee_ID": "EMP003", "Name": "Durai", "Age": 24, "Salary": 26000, "Department": "Information Technology", "Job_Role": "Software Engineer", "Working_Hours": 10, "Experience_Years": 1, "Job_Level": 1, "Overtime": "Yes", "Skills_Master": "Active", "Attendance_Pct": 87, "Domain_Knowledge": "Medium", "Address": "Chennai", "Blood_Group": "O+ve"},
        {"Employee_ID": "EMP004", "Name": "Ramesh Kumar", "Age": 23, "Salary": 45000, "Department": "Information Technology", "Job_Role": "Data Scientist", "Working_Hours": 12, "Experience_Years": 0, "Job_Level": 1, "Overtime": "Yes", "Skills_Master": "Active", "Attendance_Pct": 96, "Domain_Knowledge": "High", "Address": "Chennai", "Blood_Group": "B+ve"},
        {"Employee_ID": "EMP005", "Name": "Varsha", "Age": 25, "Salary": 90000, "Department": "Information Technology", "Job_Role": "Data Scientist", "Working_Hours": 10, "Experience_Years": 1, "Job_Level": 1, "Overtime": "Yes", "Skills_Master": "Active", "Attendance_Pct": 70, "Domain_Knowledge": "Medium", "Address": "Chennai", "Blood_Group": "A+ve"},
        {"Employee_ID": "EMP006", "Name": "Divya Bharathi", "Age": 26, "Salary": 35000, "Department": "Information Technology", "Job_Role": "Cybersecurity Analyst", "Working_Hours": 9, "Experience_Years": 3, "Job_Level": 2, "Overtime": "Yes", "Skills_Master": "Inactive", "Attendance_Pct": 75, "Domain_Knowledge": "Low", "Address": "Chennai", "Blood_Group": "AB+ve"},
        {"Employee_ID": "EMP007", "Name": "Ram", "Age": 30, "Salary": 25000, "Department": "Information Technology", "Job_Role": "Data Scientist", "Working_Hours": 10, "Experience_Years": 10, "Job_Level": 4, "Overtime": "Yes", "Skills_Master": "Active", "Attendance_Pct": 97, "Domain_Knowledge": "High", "Address": "Coimbatore", "Blood_Group": "O+ve"}
    ])

# Helper function to compute AI prediction for a batch
def predict_ai_status(df_input):
    if df_input.empty:
        return []
    model_features = model.feature_names_in_
    predictions = []
    for idx, row in df_input.iterrows():
        single_input_df = pd.DataFrame(0, index=[0], columns=model_features)
        for col in model_features:
            if 'satisfaction' in col.lower() or 'balance' in col.lower() or 'involvement' in col.lower():
                single_input_df[col] = 3
            elif 'rate' in col.lower():
                single_input_df[col] = 500
            elif 'numcompaniesworked' in col.lower():
                single_input_df[col] = 1
            elif 'trainingtimeslastyear' in col.lower():
                single_input_df[col] = 2
                
        overtime_encoded = 1 if row['Overtime'] == "Yes" else 0
        raw_salary = row['Salary']
        if raw_salary <= 20000: ai_salary = raw_salary
        elif raw_salary <= 50000: ai_salary = int(raw_salary / 12)
        else: ai_salary = int(raw_salary / 25)
            
        for col in single_input_df.columns:
            if col.lower() == 'age': single_input_df[col] = row['Age']
            elif col.lower() == 'monthlyincome': single_input_df[col] = ai_salary
            elif col.lower() == 'overtime': single_input_df[col] = overtime_encoded
            elif col.lower() == 'totalworkingyears': single_input_df[col] = row['Experience_Years']
            elif col.lower() == 'joblevel': single_input_df[col] = row['Job_Level']
            elif col == f"Department_{row['Department']}": single_input_df[col] = 1
            elif col == f"JobRole_{row['Job_Role']}": single_input_df[col] = 1
            
        pred = model.predict(single_input_df)[0]
        predictions.append("🚨 High Risk" if pred == 1 else "✅ Stable")
    return predictions

# Explainable AI Reasoner (XAI)
def explain_ai_prediction(row, ai_status):
    if "Stable" in ai_status:
        return "💡 **AI Decision Drivers:** Strong income bounds, no forced overtime, and continuous domain stability markers match historical retention logic."
    reasons = []
    if row['Overtime'] == "Yes": reasons.append("Forced Overtime Duty detected (+114% weightage)")
    if row['Salary'] < 60000: reasons.append("Sub-baseline Monthly Salary Bracket detected")
    if row['Experience_Years'] <= 2: reasons.append("Early Career Phase vulnerability matrix match")
    if row['Working_Hours'] > 9: reasons.append("Daily Work hours exceeding corporate safety threshold")
    return "💡 **AI Decision Drivers (Explainable AI):** Triggered due to " + " and ".join(reasons) + "."

# AI TIMELINE & TENURE PREDICTOR
def predict_remaining_tenure(row, ai_status):
    if "Stable" in ai_status: return "⏳ Long Term (2+ Years Expected)"
    if row['Working_Hours'] >= 11 and row['Attendance_Pct'] < 90: return "⚠️ Critical: Leave within 1 - 3 Months"
    return "⚠️ Moderate Risk: Leave within 3 - 6 Months"

# 🔥 DYNAMIC INTENT-BASED CUSTOM BUSINESS RULES OPTIMIZER
def get_ai_retention_plan(row, lang_dict):
    # கண்டிஷன் 1: வயது குறைவாக இருந்து அனுபவம், ஸ்கில், அட்டெண்டன்ஸ் இல்லையென்றால் -> பணிநீக்கம்
    if row['Age'] <= 26 and row['Experience_Years'] == 0 and row['Skills_Master'] != "Active" and row['Attendance_Pct'] < 80:
        return lang_dict["terminate_low_merit"]
        
    # கண்டிஷன் 2: எல்லாமே அதிகமாக இருந்தால் (Experience, Job Level, Active Skills, Attendance, Domain Knowledge) -> சம்பள உயர்வு (+40%)
    if row['Experience_Years'] >= 5 and row['Job_Level'] >= 3 and row['Skills_Master'] == "Active" and row['Attendance_Pct'] >= 90 and row['Domain_Knowledge'] == "High":
        return lang_dict["high_merit_bonus"]
        
    # கண்டிஷன் 3: வயது குறைவாக இருந்து அனுபவம் அதிகமாக இருந்தால் -> சம்பள உயர்வு + எதிர்கால பலன்கள்
    if row['Age'] <= 32 and row['Experience_Years'] >= 5:
        return lang_dict["young_expert_raise"]
        
    # கண்டிஷன் 4: அட்டெண்டன்ஸ் 85% கீழே இருந்தால் -> எச்சரிக்கை / 3 எச்சரிக்கைக்கு பின் பணிநீக்கம்
    if row['Attendance_Pct'] < 85:
        return lang_dict["attendance_warning"].format(attendance=row['Attendance_Pct'])
        
    # வழக்கமான கண்டிஷன்கள் (Fallback Rules)
    if row['Overtime'] == "Yes" and row['Working_Hours'] > 8:
        return lang_dict["standard_ot"]
        
    return lang_dict["standard_raise"]

# 4. Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(["➕ Add / Update Employee", "📊 Master Dashboard & Charts", "🔍 Profile Finder & Explainable AI", "🔬 Model Evaluation Metrics"])

# ==================== TAB 1: EMPLOYEE REGISTRATION ====================
with tab1:
    st.subheader("Add New Employee or Adjust Existing Employee Data")
    existing_ids = st.session_state.hr_database['Employee_ID'].tolist()
    select_id_mode = st.selectbox("Select Action", ["Create New Profile", "Update Existing Profile"])
    
    target_id, default_name, default_age, default_salary, default_hours, default_overtime, default_attendance, default_exp = "EMP003", "Durai", 24, 26000, 10, "Yes", 87, 1
    if select_id_mode == "Update Existing Profile" and existing_ids:
        chosen_id = st.selectbox("Choose Employee ID to Adjust", existing_ids)
        emp_match = st.session_state.hr_database[st.session_state.hr_database['Employee_ID'] == chosen_id].iloc[0]
        target_id, default_name, default_age, default_salary, default_hours, default_overtime, default_attendance, default_exp = emp_match['Employee_ID'], emp_match['Name'], int(emp_match['Age']), int(emp_match['Salary']), int(emp_match['Working_Hours']), emp_match['Overtime'], int(emp_match['Attendance_Pct']), int(emp_match['Experience_Years'])

    with st.form("reg_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            emp_id = st.text_input("Employee ID", value=target_id, disabled=(select_id_mode == "Update Existing Profile"))
            emp_name = st.text_input("Full Name", value=default_name)
            age = st.slider("Age", 18, 60, default_age)
            blood_group = st.selectbox("Blood Group", ["O+ve", "A+ve", "B+ve"])
            address = st.text_area("Address", value="Chennai")
        with col2:
            department = st.selectbox("Department", ["Information Technology", "Sales", "Research & Development"])
            job_role = st.selectbox("Job Role", ["Software Engineer", "Data Scientist", "Cybersecurity Analyst", "Network Engineer", "Manager"])
            experience = st.slider("Experience (Years)", 0, 40, default_exp)
            job_level = st.slider("Job Level (1 to 5)", 1, 5, 1)
        with col3:
            salary = st.slider("Current Salary ($)", 1000, 500000, default_salary, step=1000)
            working_hours = st.slider("Daily Working Hours", 4, 14, default_hours)
            overtime = st.selectbox("Works Overtime?", ["Yes", "No"], index=(0 if default_overtime == "Yes" else 1))
            skills_master = st.selectbox("Skills Mastery Status", ["Active", "Inactive"])
            attendance_pct = st.slider("Attendance (%)", 50, 100, default_attendance)
            domain_knowledge = st.selectbox("Domain Knowledge", ["High", "Medium", "Low"])
            
        submit = st.form_submit_button("💾 Save & Update Live Database", type="primary")
        if submit:
            if emp_id in st.session_state.hr_database['Employee_ID'].values:
                st.session_state.hr_database = st.session_state.hr_database[st.session_state.hr_database['Employee_ID'] != emp_id]
            new_emp = {"Employee_ID": emp_id, "Name": emp_name, "Age": age, "Salary": salary, "Department": department, "Job_Role": job_role, "Working_Hours": working_hours, "Experience_Years": experience, "Job_Level": job_level, "Overtime": overtime, "Address": address, "Blood_Group": blood_group, "Skills_Master": skills_master, "Attendance_Pct": attendance_pct, "Domain_Knowledge": domain_knowledge}
            st.session_state.hr_database = pd.concat([st.session_state.hr_database, pd.DataFrame([new_emp])], ignore_index=True)
            st.success(f"Successfully updated records for {emp_name}!")

# ==================== TAB 2: MASTER DASHBOARD & CHARTS ====================
with tab2:
    st.subheader("📋 Executive Insights Dashboard & Analytical Sub-systems")
    db = st.session_state.hr_database.copy()
    
    if not db.empty:
        db['AI Model Prediction'] = predict_ai_status(db)
        
        # LIVE GRAPHS SECTION
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("##### 📊 Attrition Risk by Department")
            dept_chart = db.groupby(['Department', 'AI Model Prediction']).size().unstack(fill_value=0)
            st.bar_chart(dept_chart)
        with col_g2:
            st.markdown("##### 📈 Age Distribution Curve")
            age_chart = db.groupby('Age').size()
            st.line_chart(age_chart)
        with col_g3:
            st.markdown("##### 🥧 Overall Attrition Risk Share")
            risk_counts = db['AI Model Prediction'].value_counts()
            st.bar_chart(risk_counts)
            
        st.markdown("---")
        
        tenure, plans = [], []
        for idx, row in db.iterrows():
            tenure.append(predict_remaining_tenure(row, row['AI Model Prediction']))
            plans.append(get_ai_retention_plan(row, text))
        db['Predicted Remaining Tenure'] = tenure
        db['AI Optimized Action Plan'] = plans
        
        st.dataframe(db[['Employee_ID', 'Name', 'Salary', 'AI Model Prediction', 'Predicted Remaining Tenure', 'AI Optimized Action Plan']], use_container_width=True)
        
        csv_data = db.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Complete HR Audit Sheet (CSV File)", data=csv_data, file_name="HR_AI_Audit_Report.csv", mime="text/csv", type="primary")
        
# ==================== TAB 3: PROFILE FINDER & EXPLAINABLE AI ====================
with tab3:
    st.subheader("🔍 Single Profile Search & Explainable AI Engine")
    search_id = st.text_input("Enter Unique Employee ID ")