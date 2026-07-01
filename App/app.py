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
        "reason_title": "### 🔍 Potential Reasons for Proceeding with Termination:",
        "reason_1": "1. 📉 **Low Attendance:** Employee attendance is below 90% ({attendance}%).",
        "reason_2": "2. ⚠️ **Inactive Skills Mastery:** The employee's essential technical skill qualification status is Inactive.",
        "reason_3": "3. 💰 **Performance vs Cost:** Compared to the high salary (${salary:,}) provided for low work experience ({exp} Yrs), the current output does not fully meet organization targets."
    },
    "Tamil": {
        "title": "🏢 AI HR போர்டல்: இண்டஸ்ட்ரி-லெவல் டேட்டா சயின்ஸ் & ரிடென்ஷன் பிளாட்பார்ம்",
        "subtitle": "லைவ் அனலிட்டிக்ஸ், அசல் ML மெட்ரிக்ஸ் மற்றும் எக்ஸ்ப்ளெய்னபிள் AI கொண்ட எண்ட்-டு-எண்ட் மெஷின் லேர்னிங் சிஸ்டம்.",
        "reason_title": "### 🔍 தக்க வைக்காமல் விடுவிப்பதற்கான சாத்தியமான காரணங்கள்:",
        "reason_1": "1. 📉 **குறைந்த வருகைப்பதிவு (Low Attendance):** ஊழியரின் அட்டெண்டன்ஸ் 90%-க்கும் குறைவாக ({attendance}%) உள்ளது.",
        "reason_2": "2. ⚠️ **திறன் நிலைത്തன்மை குறைபாடு (Inactive Skills Master):** ஊழியரின் அத்தியாவசிய தொழில்நுட்ப திறன் தகுதி சுணக்கமாக (Inactive) உள்ளது.",
        "reason_3": "3. 💰 **செயல்திறன் மதிப்பீடு (Performance vs Cost):** ஊழியரின் குறைந்த பணி அனுபவத்திற்கு ({exp} வருடம்) வழங்கப்படும் அதிக சம்பளத்துடன் (${salary:,}) ஒப்பிடும்போது, தற்போதைய அவுட்புட் நிறுவனத்தின் இலக்குகளை முழுமையாக ஈடுகட்டவில்லை."
    },
    "Malayalam": {
        "title": "🏢 AI HR പോർട്ടൽ: ഇൻഡസ്ട്രി-ലെവൽ ഡാറ്റ സയൻസ് & റിറ്റെൻഷൻ പ്ലാറ്റ്‌ഫോം",
        "subtitle": "ലൈവ് അനലിറ്റിക്സ്, യഥാർത്ഥ ML മെട്രിക്സ്, എക്സ്പ്ലെയ്നബിൾ AI എന്നിവ അടങ്ങിയ ഒരു എൻഡ്-ടു-എൻഡ് മെഷീൻ ലേണിംഗ് സിസ്റ്റം.",
        "reason_title": "### 🔍 ഒഴിവാക്കുന്നതിനുള്ള സാധ്യമായ കാരണങ്ങൾ:",
        "reason_1": "1. 📉 **കുറഞ്ഞ ഹാജർനില (Low Attendance):** ജീവനക്കാരൻ്റെ ഹാജർനില 90%-ൽ താഴെയാണ് ({attendance}%).",
        "reason_2": "2. ⚠️ **നൈപുണ്യ കുറവ് (Inactive Skills Master):** ജീവനക്കാരൻ്റെ സാങ്കേതിക നൈപുണ്യ യോഗ്യത നിഷ്ക്രിയമാണ് (Inactive).",
        "reason_3": "3. 💰 **പ്രകടനവും ചെലവും (Performance vs Cost):** കുറഞ്ഞ പ്രവൃത്തിപരിചയത്തിന് ({exp} വർഷം) നൽകുന്ന ഉയർന്ന ശമ്പളവുമായി (${salary:,}) താരതമ്യം ചെയ്യുമ്പോൾ, നിലവിലെ ഔട്ട്പുട്ട് കമ്പനിയുടെ ലക്ഷ്യങ്ങൾ പൂർണ്ണമായി വ്യക്തമാക്കുന്നില്ല."
    },
    "Telugu": {
        "title": "🏢 AI HR పోర్టల్: ఇండస్ట్రీ-లెవెల్ డేటా సైన్స్ & రిటెన్షన్ ప్లాట్‌ఫారమ్",
        "subtitle": "లైవ్ అనలిటిక్స్, రియల్ ML మెట్రిక్స్ మరియు ఎక్స్‌ప్లైనబుల్ AI ఫీచర్లతో ఎండ్-టు-ఎండ్ మెషీన్ లెర్నింగ్ సిస్టమ్.",
        "reason_title": "### 🔍 తొలగించడానికి గల సంభావ్య కారణాలు:",
        "reason_1": "1. 📉 **తక్కువ హాజరు (Low Attendance):** ఉద్యోగి హాజరు 90% కంటే తక్కువగా ({attendance}%) ఉంది.",
        "reason_2": "2. ⚠️ **నైపుణ్యాల నిష్క్రియత (Inactive Skills Master):** ఉద్యోగి యొక్క సాంకేతిక నైపుణ్యాల అర్హత నిష్క్రియంగా (Inactive) ఉంది.",
        "reason_3": "3. 💰 **పనితీరు వర్సెస్ ఖర్చు (Performance vs Cost):** తక్కువ పని అనుభవానికి ({exp} సంవత్సరాలు) ఇచ్చే అధిక జీతంతో (${salary:,}) పోల్చితే, ప్రస్తుత అవుట్‌పుట్ సంస్థ లక్ష్యాలను పూర్తిగా అందుకోలేదు."
    },
    "Hindi": {
        "title": "🏢 AI HR पोर्टल: इंडस्ट्री-关 level डेटा साइंस और रिटेंशन प्लेटफॉर्म",
        "subtitle": "लाइव एनालिटिक्स, वास्तविक ML मेट्रिक्स और एक्सप्लेनेबल AI की विशेषता वाला एक एंड-टू-एंड मशीन लर्निंग सिस्टम।",
        "reason_title": "### 🔍 सेवा समाप्ति के संभावित कारण:",
        "reason_1": "1. 📉 **कम उपस्थिति (Low Attendance):** कर्मचारी की उपस्थिति 90% से कम ({attendance}%) है।",
        "reason_2": "2. ⚠️ **निष्क्रिय कौशल योग्यता (Inactive Skills Master):** कर्मचारी की आवश्यक तकनीकी कौशल स्थिति निष्क्रिय (Inactive) है।",
        "reason_3": "3. 💰 **प्रदर्शन बनाम लागत (Performance vs Cost):** कम कार्य अनुभव ({exp} वर्ष) के लिए दिए जा रहे उच्च वेतन (${salary:,}) की तुलना में, वर्तमान आउटपुट कंपनी के लक्ष्यों को पूरी तरह से पूरा नहीं करता है।"
    }
}

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

# 3. Initialize Session State Databases (Main HR Database & Terminated History)
if 'hr_database' not in st.session_state:
    st.session_state.hr_database = pd.DataFrame([
        {"Employee_ID": "EMP001", "Name": "Arun Kumar", "Age": 28, "Salary": 85000, "Department": "Information Technology", "Job_Role": "Software Engineer", "Working_Hours": 10, "Experience_Years": 4, "Job_Level": 2, "Overtime": "Yes", "Skills_Master": "Active", "Attendance_Pct": 92, "Domain_Knowledge": "High", "Address": "Chennai", "Blood_Group": "O+ve", "Phone": "+91 98765 43210"},
        {"Employee_ID": "EMP002", "Name": "Priya Sharma", "Age": 35, "Salary": 210000, "Department": "Sales", "Job_Role": "Manager", "Working_Hours": 8, "Experience_Years": 10, "Job_Level": 4, "Overtime": "No", "Skills_Master": "Active", "Attendance_Pct": 95, "Domain_Knowledge": "High", "Address": "Coimbatore", "Blood_Group": "A+ve", "Phone": "+91 87654 32109"},
        {"Employee_ID": "EMP003", "Name": "Durai", "Age": 24, "Salary": 26000, "Department": "Information Technology", "Job_Role": "Software Engineer", "Working_Hours": 10, "Experience_Years": 1, "Job_Level": 1, "Overtime": "Yes", "Skills_Master": "Active", "Attendance_Pct": 87, "Domain_Knowledge": "Medium", "Address": "Chennai", "Blood_Group": "O+ve", "Phone": "+91 76543 21098"},
        {"Employee_ID": "EMP004", "Name": "Ramesh Kumar", "Age": 23, "Salary": 45000, "Department": "Information Technology", "Job_Role": "Data Scientist", "Working_Hours": 12, "Experience_Years": 0, "Job_Level": 1, "Overtime": "Yes", "Skills_Master": "Active", "Attendance_Pct": 96, "Domain_Knowledge": "High", "Address": "Chennai", "Blood_Group": "B+ve", "Phone": "+91 65432 10987"},
        {"Employee_ID": "EMP005", "Name": "Varsha", "Age": 25, "Salary": 90000, "Department": "Information Technology", "Job_Role": "Data Scientist", "Working_Hours": 10, "Experience_Years": 1, "Job_Level": 1, "Overtime": "Yes", "Skills_Master": "Inactive", "Attendance_Pct": 70, "Domain_Knowledge": "Medium", "Address": "Chennai", "Blood_Group": "A+ve", "Phone": "+91 54321 09876"},
        {"Employee_ID": "EMP006", "Name": "Divya Bharathi", "Age": 26, "Salary": 35000, "Department": "Information Technology", "Job_Role": "Cybersecurity Analyst", "Working_Hours": 9, "Experience_Years": 3, "Job_Level": 2, "Overtime": "Yes", "Skills_Master": "Inactive", "Attendance_Pct": 75, "Domain_Knowledge": "Low", "Address": "Chennai", "Blood_Group": "AB+ve", "Phone": "+91 43210 98765"},
        {"Employee_ID": "EMP007", "Name": "Vikram Seth", "Age": 42, "Salary": 180000, "Department": "Information Technology", "Job_Role": "Network Engineer", "Working_Hours": 8, "Experience_Years": 15, "Job_Level": 5, "Overtime": "No", "Skills_Master": "Active", "Attendance_Pct": 98, "Domain_Knowledge": "High", "Address": "Coimbatore", "Blood_Group": "O+ve", "Phone": "+91 32109 87654"}
    ])

if 'terminated_database' not in st.session_state:
    st.session_state.terminated_database = pd.DataFrame(columns=[
        "Employee_ID", "Name", "Age", "Salary", "Department", "Job_Role", "Address", "Blood_Group", "Phone", "Reason"
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

# AI LOSS-CONTROL & RETENTION PLAN OPTIMIZER (CRITICAL CRITERIA ADDED)
def get_ai_retention_plan(row):
    # 💥 நிபந்தனை: Skill-ம் இல்லாமல், Attendance-ம் 80% கீழே இருந்தால் சிஸ்டம் தானாக பணிநீக்க உத்தரவு பிறப்பிக்கும்
    if row['Skills_Master'] != "Active" and row['Attendance_Pct'] < 80:
        return "❌ CRITICAL: No Skill & Low Attendance Detected. SYSTEM RECOMMENDATION: TERMINATE IMMEDIATELY."
    if row['Skills_Master'] != "Active" or row['Attendance_Pct'] < 90:
        return "❌ No Counter-Offer (Merits Not Met). Proceed with Termination/Exit."
    if row['Overtime'] == "Yes" and row['Working_Hours'] > 8:
        return "🛡️ Strategy: Remove Overtime & Set Max 8 Working Hours. Retain with standard benefits."
    return f"🛡️ Strategy: Provide High Salary Raise (+30% -> ${int(row['Salary'] * 1.3):,}). Safe investment due to excellent merits."

# 4. Navigation Tabs (Added 5th Tab for Terminated Archives)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "➕ Add / Update Employee", 
    "📊 Master Dashboard & Charts", 
    "🔍 Profile Finder & Explainable AI", 
    "🔬 Model Evaluation Metrics",
    "🗑️ Terminated Employees History"
])

# ==================== TAB 1: EMPLOYEE REGISTRATION ====================
with tab1:
    st.subheader("Add New Employee or Adjust Existing Employee Data")
    existing_ids = st.session_state.hr_database['Employee_ID'].tolist()
    select_id_mode = st.selectbox("Select Action", ["Create New Profile", "Update Existing Profile"])
    
    target_id, default_name, default_age, default_salary, default_hours, default_overtime, default_attendance, default_exp, default_phone = "EMP003", "Durai", 24, 26000, 10, "Yes", 87, 1, "+91 99999 88888"
    if select_id_mode == "Update Existing Profile" and existing_ids:
        chosen_id = st.selectbox("Choose Employee ID to Adjust", existing_ids)
        emp_match = st.session_state.hr_database[st.session_state.hr_database['Employee_ID'] == chosen_id].iloc[0]
        target_id, default_name, default_age, default_salary, default_hours, default_overtime, default_attendance, default_exp, default_phone = emp_match['Employee_ID'], emp_match['Name'], int(emp_match['Age']), int(emp_match['Salary']), int(emp_match['Working_Hours']), emp_match['Overtime'], int(emp_match['Attendance_Pct']), int(emp_match['Experience_Years']), emp_match.get('Phone', "+91 99999 88888")

    with st.form("reg_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            emp_id = st.text_input("Employee ID", value=target_id, disabled=(select_id_mode == "Update Existing Profile"))
            emp_name = st.text_input("Full Name", value=default_name)
            age = st.slider("Age", 18, 60, default_age)
            blood_group = st.selectbox("Blood Group", ["O+ve", "A+ve", "B+ve", "AB+ve"])
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
            phone = st.text_input("Phone Number", value=default_phone)
            
        submit = st.form_submit_button("💾 Save & Update Live Database", type="primary")
        if submit:
            if emp_id in st.session_state.hr_database['Employee_ID'].values:
                st.session_state.hr_database = st.session_state.hr_database[st.session_state.hr_database['Employee_ID'] != emp_id]
            new_emp = {"Employee_ID": emp_id, "Name": emp_name, "Age": age, "Salary": salary, "Department": department, "Job_Role": job_role, "Working_Hours": working_hours, "Experience_Years": experience, "Job_Level": job_level, "Overtime": overtime, "Address": address, "Blood_Group": blood_group, "Skills_Master": skills_master, "Attendance_Pct": attendance_pct, "Domain_Knowledge": domain_knowledge, "Phone": phone}
            st.session_state.hr_database = pd.concat([st.session_state.hr_database, pd.DataFrame([new_emp])], ignore_index=True)
            st.success(f"Successfully updated records for {emp_name}!")

# ==================== TAB 2: MASTER DASHBOARD & CHARTS ====================
with tab2:
    st.subheader("📋 Executive Insights Dashboard & Analytical Sub-systems")
    db = st.session_state.hr_database.copy()
    
    if not db.empty:
        db['AI Model Prediction'] = predict_ai_status(db)
        
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
            plans.append(get_ai_retention_plan(row))
        db['Predicted Remaining Tenure'] = tenure
        db['AI Optimized Action Plan'] = plans
        
        st.dataframe(db[['Employee_ID', 'Name', 'Salary', 'AI Model Prediction', 'Predicted Remaining Tenure', 'AI Optimized Action Plan']], use_container_width=True)
        
        csv_data = db.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Complete HR Audit Sheet (CSV File)", data=csv_data, file_name="HR_AI_Audit_Report.csv", mime="text/csv", type="primary")

# ==================== TAB 3: PROFILE FINDER & EXPLAINABLE AI ====================
with tab3:
    st.subheader("🔍 Single Profile Search & Explainable AI Engine")
    search_id = st.text_input("Enter Unique Employee ID (e.g., EMP005)")
    
    if search_id:
        db = st.session_state.hr_database
        profile = db[db['Employee_ID'].str.lower() == search_id.strip().lower()]
        
        if not profile.empty:
            p = profile.iloc[0]
            single_ai = predict_ai_status(pd.DataFrame([p]))[0]
            single_tenure = predict_remaining_tenure(p, single_ai)
            single_xai = explain_ai_prediction(p, single_ai)
            single_plan = get_ai_retention_plan(p)
            
            st.markdown(f"### 👤 Profile Card: {p['Name']} ({p['Employee_ID']})")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.write(f"💵 **Salary:** ${p['Salary']:,} | **Working Hours:** {p['Working_Hours']} hrs (Overtime: {p['Overtime']})")
                st.write(f"🏢 **Department:** {p['Department']} | **Role:** {p['Job_Role']} | **Experience:** {p['Experience_Years']} Yrs")
            with col_p2:
                st.write(f"🎯 **Skills Status:** {p['Skills_Master']} | **Attendance:** {p['Attendance_Pct']}% | 📞 **Phone:** {p.get('Phone', 'N/A')}")
                
            st.write("---")
            st.write(f"🤖 **Live AI Risk Assessment:** {single_ai} | **Expected Tenure:** {single_tenure}")
            
            st.info(single_xai)
            
            if "❌" in single_plan: 
                st.error(f"📋 **AI Corporate Retention Order:** {single_plan}")
                
            
                if "SYSTEM RECOMMENDATION: TERMINATE IMMEDIATELY." in single_plan:
                    st.warning("⚠️ **Management Action Required:** This employee triggers zero-merit threshold guidelines.")
                    if st.button(f"💥 Terminate {p['Name']} (Delete & Archive)", type="primary"):
                        
                        terminated_employee = {
                            "Employee_ID": p['Employee_ID'], "Name": p['Name'], "Age": p['Age'], 
                            "Salary": p['Salary'], "Department": p['Department'], "Job_Role": p['Job_Role'],