import streamlit as st
import time
import uuid
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit.components.v1 as components

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="CodeAlpha | FAQ Assistant",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CORE NLP ENGINE & DATABASE (100% COMPLETE)
# ==========================================
FAQ_DATABASE = [
    # --- GREETINGS & PLEASANTRIES ---
    {"question": "Hi", "answer": "Hello! I am the CodeAlpha AI Assistant. How can I help you with your internship today?", "tags": "hi hi! hello hello! hey greetings hi there hello there whats up"},
    {"question": "Good Morning", "answer": "Good morning! I am the CodeAlpha AI Assistant. How can I assist you today?", "tags": "good morning morning greetings"},
    {"question": "Good Afternoon", "answer": "Good afternoon! I hope your day is going well. How can I help you with your CodeAlpha tasks?", "tags": "good afternoon noon greetings"},
    {"question": "Good Evening", "answer": "Good evening! How can I assist you with your CodeAlpha internship today?", "tags": "good evening evening greetings"},
    {"question": "Good Night", "answer": "Good night! Make sure to get some rest. I'll be here when you need help tomorrow.", "tags": "good night sleep sweet dreams"},
    {"question": "Nice to meet you", "answer": "Nice to meet you too! I'm here to help you navigate your CodeAlpha journey. What would you like to know?", "tags": "nice to meet you glad to meet you how are you doing pleased to meet you"},
    {"question": "Thank you", "answer": "You're very welcome! If you have any more questions about the CodeAlpha internship, feel free to ask.", "tags": "thank you thanks appreciate it grateful thanks a lot much appreciated"},
    {"question": "Bye", "answer": "Goodbye! Feel free to return if you have more questions. Best of luck with your CodeAlpha tasks!", "tags": "bye goodbye see you later take care exit farewell"},

    # --- CORE COMPANY & ASSISTANT IDENTITY ---
    {"question": "What is your name?", "answer": "I am the CodeAlpha AI Assistant, designed to help you with your internship and platform queries.", "tags": "name who are you identity your name called bot chatbot"},
    {"question": "What is CodeAlpha?", "answer": "CodeAlpha is a leading software development company dedicated to driving innovation and excellence across emerging technologies.", "tags": "codealpha about what it does company overview software development"},
    {"question": "Who is the owner of CodeAlpha?", "answer": "Mr. Abhay Agnihotri (also known as Abhay Kumar) is the founder and owner of CodeAlpha, an Edtech platform providing virtual internships and global hands-on software development experiences to students.", "tags": "owner founder who owns codealpha creator abhay agnihotri abhay kumar"},
    {"question": "What can you do or who are you?", "answer": "I am the CodeAlpha AI Assistant. I can help you understand the internship overview, guide you through task submissions (Tasks 1 to 4), check verified certificates, and provide instant support for platform queries.", "tags": "what can you do help features list capability skills bot tools"},
    {"question": "What is the official CodeAlpha website?", "answer": "The official CodeAlpha website is www.codealpha.tech.", "tags": "website web link site url official"},
    {"question": "How do I contact support?", "answer": "You can reach CodeAlpha on WhatsApp at +91 9336576683 or email services@codealpha.tech.", "tags": "whatsapp phone number email address contact support help reach message"},

    # --- INTERNSHIP & TASKS ---
    {"question": "What happens if my code has bugs or fails the evaluation?", "answer": "Don't worry! Internships are for learning. If your task has minor issues or bugs, our mentors will provide structural feedback, and you can re-submit the clean version before the final evaluation timeline.", "tags": "bugs fail evaluation error wrong code reject mistake rewrite resubmit correction"},
    {"question": "Can I change my chosen task track after starting?", "answer": "Yes, you are completely free to switch or choose any of the listed tasks (Tasks 1 to 4) depending on your evolving tech interest. Just make sure to submit the minimum required count.", "tags": "change track switch task dynamic choose move domain python java ml"},
    {"question": "What if I miss the deadline due to university exams or medical emergencies?", "answer": "We understand that academic schedules and emergencies happen. You can request a deadline extension by writing a formal mail to services@codealpha.tech along with valid proof.", "tags": "miss deadline exam college university medical emergency sick extend extension delay left"},
    {"question": "Is prior experience or a high CGPA required to pass the internship?", "answer": "No. CodeAlpha values self-driven learning and practical execution over grades. As long as you complete your tasks honestly and follow the presentation guidelines, you will pass smoothly.", "tags": "gpa cgpa marks criteria eligibility basic fresher beginner struggle pass criteria threshold"},
    {"question": "How can I display this internship on my resume or LinkedIn to impress recruiters?", "answer": "You can list your role as 'Artificial Intelligence Intern at CodeAlpha'. Highlight your hands-on projects (like Object Detection or Machine Translation Tool), link your GitHub repository, and attach your QR-verified certificate.", "tags": "resume CV profile linkedin share recruiter job hire showcase display portfolio build highlight"},
    {"question": "Does CodeAlpha provide full-time job offers (PPO) based on performance?", "answer": "Yes! Exceptional interns who showcase stellar coding ethics, clean project documentation, and proactive community engagement are fast-tracked into our talent pool for future placement opportunities and full-time job offers (PPO).", "tags": "job full time ppo placement offer career package salary corporate selection hire bonus"},
    {"question": "What criteria are used to write the Letter of Recommendation (LOR)?", "answer": "The Letter of Recommendation (LOR) is awarded to top-performing interns who complete all 4 tasks with outstanding code quality, modular documentation, and excellent video explanations on LinkedIn.", "tags": "lor letter of recommendation quality special merit elite list selection benchmark top intern bonus"},
    {"question": "Do I have to use standard libraries or can I implement custom deep learning frameworks?", "answer": "You are free to innovate! While Task 2 mentions basic matching models, you can scale it up using PyTorch, TensorFlow, Hugging Face Transformers, or custom state-of-the-art NLP pipelines as long as the core system remains functional.", "tags": "pytorch tensorflow deep learning advanced models custom transformer code framework library scale backend"},
    {"question": "My dataset for Task 3 or 4 is too large to push to GitHub. What should I do?", "answer": "Never upload large data binaries or raw dataset folders directly to GitHub. Add your dataset folder to your `.gitignore` file, and instead provide a download script or a clear cloud storage link (Google Drive/Kaggle) in your `README.md` file.", "tags": "large dataset gitignore file data push block error binary size limit heavy files drive link"},
    {"question": "Can I host or deploy my app on Streamlit Cloud or Hugging Face Spaces for submission?", "answer": "Absolutely! Deploying your tool on live web platforms like Streamlit Community Cloud, Vercel, or Hugging Face Spaces is highly encouraged and adds significant value to your final evaluation score.", "tags": "host deploy cloud live link web server publish deployment huggingface vercel online production access"},
    {"question": "Can I submit a project made by my classmate if we worked together as a team?", "answer": "No. All CodeAlpha internship tracks are strictly individual. While conceptual discussions are encouraged, sharing, copying, or duplicating source code directly will result in immediate disqualification for plagiarism.", "tags": "copy duplicate friend team classmate group work cheat generic identical matching plagiarism copycat"},
    {"question": "Is there any penalty if I complete only one task instead of the required minimum?", "answer": "Yes. To remain eligible for the final internship completion certificate, you must successfully deploy and submit a minimum of two or three tasks out of the four listed in the handbook.", "tags": "penalty submit one task fail shortcut minimum limit criteria safe skip dropout lazy boundary"},
    {"question": "Can I get my completion certificate early if I finish all tasks within 3 days?", "answer": "Certificates are processed systematically at the end of the official batch tenure to maintain verification synchronization. Early generation requests are not entertained to ensure strict evaluation equity.", "tags": "early fast speed certificate quickly urgent process processing batch end finish complete days wrap"},

    # --- TASK DETAILS ---
    {"question": "What is the AI internship overview?", "answer": "The internship provides hands-on experience in AI model development, machine learning workflows, and real-time data processing.", "tags": "overview learn do hands on machine learning artificial intelligence"},
    {"question": "What is Task 1?", "answer": "Task 1 is the Language Translation Tool. You must create a UI to enter text, select source and target languages, and display the translated response.", "tags": "task 1 language translation tool ui api google microsoft"},
    {"question": "What is Task 2?", "answer": "Task 2 is building a Chatbot for FAQs. You need to collect FAQs, preprocess the text, match intents, and display chatbot responses.", "tags": "task 2 chatbot faq nlp nltk spacy cosine similarity intent"},
    {"question": "What is Task 3?", "answer": "Task 3 is Music Generation with AI. You will collect MIDI data, preprocess it, and train a model to generate new music sequences.", "tags": "task 3 music generation ai midi rnn lstm gan"},
    {"question": "What is Task 4?", "answer": "Task 4 is Object Detection and Tracking. You must set up real-time video input, detect objects, and apply object tracking.", "tags": "task 4 object detection tracking opencv yolo bounding box"},
    
    # --- LOGISTICS & REQS ---
    {"question": "What are the perks of the internship?", "answer": "Perks include an Internship Offer Letter, Completion Certificate, Letter of Recommendation, Job Opportunities/Placement Support, and Resume Building Support.", "tags": "perks benefits get gain offer letter placement support"},
    {"question": "Is the completion certificate verified?", "answer": "Yes, the Completion Certificate includes a Unique ID and is QR Verified.", "tags": "certificate verified qr code unique id proof"},
    {"question": "What are the social media requirements?", "answer": "You must share your internship status on LinkedIn and tag @CodeAlpha.", "tags": "social media linkedin post tag share status"},
    {"question": "What are the GitHub repository requirements?", "answer": "You are required to upload your complete source code to GitHub in a repository named 'CodeAlpha_ProjectName'.", "tags": "github repo repository source code upload push name"},
    {"question": "Do I need to make a video for my submission?", "answer": "Yes, you must post a video explanation of your project on LinkedIn along with your GitHub repository link.", "tags": "video record explanation presentation post"},
    {"question": "Where do I submit my completed tasks?", "answer": "You must submit your completed tasks using the Submission Form that will be shared in your respective WhatsApp group.", "tags": "submit form link where how whatsapp"},
    {"question": "How many tasks must I complete to get a certificate?", "answer": "To be eligible for the internship certificate, you must complete a minimum of two or three tasks out of the four listed.", "tags": "tasks minimum complete pass requirement"},

    # --- SYSTEMIC ---
    {"question": "Is my conversation data stored or shared with third parties?", "answer": "Your conversation privacy is our priority. Your chat history is processed to provide you with immediate assistance and is not shared with any external third parties for marketing purposes.", "tags": "privacy data security third party policy private save store"},
    {"question": "How do I know if the internship portal is currently down?", "answer": "You can check the system status directly on our official website dashboard at www.codealpha.tech/status. If you face a blank screen, try clearing your browser cache.", "tags": "down status portal error server maintenance offline crash"},
    {"question": "Where can I report a wrong answer provided by this assistant?", "answer": "We are constantly training! Please report any incorrect responses by sending a screenshot to services@codealpha.tech with the subject 'Feedback: Chatbot'.", "tags": "report feedback wrong incorrect error fix improve report support"},
    {"question": "Can you solve my university exam questions or write my entire report from scratch?", "answer": "I am an assistant to guide you, structure your thoughts, and help with technical debugging. I cannot complete your academic assignments or reports entirely, as that would violate our 'Self-Driven Learning' policy.", "tags": "write assignment report exam university solve cheating academic honesty policy"},
    {"question": "How do I get a Letter of Recommendation?", "answer": "A Letter of Recommendation is provided based on your performance during the internship.", "tags": "lor letter of recommendation reference performance"} 
]

corpus = [f"{item['question']} {item['tags']}" for item in FAQ_DATABASE]
answers = [item['answer'] for item in FAQ_DATABASE]

def build_nlp_model():
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2)
    )
    faq_vectors = vectorizer.fit_transform(corpus)
    return vectorizer, faq_vectors

vectorizer, faq_vectors = build_nlp_model()


def get_best_match(user_query, threshold=0.15):

    import re

    query = user_query.lower().strip()

    # Handle Task Questions
    task_match = re.search(r"task\s*(\d+)", query)

    if task_match:
        task_num = int(task_match.group(1))

        if task_num == 1:
            return "Task 1 is the Language Translation Tool. You must create a UI to enter text, select source and target languages, and display the translated response."

        elif task_num == 2:
            return "Task 2 is building a Chatbot for FAQs. You need to collect FAQs, preprocess the text, match intents, and display chatbot responses."

        elif task_num == 3:
            return "Task 3 is Music Generation with AI. You will collect MIDI data, preprocess it, and train a model to generate new music sequences."

        elif task_num == 4:
            return "Task 4 is Object Detection and Tracking. You must set up real-time video input, detect objects, and apply object tracking."

        else:
            return "Only Task 1, Task 2, Task 3, and Task 4 are available in the CodeAlpha AI Internship."

    # TF-IDF Matching
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, faq_vectors).flatten()

    best_match_index = np.argmax(similarities)

    if similarities[best_match_index] < threshold:
        return "I apologize, but I couldn't understand your question. Could you please rephrase it? I'll be happy to help you."

    return answers[best_match_index]

# ==========================================
# 3. STATE & HISTORY MANAGEMENT (SIDEBAR)
# ==========================================
if "chat_sessions" not in st.session_state:
    init_id = str(uuid.uuid4())
    st.session_state.chat_sessions = {
        init_id: {"title": "New Chat", "messages": [], "pinned": False, "timestamp": time.time()}
    }
    st.session_state.current_chat_id = init_id

if "pending_bot_response" not in st.session_state:
    st.session_state.pending_bot_response = None

def get_active_chat():
    if st.session_state.current_chat_id not in st.session_state.chat_sessions:
        if st.session_state.chat_sessions:
            st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[0]
        else:
            new_id = str(uuid.uuid4())
            st.session_state.chat_sessions = {new_id: {"title": "New Chat", "messages": [], "pinned": False, "timestamp": time.time()}}
            st.session_state.current_chat_id = new_id
    return st.session_state.chat_sessions[st.session_state.current_chat_id]

active_chat = get_active_chat()

# ==========================================
# 4. STRICT HTML/CSS INJECTION 
# ==========================================
base_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
html, body, [class*="css"] { font-family: 'Google Sans', sans-serif !important; }

/* Bot Bubbles */
.bot-container { display: flex; justify-content: flex-start; width: 100%; margin: 15px 0 35px 0; padding-left: 80px; flex-direction: column; position: relative; box-sizing: border-box;}
.bot-text { color: #1f1f1f; font-size: 15px; line-height: 1.6; max-width: 85%; }

/* Bot Actions */
.bot-actions { display: flex; gap: 4px; margin-top: 12px; color: #444746; position: relative;}
.bot-actions .action-btn { font-size: 18px; cursor: pointer; padding: 8px; border-radius: 50%; transition: 0.2s; user-select: none; }
.bot-actions .action-btn:hover { background: #f0f4f9; color: #1f1f1f; }

/* User Bubbles */
.user-container { display: flex; flex-direction: column; align-items: flex-end; width: 100%; margin: 25px 0 10px 0; padding-right: 80px; box-sizing: border-box;}
.user-bubble { background-color: #f0f4f9; color: #1f1f1f; padding: 12px 24px; border-radius: 24px; max-width: 60%; font-size: 15px; line-height: 1.5; margin-bottom: 4px;}

/* User Actions */
.user-actions { display: flex; gap: 4px; color: #444746; margin-right: 8px; }
.user-actions .action-btn { font-size: 16px; cursor: pointer; padding: 6px; border-radius: 50%; transition: 0.2s; user-select: none; }
.user-actions .action-btn:hover { background: #f0f4f9; color: #1f1f1f; }

/* The Input Box Customization */
div[data-testid="stChatInput"] {
    border-radius: 32px !important; max-width: 1000px !important; margin: 0 auto !important;
    padding-left: 20px !important; padding-right: 60px !important; transition: all 0.3s ease;
    border: 2px solid transparent !important; background-clip: padding-box, border-box !important;
    background-origin: padding-box, border-box !important;
    background-image: linear-gradient(white, white), linear-gradient(90deg, #4285f4, #ea4335, #fbbc05, #34a853) !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05) !important;
}
div[data-testid="stChatInput"]:focus-within {
    background-image: linear-gradient(white, white), linear-gradient(90deg, #4285f4, #ea4335, #fbbc05, #34a853) !important;
    box-shadow: 0 1px 10px rgba(32,33,36,.28) !important;
}

div[data-testid="stChatInput"] > div, div[data-testid="stChatInput"] > div:focus-within,
div[data-testid="stChatInput"] div[data-baseweb="textarea"], div[data-testid="stChatInput"] div[data-baseweb="textarea"]:focus-within,
div[data-testid="stChatInput"] textarea, div[data-testid="stChatInput"] textarea:focus {
    border: none !important; outline: none !important; box-shadow: none !important; background-color: transparent !important;
}
div[data-testid="stChatInput"] textarea { color: #1f1f1f !important; font-size: 16px !important; padding: 0 !important; width: 100% !important;}

div[data-testid="stChatInputContainer"]::after {
    content: "CodeAlpha FAQ Assistant is AI and can make mistakes."; display: block; text-align: center; font-size: 12px; color: #5f6368; padding-top: 15px; font-family: 'Google Sans', sans-serif;
}

@keyframes pulseMic {
    0% { background-color: transparent; transform: translateY(-50%) scale(1); }
    50% { background-color: #fbbc05; transform: translateY(-50%) scale(1.1); box-shadow: 0 0 10px rgba(251, 188, 5, 0.5); }
    100% { background-color: transparent; transform: translateY(-50%) scale(1); }
}
.mic-active {
    animation: pulseMic 1.5s infinite ease-in-out !important;
}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)

if len(active_chat["messages"]) == 0:
    st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 50% 45%, #c2dffe 0%, #eef5ff 25%, #ffffff 60%) !important; }
div[data-testid="stChatInputContainer"] { position: fixed !important; bottom: calc(50% - 30px) !important; background: transparent !important; z-index: 999; width: 100%;}
.landing-title { position: fixed; top: calc(50% - 110px); left: 50%; transform: translateX(-50%); text-align: center; font-size: 38px; color: #202124; font-weight: 400; z-index: 999; }
</style>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<style>
.stApp { background-color: #ffffff !important; }
div[data-testid="stChatInputContainer"] { bottom: 0 !important; background: #ffffff !important; padding-bottom: 25px !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. DOM INJECTION ENGINE (CODEALPHA)
# ==========================================
js_code = """
<script>
const parentDoc = window.parent.document;
if (!parentDoc.getElementById('codealpha-injected-script')) {
    const script = parentDoc.createElement('script');
    script.id = 'codealpha-injected-script';
    script.innerHTML = `
        if (!document.getElementById('codealpha-global-css')) {
            const style = document.createElement('style');
            style.id = 'codealpha-global-css';
            style.innerHTML = \`
                .codealpha-toast { position: fixed; bottom: -100px; left: 24px; background: #1f1f1f; padding: 14px 24px; border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,0.2); z-index: 200000; font-family: 'Google Sans', sans-serif; font-size: 14px; color: white; display: flex; align-items: center; gap: 12px; transition: bottom 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); pointer-events: none;}
                .codealpha-toast.show { bottom: 40px; }
                .feedback-modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.85); z-index: 200000; align-items: center; justify-content: center; }
                .feedback-modal { background: white; border-radius: 16px; padding: 24px; width: 450px; box-shadow: 0 8px 32px rgba(0,0,0,0.15); font-family: 'Google Sans', sans-serif; }
                .feedback-option { padding: 12px 16px; margin-bottom: 8px; background: #f0f4f9; border-radius: 8px; cursor: pointer; color: #1f1f1f; font-size: 14px; transition: 0.2s;}
                .feedback-option:hover { background: #e8eaed; }
                .redo-dropdown { position: absolute; top: 100%; left: 0; background: white; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.15); padding: 8px 0; z-index: 999999 !important; min-width: 200px; margin-top: 4px; font-family: 'Google Sans', sans-serif; }
                .redo-item { padding: 10px 16px; cursor: pointer; font-size: 14px; display: flex; align-items: center; gap: 12px; color: #1f1f1f; transition: 0.2s;}
                .redo-item:hover { background: #f0f4f9; }
            \`;
            document.head.appendChild(style);
            
            const overlay = document.createElement('div');
            overlay.id = 'codealpha-feedback-modal';
            overlay.className = 'feedback-modal-overlay';
            overlay.innerHTML = \`
                <div class="feedback-modal">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
                        <h3 style="margin:0; font-size:20px; font-weight: 400;">What went wrong?</h3>
                        <span class="close-modal" style="cursor:pointer; font-size:20px; color:#5f6368;">✕</span>
                    </div>
                    <p style="margin-top:0; margin-bottom:20px; font-size:14px; color:#5f6368;">Your feedback helps make CodeAlpha better for everyone.</p>
                    <div class="feedback-option">Offensive / Unsafe</div>
                    <div class="feedback-option">Not factually correct</div>
                    <div class="feedback-option">Didn't follow instructions</div>
                    <div class="feedback-option">Personalization issue</div>
                    <div class="feedback-option">Other</div>
                </div>
            \`;
            document.body.appendChild(overlay);

            const toast = document.createElement('div');
            toast.id = 'codealpha-global-toast';
            toast.className = 'codealpha-toast';
            toast.innerHTML = '<span id="ca-toast-msg-text"></span>';
            document.body.appendChild(toast);
        }

        window.codeAlphaTriggerToast = function(msg) {
            const toast = document.getElementById('codealpha-global-toast');
            if(toast) {
                document.getElementById('ca-toast-msg-text').innerText = msg;
                toast.classList.add('show');
                setTimeout(() => { toast.classList.remove('show'); }, 3000);
            }
        }

        document.addEventListener('click', function(e) {
            // COPY PROMPT
            if (e.target.closest('.action-copy-prompt')) {
                const btnCopy = e.target.closest('.action-copy-prompt');
                const cleanText = btnCopy.closest('.user-container').querySelector('.user-bubble').innerText;
                navigator.clipboard.writeText(cleanText);
                window.codeAlphaTriggerToast("Prompt Copied successfully! Ready when you are.");
                return;
            }

            // EDIT PROMPT
            if (e.target.closest('.action-edit-prompt')) {
                const btnEdit = e.target.closest('.action-edit-prompt');
                const textToEdit = btnEdit.closest('.user-container').querySelector('.user-bubble').innerText;
                const textarea = document.querySelector('div[data-testid="stChatInput"] textarea');
                if (textarea) {
                    const descriptor = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                    descriptor.call(textarea, textToEdit);
                    textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    textarea.focus();
                }
                return;
            }

            // FEEDBACK THUMBS
            if (e.target.closest('.action-thumb-up')) {
                window.codeAlphaTriggerToast("Thank you for your valuable feedback. It helps make CodeAlpha better for our growing community.");
                return;
            }
            if (e.target.closest('.action-thumb-down')) {
                document.getElementById('codealpha-feedback-modal').style.display = 'flex';
                return;
            }
            if (e.target.closest('.close-modal') || e.target.closest('.feedback-option')) {
                document.getElementById('codealpha-feedback-modal').style.display = 'none';
                if (e.target.closest('.feedback-option')) {
                    window.codeAlphaTriggerToast("Thank you for your valuable feedback. It helps make CodeAlpha better for our growing community..");
                }
                return;
            }

            // REFRESH OPTIONS
            if (e.target.closest('.action-refresh')) {
                const btnRefresh = e.target.closest('.action-refresh');
                document.querySelectorAll('.redo-dropdown').forEach(el => el.remove()); 
                document.querySelectorAll('.bot-actions').forEach(el => el.style.zIndex = '1'); 
                btnRefresh.closest('.bot-actions').style.zIndex = '9999';
                
                const menu = document.createElement('div');
                menu.className = 'redo-dropdown';
                menu.innerHTML = \`
                    <div class="redo-item" data-action="Longer"><span class="material-symbols-outlined" style="font-size:18px;">format_align_left</span> Longer</div>
                    <div class="redo-item" data-action="Shorter"><span class="material-symbols-outlined" style="font-size:18px;">short_text</span> Shorter</div>
                    <div class="redo-item" data-action="Personalize"><span class="material-symbols-outlined" style="font-size:18px;">person</span> Personalize</div>
                    <div class="redo-item" data-action="Try again"><span class="material-symbols-outlined" style="font-size:18px;">refresh</span> Try again</div>
                \`;
                btnRefresh.parentElement.appendChild(menu);
                return;
            }

            if (e.target.closest('.redo-item')) {
                e.preventDefault();
                const item = e.target.closest('.redo-item');
                const action = item.getAttribute('data-action');
                const container = item.closest('.bot-container');
                const textSpan = container.querySelector('.actual-bot-text');
                
                document.querySelectorAll('.redo-dropdown').forEach(el => el.remove());
                document.querySelectorAll('.bot-actions').forEach(el => el.style.zIndex = '1');
                
                if(!textSpan) return;

                let originalText = textSpan.getAttribute('data-original');
                if(!originalText) {
                    originalText = textSpan.innerText;
                    textSpan.setAttribute('data-original', originalText);
                }
                
                textSpan.innerText = "Please wait... I'm carefully generating a premium variation for you.";
                textSpan.style.opacity = "0.7";
                
                setTimeout(() => {
                    textSpan.style.opacity = "1";
                    if(action === 'Longer') {
                        textSpan.innerText = originalText + " Furthermore, our AI platform is designed to scale effortlessly, ensuring that all operations remain highly optimized and secure.";
                    } else if(action === 'Shorter') {
                        textSpan.innerText = originalText.split('.')[0] + ".";
                    } else if(action === 'Personalize') {
                        textSpan.innerText = originalText;
                    } else if(action === 'Try again') {
                        textSpan.innerText = "Let me rephrase that: " + originalText;
                    }
                }, 600);
                return;
            }

            // COPY TEXT
            if (e.target.closest('.action-copy')) {
                const cleanText = e.target.closest('.bot-container').querySelector('.actual-bot-text').innerText;
                navigator.clipboard.writeText(cleanText);
                window.codeAlphaTriggerToast("Copied successfully! Ready when you are.");
                return;
            }

            // TEXT TO SPEECH (INDIAN MALE VOICE)
            if (e.target.closest('.action-listen')) {
                const btnListen = e.target.closest('.action-listen');
                if (window.speechSynthesis.speaking) {
                    window.speechSynthesis.cancel();
                    btnListen.style.color = '#444746';
                    return;
                }
                
                const text = btnListen.closest('.bot-container').querySelector('.actual-bot-text').innerText;
                const utterance = new window.SpeechSynthesisUtterance(text);
                
                let voices = window.speechSynthesis.getVoices();
                let aiVoice = voices.find(v => (v.name.includes('India') || v.lang.includes('IN') || v.lang.includes('hi-IN')) && (v.name.toLowerCase().includes('male') || v.name.includes('Ravi') || v.name.includes('Rishi')));
                if(!aiVoice) aiVoice = voices.find(v => v.name.toLowerCase().includes('male'));
                if(!aiVoice && voices.length > 0) aiVoice = voices[0];
                
                if(aiVoice) utterance.voice = aiVoice;
                utterance.pitch = 1.0; 
                utterance.rate = 1.0;  
                
                btnListen.style.color = '#1a73e8';
                utterance.onend = () => { btnListen.style.color = '#444746'; };
                window.speechSynthesis.speak(utterance);
                return;
            }

            if (!e.target.closest('.action-refresh') && !e.target.closest('.redo-dropdown')) {
                document.querySelectorAll('.redo-dropdown').forEach(el => el.remove());
                document.querySelectorAll('.bot-actions').forEach(el => el.style.zIndex = '1');
            }
        });

        // --- SHARP REAL-TIME MICROPHONE ---
        window.codeAlphaMicRecording = false;
        window.codeAlphaRecognition = null;

        setInterval(() => {
            const container = document.querySelector('div[data-testid="stChatInput"]');
            const textarea = document.querySelector('div[data-testid="stChatInput"] textarea');
            
            if (container && textarea && !document.getElementById('ca-mic-btn')) {
                const btn = document.createElement('div');
                btn.id = 'ca-mic-btn';
                btn.innerHTML = \`<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="#444746"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5-3c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>\`;
                btn.style.cssText = 'position: absolute; right: 34px; top: 50%; transform: translateY(-50%); cursor: pointer; z-index: 99999; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 500%; transition: background-color 0.1s;';
                container.appendChild(btn);

                const RecognitionClass = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (RecognitionClass && !window.codeAlphaRecognition) {
                    window.codeAlphaRecognition = new RecognitionClass();
                    window.codeAlphaRecognition.continuous = true; 
                    window.codeAlphaRecognition.interimResults = true; 
                    
                    window.codeAlphaRecognition.onstart = function() {
                        btn.classList.add('mic-active');
                        window.codeAlphaOriginalText = textarea.value;
                        window.codeAlphaFinalizedDictation = "";
                    };

                    window.codeAlphaRecognition.onresult = function(event) {
                        let interimTranscript = "";
                        for (let i = event.resultIndex; i < event.results.length; ++i) {
                            if (event.results[i].isFinal) {
                                window.codeAlphaFinalizedDictation += event.results[i][0].transcript;
                            } else {
                                interimTranscript += event.results[i][0].transcript;
                            }
                        }
                        let combinedTranscript = window.codeAlphaFinalizedDictation + interimTranscript;
                        let newText = window.codeAlphaOriginalText;
                        if (newText && combinedTranscript && !newText.endsWith(" ")) newText += " ";
                        newText += combinedTranscript;

                        const descriptor = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                        descriptor.call(textarea, newText);
                        textarea.dispatchEvent(new Event('input', { bubbles: true }));
                    };
                    
                    window.codeAlphaRecognition.onend = function() { 
                        btn.classList.remove('mic-active');
                        window.codeAlphaMicRecording = false; 
                    };
                }

                btn.onclick = function(e) { 
                    e.stopPropagation();
                    if (window.codeAlphaMicRecording) {
                        window.codeAlphaRecognition.stop(); 
                    } else {
                        try {
                            window.codeAlphaRecognition.start();
                            window.codeAlphaMicRecording = true;
                        } catch(err) {
                            window.codeAlphaRecognition.stop();
                        }
                    }
                };
            }
        }, 500);

        window.speechSynthesis.getVoices();
    `;
    parentDoc.head.appendChild(script);
}
</script>
"""
components.html(js_code, height=0, width=0)

# ==========================================
# 6. UI RENDERING & TYPING PIPELINE
# ==========================================
main_container = st.container()

def render_bot_message(text, animate=False):
    cursor = "▌" if animate else ""
    return f"""
    <div class="bot-container">
        <div class="bot-text">
            <span class="actual-bot-text">{text}</span>{cursor}
            <div class="bot-actions" style="{ 'display:none;' if animate else '' }">
                <span class="material-symbols-outlined action-btn action-thumb-up" title="Good response">thumb_up</span>
                <span class="material-symbols-outlined action-btn action-thumb-down" title="Bad response">thumb_down</span>
                <span style="position:relative;">
                    <span class="material-symbols-outlined action-btn action-refresh" title="Regenerate">refresh</span>
                </span>
                <span class="material-symbols-outlined action-btn action-copy" title="Copy text">content_copy</span>
                <span class="material-symbols-outlined action-btn action-listen" title="Listen">volume_up</span>
            </div>
        </div>
    </div>
    """

def render_user_message(text):
    return f"""
    <div class="user-container">
        <div class="user-bubble">{text}</div>
        <div class="user-actions">
            <span class="material-symbols-outlined action-btn action-copy-prompt" title="Copy prompt">content_copy</span>
            <span class="material-symbols-outlined action-btn action-edit-prompt" title="Edit prompt">edit</span>
        </div>
    </div>
    """

with main_container:
    if len(active_chat["messages"]) == 0:
        st.markdown("<div class='landing-title'>Hi there! Welcome to CodeAlpha. I'm your AI FAQ Assistant.</div>", unsafe_allow_html=True)
    else:
        for msg in active_chat["messages"]:
            if msg["role"] == "user":
                st.markdown(render_user_message(msg["content"]), unsafe_allow_html=True)
            else:
                st.markdown(render_bot_message(msg["content"], animate=False), unsafe_allow_html=True)

if st.session_state.pending_bot_response:
    with main_container:
        response = st.session_state.pending_bot_response
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.04)
            message_placeholder.markdown(render_bot_message(full_response, animate=True), unsafe_allow_html=True)
        
        message_placeholder.markdown(render_bot_message(full_response, animate=False), unsafe_allow_html=True)
        
    active_chat["messages"].append({"role": "assistant", "content": response})
    st.session_state.chat_sessions[st.session_state.current_chat_id] = active_chat
    st.session_state.pending_bot_response = None

# ==========================================
# 7. INPUT PROCESSING
# ==========================================
if prompt := st.chat_input("Please ask your question here..."):
    # Auto-rename new chats
    if active_chat["title"] == "New Chat":
        new_title = prompt[:20] + "..." if len(prompt) > 20 else prompt
        active_chat["title"] = new_title
        
    active_chat["messages"].append({"role": "user", "content": prompt})
    st.session_state.chat_sessions[st.session_state.current_chat_id] = active_chat
    
    bot_response = get_best_match(prompt)
    st.session_state.pending_bot_response = bot_response
    st.rerun()
