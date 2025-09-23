# ASLWebApp: Live ASL Translation with AI-Powered Summaries

**2025 Purdue Capstone Project**

**ASLWebApp** is a state-of-the-art, full-stack web application that brings real-time American Sign Language (ASL) translation to your browser. Using MediaPipe for ultra-fast landmark detection, a custom TensorFlow/Keras gesture recognition model, and OpenAI’s GPT-3.5 Turbo for natural-language summarization, ASLWebApp delivers:

- **Instant Gesture Recognition**  
  A 3-class ASL model (`hello`, `thanks`, `I love you`) runs entirely in your browser via a Flask API, smoothly translating your signs into live subtitles.

- **AI-Powered Contextual Summaries**  
  Every 10 seconds, the app composes 2–3 simple English sentences describing your gestures—no filler words, just clear context—leveraging OpenAI’s API.

- **Full Finger-Spelling Support**  
  A separate finger-signing LSTM model decodes all 26 letters of the ASL alphabet. Hold a sign for 1.5 seconds to lock in each character, building words and phrases on screen.

DEMO:

![ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/7418cc67-5bcd-45b7-bc47-4846b0848a7e)
![gif2-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/293c2e33-9c7b-416c-b5b8-e6d527bf8b5b)


Clone & Enter  
git clone https://github.com/Pranaav003/ASLWebSite.git  
cd ASLWebSite/backend  
Python Env & Dependencies  
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  
Set Environment Variables  
Create a .env file with:  
OPENAI_API_KEY=sk-…  
CAMERA_INDEX=0  
PORT=5001  
Build & Serve Frontend  
cd frontend  
npm ci  
npm run build  
cd ..  
Launch Flask Server  
flask run --host=0.0.0.0 --port=$PORT  
Open http://localhost:5001 in your browser.  
🚀 Deployment  

Build Command:  
pip install -r requirements.txt && cd frontend && npm ci && npm run build  
Start Command:  
flask run --host=0.0.0.0 --port=5001  
Environment: Python 3.11, Node 22.x  
All camera capture and inference happens in the browser—no GPU on the server required!  

🤝 Support & Sponsorship

Pranaav Iyer
📧 pranaav.iyer@gmail.com
📞 408-863-2110
🔗 linkedin.com/in/pranaav-iyer
🔗 Related Repositories

ASL Machine Learning Core (model training & preprocessing):
https://github.com/Pranaav003/ASLMachineLearning
Made with ❤️ by Pranaav Iyer, in partnership with Purdue ECE.
© 2025
