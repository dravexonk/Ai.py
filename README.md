# 🤖 Termux Universal Local AI Engine (Offline GGUF Runner)

> **Run powerful Large Language Models (LLMs) locally on your Android phone via Termux without PyTorch, API keys, or heavy dependencies!**

---

## 🌟 Introduction / परिचय

यह प्रोजेक्ट विशेष रूप से **Termux (Android)** और Linux users के लिए बनाया गया है। इसकी मदद से आप अपने मोबाइल फोन पर बिना इंटरनेट (चैट के दौरान), बिना किसी API Key और बिना किसी झंझट के खुद का **Personal Offline AI Assistant** चला सकते हैं। 

यह C++ आधारित `ctransformers` इंजन का उपयोग करता है, जो मोबाइल के CPU पर बेहद तेज और कम RAM खपत के साथ काम करता है।

---

## ✨ Features / विशेषताएँ

* **🚀 Zero PyTorch Dependency:** भारी-भरकम PyTorch या GPU की कोई आवश्यकता नहीं; यह सीधे C++ बैकएंड पर चलता है।
* **🛠️ Auto-Dependency & Setup:** टर्मक्स में रन करते ही यह खुद-ब-खुद ज़रूरी कंपाइलर (`clang`, `cmake`, `rust`, `binutils`) और पाइथन लाइब्रेरीज को सही फ्लैग्स के साथ इंस्टॉल कर लेता है।
* **📥 Interactive Hugging Face Downloader:** ऐप शुरू होते ही आपके सामने मॉडल्स की लिस्ट आती है। नंबर दबाकर मॉडल चुनें और यह उसे सीधे Hugging Face से डाउनलोड कर लेगा।
* **🎛️ Custom Model Support:** आप चाहें तो लिस्ट के अलावा Hugging Face से अपनी पसंद का कोई भी दूसरा GGUF मॉडल (Custom Repo ID & Filename देकर) डाउनलोड कर सकते हैं।
* **🧠 Multi-Turn Chat Memory:** यह पिछली बातचीत (Context) को याद रखता है, जिससे आप एक नेचुरल बातचीत कर सकते हैं।
* **⚡ Real-Time Streaming Output:** AI का जवाब एक बार में लोड होने के बजाय लाइव टाइपराइटर की तरह स्क्रीन पर आता है।

---
# ​/clear - अगर आप पुरानी बातचीत की याददाश्त (Memory) साफ़ करके नए सिरे से बात शुरू करना चाहते हैं।
​# /help - अगर कोई कमांड भूल जाएँ तो मदद देखने के लिए।
# ​/exit या बाहर - जब चैट खत्म करके बाहर आना हो।

## ⚙️ Quick Start Commands (टर्मक्स में चलाने के कमांड्स)

अपने Termux में इन आसान कमांड्स को कॉपी करके पेस्ट करें:

```bash
# 1. Termux packages update aur python/git install karein
pkg update && pkg install python git -y

# 2. Apni GitHub repository ko clone karein (Apna link daalein)
git clone [https://github.com/dravexonk/Ai.py.git]
# 3. Project folder ke andar jayein
cd Ai.py

# 4. Program run karein (Baki sabhi libraries yeh khud auto-install kar lega)
python main.py
