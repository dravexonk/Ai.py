# 🤖 Termux Local Hindi AI

यह एंड्रॉइड (Termux) के लिए डिज़ाइन किया गया एक पूरी तरह से ऑफ़लाइन, लोकल और हल्का AI असिस्टेंट है जो बिना किसी API Key या भारी PyTorch फ़्रेमवर्क के सीधे आपके फ़ोन के CPU पर चलता है।

## ✨ विशेषताएँ (Features)
- 🔒 **100% Offline & Private:** बिना इंटरनेट के काम करता है।
- ⚡ **No PyTorch Needed:** C++ आधारित `ctransformers` इंजन का उपयोग करता है जो अत्यंत तेज़ है।
- 🎨 **Rich Terminal UI:** आकर्षक कलर बॉक्सेस, लोडिंग स्पिनर और मार्कडाउन टेक्स्ट फॉर्मेटिंग।
- 🇮🇳 **Hindi Optimized:** `Qwen2.5-0.5B-Instruct` GGUF मॉडल पर आधारित जो हिंदी भाषा को बेहतरीन तरीक़े से समझता है।

## 🚀 Termux में इंस्टॉलेशन (Step-by-Step)

### 1. सिस्टम अपडेट और Git इनस्टॉल करें
```bash
pkg update && pkg upgrade -y
pkg install python git clang cmake -y
