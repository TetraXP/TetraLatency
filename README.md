# ⚡ TetraLatency: The LLM Performance Command Center
<img width="1137" height="1005" alt="image" src="https://github.com/user-attachments/assets/0105a06f-a23c-410b-a6d0-4b934521c63b" />

**TetraLatency** (`tlate`) is a high-performance terminal dashboard designed for developers who demand the absolute lowest latency from their AI agents. It provides a real-time, concurrent "Inference Matrix" to scout, benchmark, and compare LLM providers in seconds.

## 🚀 Why Use TetraLatency?
Modern AI development often relies on finding the best performance-to-cost ratio. **TetraLatency** was built to solve two main problems:
1. **Finding the Fastest Path:** Stop guessing which provider is currently under load. See real-time latency bars across every major FREE and premium provider.
2. **Instant Agent Hot-Swapping:** Designed for **OpenCode** power users, it allows you to instantly re-route your active agent's brain to the fastest available model with a single keystroke.

## 💎 Supported "Free Tier" Providers
This tool is optimized for developers taking advantage of the generous free usage limits provided by:
- 🟢 **NVIDIA NIM:** High-performance, enterprise-grade inference.
- 🟢 **Groq LPU:** The world's fastest inference engine.
- 🟢 **Cerebras:** Wafer-scale speed for instant responses.
- 🟢 **Google Gemini API:** Huge context windows and multimodal power.
- 🟢 **Mistral / Codestral:** Top-tier open-weight models for coding.
- 🟢 **Cohere:** Excellent classification and RAG performance.
- 🟢 **OpenRouter:** A massive directory of free-to-use community models.

## 🔧 Seamless OpenCode Integration
I personally use **TetraLatency** as the primary control plane for my local LLM environment.

### The "Hot-Swap" Workflow:
1. **Scout:** Run `tlate` to see which models are currently hitting the lowest latency (usually Groq or NVIDIA).
2. **Select:** Use the arrow keys to highlight the fastest healthy model.
3. **Deploy:** Press `ENTER`. 

**What happens next?** The tool instantly updates your local **OpenCode** and **Oh-My-OpenCode** configuration files. Your active agents will immediately begin using the new model on their next request—no manual config editing required.

---

## 🛠️ Installation

### Prerequisites
- **Node.js** (v18+)
- **Python** (v3.10+)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/tetraxp/TetraLatency.git
   cd TetraLatency
   ```

2. Install the binary globally:
   ```bash
   npm link
   ```

## 🔑 Configuration
**TetraLatency** is built for zero-config simplicity if you are already an OpenCode user. It automatically detects your keys from:
`~/.local/share/opencode/auth.json`

Alternatively, you can manually export your keys in your `.bashrc` or `.zshrc`:
```bash
# Core Providers
export NVIDIA_API_KEY="your_key"
export OPENROUTER_API_KEY="your_key"
export GROQ_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
export CEREBRAS_API_KEY="your_key"
export MISTRAL_API_KEY="your_key"
export CODESTRAL_API_KEY="your_key"
export COHERE_API_KEY="your_key"
```

## ⌨️ Controls
- **Up / Down Arrows:** Navigate the model matrix.
- **Enter:** Apply the selected model to your OpenCode configuration.
- **Click Headers:** Toggle sorting by Latency, Parameters, or Context.
- **Type Anywhere:** Real-time search/filter for specific models.
- **ESC / Q:** Exit.

## 📜 License
MIT
