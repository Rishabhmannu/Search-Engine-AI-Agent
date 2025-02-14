# Intelligent AI Search Engine Agent 🤖🔎

An intelligent agentic search engine powered by **Groq's Llama3-8b**, Streamlit, and multi-source search tools (DuckDuckGo, Arxiv, Wikipedia). Built with LangChain for robust AI agent orchestration.

## Features ✨
- **Multi-Source Search**: Integrates 3 powerful tools:
  - 🦆 DuckDuckGo (web search)
  - 📚 Arxiv (scientific papers)
  - 🌐 Wikipedia (encyclopedic knowledge)
- **Groq-Powered AI**: Utilizes Groq's ultra-fast Llama3-8b model
- **Smart Agent Optimization**:
  - Auto-stop when sufficient information is found
  - Max 3 iterations to prevent infinite loops
  - Concise answer prioritization
- **Streamlit UI**: User-friendly interface with chat history
- **Error Handling**: Robust parsing and API error recovery

## Getting Started 🚀

### Prerequisites
- Python 3.8+
- [Groq API Key](https://console.groq.com/)
- `.env` file for API keys (optional)

### Installation
```bash
git clone https://github.com/yourusername/ai-search-agent.git
cd ai-search-agent
pip install -r requirements.txt
```

### Setup
1. Get Groq API key from [Groq Console](https://console.groq.com/)
2. Create `.env` file:
```env
GROQ_API_KEY=your_key_here
```

### Run the App
```bash
streamlit run app2.py  # Recommended: optimized version
# or
streamlit run app.py
```

## Usage 💡
1. Enter Groq API key in sidebar
2. Ask questions in natural language:
   - "Explain quantum computing basics"
   - "Latest research on LLM optimization"
   - "Compare CNN and RNN architectures"
3. Watch the agent intelligently combine search sources!

## Code Versions 🔄
- **`app.py`**: Base implementation with custom DuckDuckGo search
- **`app2.py` (Recommended)**: Enhanced version with:
  - Official DuckDuckGoSearchRun tool
  - System instruction prompts
  - Limited iterations (3 max)
  - Early stopping mechanism
  - More robust error handling

## Deployment ☁️
Project is Deployed at Streamlit Community Cloud:

1. Go to [Streamlit Console](https://search-engine-ai-agent-mfwywkerzwtdxllpt9mhj6.streamlit.app/)
2. Enter your Groq API Key
3. Search your query with Intelligent AI Search-Engine

## Contributing 🤝
PRs welcome! Please follow:
1. Fork the repository
2. Create feature branch
3. Submit PR with detailed description

## License 📄
MIT License - See [LICENSE](LICENSE)

## Acknowledgments 🙏
- Groq for LLM infrastructure
- LangChain team for agent framework
- Streamlit for UI components
