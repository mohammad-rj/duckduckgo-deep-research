# DuckDuckGo Deep Research

the **DuckDuckGo Deep Research** its free fast privacy alternative of Google Deep Research.

## Features
- **Web Search Integration:** Uses DuckDuckGo for secure, privacy-focused web searches.
- **AI Summarization:** Implements advanced AI models to generate concise content summaries.
- **Real-Time Logs:** Displays live, detailed feedback on the summarization process.
- **Customizable Prompts:** Allows customization of the summarization process to better fit user needs.
- **Async Processing:** Efficiently handles multiple tasks simultaneously using asynchronous processing.

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/duckduckgo-deep-research.git
   cd duckduckgo-deep-research
   ```

    ## Installation
    To get started with this tool, you need Python installed on your system. Then, you can install all necessary dependencies as follows:
    
    ```markdown
    pip install -r requirements.txt
    ```
    
    After installing the Python requirements, you also need to install browser support for Playwright:
    
    ```bash
    playwright install
    ```


2. Run the application:
   ```bash
   python DeepResearch.py
   ```
3. Use the Gradio interface to interact with the application locally by inputting queries and viewing summarized results.

    ## Run Instructions
   - Once the application is running, it will provide a local URL like this:
     ```markdown
       Running on local URL: http://127.0.0.1:7860
       ``` 
   - Open the provided URL in your browser to access the Gradio interface and start using the tool.
    
    



## How It Works
1. **Input:** Enter a search query in the Gradio interface.
2. **Fetch and Summarize:** The application uses DuckDuckGo to fetch relevant web pages and generates AI-powered summaries.
3. **Real-Time Monitoring:** View live logs to track the progress of each query and summarization task.
4. **Iterative Refinement:** Summaries are refined iteratively to ensure clarity and conciseness.


## Contributing
Contributions to this project are welcome! To contribute:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Submit a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.

## Support
If you need help using this tool or have any questions, please open an issue in the repository.

## Acknowledgments
Thank you to all contributors and DuckDuckGo for providing a privacy-focused search engine platform.
