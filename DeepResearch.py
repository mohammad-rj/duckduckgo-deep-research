import asyncio
import re
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import tiktoken
from playwright.async_api import async_playwright
import gradio as gr

# -------------------------------------------------------
# Settings and Helper Functions
# -------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean text: remove empty lines and duplicates."""
    text = text.replace("‍", "").replace("‌", "").replace(" ", "")
    text = re.sub(r'\n\s*\n', '\n', text.strip())
    unique_lines = set()
    cleaned_lines = []
    for line in text.splitlines():
        if line not in unique_lines:
            unique_lines.add(line)
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def tokenize_and_split(text: str, max_tokens=4000) -> list[str]:
    """
    Split text into sections with a maximum of max_tokens using tiktoken.
    """
    tokenizer = tiktoken.encoding_for_model("gpt-4")  # Change model if needed
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
    return chunks

def _sync_summarize_text(
        text: str,
        query: str,
        refined_query_template: str,
        model="gpt-4o-mini"
) -> str:
    """
    Synchronous function to summarize a text chunk using the specified model
    and the user-provided refined_query_template.
    """
    # Use the refined query template from the user
    # (You can assume refined_query_template includes placeholders like {query} if desired)
    refined_query = refined_query_template.format(query=query)

    try:
        print(f"📝 (SYNC) Summarizing a text chunk with model {model}...")
        with DDGS() as ddgs:
            summary = ddgs.chat(refined_query + f"\n\n{text}", model=model)
        print("✅ Chunk summarization completed.")
        return summary
    except Exception as e:
        print(f"❌ Error during summarization: {e}")
        return "Error during summarization."

async def summarize_chunk_async(
        text: str,
        query: str,
        refined_query_template: str,
        model="gpt-4o-mini"
) -> str:
    """
    Asynchronous function to execute synchronous summarization
    within a Thread Executor to enable parallel processing.
    """
    return await asyncio.to_thread(
        _sync_summarize_text,
        text,
        query,
        refined_query_template,
        model
    )

def iterative_summarization(
        text: str,
        query: str,
        final_refined_query_template: str,
        summary_length: int,
        max_tokens=4000,
        model="gpt-4o-mini"
) -> str:
    """
    Iterative summarization until the text is below max_tokens;
    and finally a summary with a specified word count (summary_length).
    """
    tokenizer = tiktoken.encoding_for_model("gpt-4")

    # 1) Iterative summarization until below max_tokens
    while True:
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            break  # Within the allowed limit

        print("🔄 Starting iterative summarization process, text exceeds max_tokens.")
        # Split text into permissible sections
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = tokens[i:i + max_tokens]
            chunks.append(tokenizer.decode(chunk))

        # Summarize each section
        new_summaries = []
        for idx, chunk in enumerate(chunks, 1):
            # Re-use the chunk summarization approach or a minimal approach
            part_summary = _sync_summarize_text(chunk, query, final_refined_query_template, model=model)
            new_summaries.append(part_summary)

        # Combine summaries for the next round
        text = "\n".join(new_summaries)
        print("🔄 Iterative summarization round completed.")

    # 2) Create the final summary with specified word count
    # Use the final refined query template from the user
    # (Likewise assume it might include placeholders like {summary_length} and {query} if desired)
    final_refined_query = final_refined_query_template.format(
        summary_length=summary_length,
        query=query
    )
    try:
        print(f"📝 Creating final {summary_length}-word summary with model {model}...")
        with DDGS() as ddgs:
            final_summary = ddgs.chat(final_refined_query + f"\n\n{text}", model=model)
        print("✅ Final summarization completed.")
        return final_summary
    except Exception as e:
        print(f"❌ Error during final summarization: {e}")
        return "Error during final summarization."

# -------------------------------------------------------
# Main Functions for Playwright
# -------------------------------------------------------

async def fetch_and_summarize(
        sem,
        context,
        url,
        query,
        refined_query_template,
        model="gpt-4o-mini",
        log_callback=None
):
    """
    An async function that:
    1. Queues in max_concurrency using sem.
    2. Opens the page, retrieves and cleans its content.
    3. Processes chunking and summarization in parallel once completed.
    4. Returns the final summary of that page.
    """
    async with sem:  # Limit concurrency
        page = await context.new_page()
        try:
            await log_callback(f"🌐 Loading: {url}")
            await page.goto(url, timeout=60000)
            # Extract content
            content = await page.content()
            # Clean the content
            cleaned = clean_text(BeautifulSoup(content, "html.parser").get_text(separator="\n"))
            # If cleaned content is empty
            if not cleaned.strip():
                await log_callback(f"⚠️ Empty or invalid content for {url}")
                return url, "Empty content"

            # Split content into 4000-token chunks
            content_chunks = tokenize_and_split(cleaned, max_tokens=4000)

            # Summarize each chunk - parallel!
            tasks = []
            for idx, chunk in enumerate(content_chunks, 1):
                await log_callback(f"🗂️ (Async) Summarizing {url}, chunk {idx}/{len(content_chunks)}...")
                # Create async task for each chunk's summarization
                tasks.append(
                    asyncio.create_task(
                        summarize_chunk_async(chunk, query, refined_query_template, model=model)
                    )
                )

            # Wait for all summarization tasks to complete
            partial_summaries = await asyncio.gather(*tasks)

            full_summary = "\n".join(partial_summaries)
            return url, full_summary
        except Exception as e:
            await log_callback(f"❌ Error in loading or summarizing {url}: {e}")
            return url, "Error"
        finally:
            await page.close()

async def run_fetch_and_summarize(
        urls,
        query,
        refined_query_template,
        max_concurrency=5,
        model="gpt-4o-mini",
        log_callback=None
):
    """
    Using Playwright, launch a browser
    and perform controlled concurrency for each URL.
    Summarize chunks in parallel as well.
    """
    async with async_playwright() as p:
        # Choose Chromium browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # Control concurrency with Semaphore
        sem = asyncio.Semaphore(max_concurrency)

        # Create async tasks for each website
        tasks = [
            fetch_and_summarize(
                sem, context, url, query, refined_query_template, model=model, log_callback=log_callback
            )
            for url in urls
        ]

        summaries = []
        # Get results as each task completes
        for task in asyncio.as_completed(tasks):
            url, summary = await task
            await log_callback(f"✅ Summarization of {url} completed.")
            summaries.append(summary)

        await browser.close()

    # Combine all website summaries
    combined_summaries = "\n".join(summaries)
    return combined_summaries

# -------------------------------------------------------
# Gradio Wrapper with Real-Time Logging
# -------------------------------------------------------

async def fetch_and_summarize_gradio(
        query: str,
        max_results: int,
        max_concurrency: int,
        model: str,
        summary_length: int,
        refined_query_template: str,
        final_refined_query_template: str
):
    """
    Async generator function for Gradio that yields updates for logs and final summary.
    """
    logs = []
    log_queue = asyncio.Queue()

    async def log_callback(message: str):
        await log_queue.put(message)

    async def main_process():
        # 1) Search DuckDuckGo
        await log_callback(f"🔍 Query: {query}")

        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_results))
            await log_callback(f"🔎 Found {len(search_results)} search results.")
        except Exception as e:
            await log_callback(f"❌ Error during DuckDuckGo search: {e}")
            return None

        # Extract URLs
        urls = []
        for item in search_results:
            if item.get("href"):
                urls.append(item["href"])

        if not urls:
            await log_callback("❌ No URLs found for the query.")
            return None

        # 2) Fetch content and summarize
        await log_callback(f"🌐 Fetching and summarizing {len(urls)} URLs with max concurrency {max_concurrency}...")

        try:
            combined_summary = await run_fetch_and_summarize(
                urls,
                query,
                refined_query_template,
                max_concurrency=max_concurrency,
                model=model,
                log_callback=log_callback
            )
        except Exception as e:
            await log_callback(f"❌ Error during fetching and summarizing: {e}")
            return None

        # 3) Iterative summarization
        await log_callback("📝 Performing iterative summarization...")
        try:
            final_summary = iterative_summarization(
                combined_summary,
                query=query,
                final_refined_query_template=final_refined_query_template,
                summary_length=summary_length,
                max_tokens=4000,
                model=model
            )
            await log_callback("✅ Final summarization completed.")
            return final_summary
        except Exception as e:
            await log_callback(f"❌ Error during final summarization: {e}")
            return None

    # Start the main processing as a background task
    processing_task = asyncio.create_task(main_process())

    # Continuously yield logs as they come
    while True:
        try:
            # Wait for the next log message with a timeout
            message = await asyncio.wait_for(log_queue.get(), timeout=0.1)
            yield "", "\n".join(logs + [message])
            logs.append(message)
        except asyncio.TimeoutError:
            # Check if the processing task is done
            if processing_task.done():
                break

    # Collect any remaining logs
    while not log_queue.empty():
        message = await log_queue.get()
        logs.append(message)
        yield "", "\n".join(logs)

    # Get the final summary
    final_summary = processing_task.result()

    # Yield the final summary and logs
    if final_summary:
        yield final_summary, "\n".join(logs)
    else:
        yield "", "\n".join(logs)

# -------------------------------------------------------
# Gradio Interface Setup
# -------------------------------------------------------

interface = gr.Interface(
    fn=fetch_and_summarize_gradio,
    inputs=[
        gr.Textbox(label="Enter your query", placeholder="Search query here..."),
        gr.Number(label="Max results", value=20, precision=0),
        gr.Number(label="Max concurrency", value=10, precision=0),
        gr.Textbox(label="Model", value="gpt-4o-mini"),
        gr.Number(label="Summary length", value=200, precision=0),
        gr.Textbox(
            label="Refined Query Template",
            lines=3,
            # Default matches your original:
            value=(
                "Please review and distill the essential insights from the following content in the context "
                "of '{query}'. Focus on the most pertinent details, providing a clear and concise summary that "
                "highlights the critical information."
            )
        ),
        gr.Textbox(
            label="Final Refined Query Template",
            lines=3,
            # Default matches your original:
            value=(
                "Please compose a well-structured, {summary_length}-word summary of the following material with "
                "a focus on '{query}'. Prioritize the most impactful information, ensuring clarity and relevance "
                "throughout your synopsis."
            )
        ),
    ],
    outputs=[
        gr.Textbox(label="Summarized Result", lines=5),
        gr.Textbox(label="Logs", lines=10, interactive=False)
    ],
    title="DuckDuckGo Deep Research",
    description=(
        "Searches the web for a query, fetches content, and summarizes it. "
        "View logs in real-time. You can also customize the prompt templates below."
    ),
    allow_flagging='never',
)

# -------------------------------------------------------
# Launch the Gradio App
# -------------------------------------------------------

if __name__ == "__main__":
    interface.launch(share=True)
