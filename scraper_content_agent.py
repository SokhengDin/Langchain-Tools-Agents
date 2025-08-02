import requests
import base64
import re
import os
from urllib.parse import urljoin
from typing import List
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()


@tool
def scrape_website(url: str) -> str:
    """Scrape content from a website and extract clean text."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response    = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        text = soup.get_text()
        
        lines   = (line.strip() for line in text.splitlines())
        chunks  = (phrase.strip() for line in lines for phrase in line.split("  "))
        text    = ' '.join(chunk for chunk in chunks if chunk)
        
        if len(text) > 5000:
            text = text[:5000] + "... [Content truncated]"
        
        return f"✅ Scraped content from {url}:\n\n{text}"
        
    except Exception as e:
        return f"❌ Error scraping {url}: {str(e)}"


@tool
def extract_links_from_page(url: str) -> str:
    """Extract all links from a webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup    = BeautifulSoup(response.content, 'html.parser')
        links   = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            if href.startswith('/'):
                href = urljoin(url, href)
            elif not href.startswith('http'):
                continue
                
            if text:
                links.append(f"• {text}: {href}")
        
        if links:
            return f"🔗 Found {len(links)} links on {url}:\n\n" + "\n".join(links[:20])
        else:
            return f"No links found on {url}"
            
    except Exception as e:
        return f"❌ Error extracting links from {url}: {str(e)}"


@tool
def extract_images_from_page(url: str) -> str:
    """Extract image URLs from a webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup    = BeautifulSoup(response.content, 'html.parser')
        images  = []
        
        for img in soup.find_all('img', src=True):
            src = img['src']
            alt = img.get('alt', 'No alt text')
            
            if src.startswith('/'):
                src = urljoin(url, src)
            elif not src.startswith('http'):
                continue
                
            images.append(f"🖼️ {alt}: {src}")
        
        if images:
            return f"🖼️ Found {len(images)} images on {url}:\n\n" + "\n".join(images[:15])
        else:
            return f"No images found on {url}"
            
    except Exception as e:
        return f"❌ Error extracting images from {url}: {str(e)}"


@tool
def analyze_image_with_gemma3(image_path_or_url: str, question: str = "Describe this image") -> str:
    """Analyze an image using Gemma3's vision capabilities."""
    try:
        if image_path_or_url.startswith('http'):
            response    = requests.get(image_path_or_url, timeout=10)
            response.raise_for_status()
            image_data  = response.content
        else:
            with open(image_path_or_url, 'rb') as f:
                image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "model"     : "gemma3:27b",
            "prompt"    : question,
            "images"    : [image_base64],
            "stream"    : False
        }
        
        api_base_url = os.getenv('LLM_API_BASE_URL', 'http://localhost:11434')
        response = requests.post(
            f"{api_base_url}/api/generate",
            json        = payload,
            timeout     = 30
        )
        
        if response.status_code == 200:
            result      = response.json()
            analysis    = result.get('response', 'No analysis generated')
            return f"🔍 Gemma3 Image Analysis:\n{analysis}"
        else:
            return f"❌ Error analyzing image: HTTP {response.status_code}"
            
    except Exception as e:
        return f"❌ Error analyzing image: {str(e)}"


@tool
def extract_text_from_image(image_path_or_url: str) -> str:
    """Extract text from an image using OCR (supports Khmer and English)."""
    try:
        if image_path_or_url.startswith('http'):
            response    = requests.get(image_path_or_url, timeout=10)
            response.raise_for_status()
            image       = Image.open(io.BytesIO(response.content))
        else:
            image       = Image.open(image_path_or_url)
        
        try:
            text_khmer  = pytesseract.image_to_string(image, lang='khm+eng')
            if text_khmer.strip():
                extracted_text = text_khmer.strip()
            else:
                extracted_text = pytesseract.image_to_string(image, lang='eng').strip()
        except:
            extracted_text = pytesseract.image_to_string(image).strip()
        
        if extracted_text:
            return f"📝 Extracted text from image:\n\n{extracted_text}"
        else:
            return "❌ No text found in the image"
            
    except Exception as e:
        return f"❌ Error extracting text from image: {str(e)}"


@tool
def detect_khmer_text(text: str) -> str:
    """Detect and analyze Khmer text content."""
    khmer_pattern   = re.compile(r'[\u1780-\u17FF]+')
    khmer_matches   = khmer_pattern.findall(text)
    
    if khmer_matches:
        khmer_text  = ' '.join(khmer_matches)
        word_count  = len(khmer_matches)
        char_count  = len(khmer_text.replace(' ', ''))
        
        return f"""🇰🇭 Khmer Text Detected:
        
📊 Statistics:
• Khmer segments: {word_count}
• Khmer characters: {char_count}

📝 Extracted Khmer text:
{khmer_text}

🔍 Analysis: This text contains Khmer script, which is the writing system for the Cambodian language."""
    else:
        return "❌ No Khmer text detected in the provided content"


@tool
def summarize_content(content: str, focus: str = "general") -> str:
    """Summarize long content with specific focus areas."""
    if len(content) < 200:
        return f"✅ Content is already concise ({len(content)} characters)"
    
    sentences = re.split(r'[.!?]+', content)
    
    important_words = {
        'general'   : ['important', 'key', 'main', 'significant', 'major', 'primary'],
        'technical' : ['algorithm', 'system', 'method', 'implementation', 'architecture'],
        'news'      : ['announced', 'reported', 'breaking', 'update', 'latest', 'today'],
        'academic'  : ['research', 'study', 'findings', 'analysis', 'methodology', 'conclusion']
    }
    
    focus_words = important_words.get(focus, important_words['general'])
    
    scored_sentences    = []
    for sentence in sentences[:10]:
        sentence        = sentence.strip()
        if len(sentence) > 20:
            score       = sum(1 for word in focus_words if word.lower() in sentence.lower())
            scored_sentences.append((score, sentence))
    
    top_sentences   = sorted(scored_sentences, reverse=True)[:3]
    summary         = '. '.join([sent[1] for sent in top_sentences]) + '.'
    
    return f"📋 Summary ({focus} focus):\n\n{summary}"


@tool
def evaluate_extraction_quality(extracted_content: str, original_query: str) -> str:
    """Evaluate the quality and completeness of content extraction."""
    quality_metrics = {
        'completeness'  : 0,
        'relevance'     : 0,
        'accuracy'      : 0,
        'usefulness'    : 0
    }
    
    content_length = len(extracted_content)
    
    if content_length > 500:
        quality_metrics['completeness'] = 10
    elif content_length > 200:
        quality_metrics['completeness'] = 7
    else:
        quality_metrics['completeness'] = 4
    
    query_words = original_query.lower().split()
    content_lower = extracted_content.lower()
    matching_words = sum(1 for word in query_words if word in content_lower)
    quality_metrics['relevance'] = min(10, (matching_words / len(query_words)) * 10)
    
    if "error" in extracted_content.lower():
        quality_metrics['accuracy'] = 3
    elif "✅" in extracted_content:
        quality_metrics['accuracy'] = 9
    else:
        quality_metrics['accuracy'] = 7
    
    if any(indicator in extracted_content for indicator in ["🔗", "🖼️", "📝", "🇰🇭"]):
        quality_metrics['usefulness'] = 9
    else:
        quality_metrics['usefulness'] = 6
    
    avg_score = sum(quality_metrics.values()) / len(quality_metrics)
    
    return f"""📊 Extraction Quality Evaluation:

🎯 Overall Score: {avg_score:.1f}/10

📈 Detailed Metrics:
• Completeness: {quality_metrics['completeness']}/10
• Relevance: {quality_metrics['relevance']:.1f}/10  
• Accuracy: {quality_metrics['accuracy']}/10
• Usefulness: {quality_metrics['usefulness']}/10

💡 Assessment: {'Excellent' if avg_score >= 8 else 'Good' if avg_score >= 6 else 'Needs Improvement'}"""


@tool
def write_comprehensive_content(source_data: str, content_type: str = "article", style: str = "professional") -> str:
    """Generate comprehensive written content based on extracted data using Qwen3 writer.
    
    Args:
        source_data: The extracted content to base the writing on
        content_type: Type of content (news, report, article, blog, academic, summary)
        style: Writing style (professional, casual, academic, journalistic, creative)
        
    Returns:
        Well-written comprehensive content
    """
    try:
        api_base_url = os.getenv('LLM_API_BASE_URL', 'http://localhost:11434')
        qwen3_writer = ChatOllama(
            model       = "qwen3:8b",
            base_url    = api_base_url, 
            temperature = 0.7,
            streaming   = False
        )
        
        style_prompts = {
            'professional': "Write in a clear, professional tone suitable for business communication.",
            'casual': "Write in a friendly, conversational tone that's easy to understand.",
            'academic': "Write in a formal, scholarly tone with proper citations and analysis.",
            'journalistic': "Write in an objective, informative news style with key facts highlighted.",
            'creative': "Write with engaging, vivid language that captures the reader's attention."
        }
        
        content_formats = {
            'news': """Structure as a news article with:
- Compelling headline
- Lead paragraph with key facts
- Supporting details in inverted pyramid format
- Quotes and context
- Conclusion with implications""",
            
            'report': """Structure as a comprehensive report with:
- Executive summary
- Key findings and insights
- Detailed analysis
- Supporting data and evidence
- Recommendations and next steps""",
            
            'article': """Structure as an informative article with:
- Engaging introduction
- Well-organized main points
- Supporting details and examples
- Clear transitions between sections
- Strong conclusion""",
            
            'blog': """Structure as an engaging blog post with:
- Catchy title and hook
- Personal/relatable introduction
- Main points with subheadings
- Practical examples or tips
- Call-to-action conclusion""",
            
            'academic': """Structure as an academic piece with:
- Abstract/introduction
- Literature review of sources
- Methodology and analysis
- Findings and discussion
- Conclusion and implications""",
            
            'summary': """Create a comprehensive executive summary with:
- Key highlights upfront
- Main findings organized by importance
- Supporting evidence
- Actionable insights
- Brief conclusion"""
        }
        
        writing_prompt = f"""<thinking>
I need to write a comprehensive {content_type} based on the provided source data. Let me analyze what I have and structure it according to the requested format and style.

The user wants: {content_type} in {style} style
Source data length: {len(source_data)} characters

I should create well-structured, comprehensive content that maximizes the value of the extracted information.
</thinking>

You are an expert writer specializing in creating comprehensive, high-quality content. Create a {content_type} based on the following extracted data:

**Source Data:**
{source_data}

**Requirements:**
- Content Type: {content_type.title()}
- Writing Style: {style_prompts.get(style, style_prompts['professional'])}
- Length: Comprehensive and detailed (800-1200 words)

**Structure Guidelines:**
{content_formats.get(content_type, content_formats['article'])}

**Key Instructions:**
1. Make the content engaging and valuable to readers
2. Organize information logically with clear headings
3. Include specific details and examples from the source data
4. Ensure accuracy while making it accessible
5. Add insights and analysis beyond just summarizing
6. Use proper formatting with headings, bullet points where appropriate
7. Include a compelling introduction and strong conclusion

Create comprehensive, publication-ready content that transforms the raw data into valuable, readable material."""
        
        response = qwen3_writer.invoke([HumanMessage(content=writing_prompt)])
        written_content = response.content if hasattr(response, 'content') else str(response)
        
        return f"✍️ Qwen3 Writer Output ({content_type} - {style} style):\n\n{written_content}"
        
    except Exception as e:
        return f"❌ Writing error: {str(e)}"


class ScrapeContentAgent:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('LLM_API_BASE_URL', 'http://localhost:11434')
        
        self.tools = [
            scrape_website,
            extract_links_from_page,
            extract_images_from_page,
            analyze_image_with_gemma3,
            extract_text_from_image,
            detect_khmer_text,
            summarize_content,
            evaluate_extraction_quality,
            write_comprehensive_content
        ]
        
        # Llama3.1 for tool orchestration (supports tool calling)
        self.llama3_orchestrator = ChatOllama(
            model       = "llama3.1:latest",
            base_url    = self.base_url,
            temperature = 0.4,
            streaming   = True
        ).bind_tools(self.tools)
        
        # Gemma3 for vision-only tasks (no tool calling)
        self.gemma3_vision = ChatOllama(
            model       = "gemma3:27b",
            base_url    = self.base_url,
            temperature = 0.4,
            streaming   = False
        )
        
        # Qwen3 for reflection and writing
        self.qwen3_reflector = ChatOllama(
            model       = "qwen3:8b",
            base_url    = self.base_url,
            temperature = 0.2,
            streaming   = True
        )
        
        self.conversation_history = []
        self.workflow_graph = self._create_content_workflow()
    
    def _create_content_workflow(self):
        class ContentState(TypedDict):
            user_query          : str
            content_type        : str
            extracted_data      : List[str]
            analysis_results    : List[str]
            gemma3_insights     : str
            reflection_feedback : str
            written_content     : str
            final_response      : str
            needs_writing       : bool
            needs_reflection    : bool
            has_khmer          : bool
        
        def input_analysis_node(state: ContentState):
            return {
                "user_query"    : "User provides content extraction request",
                "content_type"  : "Analyzing input type..."
            }
        
        def llama3_orchestration_node(state: ContentState):
            return {
                "extracted_data": ["Llama3.1 orchestrates tools", "Web scraping", "Text processing"],
                "content_type"  : state.get("content_type", "mixed")
            }
        
        def gemma3_vision_node(state: ContentState):
            return {
                "gemma3_insights"   : "Gemma3 analyzes images and visual content (vision-only)",
                "analysis_results"  : state.get("analysis_results", [])
            }
        
        def reflection_node(state: ContentState):
            return {
                "reflection_feedback"   : "Qwen3 evaluates extraction quality and provides improvement suggestions",
                "analysis_results"      : state.get("analysis_results", [])
            }
        
        def qwen3_writer_node(state: ContentState):
            return {
                "written_content"       : "Qwen3 creates comprehensive written content based on extracted data",
                "analysis_results"      : state.get("analysis_results", [])
            }
        
        def khmer_processing_node(state: ContentState):
            return {
                "analysis_results": state.get("analysis_results", []) + ["Khmer text detection and processing"]
            }
        
        def content_synthesis_node(state: ContentState):
            return {
                "final_response": "Comprehensive content analysis with multimodal insights, quality evaluation, and written output"
            }
        
        def needs_khmer_processing(state: ContentState):
            return "khmer_processing" if state.get("has_khmer", False) else "reflection"
        
        def content_type_router(state: ContentState):
            content_type = state.get("content_type", "mixed")
            if "image" in content_type:
                return "gemma3_vision"
            else:
                return "reflection"
        
        def needs_writing(state: ContentState):
            return "qwen3_writer" if state.get("needs_writing", True) else "content_synthesis"
        
        builder = StateGraph(ContentState)
        
        builder.add_node("📥 input_analysis", input_analysis_node)
        builder.add_node("🤖 llama3_orchestration", llama3_orchestration_node)
        builder.add_node("👁️ gemma3_vision", gemma3_vision_node)
        builder.add_node("🔍 qwen3_reflection", reflection_node)
        builder.add_node("✍️ qwen3_writer", qwen3_writer_node)
        builder.add_node("🇰🇭 khmer_processing", khmer_processing_node)
        builder.add_node("📋 content_synthesis", content_synthesis_node)
        
        builder.add_edge(START, "📥 input_analysis")
        builder.add_edge("📥 input_analysis", "🤖 llama3_orchestration")
        builder.add_conditional_edges(
            "🤖 llama3_orchestration",
            content_type_router,
            {
                "gemma3_vision"     : "👁️ gemma3_vision",
                "reflection"        : "🔍 qwen3_reflection"
            }
        )
        builder.add_conditional_edges(
            "👁️ gemma3_vision",
            needs_khmer_processing,
            {
                "khmer_processing"  : "🇰🇭 khmer_processing",
                "reflection"        : "🔍 qwen3_reflection"
            }
        )
        builder.add_edge("🇰🇭 khmer_processing", "🔍 qwen3_reflection")
        builder.add_conditional_edges(
            "🔍 qwen3_reflection",
            needs_writing,
            {
                "qwen3_writer"      : "✍️ qwen3_writer",
                "content_synthesis" : "📋 content_synthesis"
            }
        )
        builder.add_edge("✍️ qwen3_writer", "📋 content_synthesis")
        builder.add_edge("📋 content_synthesis", END)
        
        checkpointer     = MemorySaver()
        compiled_graph  = builder.compile(checkpointer=checkpointer)
        
        try:
            import os
            graph_image = compiled_graph.get_graph().draw_mermaid_png()
            save_path   = os.path.abspath("scrape_content_agent_workflow.png")
            
            with open(save_path, "wb") as f:
                f.write(graph_image)
            
            print(f"🎨 Content workflow diagram auto-generated: {save_path}")
        except Exception as e:
            print(f"⚠️ Could not auto-generate workflow image: {e}")
        
        return compiled_graph
    
    def _reflect_on_results(self, user_query: str, extraction_results: str) -> str:
        """Use Qwen3 reflector to evaluate extraction quality and provide feedback"""
        
        reflection_prompt = f"""<thinking>
Let me analyze the quality of this content extraction result and provide constructive feedback.

Original Query: {user_query}
Extraction Results: {extraction_results}

I should evaluate:
1. Did the extraction address the user's request completely?
2. Is the extracted content relevant and useful?
3. Are there any errors or missing elements?
4. What improvements could be made?
5. Is the format clear and well-organized?

Let me provide specific, actionable feedback.
</thinking>

Please evaluate this content extraction result:

User requested: "{user_query}"

Extraction results:
{extraction_results}

Provide detailed feedback on:
1. Quality and completeness of extraction
2. Relevance to the user's request
3. Any errors or issues identified
4. Suggestions for improvement
5. Overall assessment

Be specific and constructive in your evaluation."""
        
        try:
            print(f"\n🔍 Qwen3 Reflector - Evaluating extraction quality...")
            print(f"🧠 Qwen3 Reflector: ", end="", flush=True)
            
            reflection_response = ""
            
            for chunk in self.qwen3_reflector.stream([HumanMessage(content=reflection_prompt)]):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)
                    reflection_response += chunk.content
            
            print()
            return reflection_response
            
        except Exception as e:
            return f"Reflection error: {str(e)}"
    
    def process_content(self, user_query: str) -> str:
        """Process content extraction request using Gemma3 and specialized tools"""
        
        print(f"\n📋 Gemma3 Content Agent - Processing Request")
        print(f"🔍 Query: {user_query}")
        
        system_prompt = """You are an advanced content extraction and writing orchestrator powered by multiple specialized models:

🤖 LLAMA3.1: Tool orchestration and coordination (YOU are this model)
👁️ GEMMA3: Vision analysis for images (vision-only, no tools)
🧠 QWEN3: Writing and reflection capabilities

You specialize in:
🌐 Web scraping and content extraction
📝 Text processing and summarization
🔗 Link and metadata extraction  
🇰🇭 Khmer language detection and processing
✍️ Comprehensive content writing coordination

ARCHITECTURE:
- YOU (Llama3.1) orchestrate all tools and coordinate the workflow
- Gemma3 handles image analysis separately (through direct API calls)
- Qwen3 handles writing and reflection tasks

WORKFLOW:
1. Analyze user request and determine required tools
2. Execute content extraction tools as needed
3. Coordinate with specialized models when required
4. Provide comprehensive results

When users want written content, use the write_comprehensive_content tool with appropriate content_type and style parameters.

Always be thorough and use multiple tools when beneficial. You have full access to all extraction and processing tools."""
        
        try:
            if not self.conversation_history:
                self.conversation_history.append(SystemMessage(content=system_prompt))
            
            user_msg = HumanMessage(content=user_query)
            self.conversation_history.append(user_msg)
            
            print(f"\n🤖 Llama3.1 Orchestrator: ", end="", flush=True)
            
            full_response   = ""
            tool_calls      = []
            
            for chunk in self.llama3_orchestrator.stream(self.conversation_history):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)
                    full_response += chunk.content
                
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)
            
            print()
            
            if tool_calls:
                print(f"\n🔧 Executing content extraction tools:")
                
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call.get('args', {})
                    
                    print(f"   🔨 {tool_name}({tool_args})")
                    
                    for tool in self.tools:
                        if tool.name == tool_name:
                            try:
                                if tool_args:
                                    if len(tool_args) == 1:
                                        result = tool.func(list(tool_args.values())[0])
                                    else:
                                        result = tool.func(**tool_args)
                                else:
                                    result = tool.func()
                                
                                tool_results.append(f"🔧 {tool_name}: {result}")
                                print(f"   ✅ Completed")
                                break
                            except Exception as e:
                                error_result = f"❌ Error in {tool_name}: {str(e)}"
                                tool_results.append(error_result)
                                print(f"   {error_result}")
                                break
                
                if tool_results:
                    extraction_output = f"{full_response}\n\n" + "\n\n".join(tool_results)
                    
                    reflection_feedback = self._reflect_on_results(user_query, extraction_output)
                    
                    final_output = f"{extraction_output}\n\n🔍 Qwen3 Quality Reflection:\n{reflection_feedback}"
                else:
                    final_output = full_response
            else:
                final_output = full_response
            
            ai_response = AIMessage(content=final_output)
            if tool_calls:
                ai_response.tool_calls = tool_calls
            self.conversation_history.append(ai_response)
            
            return final_output
            
        except Exception as e:
            error_msg = f"❌ Content processing error: {str(e)}"
            print(f"\n{error_msg}")
            return error_msg
    
    def save_workflow_image(self, filename="scraper_content_agent.png"):
        """Save the content extraction workflow diagram"""
        try:
            import os
            
            graph_image = self.workflow_graph.get_graph().draw_mermaid_png()
            save_path = os.path.abspath(filename)
            
            with open(save_path, "wb") as f:
                f.write(graph_image)
            
            print(f"✅ Content workflow diagram saved as: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ Error generating workflow image: {e}")
            return None


def main():
    print("🚀 Multi-Model Content Extraction Agent")
    print("=" * 60)
    
    try:
        agent = ScrapeContentAgent()
        print("✅ Multi-Model Agent initialized!")
        print("🤖 Llama3.1: Tool orchestration and coordination")
        print("👁️ Gemma3: Vision analysis for images")
        print("🧠 Qwen3: Writing and reflection capabilities")
        print("🌐 Web scraping and content extraction ready")
        print("🇰🇭 Khmer text processing enabled")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Make sure you have: ollama pull llama3.1")
        print("Make sure you have: ollama pull gemma3:27b") 
        print("Make sure you have: ollama pull qwen3:8b")
        return
    
    print("\n🔧 Available Capabilities:")
    print("• Website content scraping")
    print("• Image analysis with Gemma3 vision")
    print("• OCR text extraction (Khmer + English)")
    print("• Link and image extraction")
    print("• Content summarization")
    print("• Khmer text detection and processing")
    print("• Quality evaluation and reflection")
    print("• Comprehensive content writing (Qwen3)")
    
    print("\n✍️ Writing Formats Available:")
    print("• News articles (journalistic style)")
    print("• Professional reports (business style)")
    print("• Academic papers (scholarly style)")
    print("• Blog posts (engaging style)")
    print("• Creative content (vivid style)")
    print("• Executive summaries (concise style)")
    
    print("\n💡 Example Queries:")
    examples = [
        "Scrape https://example.com and write a news article about it",
        "Extract content from this site and create a professional report",
        "Analyze this image and write a detailed blog post about it",
        "Get content from https://tech-site.com and write an academic analysis",
        "Scrape news from multiple sources and create a comprehensive summary",
        "Extract text from this image and write a creative story based on it",
        "Process Khmer content and write a bilingual report"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example}")
    
    print(f"\nCommands: 'quit' to exit, 'workflow' to regenerate diagram")
    print(f"{'-' * 60}")
    
    while True:
        try:
            user_input = input("\n🔍 Enter your content extraction request: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'workflow':
                agent.save_workflow_image()
                continue
                
            if not user_input:
                print("Please enter a request.")
                continue
            
            result = agent.process_content(user_input)
            
            print(f"\n📋 Final Result:")
            print(f"{result}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()