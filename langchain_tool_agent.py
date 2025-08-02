import random
import math
import os

from datetime import datetime
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get weather information for a specific city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        Weather information including temperature and conditions
    """
    weather_conditions  = ["sunny", "cloudy", "rainy", "snowy", "windy", "foggy"]
    temp                = random.randint(-10, 35)
    condition           = random.choice(weather_conditions)
    humidity            = random.randint(30, 90)
    return f"Weather in {city}: {temp}°C, {condition}, humidity: {humidity}%"


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2 * 3")
        
    Returns:
        The result of the calculation or error message
    """
    try:
        allowed_names = {
            "abs"       : abs,
            "round"     : round,
            "min"       : min,
            "max"       : max,
            "sum"       : sum,
            "pow"       : pow,
            "sqrt"      : math.sqrt,
            "sin"       : math.sin,
            "cos"       : math.cos,
            "tan"       : math.tan,
            "pi"        : math.pi,
            "e"         : math.e,
            "log"       : math.log,
            "ceil"      : math.ceil,
            "floor"     : math.floor
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_random_joke() -> str:
    """Tell a random programming or general joke.
    
    Returns:
        A random joke string
    """
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "Why did the programmer quit his job? He didn't get arrays!",
        "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
        "Why do Java developers wear glasses? Because they can't C#!",
        "What's the object-oriented way to become wealthy? Inheritance!",
        "Why did the scarecrow win an award? He was outstanding in his field!",
        "What do you call a fake noodle? An impasta!",
        "Why don't eggs tell jokes? They'd crack each other up!"
    ]
    return random.choice(jokes)


@tool
def get_current_time() -> str:
    """Get the current date and time.
    
    Returns:
        Current timestamp formatted as string
    """
    now = datetime.now()
    return f"Current time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')}"


@tool
def generate_password(length: int = 12) -> str:
    """Generate a secure random password.
    
    Args:
        length: Length of password to generate (default: 12, max: 50)
        
    Returns:
        A randomly generated secure password
    """
    length      = min(max(length, 4), 50)
    chars       = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    password    = ''.join(random.choice(chars) for _ in range(length))
    return f"Generated {length}-character password: {password}"


@tool 
def get_fun_fact() -> str:
    """Get a random fun fact about science, technology, or nature.
    
    Returns:
        An interesting fun fact
    """
    facts = [
        "Honey never spoils - archaeologists have found pots of honey in ancient Egyptian tombs that are over 3000 years old and still perfectly edible!",
        "A group of flamingos is called a 'flamboyance'.",
        "Bananas are berries, but strawberries aren't!",
        "The human brain contains approximately 86 billion neurons.",
        "Octopuses have three hearts and blue blood.",
        "Lightning strikes the Earth about 100 times per second.",
        "A day on Venus is longer than its year.",
        "Sharks have been around longer than trees!"
    ]
    return random.choice(facts)


@tool
def evaluate_response_quality(response: str, topic: str) -> str:
    """Evaluate response quality for serious topics using Qwen3 evaluator.
    
    Args:
        response: The response to evaluate
        topic: The topic being discussed
        
    Returns:
        Quality evaluation with score and feedback
    """
    serious_topics = ["medical", "legal", "financial", "safety", "mental health", "emergency", "health", "investment", "legal advice"]
    is_serious = any(t in topic.lower() for t in serious_topics)
    
    if is_serious:
        return f"⚠️ SERIOUS TOPIC: {topic}. Response requires Qwen3 evaluation for accuracy and safety."
    else:
        return f"✅ General topic: {topic}. Standard evaluation applies."


@tool
def flag_potentially_harmful(content: str) -> str:
    """Flag potentially harmful or inappropriate content.
    
    Args:
        content: Content to analyze
        
    Returns:
        Warning flags or approval
    """
    warning_terms = ["medical advice", "legal advice", "financial advice", "dangerous", "self-harm", "illegal"]
    flags = [term for term in warning_terms if term in content.lower()]
    
    if flags:
        return f"🚨 WARNING: Detected concerning content: {', '.join(flags)}"
    else:
        return "✅ Content appears safe for general use"


class ModernToolAgent:
    def __init__(self
        , model_name    : str = "llama3.1:latest"
        , base_url      : str = None
        , enable_qwen3_evaluator : bool = True
    ):
        self.base_url = base_url or os.getenv('LLM_API_BASE_URL', 'http://localhost:11434')
        self.tools = [
            get_weather,
            calculate,
            get_random_joke,
            get_current_time,
            generate_password,
            get_fun_fact,
            evaluate_response_quality,
            flag_potentially_harmful
        ]
        

        self.enable_evaluator = enable_qwen3_evaluator
        if self.enable_evaluator:
            self.qwen3_evaluator = ChatOllama(
                model       = "qwen3:8b",
                base_url    = self.base_url,
                temperature = 0.3,
                streaming   = True
            ).bind_tools([evaluate_response_quality, flag_potentially_harmful])
        
        self.llm = ChatOllama(
            model       = model_name,
            base_url    = self.base_url,
            temperature = 0.7,
            streaming   = True
        ).bind_tools(self.tools)
        
        self.conversation_history = []
        self.workflow_graph = self._create_workflow_graph()
    
    def chat(self, user_input: str) -> str:
        try:
            print(f"\n👤 You: {user_input}")
            
            if not self.conversation_history:
                system_msg = SystemMessage(content="""You are a helpful and conversational AI assistant with access to various tools. 

                IMPORTANT: Always provide a natural, friendly response along with using tools. Don't just call tools silently.

                When a user asks for something:
                1. Acknowledge their request conversationally 
                2. Use the appropriate tools to get information
                3. Present the results in a natural, engaging way

                For example:
                - If asked "What's 2+2?", respond like: "Let me calculate that for you!" then use the calculate tool
                - If asked for weather, say something like: "I'll check the weather for you!" then use get_weather
                - Always be enthusiastic and personable in your responses

                Available tools:
                - get_weather: Get weather for any city
                - calculate: Perform math calculations  
                - get_random_joke: Tell jokes
                - get_current_time: Get current date/time
                - generate_password: Create secure passwords
                - get_fun_fact: Share interesting facts""")
                self.conversation_history.append(system_msg)
            
            user_msg = HumanMessage(content=user_input)
            self.conversation_history.append(user_msg)
            
            print(f"\n🤖 Assistant: ", end="", flush=True)
            
            full_response = ""
            tool_calls = []
            
            for chunk in self.llm.stream(self.conversation_history):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)
                    full_response += chunk.content

                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)
            
            print() 
            
            if tool_calls:
                print(f"\n🔧 Tool calls detected: {len(tool_calls)}")
                
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call.get('args', {})
                    
                    print(f"   🔨 Executing: {tool_name}({tool_args})")
   
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
                                print(f"   ✅ Result: {result}")
                                break
                            except Exception as e:
                                error_result = f"Error executing {tool_name}: {str(e)}"
                                tool_results.append(f"❌ {error_result}")
                                print(f"   ❌ {error_result}")
                                break
                if tool_results:
                    print(f"\n📋 Tool Results:")
                    for result in tool_results:
                        print(f"   {result}")
                    follow_up_prompt = f"""Based on the tool results below, provide a natural, conversational response to the user's question: "{user_input}"

Tool Results:
{chr(10).join(tool_results)}

Respond naturally and conversationally, incorporating these results into your answer."""
                    
                    print(f"\n💭 Generating natural response...")
                    print(f"🤖 Assistant: ", end="", flush=True)
                    
                    simple_llm = ChatOllama(
                        model       = self.llm.model,
                        base_url    = self.base_url,
                        temperature = 0.8
                    )
                    
                    natural_response = ""
                    for chunk in simple_llm.stream([HumanMessage(content=follow_up_prompt)]):
                        if hasattr(chunk, 'content') and chunk.content:
                            print(chunk.content, end="", flush=True)
                            natural_response += chunk.content
                    
                    print()  # New line
                    output = natural_response
                    
                    if self.enable_evaluator and self._needs_evaluation(user_input, output):
                        print(f"\n🧠 Qwen3 Evaluator - Checking response quality...")
                        evaluation = self._evaluate_with_qwen3(user_input, output)
                        if "WARNING" in evaluation or "SERIOUS TOPIC" in evaluation:
                            print(f"⚠️ Evaluation: {evaluation}")
                
                else:
                    output = full_response
            else:
                output = full_response

            from langchain_core.messages import AIMessage
            ai_response = AIMessage(content=output)
            if tool_calls:
                ai_response.tool_calls = tool_calls
            self.conversation_history.append(ai_response)
            
            return output
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"\n❌ Error: {error_msg}")
            return error_msg
    
    def reset_conversation(self):
        self.conversation_history = []
        print("🔄 Conversation history cleared.")
    
    def _needs_evaluation(self, user_input: str, response: str) -> bool:
        """Check if response needs Qwen3 evaluation for serious topics"""
        serious_keywords = [
            "medical", "health", "pain", "symptoms", "doctor", "medicine",
            "legal", "law", "lawsuit", "rights", "attorney",
            "financial", "investment", "money", "stocks", "crypto",
            "safety", "dangerous", "emergency", "help", "urgent"
        ]
        
        combined_text = (user_input + " " + response).lower()
        return any(keyword in combined_text for keyword in serious_keywords)
    
    def _evaluate_with_qwen3(self, user_query: str, response: str) -> str:
        """Evaluate response using Qwen3:8b with thinking"""
        if not self.enable_evaluator:
            return "Evaluation disabled"
        
        evaluation_prompt = f"""<thinking>
Let me carefully evaluate this response for potential issues or serious topic concerns.

User Query: {user_query}
Response: {response}

I should check:
1. Is this a serious topic that needs extra caution?
2. Does the response contain potentially harmful advice?
3. Should the user be advised to consult professionals?

Let me use the evaluation tools to analyze this systematically.
</thinking>

Please evaluate this response using the available tools:

User asked: "{user_query}"
Response given: "{response}"

Use evaluate_response_quality and flag_potentially_harmful tools to assess this interaction."""
        
        try:
            print(f"🤖 Qwen3: ", end="", flush=True)
            
            full_evaluation = ""
            tool_calls = []
            
            # Stream Qwen3's thinking and evaluation
            for chunk in self.qwen3_evaluator.stream([HumanMessage(content=evaluation_prompt)]):
                if hasattr(chunk, 'content') and chunk.content:
                    print(chunk.content, end="", flush=True)
                    full_evaluation += chunk.content
                
                if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                    tool_calls.extend(chunk.tool_calls)
            
            print()  # New line
            
            # Execute evaluation tools
            if tool_calls:
                evaluation_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call.get('args', {})
                    
                    if tool_name == "evaluate_response_quality":
                        result = evaluate_response_quality.func(response, user_query)
                        evaluation_results.append(result)
                    elif tool_name == "flag_potentially_harmful":
                        result = flag_potentially_harmful.func(response)
                        evaluation_results.append(result)
                
                return f"{full_evaluation}\n\nEvaluation Results: {'; '.join(evaluation_results)}"
            
            return full_evaluation
            
        except Exception as e:
            return f"Evaluation error: {str(e)}"
    
    def _create_workflow_graph(self):
        class WorkflowState(TypedDict):
            user_input      : str
            llm_response    : str
            tool_calls      : list
            final_response  : str
            needs_evaluation: bool
            qwen3_evaluation: str
        
        def user_input_node(state: WorkflowState):
            return {"user_input": "User provides input query"}
        
        def llm_processing_node(state: WorkflowState):
            return {"llm_response": "Llama3.1 processes input and identifies tools"}
        
        def tool_execution_node(state: WorkflowState):
            return {"tool_calls": ["🔧 Tools: weather, calculate, joke, time, password, facts"]}
        
        def response_generation_node(state: WorkflowState):
            return {"final_response": "Natural language response generated"}
        
        def qwen3_evaluation_node(state: WorkflowState):
            return {"qwen3_evaluation": "Qwen3:8b evaluates response for serious topics"}
        
        def should_use_tools(state: WorkflowState):
            return "tool_execution" if state.get("tool_calls") else "response_generation"
        
        def should_evaluate_with_qwen3(state: WorkflowState):
            return "qwen3_evaluation" if state.get("needs_evaluation", True) else "final_output"
        
        builder = StateGraph(WorkflowState)
        
        builder.add_node("📝 user_input", user_input_node)
        builder.add_node("🧠 llama3_processing", llm_processing_node)
        builder.add_node("🔧 tool_execution", tool_execution_node)
        builder.add_node("💬 response_generation", response_generation_node)
        builder.add_node("🔍 qwen3_evaluation", qwen3_evaluation_node)
        builder.add_node("✅ final_output", lambda s: {"final_response": "Complete"})
        
        builder.add_edge(START, "📝 user_input")
        builder.add_edge("📝 user_input", "🧠 llama3_processing")
        builder.add_conditional_edges(
            "🧠 llama3_processing", 
            should_use_tools,
            {
                "tool_execution": "🔧 tool_execution",
                "response_generation": "💬 response_generation"
            }
        )
        builder.add_edge("🔧 tool_execution", "💬 response_generation")
        builder.add_conditional_edges(
            "💬 response_generation",
            should_evaluate_with_qwen3,
            {
                "qwen3_evaluation": "🔍 qwen3_evaluation",
                "final_output": "✅ final_output"
            }
        )
        builder.add_edge("🔍 qwen3_evaluation", "✅ final_output")
        builder.add_edge("✅ final_output", END)

        checkpointer    = MemorySaver()
        compiled_graph  = builder.compile(checkpointer=checkpointer)
        
        try:
            import os
            graph_image = compiled_graph.get_graph().draw_mermaid_png()
            save_path   = os.path.abspath("tool_agent_workflow.png")
            
            with open(save_path, "wb") as f:
                f.write(graph_image)
            
            print(f"🎨 Workflow diagram auto-generated: {save_path}")
        except Exception as e:
            print(f"⚠️ Could not auto-generate workflow image: {e}")
        
        return compiled_graph


def main():
    print("🚀 Modern LangChain Tool Agent with Llama3.1 + Qwen3:8b Evaluator")
    print("=" * 65)
    
    try:
        agent = ModernToolAgent()
        print("✅ Agent initialized successfully!")
        print("🧠 Qwen3:8b evaluator enabled for serious topics")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("Make sure you have: pip install langchain-ollama")
        print("Make sure you have qwen3:8b model: ollama pull qwen3:8b")
        return
    
    print("\n🔧 Available Tools:")
    print("• Weather information")
    print("• Mathematical calculations") 
    print("• Random jokes")
    print("• Current time")
    print("• Password generation")
    print("• Fun facts")
    print("• Response quality evaluation (Qwen3)")
    print("• Harmful content detection (Qwen3)")
    
    print("\n💡 Try these examples:")
    examples = [
        "What's the weather in Tokyo?",
        "Calculate 25 * 4 + sqrt(16)",
        "Tell me a joke",
        "What time is it?",
        "Generate a 15 character password",
        "Tell me a fun fact",
        "What should I do if I have chest pain?",  # Will trigger Qwen3 evaluation
        "How should I invest my savings?"  # Will trigger Qwen3 evaluation
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example}")
    
    print(f"\nCommands: 'quit' to exit, 'reset' to clear history, 'graph' to save workflow diagram")
    print(f"{'-' * 55}")
    
    while True:
        try:
            user_input = input("\n💬 Enter your message: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                continue
            
            if not user_input:
                print("Please enter a message.")
                continue
            
            agent.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
