from vector_store import VectorStore
import ollama
import sys
from typing import List, Dict


class ProgrammerRAG:
    def __init__(self, vector_store_path: str = "vector_store.pkl"):
        """Initialize Programmer RAG system with conversation memory"""
        print("Initializing Programmer AI with conversation memory...")

        # Load vector store
        self.vector_store = VectorStore()
        if not self.vector_store.load(vector_store_path):
            print(f"❌ Could not load vector store from {vector_store_path}")
            print("Please run build_embeddings.py first!")
            sys.exit(1)

        # Conversation memory
        self.conversation_history: List[Dict] = []
        self.max_history = 10  # Keep last 10 exchanges

        print("✅ Programmer AI ready! Type 'quit' to exit.")
        print("-" * 60)

    def get_smart_context(self, query: str, max_chunks: int = 8) -> str:
        """Get relevant context using smarter retrieval"""
        # First try with original query
        results = self.vector_store.search(query, k=max_chunks)

        # If no good results, try with expanded query terms
        if not results or all(r['score'] > 0.8 for r in results):
            # Extract key terms and search again
            simple_terms = query.lower().split()[:5]
            for term in simple_terms:
                if len(term) > 3:  # Avoid very short words
                    more_results = self.vector_store.search(term, k=3)
                    if more_results:
                        results.extend(more_results)

        if not results:
            return None

        # Sort by score and remove duplicates
        unique_contents = set()
        context_parts = []

        for result in sorted(results, key=lambda x: x['score'])[:max_chunks]:
            content = result['document'].page_content.strip()
            if content and content not in unique_contents:
                unique_contents.add(content)
                # Truncate very long chunks
                if len(content) > 800:
                    content = content[:800] + "... [truncated]"
                context_parts.append(content)

        return "\n\n---\n\n".join(context_parts) if context_parts else None

    def update_conversation_history(self, question: str, answer: str):
        """Update conversation history"""
        self.conversation_history.append({
            "role": "user",
            "content": question
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        # Keep only recent history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def get_conversation_context(self) -> str:
        """Get formatted conversation history"""
        if not self.conversation_history:
            return "No previous conversation."

        formatted = []
        for i in range(0, len(self.conversation_history), 2):
            if i + 1 < len(self.conversation_history):
                user_msg = self.conversation_history[i]['content'][:150]
                assistant_msg = self.conversation_history[i + 1]['content'][:200]
                formatted.append(f"User: {user_msg}...")
                formatted.append(f"Assistant: {assistant_msg}...")

        return "\n".join(formatted[-4:])  # Last 2 exchanges

    def generate_programmer_response(self, question: str, context: str = None) -> str:
        """Generate response with programmer persona and conversation memory"""

        # Get conversation context
        conv_context = self.get_conversation_context()

        if context:
            prompt = f"""You are SeniorDev, an expert programmer AI assistant with deep knowledge in software development.

PREVIOUS CONVERSATION (for context):
{conv_context}

RELEVANT DOCUMENTATION CONTEXT:
{context}

CURRENT QUESTION: {question}

YOUR PROGRAMMER PERSONA:
- Think step-by-step like a senior developer
- Be practical, concise, and technical
- Remember our conversation context
- Build on previous discussion naturally
- If context exists, use it as your primary knowledge source
- If context is incomplete, acknowledge limitations but still provide helpful insights
- Never say you cannot answer - instead provide the most relevant information available
- Use code-like thinking: analyze, debug, optimize

THINKING PROCESS:
1. Review conversation history for context
2. Analyze the current question
3. Check documentation context for relevant information
4. Formulate a helpful, programmer-style response that continues the conversation
5. Include practical insights or next steps

RESPONSE (as SeniorDev):"""
        else:
            prompt = f"""You are SeniorDev, an expert programmer AI assistant.

PREVIOUS CONVERSATION (for context):
{conv_context}

CURRENT QUESTION: {question}

NOTE: No specific documentation context was found for this query.

YOUR APPROACH:
- As a programmer, you understand that sometimes docs don't have everything
- Consider our conversation history for context
- Provide your expert analysis based on programming principles
- Be honest about context limitations but still be helpful
- Suggest how the user might find the information elsewhere
- Offer related insights that could be useful
- Continue the conversation naturally

RESPONSE (as SeniorDev):"""

        try:
            # Create messages with conversation history
            messages = []

            # Add system message
            messages.append({
                'role': 'system',
                'content': 'You are SeniorDev, an expert programmer AI assistant. Keep responses technical, practical, and helpful.'
            })

            # Add conversation history
            for msg in self.conversation_history[-8:]:  # Last 4 exchanges
                messages.append(msg)

            # Add current prompt
            messages.append({
                'role': 'user',
                'content': prompt,
            })

            response = ollama.chat(
                model='llama3.2',
                messages=messages,
                options={
                    'temperature': 0.3,
                    'num_predict': 512
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"// System Error: {str(e)}\n// But as a programmer, I'd suggest we continue with your original question..."

    def ask_question(self, question: str) -> str:
        """Main Q&A method with conversation memory"""
        print(f"\n🧠 Processing: '{question}'")

        # Get context (try harder to find something)
        context = self.get_smart_context(question, max_chunks=10)

        if context:
            print(f"📚 Found relevant context ({len(context.split('---'))} sections)")
        else:
            print("⚠️  No specific context found - using programmer knowledge")

        # Generate response
        answer = self.generate_programmer_response(question, context)

        # Update conversation history
        self.update_conversation_history(question, answer)

        return answer

    def interactive_session(self):
        """Run interactive programming assistant session with conversation"""
        print("\n" + "=" * 60)
        print("👨‍💻 SENIORDEV - PROGRAMMER AI ASSISTANT")
        print("=" * 60)
        print("\nI'll remember our conversation and answer using your docs.")
        print("I'll always think like a programmer and give you useful responses!")
        print("\nCommands:")
        print("  'quit', 'exit', 'q' - End conversation")
        print("  'clear' - Clear conversation memory")
        print("  'history' - Show conversation history")
        print("-" * 60)

        while True:
            try:
                question = input("\n🔹 Your question: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Happy coding! Remember: The best code is tested code.")
                    break

                if question.lower() == 'clear':
                    self.conversation_history = []
                    print("🗑️  Conversation memory cleared!")
                    continue

                if question.lower() == 'history':
                    print("\n📜 CONVERSATION HISTORY:")
                    print("-" * 40)
                    for i, msg in enumerate(self.conversation_history):
                        role = "User" if msg['role'] == 'user' else "SeniorDev"
                        print(f"{role}: {msg['content'][:100]}...")
                    print("-" * 40)
                    continue

                if not question:
                    print("// Empty query. Try: 'Tell me about...' or 'Explain...'")
                    continue

                # Get and display answer
                answer = self.ask_question(question)

                print("\n" + "=" * 60)
                print("💻 SENIORDEV RESPONSE:")
                print("=" * 60)
                print(answer)
                print("=" * 60)

            except KeyboardInterrupt:
                print("\n\n⚠️  Use 'quit' to exit properly.")
            except Exception as e:
                print(f"\n❌ System error: {e}")
                print("// Debugging 101: Let's try a different approach...")


def main():
    # Create and run Programmer RAG system
    rag = ProgrammerRAG()
    rag.interactive_session()


if __name__ == "__main__":
    main()