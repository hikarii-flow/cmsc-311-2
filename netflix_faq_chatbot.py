import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from pydantic import SecretStr
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

warnings.filterwarnings("ignore")

# Netflix Service FAQ Knowledge Base - sourced from Netflix Help Center
NETFLIX_SERVICE_FAQ = """
# Netflix Plans and Pricing FAQ

Q: What are the Netflix subscription plans?
A: Netflix offers several subscription plans. The Standard with Ads plan allows streaming on two devices with occasional commercial breaks. The Standard plan allows two simultaneous streams without ads. The Premium plan supports up to four devices streaming at once and includes 4K Ultra HD.

Q: How much does Netflix cost?
A: Standard with ads is $7.99 per month. Standard is $17.99 per month (Add 1 extra member for $6.99 each month with ads or $8.99 each month without ads). Premium is $24.99 per month (Add up to 2 extra members for $6.99 each month with ads or $8.99 each month without ads). You are charged monthly on the date you signed up. The Basic plan has been discontinued. You can change your plan or cancel at any time. Depending on your location, taxes may be added to your subscription price.

Q: Can I cancel my Netflix subscription anytime?
A: Yes, you can cancel your Netflix membership at any time. To cancel, go to your Account Settings and select "Cancel Membership" under the Membership & Billing section. You'll continue to have access until the end of your current billing period.

Q: How do I change my Netflix plan?
A: To change your plan, visit Netflix.com/YourAccount and select Change Plan from the Plan Details section. Changes take effect on your next billing date.

# Netflix Account and Billing FAQ

Q: How can I pay for Netflix?
A: Netflix accepts various payment methods including credit cards, debit cards, PayPal (where available), and Netflix gift cards. You can redeem multiple gift cards on your account. Some partners allow you to add Netflix to their bill.

Q: How do I update my payment method?
A: You can update your payment method by going to your Account page and selecting the payment details section. If Netflix has problems with your payment, you'll be notified to update your billing information.

Q: When is my Netflix billing date?
A: Netflix members are automatically charged monthly on the date they signed up. You can see your plan and price at Netflix.com/YourAccount by selecting Billing details from the Billing & Membership section.

Q: Can I change my billing date?
A: Yes, you can change your billing date. Go to your Account page, select Change billing day and follow the instructions. You may be charged a prorated amount to cover the days between your current and new billing dates.

# Netflix Streaming and Technical FAQ

Q: Why is Netflix buffering or loading slowly?
A: Buffering usually indicates a slow or unstable internet connection. Netflix recommends at least 5 Mbps for HD streaming and 15 Mbps for 4K. Try restarting your router, closing background apps, or switching to a wired connection for better stability.

Q: Can I download Netflix shows for offline viewing?
A: Yes, many Netflix titles are available for offline viewing. You can download shows or movies using the Netflix app on iOS, Android, or Windows devices. Downloads typically stay available for 30 days or 48 hours after you start watching.

Q: Does Netflix support 4K streaming?
A: Yes, many Netflix titles are available in 4K Ultra HD, HDR10, HDR10+, and Dolby Vision. To stream in 4K, you need a compatible device, a Premium Netflix plan, and an internet connection of at least 15 Mbps.

Q: How do I change audio or subtitle language?
A: While watching a title, open the Audio & Subtitles menu and select your preferred language and subtitle options. Netflix supports multiple audio languages and subtitles for most titles.

# Netflix Account Access FAQ

Q: I forgot my Netflix password. How do I reset it?
A: If you forgot your password, click "Forgot Password" on the login screen and follow the reset instructions. Netflix will send a password reset link to your registered email address.

Q: Can I share my Netflix account?
A: A Netflix account is designed for people who live together in a single household. Your plan determines how many extra member slots you can add. Extra members have their own account and password but their membership is paid by the primary account holder.

Q: Does Netflix have parental controls?
A: Yes, Netflix offers detailed parental controls. You can create Kids profiles, filter content by maturity rating, and set a PIN lock for specific titles.

Q: How many people can watch Netflix at the same time?
A: The number of simultaneous streams depends on your plan. Standard with Ads and Standard plans allow up to two devices. Premium plan allows up to four devices streaming at once.

# Netflix Availability FAQ

Q: Where is Netflix available?
A: Netflix is available in over 190 countries worldwide including the Americas, Europe, Asia-Pacific (India, Japan, South Korea, Australia, etc.), Middle East, and Africa. Netflix is currently not available in China, North Korea, Crimea, Russia, and Syria.

Q: How do I request a TV show or movie?
A: You can request TV shows or movies through the Netflix Help Center. Visit the title request page to submit your suggestions.
"""


class NetflixKnowledgeBase:
    """
    Manages the Netflix knowledge base including FAQ and titles data.
    Uses vector embeddings for semantic search capabilities.

    This class implements knowledge representation methods using vector embeddings
    to store and retrieve information efficiently, enabling the chatbot to find
    relevant answers based on semantic similarity rather than exact keyword matching.
    """

    def __init__(self, titles_csv_path: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the knowledge base with Netflix titles and FAQ data.

        Args:
            titles_csv_path: Path to the Netflix titles CSV file
            persist_directory: Directory for ChromaDB persistence
        """
        self.titles_csv_path = titles_csv_path
        self.persist_directory = persist_directory

        # Initialize HuggingFace embeddings for vector representation
        # Using sentence-transformers for high-quality semantic embeddings
        self.embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
        )

        self.titles_df: Optional[pd.DataFrame] = None
        self.vectorstore: Optional[Chroma] = None
        self.documents: List[Document] = []

    def load_titles_data(self) -> pd.DataFrame:
        """
        Load and preprocess the Netflix titles dataset.

        Preprocessing steps include handling missing values, data cleaning,
        and formatting text for optimal embedding generation.

        Returns:
            Preprocessed pandas DataFrame containing Netflix titles
        """
        print("Loading Netflix titles dataset...")

        # Load the CSV file
        self.titles_df = pd.read_csv(self.titles_csv_path)

        # Preprocessing: Handle missing values
        # Fill missing text fields with empty strings to avoid NaN issues
        text_columns = [
            "director",
            "cast",
            "country",
            "date_added",
            "rating",
            "duration",
            "listed_in",
            "description",
        ]
        for col in text_columns:
            self.titles_df[col] = self.titles_df[col].fillna("")

        # Clean whitespace from string columns
        for col in self.titles_df.select_dtypes(include=["object"]).columns:
            self.titles_df[col] = self.titles_df[col].astype(str).str.strip()

        print(f"Loaded {len(self.titles_df)} Netflix titles")
        return self.titles_df

    def _create_title_document(self, row: pd.Series) -> Document:
        """
        Create a LangChain Document from a Netflix title row.

        This method structures the content for optimal retrieval, including
        all relevant metadata that can be used for citation purposes.

        Args:
            row: A pandas Series representing a single Netflix title

        Returns:
            A LangChain Document with formatted content and metadata
        """
        # Create a rich text representation of the title
        content_parts = [
            f"Title: {row['title']}",
            f"Type: {row['type']}",
            f"Release Year: {row['release_year']}",
        ]

        if row["director"]:
            content_parts.append(f"Director: {row['director']}")
        if row["cast"]:
            content_parts.append(f"Cast: {row['cast']}")
        if row["country"]:
            content_parts.append(f"Country: {row['country']}")
        if row["rating"]:
            content_parts.append(f"Rating: {row['rating']}")
        if row["duration"]:
            content_parts.append(f"Duration: {row['duration']}")
        if row["listed_in"]:
            content_parts.append(f"Genres: {row['listed_in']}")
        if row["description"]:
            content_parts.append(f"Description: {row['description']}")

        content = "\n".join(content_parts)

        # Metadata for source citation
        metadata = {
            "source": "Netflix Titles Dataset (Kaggle)",
            "show_id": row["show_id"],
            "title": row["title"],
            "type": row["type"],
            "release_year": str(row["release_year"]),
            "content_type": "title",
        }

        return Document(page_content=content, metadata=metadata)

    def _create_faq_documents(self) -> List[Document]:
        """
        Parse the FAQ knowledge base and create documents.

        Returns:
            List of Document objects containing FAQ entries
        """
        documents: List[Document] = []

        # Split FAQ into individual Q&A pairs
        lines = NETFLIX_SERVICE_FAQ.strip().split("\n")
        current_section = ""
        current_qa: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# "):
                # New section header
                if current_qa:
                    qa_text = "\n".join(current_qa)
                    documents.append(
                        Document(
                            page_content=qa_text,
                            metadata={
                                "source": "Netflix Help Center FAQ",
                                "section": current_section,
                                "content_type": "faq",
                            },
                        )
                    )
                    current_qa = []
                current_section = line[2:]
            elif line.startswith("Q:") or line.startswith("A:"):
                if line.startswith("Q:") and current_qa:
                    # Save previous Q&A pair
                    qa_text = "\n".join(current_qa)
                    documents.append(
                        Document(
                            page_content=qa_text,
                            metadata={
                                "source": "Netflix Help Center FAQ",
                                "section": current_section,
                                "content_type": "faq",
                            },
                        )
                    )
                    current_qa = []
                current_qa.append(line)

        # Don't forget the last Q&A pair
        if current_qa:
            qa_text = "\n".join(current_qa)
            documents.append(
                Document(
                    page_content=qa_text,
                    metadata={
                        "source": "Netflix Help Center FAQ",
                        "section": current_section,
                        "content_type": "faq",
                    },
                )
            )

        return documents

    def build_vectorstore(self) -> Chroma:
        """
        Build the ChromaDB vector store from all documents.

        This method implements AI search algorithms using vector similarity
        search, which enables semantic understanding of user queries beyond
        simple keyword matching.

        Returns:
            ChromaDB vector store instance
        """
        print("Building vector store...")

        # Create documents from Netflix titles
        if self.titles_df is None:
            self.load_titles_data()

        # Ensure titles_df is loaded before iterating
        if self.titles_df is not None:
            # Create title documents
            print("Creating title documents...")
            for _, row in self.titles_df.iterrows():
                self.documents.append(self._create_title_document(row))

        # Create FAQ documents
        print("Creating FAQ documents...")
        faq_docs = self._create_faq_documents()
        self.documents.extend(faq_docs)

        print(f"Total documents: {len(self.documents)}")

        # Split documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        split_docs = text_splitter.split_documents(self.documents)

        print(f"Split into {len(split_docs)} chunks")

        # Create ChromaDB vector store
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

        print("Vector store built successfully!")
        return self.vectorstore

    def get_retriever(self, k: int = 5) -> VectorStoreRetriever:
        """
        Get a retriever for the vector store.

        Args:
            k: Number of documents to retrieve

        Returns:
            A retriever instance for semantic search
        """
        if self.vectorstore is None:
            self.build_vectorstore()

        # At this point vectorstore is guaranteed to be initialized
        assert self.vectorstore is not None, "Vectorstore failed to initialize"

        return self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )


class NetflixFAQChatbot:
    """
    Main chatbot class.

    This class implements:
    - LangChain for prompt management and chain orchestration
    - Anthropic Claude API for natural language understanding and generation
    - ChromaDB for efficient similarity search in the knowledge base

    The chatbot uses a Retrieval-Augmented Generation (RAG) architecture
    where relevant context is retrieved from the knowledge base before
    generating responses, ensuring accurate and grounded answers.
    """

    def __init__(self, api_key: str, titles_csv_path: str):
        """
        Initialize the Netflix FAQ Chatbot.

        Args:
            api_key: Anthropic API key for Claude access
            titles_csv_path: Path to the Netflix titles CSV file
        """
        # Initialize the Anthropic Claude model via LangChain
        self.llm = ChatAnthropic(
            model_name="claude-haiku-4-5-20251001",
            api_key=SecretStr(api_key),
            temperature=0.3,  # Lower temperature for factual responses
            max_tokens_to_sample=1024,
            timeout=60.0,
            stop=None,
        )

        # Initialize knowledge base
        self.knowledge_base: NetflixKnowledgeBase = NetflixKnowledgeBase(
            titles_csv_path
        )
        self.retriever: Optional[VectorStoreRetriever] = None
        self.rag_chain: Optional[Runnable] = None
        self.prompt: Optional[PromptTemplate] = None

        # Define the system prompt for the chatbot
        # This implements logical constraints (first-order logic principles)
        # to ensure the chatbot stays within its designated scope
        self.system_prompt = """You are a helpful Netflix FAQ assistant. Your role is to answer questions about:
1. Netflix service features, plans, pricing, and account management
2. Netflix movies and TV shows from the catalog

IMPORTANT RULES:
- Only answer questions related to Netflix services and content.
- For questions about movies/shows, cite the source as "Netflix Titles Dataset (Kaggle)".
- For service questions, cite "Netflix Help Center FAQ".
- If you cannot find relevant information, say so honestly.
- If the question is NOT about Netflix, respond with: "I'm sorry. I can only answer questions pertaining to Netflix."

When answering:
- Be concise and helpful.
- Cite your sources when providing specific information.
- If discussing a specific title, mention its type (Movie/TV Show), release year, and other relevant details.

Context from knowledge base:
{context}

User Question: {question}

Provide a helpful response:"""

    def initialize(self) -> None:
        """
        Initialize the chatbot components including the knowledge base
        and QA chain. This must be called before using the chatbot.
        """
        print("Initializing Netflix FAQ Chatbot...")

        # Build the knowledge base
        self.knowledge_base.build_vectorstore()
        self.retriever = self.knowledge_base.get_retriever(k=5)

        # Create the prompt template
        self.prompt = PromptTemplate(
            template=self.system_prompt, input_variables=["context", "question"]
        )

        # Create the RAG chain using LCEL (LangChain Expression Language)
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        print("Chatbot initialized successfully!")

    def _is_netflix_related(self, query: str) -> bool:
        """
        Determine if the query is related to Netflix.

        This implements a simple logical agent that uses keyword matching
        and context analysis to determine query relevance.

        Args:
            query: The user's question

        Returns:
            Boolean indicating if query is Netflix-related
        """
        # Netflix-related keywords for initial filtering
        netflix_keywords = [
            "netflix",
            "movie",
            "movies",
            "show",
            "shows",
            "tv",
            "series",
            "streaming",
            "watch",
            "subscription",
            "plan",
            "account",
            "billing",
            "password",
            "profile",
            "download",
            "offline",
            "subtitle",
            "audio",
            "cancel",
            "price",
            "cost",
            "payment",
            "cast",
            "actor",
            "director",
            "genre",
            "drama",
            "comedy",
            "action",
            "thriller",
            "documentary",
            "horror",
            "romance",
            "sci-fi",
            "animation",
            "anime",
            "film",
            "episode",
            "season",
            "release",
            "rating",
            "recommend",
            "suggestion",
        ]

        query_lower = query.lower()

        # Check for Netflix-related keywords
        for keyword in netflix_keywords:
            if keyword in query_lower:
                return True

        return False

    def _format_sources(self, source_documents: List[Document]) -> str:
        """
        Format source documents for citation.

        Args:
            source_documents: List of retrieved source documents

        Returns:
            Formatted string with source citations
        """
        if not source_documents:
            return ""

        sources: set[str] = set()
        for doc in source_documents:
            source = doc.metadata.get("source", "Unknown")
            if "title" in doc.metadata:
                sources.add(f"{source} - {doc.metadata['title']}")
            else:
                sources.add(source)

        return "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

    def chat(self, user_input: str) -> str:
        """
        Process a user message and generate a response.

        This method implements the main conversation flow:
        1. Check if query is Netflix-related (logical agent decision)
        2. Retrieve relevant context (AI search)
        3. Generate response using LLM (NLP with transformers)
        4. Format and return response with citations

        Args:
            user_input: The user's question or message

        Returns:
            The chatbot's response
        """
        if self.rag_chain is None or self.retriever is None:
            return "Error: Chatbot not initialized. Please call initialize() first."

        # Check if query is Netflix-related
        if not self._is_netflix_related(user_input):
            # Use LLM to make a more nuanced decision
            check_prompt = f"""Determine if this question is related to Netflix (the streaming service),
its movies, TV shows, pricing, accounts, or streaming features.
Question: {user_input}
Answer only YES or NO:"""

            try:
                llm_response = self.llm.invoke(check_prompt)
                # Handle the response content properly
                response_text = ""
                if hasattr(llm_response, "content"):
                    content = llm_response.content
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list) and len(content) > 0:
                        # Handle list of content blocks
                        first_block = content[0]
                        if isinstance(first_block, str):
                            response_text = first_block
                        elif isinstance(first_block, dict) and "text" in first_block:
                            response_text = str(first_block["text"])

                if "no" in response_text.lower():
                    return (
                        "I'm sorry. I can only answer questions pertaining to Netflix."
                    )
            except Exception:
                pass  # If check fails, proceed with the query

        try:
            # Get relevant documents for citation
            source_docs: List[Document] = self.retriever.invoke(user_input)

            # Execute the RAG chain
            response: str = self.rag_chain.invoke(user_input)

            # Add source citations
            if source_docs:
                sources = self._format_sources(source_docs)
                response += sources

            return response

        except Exception as e:
            return f"An error occurred while processing your question: {str(e)}"

    def get_title_info(self, title_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific Netflix title.

        Args:
            title_name: Name of the movie or TV show

        Returns:
            Dictionary with title information or None if not found
        """
        if self.knowledge_base.titles_df is None:
            return None

        df = self.knowledge_base.titles_df

        # Search for the title (case-insensitive)
        mask = df["title"].str.lower() == title_name.lower()
        matches = df[mask]

        if matches.empty:
            # Try partial match
            mask = df["title"].str.lower().str.contains(title_name.lower(), na=False)
            matches = df[mask]

        if not matches.empty:
            return matches.iloc[0].to_dict()

        return None


def main() -> None:
    """
    Main function to run the Netflix FAQ Chatbot.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY");

    if not api_key:
        api_key = input("Please enter your Anthropic API key: ").strip()

    if not api_key:
        print("Error: API key is required.")
        return

    # Path to Netflix titles CSV
    titles_path = "/mnt/project/netflix_titles.csv"

    # Check if file exists
    if not os.path.exists(titles_path):
        # Try alternate path
        titles_path = "netflix_titles.csv"
        if not os.path.exists(titles_path):
            print("Error: Netflix titles CSV not found.")
            return

    # Initialize chatbot
    chatbot = NetflixFAQChatbot(api_key, titles_path)
    chatbot.initialize()

    print("\n" + "=" * 60)
    print("Chatbot is ready! Ask me anything about Netflix.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 60 + "\n")

    # Main conversation loop
    while True:
        try:
            user_input = input("> You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "bye", "q"]:
                print(
                    "\n> Netflix Bot: Thank you for using Netflix FAQ Chatbot. Goodbye!\n"
                )
                break

            # Get response from chatbot
            response = chatbot.chat(user_input)
            print(f"\n> Netflix Bot: {response}\n")

        except KeyboardInterrupt:
            print("\n\n> Netflix Bot: Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")


if __name__ == "__main__":
    main()
