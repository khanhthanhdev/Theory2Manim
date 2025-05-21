import json
import os
from typing import List, Dict
import uuid
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import Language
from langchain_core.embeddings import Embeddings
import statistics
from litellm import embedding
import litellm
import tiktoken
from tqdm import tqdm
from langfuse import Langfuse
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from mllm_tools.utils import _prepare_text_inputs
from task_generator import get_prompt_detect_plugins

class RAGVectorStore:
    """A class for managing vector stores for RAG (Retrieval Augmented Generation).

    This class handles creation, loading and querying of vector stores for both Manim core
    and plugin documentation.

    Args:
        chroma_db_path (str): Path to ChromaDB storage directory
        manim_docs_path (str): Path to Manim documentation files
        embedding_model (str): Name of the embedding model to use
        trace_id (str, optional): Trace identifier for logging. Defaults to None
        session_id (str, optional): Session identifier. Defaults to None
        use_langfuse (bool, optional): Whether to use Langfuse logging. Defaults to True
        helper_model: Helper model for processing. Defaults to None
    """

    def __init__(self, 
                 chroma_db_path: str = "chroma_db",
                 manim_docs_path: str = "rag/manim_docs",
                 embedding_model: str = "hf:sentence-transformers/all-MiniLM-L6-v2",
                 trace_id: str = None,
                 session_id: str = None,
                 use_langfuse: bool = True,
                 helper_model = None):
        self.chroma_db_path = chroma_db_path
        self.manim_docs_path = manim_docs_path
        self.embedding_model = embedding_model
        self.trace_id = trace_id
        self.session_id = session_id
        self.use_langfuse = use_langfuse
        self.helper_model = helper_model
        self.enc = tiktoken.encoding_for_model("gpt-4")
        self.plugin_stores = {}
        self.vector_store = self._load_or_create_vector_store()

    def _load_or_create_vector_store(self):
        """Loads existing or creates new ChromaDB vector stores.

        Creates/loads vector stores for both Manim core documentation and any available plugins.
        Stores are persisted to disk for future reuse.

        Returns:
            Chroma: The core Manim vector store instance
        """
        print("Entering _load_or_create_vector_store with trace_id:", self.trace_id)
        core_path = os.path.join(self.chroma_db_path, "manim_core")
        
        # Load or create core vector store
        if os.path.exists(core_path):
            print("Loading existing core ChromaDB...")
            self.core_vector_store = Chroma(
                collection_name="manim_core",
                persist_directory=core_path,
                embedding_function=self._get_embedding_function()
            )
        else:
            print("Creating new core ChromaDB...")
            self.core_vector_store = self._create_core_store()
        
        # Fix: Use correct path construction for plugin_docs
        plugin_docs_path = os.path.join(self.manim_docs_path, "plugin_docs")
        print(f"Plugin docs path: {plugin_docs_path}")
        if os.path.exists(plugin_docs_path):
            for plugin_name in os.listdir(plugin_docs_path):
                plugin_store_path = os.path.join(self.chroma_db_path, f"manim_plugin_{plugin_name}")
                if os.path.exists(plugin_store_path):
                    print(f"Loading existing plugin store: {plugin_name}")
                    self.plugin_stores[plugin_name] = Chroma(
                        collection_name=f"manim_plugin_{plugin_name}",
                        persist_directory=plugin_store_path,
                        embedding_function=self._get_embedding_function()
                    )
                else:
                    print(f"Creating new plugin store: {plugin_name}")
                    plugin_path = os.path.join(plugin_docs_path, plugin_name)
                    if os.path.isdir(plugin_path):
                        plugin_store = Chroma(
                            collection_name=f"manim_plugin_{plugin_name}",
                            embedding_function=self._get_embedding_function(),
                            persist_directory=plugin_store_path
                        )
                        plugin_docs = self._process_documentation_folder(plugin_path)
                        if plugin_docs:
                            self._add_documents_to_store(plugin_store, plugin_docs, plugin_name)
                        self.plugin_stores[plugin_name] = plugin_store
        
        return self.core_vector_store  # Return core store for backward compatibility

    def _get_embedding_function(self) -> Embeddings:
        """Creates an embedding function using HuggingFaceEmbeddings locally."""
        if self.embedding_model.startswith('hf:'):
            model_name = self.embedding_model[3:]
            print(f"Using HuggingFaceEmbeddings with model: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        elif self.embedding_model == "Alibaba-NLP/gte-modernbert-base":
            print(f"Using HuggingFaceEmbeddings with model: {self.embedding_model}")
            # Add memory optimizations for the larger model
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={
                    'device': 'cpu',
                    # Force garbage collection after inference
                    'torch_dtype': 'float16'  # Use half-precision to save memory
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 8  # Reduce batch size for lower memory consumption
                }
            )
        else:
            raise ValueError("Only HuggingFace embeddings are supported in this configuration.")

    def _create_core_store(self):
        """Creates the main ChromaDB vector store for Manim core documentation.

        Returns:
            Chroma: The initialized and populated core vector store
        """
        core_vector_store = Chroma(
            collection_name="manim_core",
            embedding_function=self._get_embedding_function(),
            persist_directory=os.path.join(self.chroma_db_path, "manim_core")
        )
        
        # Process manim core docs
        core_docs = self._process_documentation_folder(os.path.join(self.manim_docs_path, "manim_core"))
        if core_docs:
            self._add_documents_to_store(core_vector_store, core_docs, "manim_core")
        
        return core_vector_store

    def _process_documentation_folder(self, folder_path: str) -> List[Document]:
        """Processes documentation files from a folder into LangChain documents.

        Args:
            folder_path (str): Path to the folder containing documentation files

        Returns:
            List[Document]: List of processed LangChain documents
        """
        all_docs = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.md', '.py')):
                    file_path = os.path.join(root, file)
                    try:
                        loader = TextLoader(file_path)
                        documents = loader.load()
                        for doc in documents:
                            doc.metadata['source'] = file_path
                        all_docs.extend(documents)
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
        
        if not all_docs:
            print(f"No markdown or python files found in {folder_path}")
            return []
        
        # Split documents using appropriate splitters
        split_docs = []
        markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN
        )
        python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON
        )
        
        for doc in all_docs:
            if doc.metadata['source'].endswith('.md'):
                temp_docs = markdown_splitter.split_documents([doc])
                for temp_doc in temp_docs:
                    temp_doc.page_content = f"Source: {doc.metadata['source']}\n\n{temp_doc.page_content}"
                split_docs.extend(temp_docs)
            elif doc.metadata['source'].endswith('.py'):
                temp_docs = python_splitter.split_documents([doc])
                for temp_doc in temp_docs:
                    temp_doc.page_content = f"Source: {doc.metadata['source']}\n\n{temp_doc.page_content}"
                split_docs.extend(temp_docs)
        
        return split_docs

    def _add_documents_to_store(self, vector_store: Chroma, documents: List[Document], store_name: str):
        """Adds documents to a vector store in batches with rate limiting.

        Args:
            vector_store (Chroma): The vector store to add documents to
            documents (List[Document]): List of documents to add
            store_name (str): Name of the store for logging purposes
        """
        print(f"Adding documents to {store_name} store")
        
        # Calculate token statistics
        token_lengths = [len(self.enc.encode(doc.page_content)) for doc in documents]
        print(f"Token length statistics for {store_name}: "
              f"Min: {min(token_lengths)}, Max: {max(token_lengths)}, "
              f"Mean: {sum(token_lengths) / len(token_lengths):.1f}, "
              f"Median: {statistics.median(token_lengths)}, "
              f"Std: {statistics.stdev(token_lengths):.1f}")
        
        import time

        batch_size = 10
        request_count = 0
        for i in tqdm(range(0, len(documents), batch_size), desc=f"Processing {store_name} batches"):
            batch_docs = documents[i:i + batch_size]
            batch_ids = [str(uuid.uuid4()) for _ in batch_docs]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            request_count += 1
            if request_count % 100 == 0:
                time.sleep(60)  # Sleep for 1 second every 100 requests
        
        vector_store.persist()

    def find_relevant_docs(self, queries: List[Dict], k: int = 5, trace_id: str = None, topic: str = None, scene_number: int = None) -> List[str]:
        """Finds relevant documentation based on the provided queries.

        Args:
            queries (List[Dict]): List of query dictionaries with 'type' and 'query' keys
            k (int, optional): Number of results to return per query. Defaults to 5
            trace_id (str, optional): Trace identifier for logging. Defaults to None
            topic (str, optional): Topic name for logging. Defaults to None
            scene_number (int, optional): Scene number for logging. Defaults to None

        Returns:
            List[str]: Formatted string containing relevant documentation snippets
        """
        manim_core_formatted_results = []
        manim_plugin_formatted_results = []
        
        # Initialize span with a dummy object if not using Langfuse
        span_id = None
        
        # Create a Langfuse span if enabled
        if self.use_langfuse:
            langfuse = Langfuse()
            span = langfuse.span(
                trace_id=trace_id,  # Use the passed trace_id
                name=f"RAG search for {topic} - scene {scene_number}",
                metadata={
                    "topic": topic,
                    "scene_number": scene_number,
                    "session_id": self.session_id
                }
            )
            span_id = span.id
        
        import gc
        
        # Process in smaller batches to prevent memory issues
        # First, separate queries by type
        manim_core_queries = [query for query in queries if query["type"] == "manim-core"]
        manim_plugin_queries = [query for query in queries if query["type"] != "manim-core" and query["type"] in self.plugin_stores]
        
        if len([q for q in queries if q["type"] != "manim-core"]) != len(manim_plugin_queries):
            print("Warning: Some plugin queries were skipped because their types weren't found in available plugin stores")
        
        # Process core queries in smaller batches to reduce memory usage
        batch_size = 2  # Process just 2 queries at a time
        for i in range(0, len(manim_core_queries), batch_size):
            batch_queries = manim_core_queries[i:i+batch_size]
            
            # Search in core manim docs for this batch
            for query in batch_queries:
                query_text = query["query"]
                if self.use_langfuse:
                    self.core_vector_store._embedding_function.parent_observation_id = span_id
                try:
                    manim_core_results = self.core_vector_store.similarity_search_with_relevance_scores(
                        query=query_text,
                        k=k,
                        score_threshold=0.5
                    )
                    for result in manim_core_results:
                        manim_core_formatted_results.append({
                            "query": query_text,
                            "source": result[0].metadata['source'],
                            "content": result[0].page_content,
                            "score": result[1]
                        })
                except Exception as e:
                    print(f"Error searching for '{query_text}': {e}")
            
            # Force garbage collection after each batch to free memory
            gc.collect()
        
        # Process plugin queries in smaller batches
        for i in range(0, len(manim_plugin_queries), batch_size):
            batch_queries = manim_plugin_queries[i:i+batch_size]
            
            # Search in relevant plugin docs for this batch
            for query in batch_queries:
                plugin_name = query["type"]
                query_text = query["query"]
                if self.use_langfuse and plugin_name in self.plugin_stores:
                    self.plugin_stores[plugin_name]._embedding_function.parent_observation_id = span_id
                if plugin_name in self.plugin_stores:
                    try:
                        plugin_results = self.plugin_stores[plugin_name].similarity_search_with_relevance_scores(
                            query=query_text,
                            k=k,
                            score_threshold=0.5
                        )
                        for result in plugin_results:
                            manim_plugin_formatted_results.append({
                                "query": query_text,
                                "source": result[0].metadata['source'],
                                "content": result[0].page_content,
                                "score": result[1]
                            })
                    except Exception as e:
                        print(f"Error searching plugin '{plugin_name}' for '{query_text}': {e}")
            
            # Force garbage collection after each batch
            gc.collect()
        
        print(f"Number of results before removing duplicates: {len(manim_core_formatted_results) + len(manim_plugin_formatted_results)}")
        
        # Remove duplicates based on content
        manim_core_unique_results = []
        manim_plugin_unique_results = []
        seen = set()
        
        # Process core results in batches to reduce memory pressure
        for item in manim_core_formatted_results:
            key = item['content']
            if key not in seen:
                manim_core_unique_results.append(item)
                seen.add(key)
        
        # Process plugin results in batches
        for item in manim_plugin_formatted_results:
            key = item['content']
            if key not in seen:
                manim_plugin_unique_results.append(item)
                seen.add(key)
        
        # Free memory by clearing the original results
        manim_core_formatted_results = []
        manim_plugin_formatted_results = []
        gc.collect()
        
        print(f"Number of results after removing duplicates: {len(manim_core_unique_results) + len(manim_plugin_unique_results)}")
        
        # Calculate total tokens in batches
        total_tokens = 0
        batch_size = 10
        for i in range(0, len(manim_core_unique_results), batch_size):
            batch = manim_core_unique_results[i:i+batch_size]
            total_tokens += sum(len(self.enc.encode(res['content'])) for res in batch)
        
        for i in range(0, len(manim_plugin_unique_results), batch_size):
            batch = manim_plugin_unique_results[i:i+batch_size]
            total_tokens += sum(len(self.enc.encode(res['content'])) for res in batch)
        
        print(f"Total tokens for the RAG search: {total_tokens}")
        
        # Update Langfuse with the deduplicated results
        if self.use_langfuse:
            # Only send summary statistics to avoid large data transfer
            span.update(
                output=f"Found {len(manim_core_unique_results)} core results and {len(manim_plugin_unique_results)} plugin results",
                metadata={
                    "total_tokens": total_tokens,
                    "initial_results_count": len(manim_core_formatted_results) + len(manim_plugin_formatted_results),
                    "filtered_results_count": len(manim_core_unique_results) + len(manim_plugin_unique_results)
                }
            )

        # Generate the formatted results strings in batches to reduce memory usage
        manim_core_results = "Please refer to the following Manim core documentation that may be helpful for the code generation:\n\n"
        
        for i in range(0, len(manim_core_unique_results), batch_size):
            batch = manim_core_unique_results[i:i+batch_size]
            manim_core_results += "\n\n".join([f"Content:\n````text\n{res['content']}\n````\nScore: {res['score']}" for res in batch])
            if i + batch_size < len(manim_core_unique_results):
                manim_core_results += "\n\n"
                
        manim_plugin_results = "Please refer to the following Manim plugin documentation that may be helpful for the code generation:\n\n"
        
        for i in range(0, len(manim_plugin_unique_results), batch_size):
            batch = manim_plugin_unique_results[i:i+batch_size]
            manim_plugin_results += "\n\n".join([f"Content:\n````text\n{res['content']}\n````\nScore: {res['score']}" for res in batch])
            if i + batch_size < len(manim_plugin_unique_results):
                manim_plugin_results += "\n\n"
        
        gc.collect()  # Final garbage collection before returning
        
        return manim_core_results + "\n\n" + manim_plugin_results