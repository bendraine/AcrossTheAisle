#!/usr/bin/env python3
"""
Debug script specifically for counterargument generation issues
Run this to diagnose why counterarguments aren't being generated
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add your project directory to Python path
sys.path.append('.')

def test_document_store():
    """Test document store functionality step by step"""
    print("=== Testing Document Store ===")
    
    try:
        from document_store import _document_store, get_store_stats, retrieve_passages
        
        # 1. Check document store stats
        print("\n1. Document Store Statistics:")
        stats = get_store_stats()
        print(json.dumps(stats, indent=2))
        
        if stats.get("total_passages", 0) == 0:
            print("❌ PROBLEM: No documents loaded in ChromaDB!")
            print("   Solution: Add PDF files to ./data/docs/ and restart the server")
            return False
        
        # 2. Test document retrieval with different parameters
        print("\n2. Testing Document Retrieval:")
        
        test_queries = [
            "income tax policy arguments",
            "tax reform benefits",
            "progressive taxation",
            "flat tax system",
            "government revenue"
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            
            # Test without topic filter
            results_no_filter = retrieve_passages(query, top_k=3)
            print(f"     No topic filter: {len(results_no_filter)} results")
            
            # Test with income_tax topic filter
            results_with_filter = retrieve_passages(query, top_k=3, topic="income_tax")
            print(f"     With income_tax filter: {len(results_with_filter)} results")
            
            # Test with different semantic types
            arg_results = retrieve_passages(query, top_k=3, semantic_types=["argument"])
            print(f"     Argument type only: {len(arg_results)} results")
            
            # Show sample results
            if results_no_filter:
                sample = results_no_filter[0]
                print(f"     Sample result: {sample.source} (Page {sample.page_number})")
                print(f"     Relevance: {sample.relevance_score:.3f}")
                print(f"     Preview: {sample.text[:100]}...")
        
        return len(results_no_filter) > 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing document store: {e}")
        return False

def test_search_query_generation():
    """Test how search queries are generated from user data"""
    print("\n=== Testing Search Query Generation ===")
    
    try:
        from response_handler import CounterArgumentGenerator
        
        generator = CounterArgumentGenerator()
        
        # Sample user data with different opinion types
        test_cases = [
            {
                "name": "Progressive Tax Supporter",
                "data": {
                    "demographics": {"political_orientation": "Progressive"},
                    "phase1": {
                        "opinion_statement": "I believe in progressive taxation where wealthy individuals and corporations pay higher rates to fund social programs and reduce inequality."
                    }
                }
            },
            {
                "name": "Flat Tax Supporter", 
                "data": {
                    "demographics": {"political_orientation": "Conservative"},
                    "phase1": {
                        "opinion_statement": "A flat tax rate for everyone would be more fair and simpler to administer than our current complex progressive system."
                    }
                }
            },
            {
                "name": "No Clear Opinion",
                "data": {
                    "demographics": {},
                    "phase1": {
                        "opinion_statement": ""
                    }
                }
            }
        ]
        
        for test_case in test_cases:
            print(f"\n   Test Case: {test_case['name']}")
            query = generator._create_search_query(test_case['data'], None)
            print(f"   Generated Query: '{query}'")
            
            # Test this query against document store
            try:
                from document_store import retrieve_passages
                results = retrieve_passages(query, top_k=3)
                print(f"   Results Found: {len(results)}")
                if results:
                    print(f"   Top Result: {results[0].source} (Score: {results[0].relevance_score:.3f})")
            except Exception as e:
                print(f"   Query test failed: {e}")
        
    except Exception as e:
        print(f"❌ Error testing query generation: {e}")

def check_document_topics():
    """Check what topics are actually in your documents"""
    print("\n=== Checking Document Topics ===")
    
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        
        # Connect to ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_store")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        collection = chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=openai_ef
        )
        
        # Get all documents and check their topics
        all_data = collection.get(include=["metadatas"])
        
        if all_data and all_data.get("metadatas"):
            topics = {}
            sources = set()
            semantic_types = {}
            
            for metadata in all_data["metadatas"]:
                # Count topics
                topic = metadata.get("topic", "unknown")
                topics[topic] = topics.get(topic, 0) + 1
                
                # Count sources
                source = metadata.get("document_source", "unknown")
                sources.add(source)
                
                # Count semantic types
                sem_type = metadata.get("semantic_type", "unknown")
                semantic_types[sem_type] = semantic_types.get(sem_type, 0) + 1
            
            print(f"   Total passages: {len(all_data['metadatas'])}")
            print(f"   Unique sources: {len(sources)}")
            print(f"   Topics found: {topics}")
            print(f"   Semantic types: {semantic_types}")
            print(f"   Document sources: {list(sources)}")
            
            # Check if income_tax topic exists
            if "income_tax" not in topics:
                print("\n   ❌ PROBLEM: No 'income_tax' topic found!")
                print("   This explains why topic filtering fails.")
                print("   Topics in your database:", list(topics.keys()))
                return False
            else:
                print(f"\n   ✅ Found {topics['income_tax']} passages with 'income_tax' topic")
                return True
        else:
            print("   ❌ No document metadata found in ChromaDB")
            return False
            
    except Exception as e:
        print(f"❌ Error checking document topics: {e}")
        return False

def test_full_counterargument_flow():
    """Test the complete counterargument generation flow"""
    print("\n=== Testing Full Counterargument Flow ===")
    
    try:
        from response_handler import get_ai_counterargument
        from value_identifier import ValueProfile
        
        # Create sample user data that should work
        sample_user_data = {
            "demographics": {
                "political_orientation": "Progressive",
                "location": "California"
            },
            "phase1": {
                "opinion_statement": "I think we need higher taxes on the wealthy to fund social programs and reduce inequality in America.",
                "reasoning": ["The rich should pay their fair share", "Income inequality is getting worse"],
                "friend_explanation": "I believe progressive taxation helps create a more equitable society."
            },
            "phase2": {
                "confidence_score": 8,
                "flexibility": "It depends",
                "flexibility_reason": "I'd consider other approaches if they worked better"
            },
            "phase3": {
                "valid_points_other_side": "Lower taxes might encourage more investment",
                "motivation_other_side": "People want economic growth and job creation"
            }
        }
        
        # Create a simple value profile
        value_profile = ValueProfile(
            individual_vs_collective=0.2,  # More collective
            government_trust=0.7,  # High trust
            change_orientation=0.8,  # Embrace change
            primary_concerns=["inequality", "social programs"],
            confidence=0.8
        )
        
        print("   Testing counterargument generation...")
        result = get_ai_counterargument(sample_user_data, value_profile)
        
        print(f"   Counterargument length: {len(result.get('counterargument', ''))}")
        print(f"   Sources found: {len(result.get('sources', []))}")
        print(f"   Confidence score: {result.get('confidence', 0):.3f}")
        
        if len(result.get('counterargument', '')) < 100:
            print("   ❌ PROBLEM: Counterargument too short or empty")
            print(f"   Actual response: '{result.get('counterargument', 'None')}'")
            return False
        else:
            print("   ✅ Counterargument generated successfully")
            print(f"   Preview: {result['counterargument'][:150]}...")
            return True
            
    except Exception as e:
        print(f"❌ Error testing counterargument flow: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_fixes():
    """Suggest specific fixes based on test results"""
    print("\n=== Suggested Fixes ===")
    
    # Check if docs folder exists and has PDFs
    docs_folder = Path("./data/docs")
    if not docs_folder.exists():
        print("1. ❌ Create docs folder: mkdir -p ./data/docs")
    else:
        pdf_files = list(docs_folder.glob("*.pdf"))
        if not pdf_files:
            print("1. ❌ Add PDF files to ./data/docs/ folder")
            print("   Your system needs PDF documents containing political arguments")
        else:
            print(f"1. ✅ Found {len(pdf_files)} PDF files")
    
    # Check ChromaDB
    chroma_folder = Path("./chroma_store")
    if not chroma_folder.exists():
        print("2. ❌ ChromaDB not initialized - restart your server")
    else:
        print("2. ✅ ChromaDB folder exists")
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("3. ❌ Set OPENAI_API_KEY in your .env file")
    else:
        print("3. ✅ OpenAI API key found")
    
    print("\n=== Quick Fix Commands ===")
    print("# If no documents are loaded:")
    print("1. Add PDF files to ./data/docs/")
    print("2. Delete ./chroma_store/ folder")
    print("3. Restart your Flask app (python app.py)")
    print("")
    print("# If topic filtering is the issue:")
    print("Edit response_handler.py, line ~60:")
    print("Change: topic='income_tax' to: topic=None")
    print("This will search all documents regardless of topic")

def main():
    """Run all diagnostic tests"""
    print("🔍 Counterargument Generation Debug Script")
    print("=" * 60)
    
    # Set up logging to see more details
    logging.basicConfig(level=logging.INFO)
    
    # Run tests in order
    doc_store_ok = test_document_store()
    topics_ok = check_document_topics()
    test_search_query_generation()
    flow_ok = test_full_counterargument_flow()
    
    print("\n" + "=" * 60)
    
    if doc_store_ok and topics_ok and flow_ok:
        print("✅ All tests passed! Your counterargument system should work.")
    else:
        print("❌ Some tests failed. See suggested fixes below.")
        suggest_fixes()

if __name__ == "__main__":
    main()