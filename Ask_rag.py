import json, numpy as np, argparse
from openai import OpenAI
from config import *

def get_embedding(text, model=embedding_model):
    return local_client.embeddings.create(input=[text.replace("\n", " ")], model=model).data[0].embedding

def similarity(v1, v2):
    return np.dot(v1, v2)

def load_embeddings(embeddings_json):
    with open(embeddings_json, 'r', encoding='utf8') as f:
        return json.load(f)

def get_best_vectors(question_vector, index_lib, num_results):
    scores = [{'content': v['content'], 'score': similarity(question_vector, v['vector']), 'source_file': v.get('source_file', 'unknown')} for v in index_lib]
    return sorted(scores, key=lambda x: x['score'], reverse=True)[:num_results]

def rag_answer(question, context, mode="local"):
    client, completion_model = api_mode(mode)
    
    prompt = f"""Answer the question based on the provided information. 
You are given extracted parts of a document and a question. Provide a direct answer.
If you don't know the answer, just say "I do not know.". Don't make up an answer. But provide a summary of the context. and suggest further place to look for the answer.
PROVIDED INFORMATION: {context}"""
    
    completion = client.chat.completions.create(
        model=completion_model[0]["model"],
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content

def enhance_question(question, mode="local"):
    client, completion_model = api_mode(mode)
    
    enhancement_prompt = f"""Improve this question for better document search results. 
    Keep it succint and concise so it can be used for vector search.
    Do not change the original meaning:

    Original: {question}

    Enhanced question:"""
                        
    response = client.chat.completions.create(
        model=completion_model[0]["model"],
        messages=[{"role": "user", "content": enhancement_prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def detect_poor_answer(answer):
    words = answer.split()
    if len(words) < 10: return True
    
    # Check for excessive repetition (like your example)
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    max_repetition = max(word_counts.values()) / len(words)
    # Remove "I do not know" check - that's a valid response
    return max_repetition > 0.3

def fallback_answer(question, context, mode="local"):
    """Provide helpful fallback when RAG fails"""
    client, completion_model = api_mode(mode)
    
    prompt = f"""The document search didn't find a complete answer to this question: "{question}"

        Based on this partial context: {context}...

        Provide:
        1. Where in building standards this answer might be found (specific sections/documents)
        2. Your best knowledge-based answer to help the user
        3. Specific next steps they should take

        Be practical and actionable."""

    response = client.chat.completions.create(
        model=completion_model[0]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def reframe_question(question, context, mode="local"):
    """Provide helpful fallback when RAG fails"""
    client, completion_model = api_mode(mode)
    
    prompt = f"""The document search didn't find a complete answer to this question: "{question}"

        Based on this context: {context}...

        Provide:
        1.Extractly how the question should be reframed to get better results
        2. Your best knowledge-based answer to help the user
        3. If the context provides where the answer might be found (specific sections/documents) reframe the question so we can use it for vector search in RAG.

        Be practical and actionable."""

    response = client.chat.completions.create(
        model=completion_model[0]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def extract_reframed_question(reframe_response):
    """Extract the reframed question from LLM response"""
    lines = reframe_response.split('\n')
    for line in lines:
        if 'reframed' in line.lower() or 'better' in line.lower():
            # Extract question after colon or quotation marks
            if ':' in line:
                return line.split(':', 1)[1].strip(' "')
            elif '"' in line:
                return line.split('"')[1]
    return lines[0]  # fallback to first line

def perform_search(query, index_lib, doc_context, args):
    question_vector = get_embedding(query)
    best_vectors = get_best_vectors(question_vector, index_lib, args.num_results)
    
    context_parts = [doc_context]
    for v in best_vectors:
        source = v.get('source_file', 'unknown').replace('.json', '')
        context_parts.append(f"[Source: {source}]\n{v['content']}")
    context = "\n\n".join(context_parts)
    
    if args.show_context:
        print(f"\nRETRIEVED CONTEXT:")
        for i, v in enumerate(best_vectors, 1):
            print(f"[{i}] Source: {v.get('source_file', 'unknown')}")
            print(f"Content: {v['content']}\n")
    
    answer = rag_answer(query, context, args.mode)
    return answer, best_vectors

def ask_rag():
    parser = argparse.ArgumentParser(description='RAG Question Answering System')
    parser.add_argument('embeddings_json', help='Path to embeddings JSON file')
    parser.add_argument('question', nargs='+', help='Question to ask')
    parser.add_argument('--num_results', '-n', type=int, default=5, help='Number of results to retrieve (default: 3)')
    parser.add_argument('--mode', '-m', choices=['local', 'openai'], default='local', help='API mode (default: local)')
    parser.add_argument('--show_context', '-c', action='store_true', help='Show retrieved context')
    
    args = parser.parse_args()
    question = ' '.join(args.question)  # Join multiple words back into single question
    
    # Get available documents
    index_lib = load_embeddings(args.embeddings_json)
    available_docs = list(set(v.get('source_file', 'unknown').replace('.json', '') for v in index_lib))
    doc_context = f"Available knowledge base: {', '.join(available_docs)}"
    
    # First attempt
    enhanced_question = enhance_question(question, args.mode)
    print(f"Enhanced question: {enhanced_question}")
    
    answer, best_vectors = perform_search(enhanced_question, index_lib, doc_context, args)
    
    # If poor answer, try reframing once
    if detect_poor_answer(answer):
        print("Initial search unsatisfactory, reframing...")
        reframe_response = reframe_question(question, "\n".join([v['content'] for v in best_vectors]), args.mode)
        reframed_question = extract_reframed_question(reframe_response)
        print(f"Reframing search with: {reframed_question}")
        
        answer, best_vectors = perform_search(reframed_question, index_lib, doc_context, args)
        
        # If still poor after reframing, use fallback
        if detect_poor_answer(answer):
            context = "\n\n".join([f"[Source: {v.get('source_file', 'unknown').replace('.json', '')}]\n{v['content']}" for v in best_vectors])
            answer = fallback_answer(question, context, args.mode)
    
    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer}")
    print(f"SOURCES: {', '.join(set(v.get('source_file', 'unknown') for v in best_vectors))}")

if __name__ == "__main__":
    ask_rag()