from flask import Flask, request, jsonify
from server.config import *

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
    If you don't know the answer, just say "I do not know.". Don't make up an answer.
    PROVIDED INFORMATION: {context}. Provide a summary of the information provided."""
    
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
    Keep it succint and concise so it can be used for RAG vector search.
    Do not change the original meaning. Do not add any additional information, just improve the wording.

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

        Provide a reframed question that would yield better results in the document search and extract the information required from the original question.        
        return the reframed question in the format: Reframed question: <your question here>. Do not provide any additional information or explanation.
        """

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

def perform_search(query, index_lib, doc_context, num_results=5, mode="local", show_context=False):
    question_vector = get_embedding(query)
    best_vectors = get_best_vectors(question_vector, index_lib, num_results)
    
    context_parts = [doc_context]
    for v in best_vectors:
        source = v.get('source_file', 'unknown').replace('.json', '')
        context_parts.append(f"[Source: {source}]\n{v['content']}")
    context = "\n\n".join(context_parts)
    
    context_results = ""
    if show_context:
        print(f"\nRETRIEVED CONTEXT:")
        for i, v in enumerate(best_vectors, 1):
            print(f"[{i}] Source: {v.get('source_file', 'unknown')}")
            print(f"Content: {v['content']}\n")
            context_results += f"[{i}] Source: {v.get('source_file', 'unknown')}\nContent: {v['content']}\n\n"
    
    answer = rag_answer(query, context,mode)
    return answer, best_vectors, context_results

def classify_answer(question,mode="local"):
    """Provide helpful fallback when RAG fails"""
    client, completion_model = api_mode(mode)
    
    prompt = f"""If the response contain I don't know or indicates that it doesn't know the answer or it doesnt answer the question correctly: "{question}"
        Return True if the answer is satisfactory, otherwise return False. Do not provide any explanation or additional information, just return True or False.
        Example:
        ANSWER: I don't know the answer to your question based on the provided information.
        RESPONSE: False
        """

    response = client.chat.completions.create(
        model=completion_model[0]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

def extract_values(text, mode="local"): 
    """Provide helpful fallback when RAG fails"""
    client, completion_model = api_mode(mode)
    
    prompt = f"""{text}
        You are a parser that extracts the upper percentage values from a Window-to-Wall Ratio recommendation. Given text with entries like North: 30–40%, South: 40–50% (shaded), East/West: 35–45%, 
        output **only** a JSON object with lowercase keys north, south, east, and west whose values are the upper bounds as integers.  
            Example input:
            The recommended WWR …  
            - North: 30-40%  
            - South: 40-50% (shaded)  
            - East/West: 35-45% 
            Expected output:
            json
            {{'north':40,'south':50,'east':35,'west':45}}
        """

    response = client.chat.completions.create(
        model=completion_model[0]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    return response.choices[0].message.content

app = Flask(__name__)

@app.route('/llm_call', methods=['POST'])
def llm_call():
    data = request.get_json()
    input_string = data.get('input', '')
    mode = "local"
    show_context = True
    
    embeddings_json = data.get('merged.json', 'knowledge_pool\merged.json')
    # --- User Input ---
    question = input_string
    
    # Get available documents
    index_lib = load_embeddings(embeddings_json)
    available_docs = list(set(v.get('source_file', 'unknown').replace('.json', '') for v in index_lib))
    doc_context = f"Available knowledge base: {', '.join(available_docs)}"
    
    answer, best_vectors, context_results = perform_search(question, index_lib, doc_context, show_context=show_context, mode=mode)
    
    json_values = extract_values(answer, mode="local")
    
    
    # print(f"Original answer: {answer},enhanced_question: {enhanced_question}")
    # # First attempt
    # enhanced_question = enhance_question(question, mode)
    # print(f"Enhanced question: {enhanced_question}")
    
    # answer, best_vectors, context_results = perform_search(enhanced_question, index_lib, doc_context, show_context=show_context, mode=mode)
    # print(f"Original answer: {answer},enhanced_question: {enhanced_question}")
    
    # if classify_answer(answer,mode):
    #     print("Initial search unsatisfactory, reframing...")
    #     reframe_response = reframe_question(enhanced_question, answer + "\n".join([v['content'] for v in best_vectors]), mode)
    #     # reframed_question = extract_reframed_question(reframe_response)
    #     print(f"Reframing search with: {reframe_response}")
        
    #     answer, best_vectors, context_results = perform_search(reframe_response.split(':')[1], index_lib, doc_context, show_context=show_context, mode=mode)
    #     question = reframe_response.split(':')[1].strip()
    

    print(f"QUESTION: {question}")
    print(f"ANSWER: {answer}")
    print(f"SOURCES: {', '.join(set(v.get('source_file', 'unknown') for v in best_vectors))}")
    final_answer = f"QUESTION: {question}\nANSWER: {answer}\nSOURCES: {', '.join(set(v.get('source_file', 'unknown') for v in best_vectors))}, \n\nCONTEXT: {context_results}"
    return jsonify({
        "response": final_answer,
        "json_values": json_values
    })


if __name__ == '__main__':
    app.run(port=6000, debug=True)