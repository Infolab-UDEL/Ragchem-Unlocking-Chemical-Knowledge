import rank_gpt
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

class PromptLibrary:   
    def format_documents_llm(docs):
        return "\n\n".join(
            f"**Passage {i+1}**: {text['fields']['content'].strip()}"
            for i, text in enumerate(docs)
    )
    
    def reranking_llm(self, query: str , context) -> str:
        num = len(context)
        prompt = f"""
        You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.   
        The following are {num} passages, each indicated by number identifier []. You should rank them based on their relevance and usefulness to query: {query}

        {self.format_documents_llm(context)}
                
        The search query is: {query}
                
        Rank the {num} passages above based on their relevance and usefulness in accurately answering the search query. Focus on factors such as topical alignment, clarity, and completeness of information
                
        The passages will be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be RANKING= [] > [] > etc, e.g., [1] > [2] > etc.

        Please return the ranking results of {num} passages using identifiers only. Ensure that:

        The output contains the exact same number of passages (20) as provided in the initial input.

        The structure and format of the output match the expected format described earlier.
        """
        return prompt

    def reranking_llm_entities(self, query: str , context, query_entities) -> str:
        num = len(context)
        words = [ents_Q["word"] for ents_Q in query_entities["pred"][0]["ents"]]
        entities = ', '.join(words)

        prompt = f"""
        You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.   
        The following are {num} passages, each indicated by number identifier []. You should rank them based on their relevance and usefulness to query: {query}, , taking acount the following entities or words should be in the passages {entities}

        {self.format_documents_llm(context)}
                
        The search query is: {query}
                
        Rank the {num} passages above based on their relevance and usefulness in accurately answering the search query. Focus on factors such as topical alignment, clarity, and completeness of information
                
        The passages will be listed in descending order using identifiers, and the most relevant passages should be listed first, and the output format should be RANKING= [] > [] > etc, e.g., [1] > [2] > etc.
        """
        # Please return the ranking results of {num} passages using identifiers only. Ensure that:

        # The output contains the exact same number of passages (20) as provided in the initial input.

        # The structure and format of the output match the expected format described earlier.
  
        return prompt    

    def rank_gpt (self, query:str, context: list, debug:bool = False):
        return rank_gpt.create_permutation_instruction(query, context )
    
    def generate_answer_openai (self, user_query: str, top_results:List[str], debug:bool = False):

        system_instruction = f"""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        """
        
        user_instruction = f"""
        Generate a comprehensive answer to the user's question based on the provided search results. 

        - Use the information from the search results to construct your answer.
        - Reference the relevant documents using their DOI in the format [1], [2], etc., whenever a specific result is cited.
        - If the available information is insufficient to answer the question, clearly state: "There is not enough context to answer this question."
        - At the end of the answer, include a list of all referenced documents with their DOI.

        TOP_RESULTS: {top_results}
        USER_QUESTION: {user_query}

        Provide as much relevant detail as possible in the answer while adhering to the format specified above.
        """  
        messages=[
            {
            "role": "system", 
            "content": system_instruction
            },
            {
            "role": "user",
            "content": user_instruction
            }
               ]
        
        return messages
    
    def generate_answer_llama (self, user_query: str, top_results:List[str], debug:bool = False):
        
        system_instruction = f"""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't have enough context to answer the question, just say that It doesn't enough context.
        Generate an answer to the user's question based on the context.
        Dont suggest others possible source of information
        Include as much information as possible in the answer. Reference the relevant context results urls as markdown links.These links should be taken by the doi present in the context retrieved.
        """
        user_instruction = f"""
        Question: {user_query}\n Context: {top_results}
        """
        message=[
            {
            "role": "system", 
            "content": system_instruction
            },
            {
            "role": "user",
            "content": user_instruction
            }
               ]
        return message

    def get_prompt(self, task: str, *args) -> str:
        """
        Calls the appropriate method based on the task name.
        """
        task_map = {
            "reranking_llm" :self.reranking_llm,
            "reranking_llm_entities" :self.reranking_llm_entities,
            "rank_gpt": self.rank_gpt,
            "generate_answer_openai": self.generate_answer_openai,
            "generate_answer_llama": self.generate_answer_llama
        }
        if task in task_map:
            return task_map[task](*args)
        else:
            raise ValueError("Unknown task")

# Example usage
library = PromptLibrary()
# print(library.get_prompt("reranking_llm", "How do airplanes fly?",["ss", "aa"]))

