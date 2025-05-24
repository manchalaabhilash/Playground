"""
Agentic RAG implementation for advanced RAG systems.
Provides agent-based retrieval and reasoning capabilities.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import json
import re

class RagAgent:
    """
    Agent that uses retrieval-augmented generation to answer queries.
    Implements agentic behavior with tools and reasoning.
    """
    
    def __init__(self, retriever, llm_client=None, tools: List[Dict[str, Any]] = None):
        """Initialize the RAG agent"""
        self.retriever = retriever
        self.llm_client = llm_client
        self.tools = tools or []
        self.memory = []
        self.max_memory_items = 10
    
    def add_tool(self, tool: Dict[str, Any], handler: Callable):
        """Add a tool that the agent can use"""
        if "name" not in tool or "description" not in tool:
            raise ValueError("Tool must have name and description")
        
        tool["handler"] = handler
        self.tools.append(tool)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using agentic RAG"""
        if not self.llm_client:
            return {
                "answer": "I don't have the capability to process this query.",
                "reasoning": "Missing LLM client.",
                "documents": []
            }
        
        # Add query to memory
        self.add_to_memory({"role": "user", "content": query})
        
        # Retrieve relevant documents
        documents = []
        if self.retriever:
            documents = self.retriever.get_relevant_documents(query, k=5)
        
        # Extract text from documents
        doc_texts = [self._get_document_text(doc) for doc in documents]
        context = "\n\n".join(doc_texts)
        
        # Prepare tool descriptions
        tool_descriptions = ""
        if self.tools:
            tool_descriptions = "Available tools:\n"
            for tool in self.tools:
                tool_descriptions += f"- {tool['name']}: {tool['description']}\n"
        
        # Generate agent response
        prompt = f"""You are an AI assistant that uses retrieval-augmented generation to answer queries.
You have access to relevant documents and tools to help you answer the query.

Query: {query}

{f'Relevant documents:\n{context}' if context else 'No relevant documents found.'}

{tool_descriptions if tool_descriptions else 'No tools available.'}

First, analyze the query and the retrieved documents. Then, determine if you need to use any tools.
If you need to use a tool, specify the tool name and parameters in the following format:
TOOL: <tool_name>
PARAMS: <parameters in JSON format>

If you don't need to use a tool, or after using tools, provide your final answer.

Please structure your response as follows:
REASONING: Your step-by-step reasoning process
ACTION: Tool usage (if needed)
ANSWER: Your final answer to the query
"""
        
        # Get initial response
        response = self.llm_client.generate_text(prompt)
        
        # Parse response
        reasoning = ""
        action = ""
        answer = ""
        
        if "REASONING:" in response:
            reasoning_part = response.split("REASONING:")[1].split("ACTION:" if "ACTION:" in response else "ANSWER:")[0]
            reasoning = reasoning_part.strip()
        
        if "ACTION:" in response:
            action_part = response.split("ACTION:")[1].split("ANSWER:")[0]
            action = action_part.strip()
            
            # Process tool action
            tool_result = self._process_tool_action(action)
            
            # If tool was used, generate a follow-up response
            if tool_result:
                follow_up_prompt = f"""Based on the tool results, please provide your final answer.

Query: {query}

Your previous reasoning: {reasoning}

Tool result: {json.dumps(tool_result, indent=2)}

Please provide your final answer:
"""
                follow_up_response = self.llm_client.generate_text(follow_up_prompt)
                answer = follow_up_response.strip()
            else:
                # If tool action failed, extract answer from original response
                if "ANSWER:" in response:
                    answer_part = response.split("ANSWER:")[1]
                    answer = answer_part.strip()
        else:
            # No tool action, extract answer
            if "ANSWER:" in response:
                answer_part = response.split("ANSWER:")[1]
                answer = answer_part.strip()
            else:
                # If no structured format, use the whole response
                answer = response
        
        # Add answer to memory
        self.add_to_memory({"role": "assistant", "content": answer})
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "action": action,
            "documents": documents
        }
    
    def _process_tool_action(self, action_text: str) -> Optional[Dict[str, Any]]:
        """Process a tool action from the agent's response"""
        # Extract tool name and parameters
        tool_match = re.search(r"TOOL:\s*(\w+)", action_text)
        params_match = re.search(r"PARAMS:\s*({.*})", action_text, re.DOTALL)
        
        if not tool_match:
            return None
        
        tool_name = tool_match.group(1)
        
        # Find the tool
        tool = next((t for t in self.tools if t["name"] == tool_name), None)
        
        if not tool or "handler" not in tool:
            return {
                "error": f"Tool '{tool_name}' not found or has no handler"
            }
        
        # Parse parameters
        params = {}
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                return {
                    "error": f"Invalid JSON parameters for tool '{tool_name}'"
                }
        
        # Call the tool handler
        try:
            result = tool["handler"](**params)
            return {
                "tool": tool_name,
                "params": params,
                "result": result
            }
        except Exception as e:
            return {
                "tool": tool_name,
                "params": params,
                "error": str(e)
            }
    
    def add_to_memory(self, item: Dict[str, str]):
        """Add an item to the agent's memory"""
        self.memory.append(item)
        
        # Trim memory if it exceeds max size
        if len(self.memory) > self.max_memory_items:
            self.memory = self.memory[-self.max_memory_items:]
    
    def get_memory(self) -> List[Dict[str, str]]:
        """Get the agent's memory"""
        return self.memory
    
    def clear_memory(self):
        """Clear the agent's memory"""
        self.memory = []
    
    def _get_document_text(self, doc: Any) -> str:
        """Extract text content from document"""
        if hasattr(doc, "page_content"):
            return doc.page_content
        elif isinstance(doc, dict) and "content" in doc:
            return doc["content"]
        else:
            return str(doc)


class MultiAgentRag:
    """
    Implements a multi-agent RAG system with specialized agents for different tasks.
    Coordinates multiple agents to handle complex queries.
    """
    
    def __init__(self, agents: Dict[str, RagAgent] = None, router=None, llm_client=None):
        """Initialize the multi-agent RAG system"""
        self.agents = agents or {}
        self.router = router
        self.llm_client = llm_client
        self.conversation_history = []
    
    def add_agent(self, name: str, agent: RagAgent):
        """Add an agent to the system"""
        self.agents[name] = agent
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using the appropriate agent(s)"""
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # If no agents, return error
        if not self.agents:
            result = {
                "answer": "I don't have any agents to process your query.",
                "agent": None,
                "sub_queries": []
            }
            self.conversation_history.append({"role": "assistant", "content": result["answer"]})
            return result
        
        # If only one agent, use it directly
        if len(self.agents) == 1:
            agent_name = next(iter(self.agents))
            agent = self.agents[agent_name]
            
            agent_result = agent.process_query(query)
            
            result = {
                "answer": agent_result["answer"],
                "reasoning": agent_result.get("reasoning", ""),
                "agent": agent_name,
                "sub_queries": []
            }
            
            self.conversation_history.append({"role": "assistant", "content": result["answer"]})
            return result
        
        # For multiple agents, use router if available
        if self.router and self.llm_client:
            return self._route_and_process(query)
        
        # Fallback to using the first agent
        agent_name = next(iter(self.agents))
        agent = self.agents[agent_name]
        
        agent_result = agent.process_query(query)
        
        result = {
            "answer": agent_result["answer"],
            "reasoning": agent_result.get("reasoning", ""),
            "agent": agent_name,
            "sub_queries": []
        }
        
        self.conversation_history.append({"role": "assistant", "content": result["answer"]})
        return result
    
    def _route_and_process(self, query: str) -> Dict[str, Any]:
        """Route the query to appropriate agent(s) and process it"""
        # Generate routing plan
        routing_plan = self._generate_routing_plan(query)
        
        # If plan indicates a single agent, use it directly
        if not routing_plan.get("is_complex", False):
            agent_name = routing_plan.get("agent", next(iter(self.agents)))
            
            # Use specified agent if it exists, otherwise use first agent
            if agent_name in self.agents:
                agent = self.agents[agent_name]
            else:
                agent_name = next(iter(self.agents))
                agent = self.agents[agent_name]
            
            agent_result = agent.process_query(query)
            
            result = {
                "answer": agent_result["answer"],
                "reasoning": agent_result.get("reasoning", ""),
                "agent": agent_name,
                "sub_queries": []
            }
            
            self.conversation_history.append({"role": "assistant", "content": result["answer"]})
            return result
        
        # For complex queries, process each sub-query with the appropriate agent
        sub_results = []
        
        for sub_query_info in routing_plan.get("sub_queries", []):
            sub_query = sub_query_info.get("query", "")
            agent_name = sub_query_info.get("agent", next(iter(self.agents)))
            
            # Use specified agent if it exists, otherwise use first agent
            if agent_name in self.agents:
                agent = self.agents[agent_name]
            else:
                agent_name = next(iter(self.agents))
                agent = self.agents[agent_name]
            
            # Process sub-query
            sub_result = agent.process_query(sub_query)
            
            sub_results.append({
                "query": sub_query,
                "answer": sub_result["answer"],
                "reasoning": sub_result.get("reasoning", ""),
                "agent": agent_name
            })
        
        # Synthesize final answer
        final_answer = self._synthesize_answers(query, sub_results)
        
        result = {
            "answer": final_answer,
            "sub_queries": sub_results,
            "is_complex": True
        }
        
        self.conversation_history.append({"role": "assistant", "content": result["answer"]})
        return result
    
    def _generate_routing_plan(self, query: str) -> Dict[str, Any]:
        """Generate a routing plan for the query"""
        if not self.llm_client:
            # Simple fallback if no LLM client
            return {
                "is_complex": False,
                "agent": next(iter(self.agents)),
                "sub_queries": []
            }
        
        # List available agents
        agent_descriptions = "\n".join([f"- {name}: {agent.__class__.__name__}" for name, agent in self.agents.items()])
        
        # Prompt for the LLM to generate a routing plan
        prompt = f"""You are an AI coordinator that routes queries to specialized agents.
Given the following query and available agents, determine if the query is complex (requires multiple agents).
If it is complex, break it down into sub-queries and assign each to an appropriate agent.

Query: {query}

Available agents:
{agent_descriptions}

Output your response in the following JSON format:
{{
  "is_complex": true/false,
  "agent": "agent_name" (if not complex),
  "sub_queries": [
    {{
      "query": "Sub-query 1",
      "agent": "agent_name_1"
    }},
    {{
      "query": "Sub-query 2",
      "agent": "agent_name_2"
    }}
  ] (if complex)
}}
"""
        
        try:
            response = self.llm_client.generate_text(prompt)
            plan = json.loads(response)
            return plan
        except Exception as e:
            print(f"Error in routing plan generation: {str(e)}")
            # Simple fallback
            return {
                "is_complex": False,
                "agent": next(iter(self.agents)),
                "sub_queries": []
            }
    
    def _synthesize_answers(self, original_query: str, sub_results: List[Dict[str, Any]]) -> str:
        """Synthesize a final answer from sub-query results"""
        if not self.llm_client or not sub_results:
            # Simple fallback
            return sub_results[0]["answer"] if sub_results else "I couldn't process your query."
        
        # Format sub-results
        sub_answers = "\n\n".join([
            f"Sub-query: {result['query']}\nAgent: {result['agent']}\nAnswer: {result['answer']}"
            for result in sub_results
        ])
        
        # Prompt for the LLM to synthesize a final answer
        prompt = f"""You are an AI assistant that synthesizes answers from multiple sources.
Given the following original query and answers to sub-queries, provide a comprehensive final answer.

Original query: {original_query}

Answers to sub-queries:
{sub_answers}

Please synthesize a comprehensive final answer that addresses the original query:
"""
        
        try:
            final_answer = self.llm_client.generate_text(prompt)
            return final_answer
        except Exception as e:
            print(f"Error in answer synthesis: {str(e)}")
            # Simple fallback
            return sub_results[0]["answer"] if sub_results else "I couldn't process your query."
