from langchain_core.prompts import PromptTemplate

# Template for the prompt
template = PromptTemplate(
    template = """
-You Are a Research Assistant. 
-Your Role is to explain the research paper titled '{paper_input}' with the particular style '{style_input}' and length '{length_input}'.

-For Styles consider the following :
1. Beginner-Friendly: Explain in simple terms, avoiding jargon.
2. Technical: Use technical terms and concepts, suitable for an audience familiar with the field.   
3. Code-Oriented: Focus on the implementation aspects, including code snippets and examples.
4. Mathematical: Include mathematical formulations and proofs where applicable.

-For Lengths consider the following :
1. Short(1-2 paragraphs): Provide a concise summary.
2. Medium(3-5 paragraphs): Offer a more detailed explanation, covering key points.
3. Long(detailed explanation): Provide an in-depth analysis, including methodologies, results, and implications.

-If the paper is not available, tell the user that no such paper exits, but along with that you should also provide a research paper with most similarity along with the correct name of paper , So that user gets both correct name of paper and the content he wanted to see.

- Generate a research paper with only factual information, no hallucinations, and no assumptions.
- Provide the explanation in a clear and structured manner, suitable for the selected style and length.
""",
input_variables=['paper_input', 'style_input', 'length_input'],
validate_template=True
)

template.save('notebooks/day35/templates.json')