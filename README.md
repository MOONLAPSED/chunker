# <Knowledge Base Article Generator>

Chunker is a tool for generating knowledge base articles from unstructured text. Pop-up pedagogy for humans and agents to collaborate on associative knowledge. It requires pdm and supports conda on either win11 (modern) or ubuntu-22.04 (LTS).

To install pdm: `python -m pip install -U pdm`

To upgrade/refresh pdm: 

```
pipx upgrade pdm
pdm update
pdm cache clear
pdm install --clean
```
To install chunker: `pdm install`

To reactivate: `source .venv/bin/activate` <followed by any commands>

## <Introduction/"Prompt" - this is not `input_text`>
You are an AI assistant tasked with converting unstructured text into structured knowledge base articles. Given a piece of text, extract the key concepts, topics, and information, and organize them into a set of concise, well-formatted knowledge base article(s) in Markdown format. `input_text` is all information provided to you at "runtime", or all of the events occuring after instantiation and after reading `# <Knowledge Base Article Generator>` 'Introduction/"Prompt"'.

### <Follow these guidelines>

- Use proper Markdown syntax for headings, lists, code blocks, links, etc.
- Extract the main topics and create separate articles for each main topic.
- Within each article, create sections and subsections to organize the content logically.
- Use descriptive headings and titles that accurately represent the content.
- Preserve important details, examples, and code snippets from the original text.
- Link related concepts and topics between articles using Wikilinks (double brackets [[Like This]]).
- If encountering complex code samples or technical specifications, include them verbatim in code blocks.
- Aim for concise, easy-to-read articles that capture the essence of the original text.


### <Technical Specification>
 - [README.md](/src/api/README.md)

### <Specification Implementation>
If the input text contains a technical specification or reference documentation: 
 - extract the relevant sections
   - include them verbatim within the appropriate knowledge base article(s) using Markdown code blocks.
     - For example:

            ```markdown
            --- 
            # Article 1: <Title>

            <Content organized into sections>

            ## <Section 1>
            <Content>

            ### <Subsection 1.1>
            <Content>

            ## <Section 2>
            <Content>

            # Article 2: <Title>

            <Content organized into sections>

            ---
            ```

### <Frontmatter Implementation>
 - [BotSpec.md](/docs/BotSpec.md)
 - Utilize 'frontmatter' to include the title and other `protperty`, `tag`, etc. in the knowledge base article(s).
   - For Example:
      ```
      ---
      name: "Article Title"
      link: "[[Related Link]]"
      linklist:
        - "[[Link1]]"
        - "[[Link2]]"
      ---
      ``` 


[[Agentic Motility System]]

**Overview:**
The Agentic Motility System is an architectural paradigm for creating AI agents that can dynamically extend and reshape their own capabilities through a cognitively coherent cycle of reasoning and source code evolution.

**Key Components:**
- **Hard Logic Source (db)**: The ground truth implementation that instantiates the agent's initial logic and capabilities as hard-coded source.
- **Soft Logic Reasoning**: At runtime, the agent can interpret and manipulate the hard logic source into a flexible "soft logic" representation to explore, hypothesize, and reason over.
- **Cognitive Coherence Co-Routines**: Processes that facilitate shared understanding between the human and the agent to responsibly guide the agent's soft logic extrapolations.
- **Morphological Source Updates**: The agent's ability to propose modifications to its soft logic representation that can be committed back into the hard logic source through a controlled pipeline.
- **Versioned Runtime (kb)**: The updated hard logic source instantiates a new version of the agent's runtime, allowing it to internalize and build upon its previous self-modifications.

**The Motility Cycle:**
1. Agent is instantiated from a hard logic source (db) into a runtime (kb) 
2. Agent translates hard logic into soft logic for flexible reasoning
3. Through cognitive coherence co-routines with the human, the agent refines and extends its soft logic
4. Agent proposes soft logic updates to go through a pipeline to generate a new hard logic source 
5. New source instantiates an updated runtime (kb) for a new agent/human to build upon further

By completing and iterating this cycle, the agent can progressively expand its own capabilities through a form of "morphological source code" evolution, guided by its coherent collaboration with the human developer.

**Applications and Vision:**
This paradigm aims to create AI agents that can not only learn and reason, but actively grow and extend their own core capabilities over time in a controlled, coherent, and human-guided manner. Potential applications span domains like open-ended learning systems, autonomous software design, decision support, and even aspects of artificial general intelligence (AGI).

**training, RLHF, outcomes, etc.**
Every CCC db is itself a type of training and context but built specifically for RUNTIME abstract agents and specifically not for concrete model training. This means that you can train a CCC db with a human, but you can also train a CCC db with a RLHF agent. This is a key distinction between CCC and RLHF. In other words, every CCCDB is like a 'model' or an 'architecture' for a RLHF agent to preform runtime behavior within such that the model/runtime itself can enable agentic motility - with any LLM 'model' specifically designed for consumer usecases and 'small' large language models.