## Pipeline Tutorial

### Step 0: Obtain Smithery API Credentials and Configuration

To enable MCP server connections, you will need access to the Smithery API.

- Log in to your Smithery account and obtain both your (a) Smithery API key(s) and (b) configuration details.
- Save this information in the [`smithery_api_pool.json`](./smithery_api_pool.json) file.

For optimal performance and faster inference, we recommend configuring at least 8 API keys.

### Step 1: Question Synthesis

**1.1 Generate Question Prompts**

This step creates structured prompts for tool-use question generation. You can control how many questions to generate, how tools/servers are selected, and whether questions involve single or multiple servers. The script supports various sampling strategies (random, uniform, power law, featured) and allows for both single-server and multi-server scenarios. The generated prompts will be used as input for the next step in the pipeline.

```
1. Single Server Mode (tools from one server per question):
   1.1. Random sampling (default) - 1000 questions from random servers:
        python step1.1_gen_questions.py --total_prompts 1000 --sampling_strategy random
   
   1.2. Uniform sampling - 5 questions per server (total = 5 * num_servers):
        python step1.1_gen_questions.py --sampling_strategy uniform --samples_per_server 5
   
   1.3. Power law sampling - favors popular servers based on usage rank:
        python step1.1_gen_questions.py --total_prompts 2000 --sampling_strategy power_law
   
   1.4. Featured sampling - only select from featured servers:
        python step1.1_gen_questions.py --total_prompts 1000 --sampling_strategy featured
   
   1.5. Generate questions for 2-tool scenarios with uniform sampling:
        python step1.1_gen_questions.py --num_tools 2 --sampling_strategy uniform --samples_per_server 10

2. Multi-Server Mode (tools from multiple servers per question):
   2.1. Random server combinations:
        python step1.1_gen_questions.py --mode multi_server --num_tools 2 --total_prompts 1000
   
   2.2. Servers from same category (same primary label):
        python step1.1_gen_questions.py --mode multi_server --num_tools 3 --multi_server_allocation_strategy same_primary_label
   
   2.3. Servers from different categories (different primary labels):
        python step1.1_gen_questions.py --mode multi_server --num_tools 3 --multi_server_allocation_strategy different_primary_labels
   
   2.4. LLM-driven server selection from featured servers (brainstorm context first):
        python step1.1_gen_questions.py --mode multi_server --num_tools 3 --multi_server_allocation_strategy random_featured
   
   2.5. Multi-server with custom server allocation:
        python step1.1_gen_questions.py --mode multi_server --num_tools 4 --multi_server_allocated_servers 3 --total_prompts 500

Key Parameters:
- --sampling_strategy: Controls server selection (random/uniform/power_law/featured) - ONLY for single_server mode
- --mode: single_server (individual servers) vs multi_server (server combinations)  
- --multi_server_allocation_strategy: For multi_server mode (random/same_primary_label/different_primary_labels/random_featured)
- --num_tools: Number of tools to include in each prompt
- --samples_per_server: For uniform sampling, questions per server
```

**1.2 Generate Questions**

In this step, you can select from a variety of language models to generate questions, depending on your requirements for quality, speed, or resource usage. Specify the model name as the second argument to the script. For example, you might use a large instruction-tuned model like `mistralai/Mistral-Small-3.2-24B-Instruct-2506`.

```
bash step1.2_completion.sh {Question File Generated From Step 1.1} {Model Name}
```

**1.3 Process Completion**

This step processes and sanitizes the raw model completions generated in Step 1.2. It removes duplicates, cleans up formatting, and prepares the questions for downstream quality assessment. The script supports deduplication using sentence embeddings, configurable thresholds, and can optionally append tool hints to each question.

```
python step1.3_process_completion.py --input_file {Response File Generated From Step 1.2}
```

### Step 2: Question Quality Check

**2.1 Generate Question Quality Check Prompts**

This step generates quality assessment prompts for each question, which will later be used to evaluate the quality, clarity, and realism of the generated tool-use questions. The script takes as input the processed questions from Step 1.3 and outputs a new file containing prompts formatted for LLM-based quality annotation. You can specify which quality criteria to assess (difficulty, quality, realism, or all), and optionally enable debug mode to process only a subset of entries.

```
python step2.1_question_quality_check.py --input_file {File Generated From Step 1.3}
```

**2.2 Generate Question Quality Check Responses**

In this step, we generate quality check responses for the questions using a language model. By default, the script utilizes the quantized Kimi-K2 model (`RedHatAI/Kimi-K2-Instruct-quantized.w4a16`) for efficient and high-quality annotation of question quality. You can specify a different model if desired by providing its name as the second argument to the script.

```
step2.2_completion_quality_check.sh {File Generated From Step 2.1}
```

**2.3 Process Completion**

This step processes the raw quality check responses generated in Step 2.2. It extracts the LLM's ratings and reasoning for each question across multiple quality dimensions (such as tool selection difficulty, uniqueness, question quality, scenario realism, verifiability, and stability). The script aggregates these results, allowing for further filtering or analysis, and can optionally save the top-rated questions for downstream use.


```
python step2.3_process_completion.py --input_file {Response File Generated From Step 2.2}
```

### Step 3: Generate Agentic Trajectories

**3.1 Generate Agent Trajectories via Qwen/OpenAI Agents**

This step generates agentic trajectories for each question using advanced agent frameworks such as Qwen or OpenAI Agents. The script takes as input the processed and quality-checked questions from Step 2.3 and outputs detailed agent trajectories, which may include tool calls, intermediate reasoning, and final answers. You can specify the agent framework and model to use by modifying the script arguments.

```
bash step3.1_completion_agent.sh {File Generated From Step 2.3}
```

**3.2 Process Completion**

This step applies rule-based filtering to the agentic trajectories generated in Step 3.1. It analyzes each trajectory and removes those that contains failure patterns.

```
python step3.2_process_completion.py --input_file {Response File Generated From Step 3.1}
```

### Step 4: Response Quality Check

**4.1 Generate Response Quality Check Prompts**

This step generates quality assessment prompts for each agentic response, enabling evaluation of the correctness and completeness of the agent's answers. The script takes as input the filtered agentic trajectories from Step 3.2 and outputs a new file containing prompts formatted for LLM-based response quality annotation. 

```
python step4.1_response_quality_check.py --input_file {File Generated From Step 3.2}
```

**4.2 Generate Response Quality Check Responses**

In this step, we generate quality check responses for the agentic outputs using a language model. By default, the script utilizes the quantized OSS-120B model (`openai/gpt-oss-120b`) for efficient and high-quality response annotation. You can specify a different model by providing its name as the second argument to the script.

```
step4.2_completion_response_check.sh {File Generated From Step 4.1}
```

**4.3 Process Completion**

This step processes the raw response quality check outputs generated in Step 4.2. It extracts the LLM's ratings and reasoning for each agentic response across multiple quality dimensions, such as completeness, conciseness, and tool call accuracy. The script aggregates these results, computes tool usage correctness, and prepares the data for further filtering.

```
python step4.3_process_completion.py --input_file {Response File Generated From Step 4.2}
```