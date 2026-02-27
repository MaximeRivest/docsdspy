---
title: "Reimplementing Chatadapter"
subtitle: "Understanding DSPy's prompt formatting by building it yourself"
---

# Reimplementing ChatAdapter with TemplateAdapter

DSPy's `ChatAdapter` can feel like magic — you give it a signature string like `"question -> answer"` and it somehow knows how to prompt a language model. In this guide, we'll demystify that by rebuilding `ChatAdapter`'s exact behavior using `TemplateAdapter`, which gives us full control over every message sent to the LM.

## What ChatAdapter Actually Produces

Before we rebuild it, let's see what `ChatAdapter` generates. Given this signature:

```python
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

And these demos:
```python
demos = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What color is the sky?", "answer": "Blue"},
]
```

ChatAdapter produces these messages:

```
[system]
Your input fields are:
1. `question` (str):
Your output fields are:
1. `answer` (str): often between 1 and 5 words
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Answer questions with short factoid answers.

[user]
[[ ## question ## ]]
What is the capital of France?

[assistant]
[[ ## answer ## ]]
Paris

[[ ## completed ## ]]

[user]
[[ ## question ## ]]
What color is the sky?

[assistant]
[[ ## answer ## ]]
Blue

[[ ## completed ## ]]

[user]
[[ ## question ## ]]
What is 2+2?

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
```

The key elements:
1. **System message**: field descriptions + structure template + task instruction
2. **Demo messages**: user/assistant pairs using `[[ ## field ## ]]` markers
3. **Final user message**: input values + request to respond in the same format
4. **Parsing**: extract values from `[[ ## field ## ]]` markers in the LM response

## Building It With TemplateAdapter

`TemplateAdapter` lets us define the exact messages sent to the LM. We'll use:
- Template functions (`{func_name()}`) for dynamic content generation
- The `{"role": "demos"}` directive for few-shot examples
- `parse_mode="chat"` to delegate parsing to ChatAdapter's `[[ ## field ## ]]` extractor

### The Helper Functions

First, we define helper functions that generate the dynamic parts of the prompt:

```python
def format_input_fields(ctx, signature, demos, **kwargs):
    """Render input field descriptions like ChatAdapter does."""
    lines = []
    for i, (name, field) in enumerate(signature.input_fields.items(), 1):
        desc = (getattr(field, "json_schema_extra", None) or {}).get("desc", "")
        type_name = getattr(field.annotation, "__name__", str(field.annotation))
        # Skip ${field} placeholder descriptions
        if desc and not desc.startswith("${"):
            lines.append(f"{i}. `{name}` ({type_name}): {desc}")
        else:
            lines.append(f"{i}. `{name}` ({type_name}):")
    return "\n".join(lines)


def format_output_fields(ctx, signature, demos, **kwargs):
    """Render output field descriptions like ChatAdapter does."""
    lines = []
    for i, (name, field) in enumerate(signature.output_fields.items(), 1):
        desc = (getattr(field, "json_schema_extra", None) or {}).get("desc", "")
        type_name = getattr(field.annotation, "__name__", str(field.annotation))
        if desc and not desc.startswith("${"):
            lines.append(f"{i}. `{name}` ({type_name}): {desc}")
        else:
            lines.append(f"{i}. `{name}` ({type_name}):")
    return "\n".join(lines)


def format_structure_template(ctx, signature, demos, **kwargs):
    """Render the [[ ## field ## ]] structure template."""
    lines = []
    for name in signature.input_fields:
        lines.append(f"[[ ## {name} ## ]]")
        lines.append(f"{{{name}}}")
        lines.append("")
    for name in signature.output_fields:
        lines.append(f"[[ ## {name} ## ]]")
        lines.append(f"{{{name}}}")
        lines.append("")
    lines.append("[[ ## completed ## ]]")
    return "\n".join(lines)


def format_inputs_chat_style(ctx, signature, demos, **kwargs):
    """Render input values in [[ ## field ## ]] format."""
    lines = []
    for name in signature.input_fields:
        lines.append(f"[[ ## {name} ## ]]")
        lines.append(str(ctx.get(name, "")))
    return "\n".join(lines)


def format_output_request(ctx, signature, demos, **kwargs):
    """Render the 'respond with...' instruction."""
    first_output = list(signature.output_fields.keys())[0]
    return (
        f"Respond with the corresponding output fields, starting with the field "
        f"`[[ ## {first_output} ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`."
    )
```

### The Adapter Factory

Since the demos directive needs field names baked into its template strings, we create a factory function that builds the adapter for a specific signature:

```python
from dspy_template_adapter import TemplateAdapter, Predict


def make_chat_adapter_template(signature):
    """Build a ChatAdapter-style TemplateAdapter for a specific signature."""
    
    # Build demo user template: "[[ ## question ## ]]\n{question}"
    demo_user_parts = []
    for name in signature.input_fields:
        demo_user_parts.append(f"[[ ## {name} ## ]]")
        demo_user_parts.append(f"{{{name}}}")
    demo_user_template = "\n".join(demo_user_parts)
    
    # Build demo assistant template: "[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]"
    demo_assistant_parts = []
    for name in signature.output_fields:
        demo_assistant_parts.append(f"[[ ## {name} ## ]]")
        demo_assistant_parts.append(f"{{{name}}}")
    demo_assistant_parts.append("")
    demo_assistant_parts.append("[[ ## completed ## ]]")
    demo_assistant_template = "\n".join(demo_assistant_parts)
    
    adapter = TemplateAdapter(
        messages=[
            {
                "role": "system",
                "content": (
                    "Your input fields are:\n"
                    "{format_input_fields()}\n"
                    "Your output fields are:\n"
                    "{format_output_fields()}\n"
                    "All interactions will be structured in the following way, "
                    "with the appropriate values filled in.\n\n"
                    "{format_structure_template()}\n"
                    "In adhering to this structure, your objective is: \n"
                    "        {instruction}"
                ),
            },
            {
                "role": "demos",
                "user": demo_user_template,
                "assistant": demo_assistant_template,
            },
            {
                "role": "user",
                "content": "{format_inputs_chat_style()}\n\n{format_output_request()}",
            },
        ],
        parse_mode="chat",  # Delegates to ChatAdapter.parse()
    )
    
    # Register our helper functions
    adapter.register_helper("format_input_fields", format_input_fields)
    adapter.register_helper("format_output_fields", format_output_fields)
    adapter.register_helper("format_structure_template", format_structure_template)
    adapter.register_helper("format_inputs_chat_style", format_inputs_chat_style)
    adapter.register_helper("format_output_request", format_output_request)
    
    return adapter
```

### Using It

```python
import dspy
from dspy_template_adapter import Predict

class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Create adapter for this signature
chat_style_adapter = make_chat_adapter_template(QA)

# Bind it to a Predict
qa = Predict(QA, adapter=chat_style_adapter)

# Use it
result = qa(question="What is the capital of Japan?")
print(result.answer)  # "Tokyo"
```

### Verifying It Matches

You can compare the messages side-by-side:

```python
# Our custom adapter
custom_messages = chat_style_adapter.preview(QA, demos=[], inputs={"question": "What is 2+2?"})

# Original ChatAdapter
original_messages = dspy.ChatAdapter().format(QA, demos=[], inputs={"question": "What is 2+2?"})

# They should be identical
for custom, original in zip(custom_messages, original_messages):
    assert custom["role"] == original["role"]
    assert custom["content"].strip() == original["content"].strip()
```

## What This Teaches Us

1. **ChatAdapter is just a formatter**: It transforms signatures into a specific message structure. There's no special LLM magic — just careful prompt engineering with `[[ ## field ## ]]` markers.

2. **The markers are arbitrary**: DSPy chose `[[ ## field ## ]]` as a delimiter. You could use XML tags, JSON, or anything else. The key is consistency between formatting and parsing.

3. **Adapters are composable**: We used `parse_mode="chat"` to reuse ChatAdapter's parsing logic while providing our own formatting. You could also write a custom parser.

4. **Full control is possible**: When ChatAdapter's format doesn't work for your use case (specific model quirks, domain requirements, etc.), you can build exactly what you need.

## Next Steps

Now that you understand how ChatAdapter works, you can:

- Build custom adapters for specific models (Claude prefers XML, for example)
- Create minimal prompts for reasoning models that don't need the full scaffold
- Debug prompt issues by inspecting exactly what's being sent
- Experiment with different prompt structures while keeping DSPy's optimization pipeline

See [dspy-template-adapter](https://github.com/dspy-community-org/dspy_template_adapter) for the full documentation.




```python
   chat_template = TemplateAdapter(
       messages=[
           {
               "role": "system",
               "content": """Your input fields are:
   {% for f in inputs %}
   {f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
   {% endfor %}
   Your output fields are:
   {% for f in outputs %}
   {f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
   {% endfor %}
   All interactions will be structured in the following way, with the appropriate values filled in.

   {% for f in inputs separator='\n\n' %}
   [[ ## {f.name} ## ]]
   {{{f.name}}}
   {% endfor %}

   {% for f in outputs separator='\n\n' %}
   [[ ## {f.name} ## ]]
   {f.typed_placeholder}
   {% endfor %}

   [[ ## completed ## ]]
   In adhering to this structure, your objective is:
           {instruction}""",
           },
           {
               "role": "demos",
               "user": """{% for f in inputs separator='\n\n' %}
   [[ ## {f.name} ## ]]
   {f.value}
   {% endfor %}""",
               "assistant": """{% for f in outputs separator='\n\n' %}
   [[ ## {f.name} ## ]]
   {f.value}
   {% endfor %}

   [[ ## completed ## ]]
   """,
           },
           {
               "role": "user",
               "content": """{% for f in inputs separator='\n\n' %}
   [[ ## {f.name} ## ]]
   {f.value}
   {% endfor %}

   Respond with the corresponding output fields, starting with the field {% for f in outputs
 separator=', then ' %}`[[ ## {f.name} ## ]]`{f.chat_type_hint}{% endfor %}, and then ending with
 the marker for `[[ ## completed ## ]]`.""",
           },
       ],
       parse_mode="chat",
   )
```
