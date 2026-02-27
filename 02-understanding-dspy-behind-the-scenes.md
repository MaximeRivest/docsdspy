---
title: "Understanding Dspy Behind The Scenes"
subtitle: "What actually happens when you call a DSPy module"
description: |
  A detailed walkthrough of DSPy's internal execution flow, from module call 
  to LM request and back. Covers how Predict, Adapter, and LM layers interact,
  with concrete examples for single calls, ChainOfThought, and custom modules.
abstract: |
  When you call a DSPy module, several layers work together: your module 
  orchestrates the task, Predict manages the signature and wraps results,
  the Adapter formats inputs into structured messages and parses outputs,
  and the LM client sends requests to the API. This document traces that 
  flow step-by-step so you understand exactly what DSPy does for you.
author: "Maxime Rivest"
date: February 27 2026
categories: [internals, architecture, tutorial]
keywords: [dspy, module, predict, adapter, lm, flow, architecture]
toc: true
toc-depth: 3
---
>
> ## Who is this for?
> You've written your first DSPy program and it works. Now you want to understand *why* it works — what's happening under the hood when you call a module.

---

## The Core Flow

Every DSPy module eventually goes through the same path: **Module → Predict → Adapter → LM**.

```
User calls module(inputs)
    → Module.__call__
    → Module.forward (your logic / submodule calls)
        ↓
        calls Predict (or ChainOfThought, ReAct, which must themselves call Predict etc.)
            → Predict.forward
                → adapter.format()      → builds structured messages
                → lm(messages=...)      → hits API, gets raw text  
                → adapter.parse()       → parses text into dict
                → _forward_postprocess  → wraps into Prediction
            → returns Prediction
        ↓
        (may call more submodules...)
        ↓
    → returns final result
```

---

## What Each Layer Does

| Layer | Responsibility |
|---{20%}|---|
| **Module** | Orchestrates the task — decides which submodules to call and in what order |
| **Predict** | The base unit — holds a signature, calls the adapter, returns a Prediction |
| **Adapter** | Formats signature/demos/inputs into structured messages, parses raw LM output back into fields |
| **LM** | Sends messages to the API (via litellm), returns raw text |

---

## Example: Single Predict Call

```python
qa = dspy.Predict("question -> answer")
pred = qa(question="What is 2+2?")
```

Flow:

```
qa(question="What is 2+2?")
    → Predict.__call__
    → Predict.forward
        → adapter.format(signature, demos, inputs)
            → builds messages like:
              [
                {"role": "system", "content": "Your input fields are..."},
                {"role": "user", "content": "[[ ## question ## ]]\nWhat is 2+2?"}
              ]
        → lm(messages=...)
            → hits OpenAI/Anthropic/etc API
            → returns raw text: "[[ ## answer ## ]]\n4"
        → adapter.parse(raw_text)
            → extracts {"answer": "4"}
        → _forward_postprocess
            → Prediction.from_completions({"answer": "4"})
    → returns Prediction(answer="4")
```

---

## Example: ChainOfThought

`ChainOfThought` doesn't have special inference logic. It just modifies the signature to ask for reasoning first, then uses normal Predict.

```python
cot = dspy.ChainOfThought("question -> answer")
pred = cot(question="What is 2+2?")
```

What ChainOfThought does internally:

```python
# Your signature
"question -> answer"

# Gets transformed to
"question -> reasoning, answer"
```

Flow:

```
cot(question="What is 2+2?")
    → ChainOfThought.forward
        → self.predict(...)          # Predict with modified signature
            → Predict.forward
                → adapter.format()   # includes "reasoning" field
                → lm(messages=...)
                → adapter.parse()    # extracts reasoning + answer
            → Prediction(reasoning="...", answer="4")
    → returns Prediction
```

Result:

```python
print(pred.reasoning)  # "Let's see, 2+2 equals 4 because..."
print(pred.answer)     # "4"
```

---

## Reimplementing ChainOfThought stack

I find it easiest to learn by looking at concrete examples. Here, we will implement simplified but correct  versions of the whole ChainOfThought call dspy call stack. With reasoning llms Chain of Thought is generally no longer useful, but its simple and known enough to be the perfect candidate for this example.

We will implement it from the top down. Remember the order is: `Module → Predict → Adapter → LM`

### Module

So how is the `dspy.ChainOfThought` module implemented?

It is essentially just that:

```python
def make_cot(YOUR_SIGNATURE_STRING):
    return dspy.Predict(
      dspy.Signature(YOUR_SIGNATURE_STRING).prepend(
        name="reasoning",
        field=dspy.OutputField(),
        type_=str
    ))

my_cot = make_cot("question -> answer")
```

Which is saying: take a signature string, prepend a new `reasoning: str` OutputField in front of all other OutputFields, and wrap that in `dspy.Predict`. This is complete and sufficient to it callable and a target for optimization.

**Usage:**

```python
pred = my_cot(question="What is 2+2?")

print(pred.reasoning)  # "First, I need to add 2 and 2..."
print(pred.answer)     # "4"
```

**Why does this work?** `dspy.Predict` is already a module. It's callable, optimizable, and complete. The function is just a factory that builds one with a modified signature.

While that way of creating ChainOfThought works it is not the unsual and standard way of doing it. Because many (maybe most) modules need to implement their custom logic when the module is called the more extensive module subclassing is used. Here is an example of that:

```python
class MyChainOfThought(dspy.Module):

    def __init__(self, signature):
        super().__init__()

        self.predict = dspy.Predict(dspy.Signature(signature).prepend(
            name="reasoning",
            field=dspy.OutputField(),
            type_=str
        ))

    def forward(self, **kwargs):
        return self.predict(**kwargs)
```

**Both approaches give you a `dspy.Module`:**
| Approach | What you get | Why it's a Module |
|---|---|---|
| `make_cot("q -> a")` | `dspy.Predict` instance | `Predict` inherits from `Module` |
| `MyChainOfThought("q -> a")` | `MyChainOfThought` instance | `MyChainOfThought` inherits from `Module` |

The difference is just where the Module comes from:

- **Function factory**: the `Predict` itself *is* your module  
- **Class**: your class *is* the module, and it *contains* a `Predict`

Both are callable, both are optimizable, both can be nested inside other modules. The class form is standard because most real modules need custom logic in `forward()` — like multiple calls, loops, or post-processing.

Every path eventually goes through `Predict → Adapter → LM`. Modules (ChainOfThought, ReAct, your custom classes) just orchestrate *when* and *how many times* that happens.`dspy.Predict` does it in 1 LM call, `dspy.ChainOfThought` also does it in 1 LM call (with modified signature) and `dspy.ReAct` does it in multiple LM calls (reasoning + tool use loop). Your custom module can do it in as many LM calls as you design and can contain any other dspy module and any other custom module.

But, maybe here you are wondering, why not defining that *orchestration of calls* in a traditional python function? You can! But you will loose on the ability of the optizisers to optimize all modules so that they work together to give you the best performance on you whole program (you user facing module, the one you collect a set of training examples for). If you dont plan on doing that you don't need to use dspy special container (the module) and you can keep things simple and use python container (the function).

### Predict

Let's now move down the stack of dspy calls to `Predict`.

`Predict` is where the actual LM call happens. Every module, no matter how complex, eventually calls a `Predict` to talk to the language model.

When you create a Predict instance (`dspy.Predict("question -> answer")`), it stores your signature and any config you provide (like temperature). It also initializes empty slots — `self.lm`, `self.demos`, `self.traces` — that start as `None` or empty. These slots are where optimizers will later put their improvements: better demos, tuned instructions, or even a fine-tuned model. Visit [[how-does-optimization-works]] for the complete guide on that. In short, Predict is an object that stores optimizable parameters.

When you call a Predict, it first figures out which LM to use. It checks `self.lm` first, if an optimizer set a specific model for this predictor, it uses that. Otherwise,  it falls back to the global `settings.lm`. It's the same pattern for any optimizable parameter. For instance, for demos: it uses `self.demos` if an optimizer filled them in, otherwise it uses nothing. While I am talking about the optimizers, you could set these things yourself and upon a call your Predict instance uses that.

*Setting lm or demos yoursel by hand:*
 ```python
# Create a Predict instance
qa = dspy.Predict("question -> answer")

# Manually set demos (few-shot examples)
qa.demos = [
   {"question": "What is 2+2?", "answer": "4"},
   {"question": "What is the capital of France?", "answer": "Paris"},
]

# Manually set a specific LM for this predictor
qa.lm = dspy.LM("anthropic/claude-3-sonnet")

# Now when you call it, it uses your demos and LM
pred = qa(question="What is 3+3?")
 ```

*Passing at call time, no storage:*
 ```python
pred = qa(
   question="What is 3+3?",
   demos=[{"question": "What is 2+2?", "answer": "4"}],
   lm=dspy.LM("openai/gpt-4o"),
)
 ```

Once Predict has figured out what parameters (lm, demos, instructions, etc) to use for this call, it hands everything to the adapter and receive back the parsed lm response from the adapter. Finally, Predict wraps the parsed output into a `Prediction` object and records a trace. That trace is a record of what predictor was called, with what inputs, and what it returned. This is how optimizers learn which examples worked well.

One detail worth noting: Predict inherits from both `Module` and `Parameter`. Module makes it callable. Parameter marks it as something optimizers should look for. When an optimizer calls `program.named_parameters()`, it finds every Predict in your program because of that Parameter inheritance.

Here's a simplified but accurate version of what `Predict` does. This is close to the real implementation, with non-essential parts removed for clarity:

```python
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction


class MyPredict(Module, Parameter):

    def __init__(self, signature, **config):
        super().__init__()

        # The signature defines inputs and outputs (e.g., "question -> answer")
        self.signature = signature

        # Config holds LM settings like temperature, max_tokens, etc.
        # These get passed to the LM at call time.
        self.config = config

        # Reset initializes the learnable/optimizable state.
        # Optimizers will fill in self.demos with few-shot examples,
        # and may set self.lm to a specific (or fine-tuned) model.
        self.reset()

    def reset(self):
        self.lm = None       # Can be set per-predictor, otherwise uses global settings.lm
                             # Supports any LM: different providers (openai, anthropic, ollama),
                             # different models, or fine-tuned versions (set by BootstrapFinetune)
        self.demos = []      # Few-shot examples, filled by optimizers like BootstrapFewShot
        self.train = []      # Training examples collected during optimization
        self.traces = []     # Execution traces for debugging/optimization
        # self.adapter = None  # Could also be optimizable (try ChatAdapter vs JSONAdapter, etc.)
                              # Currently DSPy doesn't optimize adapters, but the pattern would be the same

    # Makes the instance callable.
    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def forward(self, **kwargs):
        # 1. Preprocess: get the LM, config, signature, demos
        lm, config, signature, demos = self._forward_preprocess(**kwargs)

        # 2. Get the adapter
        adapter = settings.adapter or ChatAdapter()

        # 3. Call the adapter: format → LM call → parse
        completions = adapter(
            lm,
            lm_kwargs=config,
            signature=signature,
            demos=demos,
            inputs=kwargs,
        )

        # 4. Wrap results in a Prediction and record the trace
        return self._forward_postprocess(completions, signature, **kwargs)
```

So, Predict is the layer at which 'training out'(optimization) of different things (parameters) can happen because `Predict` can run using different language model weights, few shot examples and prompts without ever changing it's code, only the values stored in those parameters are changed.

Here is a shortlist of what different optimizers change:
| What changes | Where it lives | Changed by |
|---|---|---|
| Instructions | `predict.signature.instructions` | `MIPROv2`, `GEPA` |
| Few-shot demos | `predict.demos` | `BootstrapFewShot`, `LabeledFewShot` |
| The LM itself | `predict.lm` | `BootstrapFinetune` (swaps in fine-tuned model) |

The `forward()` method stays the same — it just uses whatever signature, demos, and LM are currently stored.

### Adapter

Now let's go one level deeper. What does the adapter actually do?

The adapter is the bridge between DSPy's structured world (signatures, fields, types) and the LLM's raw world (text messages in, text out). To make its life easier adapters in dspy also format into the openai standard list of messages and leverage litellm to do the translation between that format and the one required by your target provider (as specified in the string you provided to `LM`). 

Adapters of 3 steps, first it formats, second it communicates to the inference endpoint, third it parses.

```
adapter(...)
    → adapter.format()   # signature + demos + inputs → list of messages
    → lm(messages=...)   # send to API, get raw text back
    → adapter.parse()    # raw text → dict of field values
    → return completions
```

When first coming to dspy many people feel it is *magical* that a short signature string is enough to 'prompt' a language model the perform a specific task for them. Some even feel its too magical and dislike how much control and understanding they loose to dspy. Making your own adapter is where you can all that control, although, experience have shown me that the default dspy adapter generally perform equally as well as my own adapters or my own prompt (before optimization that is). Pedogically, I find that very helpful to know how to make my own adapter and in some advance cases (5 to 10% of the time, i need my own). 

TK


### LM

The final layer. The LM client sends messages to the actual API.

```python
raw_outputs = lm(messages=messages, **lm_kwargs)
```

Inside, it's essentially:

```python
class MyLM:

    def __init__(self, model):
        self.model = model  # e.g., "openai/gpt-4o-mini"

    def __call__(self, messages, **kwargs):
        # Call litellm (handles all providers: OpenAI, Anthropic, local, etc.)
        response = litellm.completion(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        # Extract the text from the response
        return [{"text": choice.message.content} for choice in response.choices]
```

DSPy uses [litellm](https://github.com/BerriAI/litellm) under the hood, which provides a unified interface to 100+ LLM providers. So when you write:

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
# or
dspy.configure(lm=dspy.LM("anthropic/claude-3-sonnet"))
# or
dspy.configure(lm=dspy.LM("ollama/llama3"))
```

The same `Predict` code works with all of them.

---

## Putting It All Together

Now you've seen every layer. Here's the complete flow for a single call:

```python
cot = dspy.ChainOfThought("question -> answer")
pred = cot(question="What is 2+2?")
```

```
1. ChainOfThought.__call__
   └→ ChainOfThought.forward
      └→ self.predict(question="What is 2+2?")

2. Predict.__call__
   └→ Predict.forward
      ├→ lm = dspy.settings.lm
      ├→ adapter = dspy.ChatAdapter()
      └→ completions = adapter(lm, signature, demos, inputs)

3. Adapter.__call__
   ├→ messages = adapter.format(signature, demos, inputs)
   │     → [{"role": "system", ...}, {"role": "user", ...}]
   ├→ raw_outputs = lm(messages)
   └→ completions = adapter.parse(raw_outputs)
         → [{"reasoning": "...", "answer": "4"}]

4. Back in Predict.forward
   └→ return Prediction.from_completions(completions)
         → Prediction(reasoning="...", answer="4")

5. Back in ChainOfThought.forward
   └→ return pred

6. User receives Prediction(reasoning="...", answer="4")
```

Every DSPy program, no matter how complex, is just this flow repeated — once for simple calls, many times for multi-step programs.

