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

