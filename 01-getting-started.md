---
title: "Getting Started"
---
DSPy is a Python library for building AI programs. In simple terms, an AI program is a repeatable task solved using one or more structured messages to a language model.

 ###### Repeatable Tasks

 When building an AI program, you start by defining your signature: the inputs you'll provide and
 the outputs you expect back. This is your task contract, its what goes in, and what comes out.

*Example signature:*  
```text
english_text, target_language -> translated_text
```

 Once you have a signature, you choose a module — a strategy for how the AI should approach the
 task. DSPy provides several built-in modules:
 
 - `dspy.Predict` — just answer directly
 - `dspy.ChainOfThought` — think step-by-step before answering
 - `dspy.ReAct` — reason and use tools, in a loop.
 - `dspy.RLM` — explore large contexts recursively

In this case, the simple `Predict` will do.

*Example program:*
```python
my_first_program = dspy.Predict("english_text, target_language -> translated_text")
```

*Example program usage:*
```python
my_first_program(
    english_text = "What is the capital of Canada?",
    target_language = "French"
)
```

Behind the scenes, that program is turned into a prompt, sent to a language model, the language model responds, and the response is parsed into a Python data type ready to use programmatically.

For more on: [[understanding-dspy-behind-the-scenes]]

---

## 1) Core DSPy Primitives

You only need a few concepts to start:

1. **Signature**: input/output contract (`"question -> answer"`).
2. **Module**: strategy that executes a signature (e.g., `dspy.Predict`, `dspy.ChainOfThought`).
3. **Program**: composition of modules for your task.
4. **Metric function**: scores program outputs.
5. **Optimizer**: improves your program using data + metric.

---

## 2) Minimal Example (One Module)

```python
import dspy

# Configure an LM (example provider/model string)
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

qa = dspy.Predict("question -> answer")   # module
pred = qa(question="What is the capital of France?")
print(pred.answer)
```

This is a **module call**, not yet a full multi-step program.

---

## 3) Building a Multi-Step Program (Step by Step)

Let's build a translator that: corrects typos → translates → preserves original formatting.

We'll build it piece by piece, then show the class syntax at the end.

---

### Step 1: Define Your Signatures

Signatures are your task contracts. Define them first:

```python
# Simple signatures (string form)
correct_signature = "original_text -> corrected_text"
translate_signature = "corrected_text, target_language -> translated_text"

# Signature with instructions (when you need to be specific)
apply_formatting_signature = dspy.Signature(
    "original_text, translated_text -> final_text",
    instructions="""
        Apply the formatting style from the original text
        (capitalization, markdown, punctuation) to the translated text.
    """
)
```

---

### Step 2: Create an Empty Module

A module is just a container. Create one:

```python
translator = dspy.Module()
```

---

### Step 3: Attach Submodules

Now attach a `Predict` for each signature:

```python
translator.correct = dspy.Predict(correct_signature)
translator.translate = dspy.Predict(translate_signature)
translator.apply_original_formatting = dspy.Predict(apply_formatting_signature)
```

Each of these is a **submodule** — a building block inside your program.

---

### Step 4: Define the Forward Method

The `forward` method defines *how* the submodules connect:

```python
def translator_forward(self, original_text: str, target_language: str):

    corrected = self.correct(
        original_text=original_text
    ).corrected_text

    translated = self.translate(
        corrected_text=corrected,
        target_language=target_language
    ).translated_text

    final = self.apply_original_formatting(
        original_text=original_text,
        translated_text=translated
    ).final_text

    return dspy.Prediction(final_text=final)

# Attach it to the module
import types
translator.forward = types.MethodType(translator_forward, translator)
```

---

### Step 5: Use It

```python
result = translator(
    original_text="helo wrld",
    target_language="French"
)
print(result.final_text)  # "Bonjour Le Monde"
```

Three LM calls happened: correct → translate → apply formatting.

---

### The Class Syntax (Equivalent, More Compact)

The class syntax is just a cleaner way to write the same thing:

```python
class TranslatorProgram(dspy.Module):

    def __init__(self):
        super().__init__()
        
        # Submodules go here (like Step 3)
        self.correct = dspy.Predict("original_text -> corrected_text")
        self.translate = dspy.Predict("corrected_text, target_language -> translated_text")
        self.apply_original_formatting = dspy.Predict(dspy.Signature(
            "original_text, translated_text -> final_text",
            instructions="""
                Apply the formatting style from the original text
                (capitalization, markdown, punctuation) to the translated text.
            """
        ))

    def forward(self, original_text: str, target_language: str):
        # Logic goes here (like Step 4)
        corrected = self.correct(original_text=original_text).corrected_text
        translated = self.translate(corrected_text=corrected, target_language=target_language).translated_text
        final = self.apply_original_formatting(original_text=original_text, translated_text=translated).final_text
        return dspy.Prediction(final_text=final)
```

The class gives you two containers:
- **`__init__`** — where you define *what* submodules exist
- **`forward`** — where you define *how* they connect

---

### Summary

- `self.correct`, `self.translate`, `self.apply_original_formatting` are **submodules**.
- `TranslatorProgram` (or `translator`) is the **program** (top-level DSPy object).
- If deployed with API + DB + logging, that whole deployment is the **system**.

---

## 4) Creating a Module That Modifies Signatures

DSPy's built-in modules like `ChainOfThought` work by *modifying* the signature you give them. Let's build a simple version ourselves.

### Example: AddConfidence

We'll create a module that adds a `confidence` field to any signature:

```python
class AddConfidence(dspy.Module):
    """Wraps any signature and adds a confidence score to the output."""

    def __init__(self, signature):
        super().__init__()
        
        # Take the user's signature and add a confidence field
        signature = dspy.ensure_signature(signature)
        extended_signature = signature.append(
            name="confidence",
            field=dspy.OutputField(),
            type_=float
        )
        
        # Use Predict with the extended signature
        self.predict = dspy.Predict(extended_signature)

    def forward(self, **kwargs):
        return self.predict(**kwargs)
```

### Usage

```python
# User provides a simple signature
qa_with_confidence = AddConfidence("question -> answer")

# Call it
result = qa_with_confidence(question="What is 2+2?")
print(result.answer)      # "4"
print(result.confidence)  # 0.95
```

### What Happened

1. User gave signature: `"question -> answer"`
2. `AddConfidence` transformed it to: `"question -> answer, confidence"`
3. The LM now returns both fields

This is exactly how `ChainOfThought` works — it prepends a `reasoning` field to your signature.

---

## 5) Add a Metric Function

```python
def translation_metric(example, pred, trace=None):
    # toy metric: exact match on normalized string
    gold = example.final_text.strip().lower()
    got = pred.final_text.strip().lower()
    return gold == got
```

This is a **function** (plain Python callable), not a module.

---

## 6) Optimize the Program

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=translation_metric)
optimized_program = optimizer.compile(
    student=TranslatorProgram(),
    trainset=trainset,
)
```

Now your **program** has improved prompts/demos according to your metric.

---

## 7) Language Discipline for the Rest of This Guide

When writing docs/code comments:

- Say **module** for units/components.
- Say **program** for end-to-end DSPy pipeline object.
- Say **system** for product/application context.
- Say **function** for plain Python callables.

Avoid using these interchangeably unless you explicitly explain scope.

---

## 8) What to Learn Next

1. Signatures (`dspy.Signature` and string signatures)
2. Built-in modules (`Predict`, `ChainOfThought`, `ReAct`)
3. Evaluation loops and metrics
4. Optimizers (`GEPA`, `MIPROv2`, `Bootstrap*`)
5. Production concerns (caching, observability, deployment)
