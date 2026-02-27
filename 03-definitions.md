---
title: "Definitions"
---


In this guide, these words have strict meanings:

| Term | DSPy meaning | When to use it |
|---{15%}|---|---|
| **Module** | A reusable DSPy unit (`dspy.Module`) like `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, or your own class. | When talking about one component. |
| **Program** | A composed DSPy pipeline (usually a top-level `dspy.Module`) that solves a full task and can be evaluated/optimized/saved. | When talking about the end-to-end DSPy object. |
| **System** | The broader real-world app around your program (APIs, retrieval, DBs, monitoring, UI, etc.). | When talking about production/application context. |
| **Function** | A plain Python callable (metric function, tool function, helper). | When talking about regular Python logic. |

### Rule of thumb

- **Module = building block**
- **Program = assembled DSPy solution**
- **System = whole application context**
- **Function = plain Python callable**

> Note: A DSPy program is often implemented as a `dspy.Module`.  
> So implementation-wise they overlap, but conceptually the scope is different.

---

## 1) Core DSPy Primitives

You only need a few concepts to start:

1. **Signature**: input/output contract (`"question -> answer"`).
2. **Module**: strategy that executes a signature (e.g., `dspy.Predict`, `dspy.ChainOfThought`).
3. **Program**: composition of modules for your task.
4. **Metric function**: scores program outputs.
5. **Optimizer**: improves your program using data + metric.

