### ELE results by split

| Model | Core/Test acc | Core/Test 95% CI | Challenge acc | CF pair-success | CF brittle | errors |
|---|---|---|---|---|---|---|
| gpt-5.6-terra | 100.0% (11/11) | [74.1, 100.0] | 85.3% (133/156) | 83% | 17% | 0 |
| claude-opus-5 | 81.8% (9/11) | [52.3, 94.9] | 85.9% (134/156) | 50% | 33% | 0 |
| gemma-3-27b | 72.7% (8/11) | [43.4, 90.3] | 77.6% (121/156) | 33% | 33% | 0 |
| qwen3-235b | 81.8% (9/11) | [52.3, 94.9] | 84.0% (131/156) | 50% | 33% | 0 |
| llama4-maverick | 81.8% (9/11) | [52.3, 94.9] | 84.6% (132/156) | 50% | 50% | 0 |
| deepseek-v3.2 | 100.0% (11/11) | [74.1, 100.0] | 81.4% (127/156) | 33% | 50% | 0 |
| nova-micro | 72.7% (8/11) | [43.4, 90.3] | 75.0% (117/156) | 17% | 67% | 0 |

### Per-category accuracy (Core/Test + Challenge)

| Model | entity_resolution | precedent_exception | cross_system_synthesis | policy_version | approval_chain | temporal_consistency |
|---|---|---|---|---|---|---|
| gpt-5.6-terra | 94% | 83% | 94% | 85% | 90% | 68% |
| claude-opus-5 | 90% | 78% | 84% | 85% | 97% | 76% |
| gemma-3-27b | 84% | 67% | 74% | 73% | 83% | 80% |
| qwen3-235b | 94% | 78% | 81% | 79% | 93% | 76% |
| llama4-maverick | 97% | 78% | 77% | 82% | 86% | 84% |
| deepseek-v3.2 | 97% | 72% | 74% | 79% | 97% | 72% |
| nova-micro | 94% | 72% | 68% | 70% | 83% | 60% |

---

_Development run (not headline evidence). Models evaluated via AWS Bedrock; judge: `us.anthropic.claude-opus-4-6-v1`. Scenarios 006-011 and all counterfactual pairs are `ELE Editorial Team`-authored and pending independent expert validation per §4.3. Core/Test n=11 and CF n=6 are first tranches; only the Challenge split (n=156) is well-powered. Google is represented by open-weight Gemma 3 (no Gemini available in the Bedrock account at run time)._
