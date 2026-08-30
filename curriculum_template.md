# General Curriculum Template — Reusable for Any Future Upload

*Two genuinely different entry types, not variations of one. Assessment = ongoing, one per 
concept, scheduled off that concept's own completion date. Midterm = tied to a project's due 
date, and its Part 1 covers everything completed before that date, cumulatively.*

## Required fields, per entry

| Field | Assessment | Midterm |
|---|---|---|
| `topic` | Name of the concept | Name of the project |
| `type` | `"assessment"` | `"midterm"` |
| `chapter` | Which curriculum chapter this belongs under | (usually spans multiple chapters — see Part 1 rule below) |
| `resources` | Links/articles/repos — fed directly to the exam generator, pulled exactly from your Study Checklist | README + repo URL + design-decision doc |
| `completion_date` | When you finish studying it | When the project is done |
| `max_marks` | 50 | 100 (also the natural GPA weight — see below) |

**Exam timing rule, both types:** exam window = `completion_date + 1 day` through `completion_date + 3 days`.

**Midterm Part 1 rule:** small coding/implementation questions drawn from every Assessment 
whose `completion_date` falls on or before this Midterm's `completion_date` — i.e., everything 
done up to that point, not just the topics in this midterm's own chapter.

**Midterm Part 2 rule:** questions probing the actual project submission — real decisions, 
using this entry's own resources.

## GPA

Weighted by `max_marks` — no separate credits field needed, since marks already reflect real 
scope (a Midterm is naturally worth double an Assessment). `GPA = Σ(score earned) / Σ(max_marks) × 100`.

## Optional fields

`term`, `prerequisites`, `probe_focus` — include if useful, not required.
