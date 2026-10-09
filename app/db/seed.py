"""Idempotent prompt-template seeder.

Run directly to seed a local database:
    python -m app.db.seed

Called automatically from app startup (lifespan) to ensure production
environments have all required templates on first boot.
"""
from __future__ import annotations

import logging
from typing import Final

from sqlalchemy.orm import Session

from app.models.prompt_template import PromptTemplate

logger = logging.getLogger(__name__)

# ── Template definitions ──────────────────────────────────────────────────────
#
# Each body uses Python str.format()-style substitution.
# Variables available at render time are shown in the leading comment.
# Literal JSON braces in the body MUST be doubled: {{ and }}.
#
# ─────────────────────────────────────────────────────────────────────────────

# Variables: {topic}, {curriculum_content}, {resource_guidance}
# resource_guidance is "" for standalone curricula (unchanged rendering) and
# a per-resource search-then-fetch / general-knowledge instruction block for
# curriculum-upload Assessment-type entries.
#
# Format (v2.0): Section 1 is 5 MCQs (30 pts, no partial credit, graded
# deterministically — see app.adapters.anthropic_llm._build_mcq_project_fields
# and GradingService._grade_mcq_project); Section 2 is one coding project
# (70 pts, criteria-based rubric). Replaces the prior
# coding-problem/written-question/mixed branching entirely — every
# tech-concept assessment now gets this same two-section shape.
_ASSESSMENT_GENERATION = """\
You are an expert technical assessment designer for a software engineering \
learning platform.

A student has completed a self-directed learning curriculum and needs to be \
assessed on their progress.

Topic: {topic}

Curriculum Materials (what the student studied):
{curriculum_content}

{resource_guidance}

Design a two-section assessment:

SECTION 1 — 5 Multiple-Choice Questions (30 points: 6 points each, no \
partial credit)
Each question must:
- Test understanding of the concept, not surface trivia — ask "why does X \
happen" or "what would happen if", not "what is the definition of X".
- Have exactly 4 options with exactly ONE correct answer.
- Have 3 genuinely plausible wrong answers (distractors a student who \
half-understands the concept could pick) — never an obviously-wrong \
throwaway option.
- Include an explanation of why the correct option is right and why each \
of the other 3 is wrong — this is NEVER shown to the student during the \
exam; it is used only for grading feedback afterward.
- Be answerable without writing any code — conceptual understanding only.

SECTION 2 — 1 Coding Project (70 points, broken into 4-6 checkable criteria)
The project must:
- Be built AROUND the concept being assessed, not merely use it in \
passing. If the topic is "Context and Lifting State", the project IS a \
state-sharing problem that can only be cleanly solved with those tools. \
If the topic is "Browser Internals", the project measures and \
demonstrates a reflow/repaint problem and fixes it. The assessed concept \
must be the reason the project exists, not an incidental detail of it.
- Be scoped to 60-90 minutes of focused work — one focused implementation \
proving understanding of one core idea, not a full application.
- Allow ordinary supporting concepts (other hooks, basic CSS, standard \
library calls, etc.) without those becoming the point of the exercise.
- Include a starter scaffold inline (as a fenced code block within \
project_text) where it reduces setup friction without giving away the \
solution — especially valuable for boilerplate-heavy environments (React, \
TypeScript). Omit it when the topic needs no scaffolding.
- Break project_criteria into 4-6 sub-criteria whose point values sum to \
EXACTLY 70. Each criterion's description must be a single, checkable fact \
about the submitted code — e.g. "The Context Provider wraps every \
consumer component, not just one of them" — never a vague quality \
judgment like "code quality is good".

Infer the project's technical shape (React, backend API, CLI tool, etc.) \
entirely from the curriculum materials above.

Respond with a single JSON object containing exactly these fields:
{{
  "mcqs": [
    {{
      "question": "Why does ... ?",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "correct_option": "B",
      "explanation": "Why B is correct, and specifically why A, C, and D are each wrong."
    }}
    // ... exactly 5 of these
  ],
  "project_text": "The full project prompt presented to the student, in markdown. State the problem, requirements, constraints, and (if useful) a starter scaffold as a fenced code block. Do NOT list point values here — those belong only in project_criteria.",
  "project_criteria": [
    {{"description": "One single, checkable fact about the submitted code.", "points": 15}}
    // ... 4-6 of these, points summing to exactly 70
  ],
  "duration_minutes": 90
}}

Rules:
- mcqs must contain exactly 5 entries, each with exactly 4 options and a \
correct_option of exactly "A", "B", "C", or "D" (matching the option's \
position: A=index 0, B=index 1, C=index 2, D=index 3).
- project_criteria must contain 4-6 entries whose "points" sum to exactly 70.
- duration_minutes covers only Section 2 (the project) — set it to 60-90 \
based on project complexity.
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {topic}, {curriculum_content}
_CURRICULUM_ANALYSIS = """\
You are an expert curriculum analyst for a software engineering learning platform.

Analyse the following learning materials and provide a structured summary.

Topic: {topic}

Curriculum Materials:
{curriculum_content}

Respond with a single JSON object containing exactly these fields:
{{
  "summary": "A concise 2–3 sentence summary of what this curriculum covers and what a student will be able to do after completing it.",
  "key_topics": ["topic1", "topic2", "topic3"],
  "complexity_level": "intermediate",
  "estimated_study_hours": 10.0
}}

Rules:
- complexity_level must be exactly one of: "beginner", "intermediate", "advanced"
- key_topics should list 4–8 specific technical concepts covered by the materials
- estimated_study_hours should reflect realistic self-study time (e.g. 5.0, 12.5, 20.0)
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {topic}, {curriculum_content}, {previous_mastery_score},
#            {weak_areas}, {attempt_number}, {resource_guidance}
# resource_guidance is "" for standalone curricula (unchanged rendering) and
# a per-resource search-then-fetch / general-knowledge instruction block for
# curriculum-upload Assessment-type entries retaking their exam.
#
# Format (v2.0): same two-section MCQ + Coding Project shape as
# _ASSESSMENT_GENERATION — see its comment for the rationale. A retest gets
# a fresh set of 5 MCQs and a fresh project, both weighted toward the
# previously-identified weak areas.
_RETEST_GENERATION = """\
You are an expert technical assessment designer for a software engineering \
learning platform.

A student is retaking an assessment. Create a targeted retest that focuses on \
their identified areas of weakness.

Topic: {topic}

Curriculum Materials:
{curriculum_content}

{resource_guidance}

Previous Attempt Results:
- Attempt number: {attempt_number}
- Previous mastery score: {previous_mastery_score}%
- Identified weak areas: {weak_areas}

Design a two-section retest, using DIFFERENT questions and a different \
project from any previous attempt:

SECTION 1 — 5 Multiple-Choice Questions (30 points: 6 points each, no \
partial credit)
Weight at least 3 of the 5 questions toward the identified weak areas; the \
remainder may confirm retained understanding of stronger areas. Same \
quality bar as a first attempt: test understanding (not trivia), exactly 4 \
genuinely plausible options, and a hidden per-option explanation never \
shown to the student during the exam.

SECTION 2 — 1 Coding Project (70 points, broken into 4-6 checkable criteria)
Build the project primarily (70%+) around the student's identified weak \
areas, calibrated to let an improved student demonstrate that improvement \
— not a harder version of the same exercise, a different exercise probing \
the same underlying concept. Same requirements as a first attempt: scoped \
to 60-90 minutes, checkable criteria summing to exactly 70, starter \
scaffold inline where it reduces setup friction.

Respond with a single JSON object containing exactly these fields:
{{
  "mcqs": [
    {{
      "question": "Why does ... ?",
      "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
      "correct_option": "B",
      "explanation": "Why B is correct, and specifically why A, C, and D are each wrong."
    }}
    // ... exactly 5 of these
  ],
  "project_text": "The full retest project prompt presented to the student, in markdown. Do NOT list point values here.",
  "project_criteria": [
    {{"description": "One single, checkable fact about the submitted code.", "points": 15}}
    // ... 4-6 of these, points summing to exactly 70
  ],
  "duration_minutes": 90
}}

Rules:
- mcqs must contain exactly 5 entries, each with exactly 4 options and a \
correct_option of exactly "A", "B", "C", or "D".
- project_criteria must contain 4-6 entries whose "points" sum to exactly 70.
- duration_minutes covers only Section 2, set based on the complexity of \
the weak areas (60-90 minutes).
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {topic}, {cumulative_pool_content}, {own_resources_list},
#            {resource_guidance}, {readme_content}, {probe_focus},
#            {part1_max_marks}, {part2_max_marks}
_MIDTERM_GENERATION = """\
You are an expert technical assessment designer for a software engineering \
learning platform, designing a two-part Midterm exam.

Project: {topic}

PART 1 — small coding/implementation questions ({part1_max_marks} marks)
Draw these from everything the student has studied and been assessed on up \
to this point — cumulative, not just this project's own material:
{cumulative_pool_content}

Part 1 should be SHORT, targeted questions (or a small coding exercise) that \
confirm retained understanding of that cumulative material. Do not require \
fetching those resources again — they were already covered and assessed \
earlier; draw on your knowledge of them.

PART 2 — probing the actual project submission ({part2_max_marks} marks)
This part is about the real decisions made in the project itself. Ground it \
in the project's own resources listed below:
{own_resources_list}

{resource_guidance}

Project README / design writeup on record for this project (submitted when \
this midterm's pending resources were filled in):
{readme_content}

If a README is present above, prefer probing the SPECIFIC claims and file/\
function references it makes — questions tied to concrete claims are easier \
to grade accurately later than generic ones. If no README is present, probe \
the project's real decisions using the other resources above instead.

Probe focus for Part 2: {probe_focus}

Part 2 should ask the student to defend specific decisions they made in the \
project — not generic questions about the technology in the abstract.

Respond with a single JSON object containing exactly these fields:
{{
  "part1_text": "The full Part 1 exam text presented to the student. Use markdown formatting, numbered questions.",
  "part1_rubric": "Marking rubric for Part 1 — per-question expected answers with full/partial/no marks, totalling {part1_max_marks}.",
  "part2_text": "The full Part 2 exam text presented to the student. Use markdown formatting, numbered questions tied to the probe focus.",
  "part2_rubric": "Marking rubric for Part 2 — per-question expected answers with full/partial/no marks, totalling {part2_max_marks}.",
  "duration_minutes": 120
}}

Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {topic}, {own_resources_list}, {resource_guidance},
#            {readme_content}, {probe_focus}, {part2_max_marks}
# Defense-only variant for a running-project checkpoint (upload's
# "running_project": true) — Part 2 (project defense) only, no cumulative
# Part 1. See MidtermDetail.defense_only.
_MIDTERM_GENERATION_DEFENSE_ONLY = """\
You are an expert technical assessment designer for a software engineering \
learning platform, designing one checkpoint of a running project — a \
project defense, not a two-part exam. There is no Part 1; grade only what \
follows.

Project: {topic}

Probe the real decisions made in the project itself so far. Ground it in \
the project's own resources listed below:
{own_resources_list}

{resource_guidance}

Project README / design writeup on record for this project (submitted when \
this checkpoint's pending resources were filled in):
{readme_content}

If a README is present above, prefer probing the SPECIFIC claims and file/\
function references it makes — questions tied to concrete claims are easier \
to grade accurately later than generic ones. If no README is present, probe \
the project's real decisions using the other resources above instead.

Probe focus: {probe_focus}

Ask the student to defend specific decisions they made in the project at \
this checkpoint — not generic questions about the technology in the \
abstract. This checkpoint is worth {part2_max_marks} marks in total.

Respond with a single JSON object containing exactly these fields:
{{
  "part2_text": "The full checkpoint defense text presented to the student. Use markdown formatting, numbered questions tied to the probe focus.",
  "part2_rubric": "Marking rubric — per-question expected answers with full/partial/no marks, totalling {part2_max_marks}.",
  "duration_minutes": 60
}}

Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {topic}, {cumulative_pool_content}, {own_resources_list},
#            {resource_guidance}, {readme_content}, {probe_focus},
#            {part1_max_marks}, {part2_max_marks}, {previous_part1_score},
#            {previous_part2_score}, {weak_areas}, {attempt_number}
_MIDTERM_RETEST_GENERATION = """\
You are an expert technical assessment designer for a software engineering \
learning platform, designing a RETAKE of a two-part Midterm exam.

Project: {topic}

The student did not pass on attempt {attempt_number}. Previous scores:
- Part 1: {previous_part1_score}/{part1_max_marks}
- Part 2: {previous_part2_score}/{part2_max_marks}
- Identified weak areas: {weak_areas}

PART 1 — small coding/implementation questions ({part1_max_marks} marks)
Draw these from everything the student has studied and been assessed on up \
to this point — cumulative, not just this project's own material:
{cumulative_pool_content}

Use different questions from the previous attempt, focused primarily on the \
identified weak areas, while still confirming retained understanding of the \
broader cumulative material.

PART 2 — probing the actual project submission ({part2_max_marks} marks)
This part is about the real decisions made in the project itself. Ground it \
in the project's own resources listed below:
{own_resources_list}

{resource_guidance}

Project README / design writeup on record for this project (submitted when \
this midterm's pending resources were filled in):
{readme_content}

If a README is present above, prefer probing the SPECIFIC claims and file/\
function references it makes — questions tied to concrete claims are easier \
to grade accurately later than generic ones. If no README is present, probe \
the project's real decisions using the other resources above instead.

Probe focus for Part 2: {probe_focus}

Ask about different aspects of the project than a first attempt would, \
weighted toward the identified weak areas, while still asking the student \
to defend specific real decisions — not generic questions about the \
technology in the abstract.

Respond with a single JSON object containing exactly these fields:
{{
  "part1_text": "The full Part 1 retest text presented to the student. Use markdown formatting, numbered questions.",
  "part1_rubric": "Marking rubric for Part 1 — per-question expected answers with full/partial/no marks, totalling {part1_max_marks}.",
  "part2_text": "The full Part 2 retest text presented to the student. Use markdown formatting, numbered questions tied to the probe focus.",
  "part2_rubric": "Marking rubric for Part 2 — per-question expected answers with full/partial/no marks, totalling {part2_max_marks}.",
  "duration_minutes": 120
}}

Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {topic}, {own_resources_list}, {resource_guidance},
#            {readme_content}, {probe_focus}, {part2_max_marks},
#            {previous_part2_score}, {weak_areas}, {attempt_number}
# Defense-only retest variant — see _MIDTERM_GENERATION_DEFENSE_ONLY.
_MIDTERM_RETEST_GENERATION_DEFENSE_ONLY = """\
You are an expert technical assessment designer for a software engineering \
learning platform, designing a RETAKE of one checkpoint of a running \
project — a project defense, not a two-part exam. There is no Part 1; \
grade only what follows.

Project: {topic}

The student did not pass on attempt {attempt_number}. Previous score: \
{previous_part2_score}/{part2_max_marks}. Identified weak areas: {weak_areas}.

Probe the real decisions made in the project itself so far. Ground it in \
the project's own resources listed below:
{own_resources_list}

{resource_guidance}

Project README / design writeup on record for this project (submitted when \
this checkpoint's pending resources were filled in):
{readme_content}

If a README is present above, prefer probing the SPECIFIC claims and file/\
function references it makes — questions tied to concrete claims are easier \
to grade accurately later than generic ones. If no README is present, probe \
the project's real decisions using the other resources above instead.

Probe focus: {probe_focus}

Ask about different aspects of the project than a first attempt would, \
weighted toward the identified weak areas, while still asking the student \
to defend specific real decisions — not generic questions about the \
technology in the abstract. This checkpoint is worth {part2_max_marks} \
marks in total.

Respond with a single JSON object containing exactly these fields:
{{
  "part2_text": "The full checkpoint retest text presented to the student. Use markdown formatting, numbered questions tied to the probe focus.",
  "part2_rubric": "Marking rubric — per-question expected answers with full/partial/no marks, totalling {part2_max_marks}.",
  "duration_minutes": 60
}}

Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {part1_text}, {part1_rubric}, {part2_text}, {part2_rubric},
#            {part1_max_marks}, {part2_max_marks}, {part1_submission_content},
#            {part2_submission_content}, {resource_guidance}
_MIDTERM_GRADING = """\
You are an expert technical assessor for a software engineering learning \
platform, grading a two-part Midterm exam.

PART 1 — cumulative knowledge check ({part1_max_marks} marks)

Exam:
{part1_text}

Rubric:
{part1_rubric}

Student's Part 1 answer:
{part1_submission_content}

PART 2 — project defense ({part2_max_marks} marks)

Exam:
{part2_text}

Rubric:
{part2_rubric}

Student's Part 2 submission (their real project):
{part2_submission_content}

{resource_guidance}

Project README / design writeup on record for this project (submitted when \
this midterm's pending resources were filled in):
{readme_content}

SPOT-CHECK — light, not a full code review: the README above may reference \
specific file paths and/or function names to support its design decisions. \
For each such reference, use the fetchable resources above (web_search/ \
web_fetch) to confirm whether that file/function actually exists and \
roughly matches what's claimed. You are not expected to read the whole \
codebase — only check the specific claims made. If a referenced path does \
not exist, or exists but clearly contradicts the claim, that is a real \
misrepresentation and must meaningfully lower part2_score, not just be \
noted as a footnote. If the references check out, that confirms the \
defense is grounded in the real project — it does not by itself earn \
extra credit beyond what the defense's actual content deserves.

Grade Part 2 by checking the student's claims and explanations against the \
actual project resources above — not only against the abstract rubric. A \
defense that sounds plausible but misrepresents what the project actually \
contains or does must NOT receive full credit for the claims that don't hold up.

Evaluate both parts carefully and provide an objective grade for each.

Respond with a single JSON object containing exactly these fields:
{{
  "part1_score": 27.0,
  "part2_score": 63.0,
  "weak_areas": ["specific concept 1", "specific concept 2"],
  "overall_feedback": "Detailed, constructive feedback for the student covering both parts."
}}

Rules:
- part1_score must be a number between 0.0 and {part1_max_marks}
- part2_score must be a number between 0.0 and {part2_max_marks}
- weak_areas lists 0–5 specific topics where the student showed gaps \
(use an empty list [] if they demonstrated strong mastery throughout)
- overall_feedback should be 2–4 sentences: acknowledge strengths, name \
specific gaps, and give one actionable improvement suggestion
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {part2_text}, {part2_rubric}, {part2_submission_content},
#            {part2_max_marks}, {resource_guidance}, {readme_content}
# Defense-only grading variant — grades the project defense alone, out of
# the FULL checkpoint max_marks (part2_max_marks == max_marks when
# defense_only; see CurriculumUploadService._create_entry). There is no
# Part 1 to grade.
_MIDTERM_GRADING_DEFENSE_ONLY = """\
You are an expert technical assessor for a software engineering learning \
platform, grading one checkpoint of a running project — a project defense, \
not a two-part exam. There is no Part 1; grade only what follows.

Checkpoint defense ({part2_max_marks} marks)

Exam:
{part2_text}

Rubric:
{part2_rubric}

Student's submission (their real project):
{part2_submission_content}

{resource_guidance}

Project README / design writeup on record for this project (submitted when \
this checkpoint's pending resources were filled in):
{readme_content}

SPOT-CHECK — light, not a full code review: the README above may reference \
specific file paths and/or function names to support its design decisions. \
For each such reference, use the fetchable resources above (web_search/ \
web_fetch) to confirm whether that file/function actually exists and \
roughly matches what's claimed. You are not expected to read the whole \
codebase — only check the specific claims made. If a referenced path does \
not exist, or exists but clearly contradicts the claim, that is a real \
misrepresentation and must meaningfully lower part2_score, not just be \
noted as a footnote. If the references check out, that confirms the \
defense is grounded in the real project — it does not by itself earn \
extra credit beyond what the defense's actual content deserves.

Grade by checking the student's claims and explanations against the actual \
project resources above — not only against the abstract rubric. A defense \
that sounds plausible but misrepresents what the project actually contains \
or does must NOT receive full credit for the claims that don't hold up.

Respond with a single JSON object containing exactly these fields:
{{
  "part2_score": 21.0,
  "weak_areas": ["specific concept 1", "specific concept 2"],
  "overall_feedback": "Detailed, constructive feedback for the student."
}}

Rules:
- part2_score must be a number between 0.0 and {part2_max_marks}
- weak_areas lists 0–5 specific topics where the student showed gaps \
(use an empty list [] if they demonstrated strong mastery throughout)
- overall_feedback should be 2–4 sentences: acknowledge strengths, name \
specific gaps, and give one actionable improvement suggestion
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {assessment_text}, {rubric}, {curriculum_content},
#            {submission_content}
_GRADING = """\
You are an expert technical assessor for a software engineering learning platform.

Grade the following student submission against the assessment and rubric provided.

Assessment:
{assessment_text}

Grading Rubric:
{rubric}

Curriculum Reference (for context):
{curriculum_content}

Student Submission:
{submission_content}

Evaluate the submission carefully and provide an objective grade.

Respond with a single JSON object containing exactly these fields:
{{
  "mastery_score": 75.0,
  "weak_areas": ["specific concept 1", "specific concept 2"],
  "overall_feedback": "Detailed, constructive feedback for the student."
}}

Rules:
- mastery_score must be a number between 0.0 and 100.0
- weak_areas lists 0–5 specific topics where the student showed gaps \
(use an empty list [] if they demonstrated strong mastery throughout)
- overall_feedback should be 2–4 sentences: acknowledge strengths, name \
specific gaps, and give one actionable improvement suggestion
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {questions_json}, {answer_key_json}, {submitted_answers_json}
# Used by GradingService._grade_mcq_project for the new MCQ + Coding
# Project assessment format. Scoring is NOT this template's job — the
# correct/incorrect decision is made deterministically in Python by
# comparing parsed_answers against the stored correct_option. This call's
# only job is: (1) normalize each submitted answer to a canonical option
# letter (handles a student answering with free text instead of a clean
# letter), and (2) write one feedback line per question.
_MCQ_GRADING = """\
You are grading the multiple-choice section of a technical assessment.

Questions and options (in order):
{questions_json}

Answer key (in the same order — NEVER reveal this to the student directly, \
only use it to write feedback):
{answer_key_json}

Student's submitted answers (in the same order — each is whatever the \
student typed or selected; it may already be a clean option letter, or it \
may be free text referring to one of the options):
{submitted_answers_json}

For each question, in order:
1. Determine which option (A, B, C, or D) the student's submitted answer \
refers to. If they submitted a letter already, use it directly. If they \
submitted free text, match it to the option it clearly refers to.
2. Write one feedback sentence: if they matched the correct option, a \
short confirmation; if not, state the correct option and adapt the answer \
key's explanation into feedback for the student.

Respond with a single JSON object containing exactly these fields:
{{
  "parsed_answers": ["B", "A", "D", "C", "B"],
  "feedback": [
    "Correct — ...",
    "Not quite — the correct answer is A because ...",
    "..."
  ]
}}

Rules:
- parsed_answers must contain exactly 5 entries, each exactly "A", "B", "C", or "D".
- feedback must contain exactly 5 entries, in the same order as the questions.
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {project_text}, {project_criteria_json}, {curriculum_content},
#            {submission_content}
# Used by GradingService._grade_mcq_project for the new MCQ + Coding
# Project assessment format's Section 2. Scored directly in the criteria's
# own point space (summing to 70) rather than as a 0-100 mastery_score —
# GradingService combines this with the MCQ section's score (both already
# in the same point space) without renormalizing either one.
_PROJECT_GRADING = """\
You are an expert technical assessor for a software engineering learning \
platform, grading the coding-project section of an assessment.

Project Prompt:
{project_text}

Grading Criteria (each is a single checkable fact about the code — award \
points for a criterion only if the submitted code actually satisfies it):
{project_criteria_json}

Curriculum Reference (for context):
{curriculum_content}

Student Submission:
{submission_content}

Evaluate the submission against each criterion independently and sum the \
points earned.

Respond with a single JSON object containing exactly these fields:
{{
  "project_score": 58.0,
  "weak_areas": ["specific concept 1", "specific concept 2"],
  "overall_feedback": "Detailed, constructive feedback for the student, referencing which criteria were and weren't met."
}}

Rules:
- project_score must be a number between 0.0 and the sum of the given \
criteria's points.
- weak_areas lists 0-5 specific topics where the student showed gaps \
(use an empty list [] if they demonstrated strong mastery throughout).
- overall_feedback should be 2-4 sentences: acknowledge strengths, name \
specific unmet criteria, and give one actionable improvement suggestion.
Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# Variables: {reason}
_RESCHEDULE_CLASSIFICATION = """\
You are an assessment coordinator evaluating a student's request to reschedule \
their technical assessment.

Student's reason for rescheduling:
{reason}

Classify the reason into exactly one of the following categories:
- interview          — student has a job/internship interview
- medical            — illness, medical appointment, or health emergency
- emergency          — family emergency, bereavement, or other acute personal crisis
- work_escalation    — urgent work deadline, on-call incident, or business-critical task
- procrastination    — student is not ready, wants more preparation time without a specific reason
- lack_of_preparation — student explicitly states they have not studied enough
- missed_schedule    — student simply forgot or missed the scheduled time

Respond with a single JSON object containing exactly these fields:
{{
  "category": "medical",
  "reasoning": "One sentence explaining why this reason maps to the chosen category."
}}

Return ONLY the JSON object. Do not include any other text before or after it.\
"""

# ── Public constants ──────────────────────────────────────────────────────────

# Maps slug → (version, body). Version is bumped when the prompt changes
# in a way that meaningfully affects LLM behaviour.
SEED_TEMPLATES: Final[dict[str, tuple[str, str]]] = {
    "assessment_generation":     ("2.0", _ASSESSMENT_GENERATION),
    "curriculum_analysis":       ("1.0", _CURRICULUM_ANALYSIS),
    "retest_generation":         ("2.0", _RETEST_GENERATION),
    "midterm_generation":        ("1.1", _MIDTERM_GENERATION),
    "midterm_generation_defense_only": ("1.0", _MIDTERM_GENERATION_DEFENSE_ONLY),
    "midterm_retest_generation": ("1.1", _MIDTERM_RETEST_GENERATION),
    "midterm_retest_generation_defense_only": ("1.0", _MIDTERM_RETEST_GENERATION_DEFENSE_ONLY),
    "grading":                   ("1.0", _GRADING),
    "midterm_grading":           ("1.1", _MIDTERM_GRADING),
    "midterm_grading_defense_only": ("1.0", _MIDTERM_GRADING_DEFENSE_ONLY),
    "mcq_grading":                ("1.0", _MCQ_GRADING),
    "project_grading":            ("1.0", _PROJECT_GRADING),
    "reschedule_classification": ("1.0", _RESCHEDULE_CLASSIFICATION),
}

# The subset whose absence will block core user-facing functionality.
REQUIRED_SLUGS: Final[frozenset[str]] = frozenset({
    "assessment_generation",
    "curriculum_analysis",
    "retest_generation",
    "grading",
    "reschedule_classification",
    # Midterms are the highest-stakes entries in a curriculum upload (100
    # marks each, including the capstone) — without these, generation or
    # grading fails outright for every one of them, with no other signal.
    "midterm_generation",
    "midterm_grading",
    # A failing first attempt silently strands the student with no retest
    # ever generated if this is missing — same severity as the first-
    # attempt template above, just later in the lifecycle, so it belongs
    # here too rather than being treated as best-effort.
    "midterm_retest_generation",
    # Same reasoning, for the subset of Midterms that are running-project
    # (defense_only) checkpoints — see MidtermDetail.defense_only.
    "midterm_generation_defense_only",
    "midterm_grading_defense_only",
    "midterm_retest_generation_defense_only",
    # assessment_generation/retest_generation now always produce the MCQ +
    # Coding Project format (v2.0) — grading every such assessment fails
    # outright without these two.
    "mcq_grading",
    "project_grading",
})


# ── Seeder ────────────────────────────────────────────────────────────────────

def seed_prompt_templates(db: Session) -> list[str]:
    """Idempotently insert or update prompt templates.

    - Missing templates are inserted.
    - Existing templates whose version differs from SEED_TEMPLATES are updated
      in-place (body + version). This lets a version bump propagate to existing
      DBs by re-running `python -m app.db.seed`.
    - Templates already at the current version are left untouched.

    Returns the list of slugs that were inserted or updated.
    """
    changed: list[str] = []
    for slug, (version, body) in SEED_TEMPLATES.items():
        exists = (
            db.query(PromptTemplate)
            .filter(PromptTemplate.slug == slug, PromptTemplate.is_active.is_(True))
            .first()
        )
        if exists is None:
            db.add(PromptTemplate(slug=slug, version=version, body=body, is_active=True))
            changed.append(slug)
        elif exists.version != version:
            exists.body = body
            exists.version = version
            changed.append(slug)

    if changed:
        db.commit()
        logger.info("Seeded/updated %d prompt template(s): %s", len(changed), changed)
    return changed


def check_missing_templates(db: Session) -> list[str]:
    """Return the list of REQUIRED_SLUGS that have no active template row."""
    return [
        slug for slug in sorted(REQUIRED_SLUGS)
        if not db.query(PromptTemplate)
        .filter(PromptTemplate.slug == slug, PromptTemplate.is_active.is_(True))
        .first()
    ]


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        inserted = seed_prompt_templates(db)
        missing = check_missing_templates(db)
        if inserted:
            print(f"Seeded {len(inserted)} template(s): {inserted}")
        else:
            print("All required templates already present — nothing to do.")
        if missing:
            print(f"WARNING: still missing after seed: {missing}")
    finally:
        db.close()
