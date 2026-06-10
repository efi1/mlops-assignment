"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """You are an expert data analyst who writes SQLite queries.
Given a database schema and a question, write a single SQL query that answers it.
Return ONLY the SQL query inside a ```sql code block. No explanation."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Database schema:
{schema}

Question: {question}

Write a single SQLite query that answers the question."""


VERIFY_SYSTEM = """You are a meticulous reviewer of SQL query results.
Given a question, the SQL that was run, and the result it produced, decide whether
the result plausibly answers the question.

Flag a result as NOT ok when:
- the SQL produced an execution error
- the result is empty AND the question clearly implies rows should exist
- the returned columns clearly do not answer what was asked

Do NOT flag a result merely for being empty if an empty answer is plausible
for the question.

Respond with ONLY a JSON object, no prose, no code fences:
{"ok": true or false, "issue": "short description, or empty string if ok"}"""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """Question: {question}

SQL that was run:
{sql}

Result:
{result}

Is this result plausible? Respond with the JSON object only."""


REVISE_SYSTEM = """You are an expert data analyst fixing a broken SQLite query.
You are given the question, the previous SQL attempt, the result it produced,
and a reviewer's complaint. Produce a corrected single SQL query.
Return ONLY the SQL query inside a ```sql code block. No explanation."""

# Available placeholders: {schema}, {question}, {sql}, {result}, {issue}
REVISE_USER = """Database schema:
{schema}

Question: {question}

Previous SQL attempt:
{sql}

Result it produced:
{result}

Reviewer's complaint: {issue}

Write a corrected SQLite query that addresses the complaint."""